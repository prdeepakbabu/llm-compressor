from abc import abstractmethod
from typing import Dict, Iterable, List, Optional, Tuple, TypedDict
from weakref import ref

import torch
from compressed_tensors import InternalModule
from compressed_tensors.offload.dist_utils import as_broadcastable
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam
from compressed_tensors.registry.registry import RegistryMixin
from compressed_tensors.utils import align_module_device
from torch import distributed as dist

from llmcompressor.observers.helpers import flatten_for_calibration

__all__ = ["Observer", "MinMaxTuple", "QParamsDict"]

MinMaxTuple = Tuple[torch.Tensor, torch.Tensor]


class QParamsDict(TypedDict, total=False):
    """Dictionary containing quantization parameters."""

    scale: torch.Tensor
    zero_point: torch.Tensor
    global_scale: Optional[torch.Tensor]


class Observer(InternalModule, RegistryMixin):
    """
    Base class for observers which compute quantization parameters given observerations
    of weights, activations, or attention states.

    Example:
    ```python
    module = ...
    observer = Observer.load_from_registry(
        observer, base_name="weight", args=..., module=module
    )
    # Weight observers auto-use module.weight when called with no args:
    qparams = observer().get_qparams()
    # Or pass a value explicitly:
    qparams = observer(some_tensor).get_qparams()
    ```

    :param base_name: str used to name the observer attribute
    :param args: quantization args used to calibrate and quantize the observed value
    :param module: optional module with attached quantization parameters. This argument
        is required to utilize existing qparams such as global_scale or g_idx
    :param **observer_kwargs: keyword arguments for observer initialization
    """

    # Dict of statistic attribute names to reduce operations for DDP synchronization
    # Subclasses should override this to specify which attributes to sync
    # e.g., {"min_vals": dist.ReduceOp.MIN, "max_vals": dist.ReduceOp.MAX}
    _sync_dict: Dict[str, dist.ReduceOp] = {}

    def __init__(
        self,
        base_name: str,
        args: QuantizationArgs,
        module: Optional[torch.nn.Module] = None,
        **observer_kwargs,
    ):
        super().__init__()
        self.module = ref(module) if module is not None else None
        self.base_name = base_name
        self.args = args

        # populate observer kwargs
        self.args.observer_kwargs = self.args.observer_kwargs or {}
        self.args.observer_kwargs.update(observer_kwargs)

        # Observers in fused groups (e.g. Q/K/V, gate/up) for shared global_scale
        self._fused_observers: list["Observer"] = []

        # Idempotency tracking: skip update_statistics if called with the same tensor
        # needed to avoid n^2 update calls for fused observers
        self._last_observed_id: int | None = None
        self._last_observed_version: int | None = None

    @property
    def has_statistics(self) -> bool:
        """Whether this observer has accumulated statistics (has been observed)."""
        return hasattr(self, "min_vals") and hasattr(self, "max_vals")

    @abstractmethod
    def update_statistics(self, observed: torch.Tensor) -> None:
        """
        Update internal observer statistics from observed tensor.
        This method should update the observer's statistic attributes
        (e.g., self.min_vals, self.max_vals).

        :param observed: flattened observed value of shape
                        (num_observations, *qparam_shape, group_size)
        """
        raise NotImplementedError()

    def compute_qparams_from_statistics(self) -> QParamsDict:
        """
        Compute all quantization parameters from accumulated internal statistics.

        Default implementation assumes min_vals and max_vals attributes exist.
        Computes scale, zero_point, and global_scale (if TENSOR_GROUP strategy).
        For non-TENSOR_GROUP strategies, global_scale should be None.

        For TENSOR_GROUP, global_scale is computed from the combined statistics
        of this observer and all fused observers (linked via fuse_with()).

        Subclasses can override if they need custom logic.

        :return: dict with keys "scale", "zero_point", and "global_scale"
        """
        if not self.has_statistics:
            raise RuntimeError("No statistics available. Call observer(value) first.")

        global_scale = None
        if self.args.strategy == QuantizationStrategy.TENSOR_GROUP:
            # Compute absmax across this observer and all fused observers
            global_absmax = torch.max(-self.min_vals.min(), self.max_vals.max())
            for obs in self._fused_observers:
                obs()
                absmax = torch.max(-obs.min_vals.min(), obs.max_vals.max())
                global_absmax = torch.max(global_absmax, absmax).reshape(1)
            global_scale = generate_gparam(-global_absmax, global_absmax)

        # Compute scale and zero_point using global_scale
        scale, zero_point = calculate_qparams(
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            quantization_args=self.args,
            global_scale=global_scale,
        )

        return {"scale": scale, "zero_point": zero_point, "global_scale": global_scale}

    @torch.no_grad
    def get_qparams(self) -> QParamsDict:
        """
        Compute quantization parameters from accumulated statistics.

        If this observer hasn't been observed yet, triggers observation
        automatically (lazy). Weight observers auto-use module.weight.
        Fused partner observation is handled by compute_qparams_from_statistics
        with idempotency making repeated calls free.

        :return: dict with keys "scale", "zero_point", and "global_scale"
        """
        self()  # mostly for weight observers
        return self.compute_qparams_from_statistics()

    @torch.no_grad
    def forward(self, observed: Optional[torch.Tensor] = None) -> "Observer":
        """
        Update observer statistics from observed value.

        If no value is provided and this is a weight observer, automatically
        uses the attached module's weight tensor.

        Idempotent: if called again with the same tensor (same id + _version),
        skips the update. This makes repeated observer() calls free.

        To get quantization parameters, call get_qparams() after this method.
        Can be chained: observer(value).get_qparams()

        :param observed: value being observed. If None and base_name is "weight",
            uses module.weight from the attached module
        :return: self for method chaining
        """
        # allow observer() on weight observers
        if observed is None and self.base_name == "weight" and self.module is not None:
            observed = self._get_module_param("weight")

        # Idempotency: skip if this exact tensor (same id + version) was
        # already observed. This makes repeated observer() calls free.
        if (
            observed is None
            or observed.numel() == 0
            or self._already_observed(observed)
        ):
            return

        g_idx = self._get_module_param(f"{self.base_name}_g_idx")
        observed = flatten_for_calibration(observed, self.base_name, self.args, g_idx)
        self.update_statistics(observed)
        return self

    def _get_module_param(self, name: str) -> Optional[torch.nn.Parameter]:
        if self.module is None or (module := self.module()) is None:
            return None

        with align_module_device(module):
            return getattr(module, f"{name}", None)

    def _already_observed(self, observed: torch.Tensor) -> bool:
        obs_id = id(observed)
        obs_ver = observed._version
        already_observed = (
            obs_id == self._last_observed_id and obs_ver == self._last_observed_version
        )
        if not already_observed:
            self._last_observed_id = obs_id
            self._last_observed_version = obs_ver
        return already_observed

    @staticmethod
    def fuse(observers: Iterable["Observer"]) -> None:
        """
        Link all observers in the list with each other for shared global_scale.

        :param observers: list of observers to fuse together
        """
        for obs in observers:
            for other in observers:
                if other is not obs:
                    obs._fused_observers.append(other)

    def synchronize_statistics(self) -> List[dist.Work]:
        """All-reduce accumulated statistics across DDP ranks.

        Issues async all-reduce operations on statistic attributes specified in
        _sync_dict. Each attribute is reduced using its specified operation.

        :return: list of async communication handles
        """
        comms = []
        for attr_name, reduce_op in self._sync_dict.items():
            val = getattr(self, attr_name, None)
            if val is not None:
                comms.append(
                    dist.all_reduce(as_broadcastable(val), op=reduce_op, async_op=True)
                )
        return comms

    def attach(self, module: torch.nn.Module) -> None:
        """
        Called when the observer is attached to a module.
        Subclasses can override to register hooks or initialize state.

        :param module: the module this observer is being attached to
        """
        pass

    def detach(self, module: torch.nn.Module) -> None:
        """
        Called before the observer is deleted from a module.
        Subclasses can override to remove hooks and clean up module attributes.

        :param module: the module this observer is being removed from
        """
        pass
