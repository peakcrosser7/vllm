from typing import Any, Dict, Optional, Union

import torch
import torch.distributed

from .parallel_state import get_tp_group


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(input_: torch.Tensor,
                                     dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_gather(input_: torch.Tensor,
                                 dst: int = 0,
                                 dim: int = -1) -> Optional[torch.Tensor]:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(tensor_dict: Optional[Dict[Any, Union[torch.Tensor,
                                                                Any]]] = None,
                          src: int = 0) -> Optional[Dict[str, Union[torch.Tensor, Any]]]:
    """广播发送或接收带张量的字典

    Args:
        tensor_dict (Optional[Dict[Any, Union[torch.Tensor, Any]]], optional): 待发送或接收的可能带有张量的字典. Defaults to None.
        src (int, optional): 广播源的本地rank号. Defaults to 0.

    Returns:
        Optional[Dict[str, Union[torch.Tensor, Any]]]: 发送或接收到的可能带有张量的字典
    """
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
