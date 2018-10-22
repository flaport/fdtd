## Imports

from typing import Union
import numpy # numpy has to be present
try:
    import torch
    torch.set_default_dtype(torch.float64)
    torch_available = True
    torch_cuda_available = torch.cuda.is_available()
except ImportError:
    torch_available = False
    torch_cuda_available = False


## Typing

if torch_available:
    TensorLike = Union[numpy.ndarray, torch.Tensor]
else:
    TensorLike = numpy.ndarray


## Backends

class Backend:
    def __repr__(self):
        return self.__class__.__name__


class NumpyBackend(Backend):
    import numpy
    ones = staticmethod(numpy.ones)
    zeros = staticmethod(numpy.zeros)


if torch_available:
    class TorchBackend(Backend):
        import torch
        ones = staticmethod(torch.ones)
        zeros = staticmethod(torch.zeros)


if torch_cuda_available:
    class TorchCudaBackend(TorchBackend):
        def ones(self, shape):
            return self.torch.ones(shape, device="cuda")

        def zeros(self, shape):
            return self.torch.zeros(shape, device="cuda")


## Default Backend

backend = NumpyBackend()


## Set backend

def set_backend(name):
    if name == "torch":
        if not torch_available:
            raise RuntimeError("Torch backend is not available. Is PyTorch installed?")
        backend.__class__ = TorchBackend
    elif name == "torch.cuda":
        if not torch_cuda_available:
            raise RuntimeError("Torch cuda backend is not available. "
                               "Is PyTorch with cuda support installed?")
        backend.__class__ = TorchCudaBackend
    elif name == "numpy":
        backend.__class__ = NumpyBackend
    else:
        raise RuntimeError(f"unknown backend {name}")
