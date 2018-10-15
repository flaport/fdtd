## Imports

import numpy


## Backends

class Backend:
    import numpy

    def __repr__(self):
        return self.__class__.__name__


class NumpyBackend(Backend):
    ones = staticmethod(numpy.ones)
    zeros = staticmethod(numpy.zeros)


class TorchBackend(Backend):
    try:
        import torch

        torch.set_default_dtype(torch.float64)
    except ImportError:
        pass
    else:
        ones = staticmethod(torch.ones)
        zeros = staticmethod(torch.zeros)


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
        import torch

        backend.__class__ = TorchBackend
    elif name == "torch.cuda":
        import torch

        if not torch.cuda.is_available():
            raise ValueError("CUDA not available")
        backend.__class__ = TorchCudaBackend
    elif name == "numpy":
        backend.__class__ = NumpyBackend
    else:
        raise RuntimeError(f"unknown backend {name}")
