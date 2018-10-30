## Imports

# Numpy Backend
import numpy  # numpy has to be present

# Torch Backends (and flags)
try:
    import torch

    torch.set_default_dtype(torch.float64)  # we need more precision for FDTD
    torch._C.set_grad_enabled(False)  # we don't need gradients (for now)
    torch_available = True
    torch_cuda_available = torch.cuda.is_available()
except ImportError:
    torch_available = False
    torch_cuda_available = False


## Typing
from typing import Union

if torch_available:
    Tensorlike = Union[numpy.ndarray, torch.Tensor]
else:
    Tensorlike = numpy.ndarray


## Backends

# Base Class
class Backend:
    """ Backend Base Class """

    def __repr__(self):
        return self.__class__.__name__


# Numpy Backend
class NumpyBackend(Backend):
    """ Numpy Backend """

    import numpy

    ones = staticmethod(numpy.ones)
    zeros = staticmethod(numpy.zeros)
    stack = staticmethod(numpy.stack)
    transpose = staticmethod(numpy.transpose)


# Torch Backend
if torch_available:

    class TorchBackend(Backend):
        """ Torch Backend """

        import torch

        ones = staticmethod(torch.ones)
        zeros = staticmethod(torch.zeros)
        stack = staticmethod(torch.stack)

        @staticmethod
        def transpose(tensor, axes=None):
            if axes is None:
                axes = tuple(range(len(tensor.shape)-1, -1, -1))
            return tensor.permute(*axes)


# Torch Cuda Backend
if torch_cuda_available:

    class TorchCudaBackend(TorchBackend):
        """ Torch Cuda Backend """

        def ones(self, shape):
            return self.torch.ones(shape, device="cuda")

        def zeros(self, shape):
            return self.torch.zeros(shape, device="cuda")


## Default Backend
# this backend object will be used for all array/tensor operations.
# the backend is changed by changing the class of the backend
# using the "set_backend" function. This "monkeypatch" will replace all the methods
# of the backend object by the methods supplied by the new class.
backend = NumpyBackend()


## Set backend
def set_backend(name: str):
    """ Set the backend for the FDTD simulations

    Args:
        name: name of the backend. Allowed backends: "numpy", "torch", "torch.cuda".
    """
    # this method monkeypatches the backend object and changes its class.
    if name == "torch":
        if not torch_available:
            raise RuntimeError("Torch backend is not available. Is PyTorch installed?")
        backend.__class__ = TorchBackend
    elif name == "torch.cuda":
        if not torch_cuda_available:
            raise RuntimeError(
                "Torch cuda backend is not available. "
                "Is PyTorch with cuda support installed?"
            )
        backend.__class__ = TorchCudaBackend
    elif name == "numpy":
        backend.__class__ = NumpyBackend
    else:
        raise RuntimeError(f"unknown backend {name}")
