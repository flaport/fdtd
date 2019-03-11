""" Backend for FDTD

Three backends are available:

 - numpy [default]
 - torch
 - torch.cuda

A backend can be set with the `set_backend` function:

```
    import fdtd
    fdtd.set_backend("torch")
```

"""

## Imports

# Numpy Backend
import numpy  # numpy has to be present

# Torch Backends (and flags)
try:
    import torch

    torch.set_default_dtype(torch.float64)  # we need more precision for FDTD
    torch._C.set_grad_enabled(False)  # we don't need gradients (for now)
    TORCH_AVAILABLE = True
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_CUDA_AVAILABLE = False


# Base Class
class Backend:
    """ Backend Base Class """

    # constants
    pi = numpy.pi

    def __repr__(self):
        return self.__class__.__name__


# Numpy Backend
class NumpyBackend(Backend):
    """ Numpy Backend """

    # types
    int = numpy.int64
    float = numpy.float64

    # methods
    exp = staticmethod(numpy.exp)
    sin = staticmethod(numpy.sin)
    cos = staticmethod(numpy.cos)
    sum = staticmethod(numpy.sum)
    stack = staticmethod(numpy.stack)
    transpose = staticmethod(numpy.transpose)
    reshape = staticmethod(numpy.reshape)
    squeeze = staticmethod(numpy.squeeze)

    @staticmethod
    def bmm(arr1, arr2):
        """ batch matrix multiply two arrays """
        return numpy.einsum("ijk,ikl->ijl", arr1, arr2)

    def is_array(self, arr):
        """ check if an object is an array """
        return isinstance(arr, numpy.ndarray)

    # constructors
    array = staticmethod(numpy.array)
    ones = staticmethod(numpy.ones)
    zeros = staticmethod(numpy.zeros)
    linspace = staticmethod(numpy.linspace)
    arange = staticmethod(numpy.arange)
    numpy = staticmethod(numpy.asarray)


# Torch Backend
if TORCH_AVAILABLE:

    class TorchBackend(Backend):
        """ Torch Backend """

        # types
        int = torch.int64
        float = torch.get_default_dtype()

        # methods
        exp = staticmethod(torch.exp)
        sin = staticmethod(torch.sin)
        cos = staticmethod(torch.cos)
        sum = staticmethod(torch.sum)
        stack = staticmethod(torch.stack)
        squeeze = staticmethod(torch.squeeze)
        reshape = staticmethod(torch.reshape)
        bmm = staticmethod(torch.bmm)

        @staticmethod
        def transpose(arr, axes=None):
            """ transpose an array """
            if axes is None:
                axes = tuple(range(len(arr.shape) - 1, -1, -1))
            return arr.permute(*axes)

        def is_array(self, arr):
            """ check if an object is an array """
            return torch.is_tensor(arr)

        # constructors
        ones = staticmethod(torch.ones)
        zeros = staticmethod(torch.zeros)
        arange = staticmethod(torch.arange)

        def array(self, arr, dtype=None):
            """ make an array """
            if dtype is None:
                dtype = torch.get_default_dtype()
            return torch.tensor(arr, device="cpu", dtype=dtype)

        def numpy(self, arr):
            """ convert an array to a numpy.ndarray """
            if torch.is_tensor(arr):
                return arr.numpy()
            else:
                return numpy.asarray(arr)

        def linspace(self, start, stop, num=50, endpoint=True):
            """ create a linearly spaced array between two points """
            delta = (stop - start) / float(num - float(endpoint))
            if not delta:
                return self.array([start] * num)
            return torch.arange(start, stop + 0.5 * float(endpoint) * delta, delta)


# Torch Cuda Backend
if TORCH_CUDA_AVAILABLE:

    class TorchCudaBackend(TorchBackend):
        """ Torch Cuda Backend """

        def ones(self, shape):
            """ create an array consisting of ones """
            return torch.ones(shape, device="cuda")

        def zeros(self, shape):
            """ create an array consisting of zeros """
            return torch.zeros(shape, device="cuda")

        def array(self, arr, dtype=None):
            if dtype is None:
                dtype = torch.get_default_dtype()
            return torch.tensor(arr, device="cuda", dtype=dtype)

        def numpy(self, arr):
            if torch.is_tensor(arr):
                return arr.cpu().numpy()
            else:
                return numpy.asarray(arr)

        def linspace(self, start, stop, num=50, endpoint=True):
            delta = (stop - start) / float(num - float(endpoint))
            if not delta:
                return self.array([start] * num)
            return torch.arange(
                start, stop + 0.5 * float(endpoint) * delta, delta, device="cuda"
            )


## Default Backend
# this backend object will be used for all array/tensor operations.
# the backend is changed by changing the class of the backend
# using the "set_backend" function. This "monkeypatch" will replace all the methods
# of the backend object by the methods supplied by the new class.
backend = NumpyBackend()


## Set backend
def set_backend(name: str):
    """ Set the backend for the FDTD simulations

    This function monkeypatches the backend object by changing its class.
    This way, all methods of the backend object will be replaced.

    Args:
        name: name of the backend. Allowed backend names:
            - "numpy" (defaults to float64 arrays)
            - "torch" (defaults to float64 tensors)
            - "torch.float32"
            - "torch.float64"
            - "torch.cuda" (defaults to float64 tensors)
            - "torch.cuda.float32"
            - "torch.cuda.float64"
    """
    # perform checks
    if "torch" in name and not TORCH_AVAILABLE:
        raise RuntimeError("Torch backend is not available. Is PyTorch installed?")
    if "torch.cuda" in name and not TORCH_CUDA_AVAILABLE:
        raise RuntimeError(
            "Torch cuda backend is not available.\n"
            "Do you have a GPU on your computer?\n"
            "Is PyTorch with cuda support installed?"
        )

    # change backend by monkeypatching
    if name == "numpy":
        backend.__class__ = NumpyBackend
    elif name == "torch" or name == "torch.float64":
        torch.set_default_dtype(torch.float64)
        backend.__class__ = TorchBackend
    elif name == "torch.float32":
        torch.set_default_dtype(torch.float32)
        backend.__class__ = TorchBackend
        backend.float = torch.float32
    elif name == "torch.cuda" or name == "torch.float64":
        torch.set_default_dtype(torch.float64)
        backend.__class__ = TorchCudaBackend
    elif name == "torch.cuda.float32":
        torch.set_default_dtype(torch.float32)
        backend.__class__ = TorchCudaBackend
        backend.float = torch.float32
    else:
        raise RuntimeError(f'unknown backend "{name}"')
