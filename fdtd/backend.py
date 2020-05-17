""" Selects the backend for the fdtd-package.

The `fdtd` library allows to choose a backend. The ``numpy`` backend is the
default one, but there are also several additional PyTorch backends:

    - ``numpy`` (defaults to float64 arrays)
    - ``torch`` (defaults to float64 tensors)
    - ``torch.float32``
    - ``torch.float64``
    - ``torch.cuda`` (defaults to float64 tensors)
    - ``torch.cuda.float32``
    - ``torch.cuda.float64``

For example, this is how to choose the `"torch"` backend: ::

    fdtd.set_backend("torch")

In general, the ``numpy`` backend is preferred for standard CPU calculations
with `"float64"` precision. In general, ``float64`` precision is always
preferred over ``float32`` for FDTD simulations, however, ``float32`` might
give a significant performance boost.

The ``cuda`` backends are only available for computers with a GPU.

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
    """ integer type for array"""

    float = numpy.float64
    """ floating type for array """

    # methods
    exp = staticmethod(numpy.exp)
    """ exponential of all elements in array """

    sin = staticmethod(numpy.sin)
    """ sine of all elements in array """

    cos = staticmethod(numpy.cos)
    """ cosine of all elements in array """

    sum = staticmethod(numpy.sum)
    """ sum elements in array """

    stack = staticmethod(numpy.stack)
    """ stack multiple arrays """

    transpose = staticmethod(numpy.transpose)
    """ transpose array by flipping two dimensions """

    reshape = staticmethod(numpy.reshape)
    """ reshape array into given shape """

    squeeze = staticmethod(numpy.squeeze)
    """ remove dim-1 dimensions """

    @staticmethod
    def bmm(arr1, arr2):
        """ batch matrix multiply two arrays """
        return numpy.einsum("ijk,ikl->ijl", arr1, arr2)

    @staticmethod
    def is_array(arr):
        """ check if an object is an array """
        return isinstance(arr, numpy.ndarray)

    # constructors
    array = staticmethod(numpy.array)
    """ create an array from an array-like sequence """

    ones = staticmethod(numpy.ones)
    """ create an array filled with ones """

    zeros = staticmethod(numpy.zeros)
    """ create an array filled with zeros """

    linspace = staticmethod(numpy.linspace)
    """ create a linearly spaced array between two points """

    arange = staticmethod(numpy.arange)
    """ create a range of values """

    numpy = staticmethod(numpy.asarray)
    """ convert the array to numpy array """


# Torch Backend
if TORCH_AVAILABLE:

    class TorchBackend(Backend):
        """ Torch Backend """

        # types
        int = torch.int64
        """ integer type for array"""

        float = torch.get_default_dtype()
        """ floating type for array """

        # methods
        exp = staticmethod(torch.exp)
        """ exponential of all elements in array """

        sin = staticmethod(torch.sin)
        """ sine of all elements in array """

        cos = staticmethod(torch.cos)
        """ cosine of all elements in array """

        sum = staticmethod(torch.sum)
        """ sum elements in array """

        stack = staticmethod(torch.stack)
        """ stack multiple arrays """

        @staticmethod
        def transpose(arr, axes=None):
            """ transpose array by flipping two dimensions """
            if axes is None:
                axes = tuple(range(len(arr.shape) - 1, -1, -1))
            return arr.permute(*axes)

        squeeze = staticmethod(torch.squeeze)
        """ remove dim-1 dimensions """

        reshape = staticmethod(torch.reshape)
        """ reshape array into given shape """

        bmm = staticmethod(torch.bmm)
        """ batch matrix multiply two arrays """

        @staticmethod
        def is_array(arr):
            """ check if an object is an array """

        def array(self, arr, dtype=None):
            """ create an array from an array-like sequence """
            if dtype is None:
                dtype = torch.get_default_dtype()
            return torch.tensor(arr, device="cpu", dtype=dtype)
            return torch.is_tensor(arr)

        # constructors
        ones = staticmethod(torch.ones)
        """ create an array filled with ones """

        zeros = staticmethod(torch.zeros)
        """ create an array filled with zeros """

        def linspace(self, start, stop, num=50, endpoint=True):
            """ create a linearly spaced array between two points """
            delta = (stop - start) / float(num - float(endpoint))
            if not delta:
                return self.array([start] * num)
            return torch.arange(start, stop + 0.5 * float(endpoint) * delta, delta)

        arange = staticmethod(torch.arange)
        """ create a range of values """

        def numpy(self, arr):
            """ convert the array to numpy array """
            if torch.is_tensor(arr):
                return arr.numpy()
            else:
                return numpy.asarray(arr)



# Torch Cuda Backend
if TORCH_CUDA_AVAILABLE:

    class TorchCudaBackend(TorchBackend):
        """ Torch Cuda Backend """

        def ones(self, shape):
            """ create an array filled with ones """
            return torch.ones(shape, device="cuda")

        def zeros(self, shape):
            """ create an array filled with zeros """
            return torch.zeros(shape, device="cuda")

        def array(self, arr, dtype=None):
            """ create an array from an array-like sequence """
            if dtype is None:
                dtype = torch.get_default_dtype()
            return torch.tensor(arr, device="cuda", dtype=dtype)

        def numpy(self, arr):
            """ convert the array to numpy array """
            if torch.is_tensor(arr):
                return arr.cpu().numpy()
            else:
                return numpy.asarray(arr)

        def linspace(self, start, stop, num=50, endpoint=True):
            """ convert a linearly spaced interval of values """
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
            - ``numpy`` (defaults to float64 arrays)
            - ``torch`` (defaults to float64 tensors)
            - ``torch.float32``
            - ``torch.float64``
            - ``torch.cuda`` (defaults to float64 tensors)
            - ``torch.cuda.float32``
            - ``torch.cuda.float64``

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
