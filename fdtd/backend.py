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
from functools import wraps

# used only by test runner.
# default must be idx 0.
backend_names = [
    dict(backends="numpy"),
    dict(backends="torch.float32"),
    dict(backends="torch.float64"),
    dict(backends="torch.cuda.float32"),
    dict(backends="torch.cuda.float64"),
]

numpy_float_dtypes = {
    getattr(numpy, "float_", numpy.float64),
    getattr(numpy, "float16", numpy.float64),
    getattr(numpy, "float32", numpy.float64),
    getattr(numpy, "float64", numpy.float64),
    getattr(numpy, "float128", numpy.float64),
}


# Torch Backends (and flags)
try:
    import torch

    torch.set_default_dtype(torch.float64)  # we need more precision for FDTD
    try:  # we don't need gradients (for now)
        torch._C.set_grad_enabled(False)  # type: ignore
    except AttributeError:
        torch._C._set_grad_enabled(False)
    TORCH_AVAILABLE = True
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_CUDA_AVAILABLE = False


# Base Class
class Backend:
    """Backend Base Class"""

    # constants
    pi = numpy.pi

    def __repr__(self):
        return self.__class__.__name__


def _replace_float(func):
    """replace the default dtype a function is called with"""

    @wraps(func)
    def new_func(self, *args, **kwargs):
        result = func(*args, **kwargs)
        if result.dtype in numpy_float_dtypes:
            result = numpy.asarray(result, dtype=self.float)
        return result

    return new_func


# Numpy Backend
class NumpyBackend(Backend):
    """Numpy Backend"""

    # types
    int = numpy.int64
    """ integer type for array"""

    float = numpy.float64
    """ floating type for array """

    # methods
    asarray = _replace_float(numpy.asarray)

    exp = staticmethod(numpy.exp)
    """ exponential of all elements in array """

    sin = staticmethod(numpy.sin)
    """ sine of all elements in array """

    cos = staticmethod(numpy.cos)
    """ cosine of all elements in array """

    sum = staticmethod(numpy.sum)
    """ sum elements in array """

    max = staticmethod(numpy.max)
    """ max element in array """

    stack = staticmethod(numpy.stack)
    """ stack multiple arrays """

    transpose = staticmethod(numpy.transpose)
    """ transpose array by flipping two dimensions """

    reshape = staticmethod(numpy.reshape)
    """ reshape array into given shape """

    squeeze = staticmethod(numpy.squeeze)
    """ remove dim-1 dimensions """

    broadcast_arrays = staticmethod(numpy.broadcast_arrays)
    """ broadcast arrays """

    broadcast_to = staticmethod(numpy.broadcast_to)
    """ broadcast array into shape """

    @staticmethod
    def bmm(arr1, arr2):
        """batch matrix multiply two arrays"""
        return numpy.einsum("ijk,ikl->ijl", arr1, arr2)

    @staticmethod
    def is_array(arr):
        """check if an object is an array"""
        return isinstance(arr, numpy.ndarray)

    # constructors
    array = _replace_float(numpy.array)
    """ create an array from an array-like sequence """

    ones = _replace_float(numpy.ones)
    """ create an array filled with ones """

    zeros = _replace_float(numpy.zeros)
    """ create an array filled with zeros """

    zeros_like = staticmethod(numpy.zeros_like)
    """ create an array filled with zeros """

    linspace = _replace_float(numpy.linspace)
    """ create a linearly spaced array between two points """

    arange = _replace_float(numpy.arange)
    """ create a range of values """

    pad = staticmethod(numpy.pad)

    fftfreq = staticmethod(numpy.fft.fftfreq)

    fft = staticmethod(numpy.fft.fft)

    exp = staticmethod(numpy.exp)

    divide = staticmethod(numpy.divide)

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # beware to future people:
    # because this line *redefines numpy*,
    # you have to add your new staticmethods /above/ this line to avoid mystification.
    # <3 <3 <3 <3
    #
    # could this (and below) perhaps be changed to "to_numpy()"
    # or maybe "check_numpy" ?
    numpy = _replace_float(numpy.asarray)
    """ convert the array to numpy array """


# Torch Backend
if TORCH_AVAILABLE:
    import torch

    class TorchBackend(Backend):
        """Torch Backend"""

        # types
        int = torch.int64
        """ integer type for array"""

        float = torch.get_default_dtype()
        """ floating type for array """

        # methods
        asarray = staticmethod(torch.as_tensor)
        """ create an array """

        exp = staticmethod(torch.exp)
        """ exponential of all elements in array """

        sin = staticmethod(torch.sin)
        """ sine of all elements in array """

        cos = staticmethod(torch.cos)
        """ cosine of all elements in array """

        sum = staticmethod(torch.sum)
        """ sum elements in array """

        max = staticmethod(torch.max)
        """ max element in array """

        stack = staticmethod(torch.stack)
        """ stack multiple arrays """

        @staticmethod
        def transpose(arr, axes=None):
            """transpose array by flipping two dimensions"""
            if axes is None:
                axes = tuple(range(len(arr.shape) - 1, -1, -1))
            return arr.permute(*axes)

        squeeze = staticmethod(torch.squeeze)
        """ remove dim-1 dimensions """

        broadcast_arrays = staticmethod(torch.broadcast_tensors)
        """ broadcast arrays """

        broadcast_to = staticmethod(torch.broadcast_to)
        """ broadcast array into shape """

        reshape = staticmethod(torch.reshape)
        """ reshape array into given shape """

        bmm = staticmethod(torch.bmm)
        """ batch matrix multiply two arrays """

        @staticmethod
        def is_array(arr):
            """check if an object is an array"""
            # is this a reasonable implemenation?
            return isinstance(arr, numpy.ndarray) or torch.is_tensor(arr)

        def array(self, arr, dtype=None):
            """create an array from an array-like sequence"""
            if dtype is None:
                dtype = torch.get_default_dtype()
            if torch.is_tensor(arr):
                return arr.clone().to(device="cpu", dtype=dtype)
            return torch.tensor(arr, device="cpu", dtype=dtype)

        # constructors
        ones = staticmethod(torch.ones)
        """ create an array filled with ones """

        zeros = staticmethod(torch.zeros)
        """ create an array filled with zeros """

        def linspace(self, start, stop, num=50, endpoint=True):
            """create a linearly spaced array between two points"""
            delta = (stop - start) / float(num - float(endpoint))
            if not delta:
                return self.array([start] * num)
            return torch.arange(start, stop + 0.5 * float(endpoint) * delta, delta)

        arange = staticmethod(torch.arange)
        """ create a range of values """

        pad = staticmethod(torch.nn.functional.pad)  # type: ignore

        fftfreq = staticmethod(numpy.fft.fftfreq)

        fft = staticmethod(torch.fft)  # type: ignore

        divide = staticmethod(torch.div)

        exp = staticmethod(torch.exp)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # The same warning applies here.
        # <3 <3 <3 <3

        def numpy(self, arr):
            """convert the array to numpy array"""
            if torch.is_tensor(arr):
                return arr.numpy()
            else:
                return numpy.asarray(arr)

    # Torch Cuda Backend
    if TORCH_CUDA_AVAILABLE:

        class TorchCudaBackend(TorchBackend):
            """Torch Cuda Backend"""

            def ones(self, shape):
                """create an array filled with ones"""
                return torch.ones(shape, device="cuda")

            def zeros(self, shape):
                """create an array filled with zeros"""
                return torch.zeros(shape, device="cuda")

            def array(self, arr, dtype=None):
                """create an array from an array-like sequence"""
                if dtype is None:
                    dtype = torch.get_default_dtype()
                if torch.is_tensor(arr):
                    return arr.clone().to(device="cuda", dtype=dtype)
                return torch.tensor(arr, device="cuda", dtype=dtype)

            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # The same warning applies here.
            def numpy(self, arr):
                """convert the array to numpy array"""
                if torch.is_tensor(arr):
                    return arr.cpu().numpy()
                else:
                    return numpy.asarray(arr)

            def linspace(self, start, stop, num=50, endpoint=True):
                """convert a linearly spaced interval of values"""
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
    """Set the backend for the FDTD simulations

    This function monkeypatches the backend object by changing its class.
    This way, all methods of the backend object will be replaced.

    Args:
        name: name of the backend. Allowed backend names:
            - ``numpy`` (defaults to float64 arrays)
            - ``numpy.float16``
            - ``numpy.float32``
            - ``numpy.float64``
            - ``numpy.float128``
            - ``torch`` (defaults to float64 tensors)
            - ``torch.float16``
            - ``torch.float32``
            - ``torch.float64``
            - ``torch.cuda`` (defaults to float64 tensors)
            - ``torch.cuda.float16``
            - ``torch.cuda.float32``
            - ``torch.cuda.float64``

    """
    # perform checks
    if name.startswith("torch") and not TORCH_AVAILABLE:
        raise RuntimeError("Torch backend is not available. Is PyTorch installed?")
    if name.startswith("torch.cuda") and not TORCH_CUDA_AVAILABLE:
        raise RuntimeError(
            "Torch cuda backend is not available.\n"
            "Do you have a GPU on your computer?\n"
            "Is PyTorch with cuda support installed?"
        )

    if name.count(".") == 0:
        dtype, device = "float64", "cpu"
    elif name.count(".") == 1:
        name, dtype = name.split(".")
        if dtype == "cuda":
            device, dtype = "cuda", "float64"
        else:
            device = "cpu"
    elif name.count(".") == 2:
        name, device, dtype = name.split(".")
    else:
        raise ValueError(f"Unknown backend '{name}'")

    if name == "numpy":
        if device == "cpu":
            backend.__class__ = NumpyBackend
            backend.float = getattr(numpy, dtype)
        elif device == "cuda":
            raise ValueError(
                "Device 'cuda' not available for numpy backend. Use 'torch' backend in stead."
            )
        else:
            raise ValueError(
                "Unknown device '{device}'. Available devices: 'cpu', 'cuda'"
            )
    elif name == "torch":
        if device == "cpu":
            backend.__class__ = TorchBackend
            backend.float = getattr(torch, dtype)
        elif device == "cuda":
            backend.__class__ = TorchCudaBackend
            backend.float = getattr(torch, dtype)
        else:
            raise ValueError(
                "Unknown device '{device}'. Available devices: 'cpu', 'cuda'"
            )
    else:
        raise ValueError(
            "Unknown backend '{name}'. Available backends: 'numpy', 'torch'"
        )
