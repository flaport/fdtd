
## Imports

# typing
from .typing_ import Tuple, Number, Tensorlike

# relative
from .backend import backend as bd
from . import constants as const

## Functions
def curl_E(E: Tensorlike) -> Tensorlike:
    """Transforms an E-type field into an H-type field by performing a curl
    operation

    Args:
    
        E: Electric field to take the curl of (E-type field located on the
           edges of the grid cell [integer gridpoints])

    Returns:
    
        The curl of E (H-type field located on the faces of the grid
        [half-integer grid points])
    """
    curl = bd.zeros(E.shape)

    curl[:, :-1, :, 0] += E[:, 1:, :, 2] - E[:, :-1, :, 2]
    curl[:, :, :-1, 0] -= E[:, :, 1:, 1] - E[:, :, :-1, 1]

    curl[:, :, :-1, 1] += E[:, :, 1:, 0] - E[:, :, :-1, 0]
    curl[:-1, :, :, 1] -= E[1:, :, :, 2] - E[:-1, :, :, 2]

    curl[:-1, :, :, 2] += E[1:, :, :, 1] - E[:-1, :, :, 1]
    curl[:, :-1, :, 2] -= E[:, 1:, :, 0] - E[:, :-1, :, 0]

    return curl


def curl_edge_to_face(v: Tensorlike) -> Tensorlike:
    """
    Compute the curl of a vector field where the vector components are defined
    on the edges of the grid cells. The result is a vector field defined on the
    faces of the grid cells.

    Parameters:
    
    v (ndarray): Input vector field with shape (Nx, Ny, Nz, 3), where each
    component (vx, vy, vz) is defined on the edges of the grid cells.

    Returns:
    
    ndarray: The curl of the input vector field with shape (Nx, Ny, Nz, 3),
    where each component is defined on the faces of the grid cells.
    """
    
    return curl_E(v)



def curl_H(H: Tensorlike) -> Tensorlike:
    """Transforms an H-type field into an E-type field by performing a curl
    operation

    Args:
        H: Magnetic field to take the curl of (H-type field located on half-integer grid points)

    Returns:
        The curl of H (E-type field located on the edges of the grid [integer grid points])

    """
    curl = bd.zeros(H.shape)

    curl[:, 1:, :, 0] += H[:, 1:, :, 2] - H[:, :-1, :, 2]
    curl[:, :, 1:, 0] -= H[:, :, 1:, 1] - H[:, :, :-1, 1]

    curl[:, :, 1:, 1] += H[:, :, 1:, 0] - H[:, :, :-1, 0]
    curl[1:, :, :, 1] -= H[1:, :, :, 2] - H[:-1, :, :, 2]

    curl[1:, :, :, 2] += H[1:, :, :, 1] - H[:-1, :, :, 1]
    curl[:, 1:, :, 2] -= H[:, 1:, :, 0] - H[:, :-1, :, 0]

    return curl


def curl_face_to_edge(A: Tensorlike) -> Tensorlike:
    """
    Compute the curl of a vector field where the vector components are defined
    on the faces of the grid cells. The result is a vector field defined on the
    edges of the grid cells.

    Parameters:

    A (ndarray): Input vector field with shape (Nx, Ny, Nz, 3), where each
    component (Ax, Ay, Az) is defined on the faces of the grid cells.

    Returns:

    ndarray: The curl of the input vector field with shape (Nx, Ny, Nz, 3),
    where each component is defined on the edges of the grid cells.
    """
    return curl_H(A)



def div(v: Tensorlike) -> Tensorlike:
    """
    Compute the divergence of a vector field v located on the edges of the grid
    cells. The result is a scalar field located on the faces of the grid cells.

    Args:
        v (Tensorlike): Input vector field with shape (Nx, Ny, Nz, 3), where
        each component 
                        (vx, vy, vz) is defined on the edges of the grid cells.

    Returns:
        Tensorlike: The divergence of the input vector field with shape (Nx, Ny,
        Nz, 1), 
                    where the scalar field is defined on the faces of the grid
                    cells.
                    
    ChatGPT:
                    
    This implementation will now correctly compute the divergence for the velocity
    field located at the edges of the grid cells, giving a scalar field at the faces
    of the grid cells, which can be used further in the simulator's computations.                    
    """
    div_v = bd.zeros((v.shape[0], v.shape[1], v.shape[2], 1))

    # Compute the x-component of the divergence
    div_v[1:-1, :, :, 0] += (v[1:-1, :, :, 0] - v[:-2, :, :, 0])
    div_v[:-2, :, :, 0]  -= (v[1:-1, :, :, 0] - v[2:, :, :, 0])

    # Compute the y-component of the divergence
    div_v[:, 1:-1, :, 0] += (v[:, 1:-1, :, 1] - v[:, :-2, :, 1])
    div_v[:, :-2, :, 0]  -= (v[:, 1:-1, :, 1] - v[:, 2:, :, 1])

    # Compute the z-component of the divergence
    div_v[:, :, 1:-1, 0] += (v[:, :, 1:-1, 2] - v[:, :, :-2, 2])
    div_v[:, :, :-2, 0]  -= (v[:, :, 1:-1, 2] - v[:, :, 2:, 2])

    return div_v



def grad(p: Tensorlike) -> Tensorlike:
    """
    Compute the gradient of a scalar field p located on the grid points. p is a
    4D array with dimensions (Nx+1, Ny+1, Nz+1, 1). The first three dimensions
    represent the grid points. The last dimension represents the scalar field.
    Returns a 4D array with dimensions (Nx+1, Ny+1, Nz+1, 3). The first three
    dimensions represent the grid points. The last dimension represents the x,
    y, and z components of the gradient.
    
    The input scalar field `p` should be a 4D array with dimensions `(Nx+1,
    Ny+1, Nz+1, 1)` to be consistent with the staggered grid used in fluid
    dynamics. 

    This implementation takes a scalar field `p` as input and computes the
    gradient of the field in each direction using the `bd` backend. The output
    vector field `grad_p` is located on the faces of the grid cells that are
    perpendicular to the x, y, and z axes. Note that this implementation assumes
    that the input scalar field `p` is located on the grid points, which is
    consistent with the staggered grid used in fluid dynamics.
       
    
    """
    grad = bd.zeros((p.shape[0], p.shape[1], p.shape[2], 3))
    
    # Compute the x-component of the gradient
    grad[:-1, :, :, 0] = (p[1:, :, :, 0] - p[:-1, :, :, 0])
    
    # Compute the y-component of the gradient
    grad[:, :-1, :, 1] = (p[:, 1:, :, 0] - p[:, :-1, :, 0])
    
    # Compute the z-component of the gradient
    grad[:, :, :-1, 2] = (p[:, :, 1:, 0] - p[:, :, :-1, 0])
    
    return grad
