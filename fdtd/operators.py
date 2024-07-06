
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


def delta(F: Tensorlike) -> Tensorlike:
    """
    Compute the vector Laplace operator of a vector field F located on the faces
    of the grid cells.· F is a 4D array with dimensions (Nx+1, Ny+1, Nz+1, 3).
    The last dimension represents the x, y, and z components of the vector
    field. The first three dimensions represent the grid points. Returns a 4D
    array with dimensions (Nx+1, Ny+1, Nz+1, 3). The first three dimensions
    represent the grid points. The last dimension represents the x, y, and z
    components of the vector Laplace operator.
    
    This implementation takes a vector field `F` as input and computes the
    vector Laplace operator of the field using the `bd` backend. The output
    vector field `delta_F` is located on the faces of the grid cells that are
    perpendicular to the x, y, and z axes. Note that this implementation assumes
    that the input vector field `F` is located on the faces of the grid cells
    that are perpendicular to the x, y, and z axes, which is consistent with the
    staggered grid used in fluid dynamics and electrodynamics.


    The `delta` function computes the vector Laplace operator of a vector field
    using the `div`, `grad`, `curl_edge_to_face`, and `curl_face_to_edge` functions. The
    `div` and `grad` functions compute the divergence and gradient of a vector
    field, respectively, using finite differences between the components of the
    vector field at different points in the grid cell. The output fields are
    located on the faces of the grid cells that are perpendicular to the x, y,
    and z axes, which is consistent with the staggered Yee grid.
    
    The field types are:

    - `F`: A vector field located on the edges of the grid cells (point field).
    - `phi`: A scalar field located on the grid points (cell-centered field).
    - `E`: A vector field located on the edges of the grid cells (point field).
    - `A`: A vector field located on the faces of the grid cells (surface
      field).
    - `H`: A vector field located on the edges of the grid cells (point field).
    - `delta_F`: A vector field located on the edges of the grid cells (point
    field).

    The `div` function computes the divergence of the vector field `F`, which
    results in a scalar field `phi` located on the grid points.

    The `grad` function computes the gradient of the scalar field `phi`, which
    results in a vector field `E` located on the edges of the grid cells.

    The `curl_edge_to_face` function computes the point curl of the vector field `F`,
    which results in a vector field `A` located on the faces of the grid cells.

    The `curl_face_to_edge` function computes the surface curl of the vector field
    `A`, which results in a vector field `H` located on the edges of the grid
    cells.

    Finally, the `delta_F` vector field is computed as the difference between
    the vector field `E` and the vector field `H`, both located on the edges of
    the grid cells.
    
    l4m4re: This makes sense. In the above code, it is the vector field A that
    is a surface field, while the vector field H is a point field.
    
    GitHub Copilot: Yes, that's correct. In the above code, the vector field `A`
    is a surface field located on the faces of the grid cells, while the vector
    field `H` is a point field located on the edges of the grid cells. 

    The `curl_edge_to_face` function computes the point curl of the vector field `F`,
    which results in a vector field `A` located on the faces of the grid cells.
    The `curl_face_to_edge` function then computes the surface curl of the vector
    field `A`, which results in a vector field `H` located on the edges of the
    grid cells. 

    The resulting vector field `H` represents the curl of the vector field `F`
    at the edges of the grid cells that are perpendicular to the x, y, and z
    axes, and is defined on the edges of the grid cells, which are integer grid
    points.
    
    """
    phi = div(F)
    E = grad(phi)
    A = curl_edge_to_face(F)
    H = curl_face_to_edge(A)
    delta_F = E - H
    return delta_F



def delta_C(C: Tensorlike) -> Tensorlike:
    """
    Compute the vector Laplace operator of a vector field F located on the faces
    of the grid cells.· 
    
    F is a 4D array with dimensions (Nx+1, Ny+1, Nz+1, 3). The last dimension
    represents the x, y, and z components of the vector field. The first three
    dimensions represent the grid points. Returns a 4D array with dimensions
    (Nx+1, Ny+1, Nz+1, 3). The first three dimensions represent the grid points.
    The last dimension represents the x, y, and z components of the vector
    Laplace operator.
    
    This implementation takes a vector field `F` as input and computes the
    vector Laplace operator of the field using the `bd` backend. The output
    vector field `delta_F` is located on the faces of the grid cells that are
    perpendicular to the x, y, and z axes. Note that this implementation assumes
    that the input vector field `F` is located on the faces of the grid cells
    that are perpendicular to the x, y, and z axes, which is consistent with the
    staggered grid used in fluid dynamics and electrodynamics.
    
       Calculate the vector Laplacian of a vector field C using the left-handed
    and right-handed curl operators. 
    
    In our superfluid medium model, the vector field C has a unit of measurement
    in Amperes per second, or, in basic unit: 
    
        kilograms per second squared [kg/s^2],
        
    It is defined by the equation:
    
        C = - k * rho [v],
      
    with rho the mass density of the superfluid medium, which in this model is
    taken equal to the electric permittivity of the medium, so for the vacuum 
    
        rho = epsilon_0 = eta/k = 8.854187817620389e-12 [kg/m^3].

    And, for the medium or vaccuum, k the quantum circulation constant,
    diffusivity or kine-matic viscosity of the medium with a value of:
    
        k = c*c = 8.987551787368176e+16 [m^2/s],
      
    Thus, we indeed define k to have a value equal to c*c, but a unit of
    measurement of meters squared per second [m^2/s] representing the quantum
    circulation konstant, diffusivity or kinematic viscosity of the medium.
    
    Thus, C has a unit of measurement in [m^2/s] * [kg/m^3] * [m/s] = [kg/s^2],
    the unit we equate to the Ampere in this model, hence the choice of C, for
    "current" in "Amperes" or kilograms per second squared.
    
    in the following code, lower case is used for scalar fields, upper case for
    vector fields.
            
    """
    p = div(C)  # p is a scalar field representing pressure in [Pa/m^3]
    L = grad(p) # a vector field representing linear force density in [N/m^3]
    
    A  = curl_edge_to_face(C) # the vector potential, also in [Pa/m^3].
    R  = curl_face_to_edge(A) # a vector field representing angular force density in [N/m^3] 
        
    d_C = L - R # Delta C, the vector laplace operator of C in [N/m^3]
    
    return d_C # net force density in [N/m^3].



def delta_I(I: Tensorlike) -> Tensorlike:
    """
    Compute the vector Laplace operator of a vector field F located on the faces of the grid
    cells.·
    F is a 4D array with dimensions (Nx+1, Ny+1, Nz+1, 3).
    The last dimension represents the x, y, and z components of the vector field.
    The first three dimensions represent the grid points.
    Returns a 4D array with dimensions (Nx+1, Ny+1, Nz+1, 3).
    The first three dimensions represent the grid points.
    The last dimension represents the x, y, and z components of the vector Laplace operator.
        
    
    In our superfluid medium model, the vector field I has a unit of measurement
    in Watts per second, or, in basic unit: 
       kilograms per seconds to the third [kg/s^3]
     
     This repreents the time derivative of the field C, whe have defined earlier.
            
    It is defined by the equation:
    
        I = - k * rho [a],
      
    with rho the mass density of the superfluid medium, which in this model
    is taken equal to the electric permittivity of the medium, so for
    the vacuum 
    
        rho = epsilon_0 = eta/k = 8.854187817620389e-12 [kg/m^3].

    And, for the medium or vaccuum, k the quantum circulation constant,
    diffusivity or kinematic viscosity of the medium with a value of:
    
        k = c*c = 8.987551787368176e+16 [m^2/s],
      
    Thus, I has a unit of measurement in [m^2/s] * [kg/m^3] * [m/s^2] = [kg/s^3],
    the unit we equate to Intensity in this model, hence the choice of I, for
    "intensity" in "Watts per meter squared" or kilograms per second to the third.
    
    For this field, we can also define derived fields, reesulting in a Yank
    density field Y, with a unit of measurment in Newton per cubic meter - second [N/m^3-s],
    the time derivative of Force density F.
    
    in the following code, lower case is used for scalar fields, upper case
    for vector fields.
            
    """
    t   = div(I)  # t is a scalar field representing power density in [W/m^3] , presumed to represent temperarure
    Y_l = grad(t) # a vector field representing linear yank density in [N/m^3-s]

    # the vector potential, also in [kg/m-s3], [N/m2-s], [J/m3-s], [Pa/s],
    # [W/m3], [K] or [kg/m-s3] representing angular power or heat density, force
    # flux, yank density flux, or angular temperature
    W   = curl_edge_to_face(I) 
    Y_a = curl_face_to_edge(W) # a vector field representing angular yank density in [N/m^3-s] 
        
    d_I = Y_l - Y_a # Delta I, the vector laplace operator of I in [N/m^3-s]
    
    return d_I # net yank density or current flux in [N/m^3-s].


