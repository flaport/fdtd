
''' 
This file contains the various operators used in the FDTD grid classes.

The operators are: grad, div and curl.

GitHub Copilot: The `grad`, `div`, and `curl` functions are implemented in the
`operators.py` file in the FDTD library. These functions are used to calculate
the gradient, divergence, and curl of a given field, respectively. 

Here is a brief explanation of how each function is implemented:

1. `grad`: This function calculates the gradient of a given field. The gradient
   is a vector that points in the direction of the maximum rate of change of the
   field. In the FDTD library, the gradient is calculated using the central
   difference method. This method involves calculating the difference between
   the field values at neighboring grid points in each dimension. The resulting
   gradient vector is then normalized to have a magnitude of 1.

2. `div`: This function calculates the divergence of a given field. The
   divergence is a scalar that represents the amount of "outwardness" of the
   field at a given point. In the FDTD library, the divergence is calculated
   using the central difference method. This method involves calculating the
   difference between the field values at neighboring grid points in each
   dimension and summing the resulting values. 

3. `curl`: This function calculates the curl of a given field. The curl is a
   vector that represents the amount of rotation of the field at a given point.
   In the FDTD library, the curl is calculated using the central difference
   method. This method involves calculating the difference between the field
   values at neighboring grid points in each dimension and taking the cross
   product of the resulting vectors. 

Overall, these functions are essential for calculating the behavior of
electromagnetic waves and fluid flow in FDTD simulations.


GitHub Copilot: Yes, that's correct! As long as you use the correct
implementations of the gradient, divergence, and curl functions for the
staggered grid with interlaced Yee-coordinates, you do not have to worry about
the specific locations of the fields on the grid. The `grad_p`, `div_vf`,
`curl_E`, and `curl_H` functions that we have discussed are designed to work
with the staggered grid used in electromagnetic and fluid dynamics simulations,
and they take care of the interpolation and averaging of the fields to the
correct locations on the grid. 


GitHub Copilot: In FDTD simulations, the electric and magnetic fields are
discretized on a grid with interlaced Yee-coordinates. This means that the
electric and magnetic fields are located at different points on the grid, and
they are staggered in such a way that the electric field is located on the edges
of the grid cells, while the magnetic field is located on the faces of the grid
cells. 

Because of this staggered arrangement, the curl of the electric field and the
curl of the magnetic field are calculated differently. The curl of the electric
field is calculated using the central difference method, which involves
calculating the difference between the electric field values at neighboring grid
points in each dimension and taking the cross product of the resulting vectors.
On the other hand, the curl of the magnetic field is calculated using the
backward difference method, which involves calculating the difference between
the magnetic field values at neighboring grid points in each dimension and
dividing the resulting vector by the grid spacing.

In the FDTD library, there are two different curl functions: `curl_E` and
`curl_H`. The `curl_E` function calculates the curl of the electric field, while
the `curl_H` function calculates the curl of the magnetic field. These functions
are designed to work with the staggered grid used in electromagnetic
simulations, and they take care of the interpolation and averaging of the fields
to the correct locations on the grid. 

Overall, the two different curl functions are necessary because of the staggered
arrangement of the electric and magnetic fields on the grid, and they are
implemented differently because of the different field operations involved.


This, however, our model is not an ordinary fluid dynamics model, nor is it a
rather ordinary electromagnetic model. It is a saperfluid dynamics model,
derived from a single equation:

  d[F]/dt = -k * delta[F].
  
  with F any given vector field within the superfluid model.
  
delta[F] is the vector laplace operator of F, which is defined as: 

    delta[F] = grad(div(F)) - curl(curl(F))

with grad the gradient operator, div the divergence operator and curl the curl
operator.

However, given the staggered grid with interlaced Yee-coordinates, where it was
necessary to implement two different curl functions, representing two different
specific fields on specific locations on the grid, we will have to make a
distinction between the two curl operators and redefine our vector laplace
operator as:

    delta[F] = grad(div(F)) - curl_surface(curl_point(F))


'''

## Imports

# standard library
import os
from os import path, makedirs, chdir, remove
from subprocess import check_call, CalledProcessError
from glob import glob
from datetime import datetime

# 3rd party
from tqdm import tqdm
from numpy import savez

# typing
from .typing_ import Tuple, Number, Tensorlike

# relative
from .backend import backend as bd
from . import constants as const
'''
from math import *

# some constants
c     = 299792458.0          # [m/s] - speed of light
c_l   = c * (pi/2)           # [m/s] - speed of longitudinal (sound) waves
mu    = 4*pi*1e-7            # mu_0 - 1/viscosity
#mu    = 4*pi*1.00000000082e-7# mu_0 - 1/viscosity - 2019 redefinition
eta   = 1/(4*pi*1e-7)        # viscosity      [Pa-s],[kg/m-s],  [N-s/m2], [J-s/m3]
h     = 6.62607015e-34       # Planck's constant    [J-s], [kg/m2-s]      
k     = c*c                  # dynamic viscosity    [m2/s]
rho   = eta/k                # 8.854187817620389e-12
q     = 1.602176634e-19      # elemental charge
m     = h/k                  # elemental mass   
'''

'''
First, some context on Yee grids from Copilot:

In a electromagnetic FDTD simulation, the electric and magnetic fields are
discretized on a grid with interlaced Yee-coordinates. This means that the
electric and magnetic fields are located at different points on the grid, and
they are staggered in such a way that the electric field is located on the edges
of the grid cells, while the magnetic field is located on the faces of the grid
cells.

The Yee grid is named after Kane S. Yee, who first proposed this grid in 1966.
The Yee grid is a staggered grid that is designed to accurately model the
behavior of electromagnetic waves. The Yee grid is widely used in FDTD
simulations because it is simple to implement and provides accurate results.

In the Yee grid, the electric and magnetic fields are located at different
points on the grid. Specifically, the electric field is located on the edges of
the grid cells, while the magnetic field is located on the faces of the grid
cells. This staggered arrangement ensures that the electric and magnetic fields
are updated at different times during the simulation, which is necessary to
accurately model the behavior of electromagnetic waves.

Overall, the interlaced Yee-coordinates are a critical part of the FDTD
simulation process, as they allow the electric and magnetic fields to be
discretized on a grid that accurately models the behavior of electromagnetic
waves. By understanding how the Yee grid works and how it is used in the
simulation, developers can gain a deeper understanding of how the FDTD Library
works and how to use it to simulate electromagnetic waves.


Yee grids are primarily used in computational electromagnetics to discretize the
Maxwell's equations and simulate electromagnetic waves. 

However, Yee grids can also be used in fluid dynamics to discretize the
Navier-Stokes equations and simulate fluid flow.

In fluid dynamics, the Navier-Stokes equations describe the motion of fluids and
can be used to simulate a wide range of physical phenomena, such as fluid flow
around objects, turbulence, and mixing. The Navier-Stokes equations are a set of
partial differential equations that describe the conservation of mass, momentum,
and energy in a fluid.

To simulate fluid flow using the Navier-Stokes equations, the equations must be
discretized on a grid. Yee grids can be used to discretize the Navier-Stokes
equations in a similar way to how they are used to discretize the Maxwell's
equations in computational electromagnetics. Specifically, the fluid velocity
and pressure can be discretized on a staggered grid with interlaced
Yee-coordinates, similar to how the electric and magnetic fields are discretized
in the FDTD simulation.

By discretizing the Navier-Stokes equations on a Yee grid, the equations can be
solved numerically using finite-difference methods, finite-volume methods, or
other numerical techniques. This allows researchers to simulate fluid flow and
study the behavior of fluids in a wide range of physical scenarios.

Overall, Yee grids can be used in fluid dynamics to discretize the Navier-Stokes
equations and simulate fluid flow. By using Yee grids in this way, researchers
can gain a deeper understanding of fluid dynamics and develop new technologies
that rely on fluid flow, such as aircraft, ships, and turbines.

A staggered grid with interlaced Yee-coordinates is a type of grid
discretization commonly used in FDTD simulations. In this type of grid, the
electric and magnetic fields are located on the faces of the grid cells, while
the current and charge densities are located on the grid points. This
arrangement is known as a "staggered" grid because the field components are
staggered with respect to the grid points.

The Yee-coordinate system is a specific type of staggered grid where the
electric and magnetic fields are located on the faces of the grid cells, but
they are interlaced in a specific way. In the Yee-coordinate system, the
electric field components are located on the faces of the grid cells that are
perpendicular to the x, y, and z axes, while the magnetic field components are
located on the faces of the grid cells that are parallel to the x, y, and z
axes. This interlacing of the field components ensures that the update equations
for the electric and magnetic fields are consistent with each other and with the
underlying physics.

The Yee-coordinate system is named after Kane S. Yee, who first proposed this
type of grid discretization in 1966. Since then, the Yee-coordinate system has
become a standard in FDTD simulations and is widely used in many different
applications, including electromagnetic wave propagation, antenna design, and
photonics.


In fluid dynamics, a staggered grid with interlaced Yee-coordinates is a type of
grid discretization that can be used to simulate fluid flow using the
Navier-Stokes equations. In this type of grid, the fluid velocity components are
located on the faces of the grid cells that are perpendicular to the x, y, and z
axes, while the pressure is located on the grid points. This arrangement is
similar to the Yee grid used in FDTD simulations, where the electric and
magnetic fields are located on the faces of the grid cells and the current and
charge densities are located on the grid points.

By discretizing the Navier-Stokes equations on a staggered grid with interlaced
Yee-coordinates, the equations can be solved numerically using finite-difference
methods, finite-volume methods, or other numerical techniques. This allows
researchers to simulate fluid flow and study the behavior of fluids in a wide
range of physical scenarios.

The staggered grid with interlaced Yee-coordinates ensures that the update
equations for the fluid velocity and pressure are consistent with each other and
with the underlying physics. This type of grid discretization is widely used in
fluid dynamics to simulate fluid flow around objects, turbulence, and mixing. By
using this type of grid, researchers can gain a deeper understanding of fluid
dynamics and develop new technologies that rely on fluid flow, such as aircraft,
ships, and turbines.




Let me summarize what I understood from what you said:

A staggered grid with interlaced Yee-coordinates is a type of grid
discretization commonly used in FDTD simulations.

In fluid dynamics, a staggered grid with interlaced Yee-coordinates is a type of
grid discretization that can be used to simulate fluid flow using the
Navier-Stokes equations. In this type of grid, the fluid velocity components are
located on the faces of the grid cells that are perpendicular to the x, y, and z
axes, while the pressure is located on the grid points. 

In electrodynamics, a staggered grid with interlaced Yee-coordinates is a type
of grid discretization that can be used to simulate the electromagnetic fields
using Maxwell's equations. In this type of grid, the electric and magnetic
fields are located on the faces of the grid cells, while the current and charge
densities are located on the grid points. This arrangement is known as a
"staggered" grid because the field components are staggered with respect to the
grid points.

In the electrodynamics Yee-coordinate system, the electric field components are
located on the faces of the grid cells that are perpendicular to the x, y, and z
axes, while the magnetic field components are located on the faces of the grid
cells that are parallel to the x, y, and z axes.

Do I understand correctly that with interlaced Yee-coordinates vector fields are
always located on the faces of the grid cells, either on the faces that are
parallel to the x, y, and z axes or on the faces that are perpendicular to the
x, y, and z axes? 

Do I also understand correctly that with interlaced Yee-coordinates scalar
fields are always located on the grid points?


GitHub Copilot: Yes, you have understood correctly. 

In both electrodynamics and fluid dynamics, a staggered grid with interlaced
Yee-coordinates is a type of grid discretization where the vector fields are
located on the faces of the grid cells, either on the faces that are parallel to
the x, y, and z axes or on the faces that are perpendicular to the x, y, and z
axes. This arrangement is known as a "staggered" grid because the field
components are staggered with respect to the grid points.

In contrast, scalar fields are always located on the grid points in a staggered
grid with interlaced Yee-coordinates. This is because the scalar fields are
associated with the physical quantities that are defined at the grid points,
such as the pressure in fluid dynamics or the charge density in electrodynamics.

'''

'''

In the grid class, which is to use the operators, there are two types of fields,
namely sclar fields such as:

  self.P          = bd.zeros((self.Nx, self.Ny, self.Nz, 1))

And vector fields such as:

  self.C          = bd.zeros((self.Nx, self.Ny, self.Nz, 3))

Each of these fields can be located at either the faces or the edges of the grid
cells, meaning at either integer or half-integer grid points. 

This is why we need multiple versions of the grid, div and curl operators.

The first operators we need are the curl and the gradient of a velocity field,
which is a vector field, which should be located on the faces of the grid cells.




Is it correct that in the curl_E and curl_H functions for a staggered grid with
interlaced Yee-coordinates the E and H fields are both located on the faces of
the grid, with the only difference being the faces perpendicular to the x, y,
and z axes for the E field, while for the H field they are the faces parallel to
the x, y and z axes? Thus, are both fields actually located on integer grid
points?


GitHub Copilot: Yes, that's correct! 

In the staggered grid with interlaced Yee-coordinates used in electromagnetic
simulations, the electric field `E` and magnetic field `H` are both located on
the faces of the grid cells. The only difference is that the electric field `E`
is located on the faces of the grid cells that are perpendicular to the x, y,
and z axes, while the magnetic field `H` is located on the faces of the grid
cells that are parallel to the x, y, and z axes. 

Both fields are located on half-integer grid points, which means that their
components are located halfway between the grid points. However, when we
interpolate the fields to the grid points, we use the values of the fields at
the nearest faces of the grid cells to the grid points. This means that the
electric field `E` is located on the edges of the grid cells that are
perpendicular to the x, y, and z axes, while the magnetic field `H` is located
on the faces of the grid cells that are parallel to the x, y, and z axes. 

Therefore, when we compute the curl of the electric field `E` using the `curl_E`
function, the resulting magnetic field `H` is located on the faces of the grid
cells that are parallel to the x, y, and z axes, which are integer grid points.
Similarly, when we compute the curl of the magnetic field `H` using the `curl_H`
function, the resulting electric field `E` is located on the edges of the grid
cells that are perpendicular to the x, y, and z axes, which are also integer
grid points.



l4m4re: Do I understand correctly that we do not have to worry about where the
various fields are located on the grids, provided we use the correct
implementations of grad, div and the two curl functions?

GitHub Copilot: Yes, that's correct! As long as you use the correct
implementations of the gradient, divergence, and curl functions for the
staggered grid with interlaced Yee-coordinates, you do not have to worry about
the specific locations of the fields on the grid. The `grad_p`, `div_vf`,
`curl_E`, and `curl_H` functions that we have discussed are designed to work
with the staggered grid used in electromagnetic and fluid dynamics simulations,
and they take care of the interpolation and averaging of the fields to the
correct locations on the grid. 

However, it is important to note that the input vector field `F` that you pass
to these functions must be located on the faces of the grid cells that are
perpendicular to the x, y, and z axes for fluid dynamics, or on the faces of the
grid cells that are perpendicular to the time axis for electromagnetic
simulations. If the input field is not located on the correct faces of the grid
cells, the results of the gradient, divergence, and curl operations may not be
correct.

Thus, if I pass a fluid dynamics velocity field in the shape of a 4D array ith
dimensions (Nx+1, Ny+1, Nz+1, 3) as F to the delta(F) funtion we designed, it
should correctly compute the delta vector laplace operator?

GitHub Copilot: Yes, that's correct! If you pass a fluid dynamics velocity field
in the shape of a 4D array with dimensions `(Nx+1, Ny+1, Nz+1, 3)` as `F` to the
`delta` function we designed, it should correctly compute the vector Laplace
operator `delta` of the field. 

The `delta` function takes a vector field `F` as input and computes the vector
Laplace operator of the field using the `grad_p`, `div_vf`, `curl_E`, and
`curl_H` functions that are designed to work with the staggered grid used in
fluid dynamics simulations. The output vector field `delta_F` is located on the
faces of the grid cells that are perpendicular to the x, y, and z axes, which is
consistent with the staggered grid used in fluid dynamics. 

Therefore, if you pass a fluid dynamics velocity field in the shape of a 4D
array with dimensions `(Nx+1, Ny+1, Nz+1, 3)` as `F` to the `delta` function, it
should correctly compute the vector Laplace operator of the field, regardless of
the specific locations of the field components on the grid.


Now let's refer to the original readme file of the fdtd library for more details
and edit a bit:

-+- 

According to the Yee discretization algorithm, there are inherently two types of
fields on the grid: `E`-type point fields on integer grid locations and `H`-type
surface fields on half-integer grid locations.

The beauty of these interlaced coordinates is that they enable a very natural
way of writing the curl of the electric and magnetic fields: the curl of an
H-type surface field will be an E-type point field and vice versa.

This way, the curl of E can be written as

```python
    curl(E)[m,n,p] = (dEz/dy - dEy/dz, dEx/dz - dEz/dx, dEy/dx - dEx/dy)[m,n,p]
                   =( ((Ez[m,n+1,p]-Ez[m,n,p])/dy - (Ey[m,n,p+1]-Ey[m,n,p])/dz),
                      ((Ex[m,n,p+1]-Ex[m,n,p])/dz - (Ez[m+1,n,p]-Ez[m,n,p])/dx),
                      ((Ey[m+1,n,p]-Ey[m,n,p])/dx - (Ex[m,n+1,p]-Ex[m,n,p])/dy)
                      )
                   =(1/du)*( ((Ez[m,n+1,p]-Ez[m,n,p]) -
                   (Ey[m,n,p+1]-Ey[m,n,p])), [assume dx=dy=dz=du]
                             ((Ex[m,n,p+1]-Ex[m,n,p]) -
                             (Ez[m+1,n,p]-Ez[m,n,p])), ((Ey[m+1,n,p]-Ey[m,n,p])
                             - (Ex[m,n+1,p]-Ex[m,n,p])) )

```

this can be written efficiently with array slices (note that the factor `(1/du)`
was left out):

```python def curl_E(E):
    curl_E = np.zeros(E.shape) curl_E[:,:-1,:,0] += E[:,1:,:,2] - E[:,:-1,:,2]
    curl_E[:,:,:-1,0] -= E[:,:,1:,1] - E[:,:,:-1,1]

    curl_E[:,:,:-1,1] += E[:,:,1:,0] - E[:,:,:-1,0] curl_E[:-1,:,:,1] -=
    E[1:,:,:,2] - E[:-1,:,:,2]

    curl_E[:-1,:,:,2] += E[1:,:,:,1] - E[:-1,:,:,1] curl_E[:,:-1,:,2] -=
    E[:,1:,:,0] - E[:,:-1,:,0] return curl_E
```


The curl for H can be obtained in a similar way (note again that the factor
`(1/du)` was left out):

```python def curl_H(H):
    curl_H = np.zeros(H.shape)

    curl_H[:,1:,:,0] += H[:,1:,:,2] - H[:,:-1,:,2] curl_H[:,:,1:,0] -=
    H[:,:,1:,1] - H[:,:,:-1,1]

    curl_H[:,:,1:,1] += H[:,:,1:,0] - H[:,:,:-1,0] curl_H[1:,:,:,1] -=
    H[1:,:,:,2] - H[:-1,:,:,2]

    curl_H[1:,:,:,2] += H[1:,:,:,1] - H[:-1,:,:,1] curl_H[:,1:,:,2] -=
    H[:,1:,:,0] - H[:,:-1,:,0] return curl_H
```


The update equations can now be rewritten as

```python
    E  += (c*dt/du)*inv(ε)*curl_H H  -= (c*dt/du)*inv(µ)*curl_E
```

The number `(c*dt/du)` is a dimensionless parameter called the _Courant number_
`sc`. For stability reasons, the Courant number should always be smaller than
`1/√D`, with `D` the dimension of the simulation. This can be intuitively be
understood as the condition that information should always travel slower than
the speed of light through the grid. In the FDTD method described here,
information can only travel to the neighboring grid cells (through application
of the curl). It would therefore take `D` time steps to travel over the diagonal
of a `D`-dimensional cube (square in `2D`, cube in `3D`), the Courant condition
follows then automatically from the fact that the length of this diagonal is
`1/√D`.

This yields the final update equations for the FDTD algorithm:

```python
    E  += sc*inv(ε)*curl_H H  -= sc*inv(µ)*curl_E
```

-+-

The fist thing we note is : "[assume dx=dy=dz=du]", which means that the grid
spacing is the same in all three dimensions. Further, we note that the factor
1/du is left out in the implementation of the curl functions, because it can be
taken outside the parentheses if dx=dy=dz=du as shown in the simplified
definition for curl_E:

curl(E)[m,n,p] =(1/du)*( ((Ez[m,n+1,p]-Ez[m,n,p]) - (Ey[m,n,p+1]-Ey[m,n,p])), 
                         ((Ex[m,n,p+1]-Ex[m,n,p]) - (Ez[m+1,n,p]-Ez[m,n,p])),
                         ((Ey[m+1,n,p]-Ey[m,n,p]) - (Ex[m,n+1,p]-Ex[m,n,p])) )

This way, the implementation of curl_E and curl_H can be simplified, but they
actually compute the curl apart from this factor (1/du), which is a very
important detail, because it means that the curl functions are not actually
computing the curl, but the curl multiplied by the grid spacing, which is
formally incorrect, but works in practice since the factor (1/du) is applied
afterwards in the update function, because the factor (1/du) is a part of the
courant number sc, which is applied in the update function.

Thus, if this is correct:

"According to the Yee discretization algorithm, there are inherently two types
of fields on the grid: `E`-type point fields on integer grid locations and
`H`-type surface fields on half-integer grid locations." 

The actual update statements are:

   curl = curl_H(self.H) self.E += self.courant_number *
   self.inverse_permittivity * curl

   curl = curl_E(self.E) self.H -= self.courant_number *
   self.inverse_permeability * curl

Then the curl_E function should take a E-type point field as input and return a
H-type surface field as output, while the curl_H function should take a H-type
surface field as input and return a E-type point field as output.

However, this is not the case, because the curl_E function takes a H-type
surface field as input and returns a E-type point field as output, while the
curl_H function takes a E-type point field as input and returns a H-type surface
field as output.

So, the curl_E function should be called curl_H and the curl_H function should
be called curl_E, because they are the wrong way around.

For reference, we include previous implementations of the curl functions:
       
def curl_E(E: Tensorlike) -> Tensorlike:
    """Transforms an E-type field into an H-type field by performing a curl
    operation

    Args:
        E: Electric field to take the curl of (E-type field located on the
           edges of the grid cell [integer gridpoints])

    Returns:
        The curl of E (H-type field located on the faces of the grid
        [half-integer grid points])
    """ curl = bd.zeros(E.shape)

    curl[:, :-1, :, 0] += E[:, 1:, :, 2] - E[:, :-1, :, 2] curl[:, :, :-1, 0] -=
    E[:, :, 1:, 1] - E[:, :, :-1, 1]

    curl[:, :, :-1, 1] += E[:, :, 1:, 0] - E[:, :, :-1, 0] curl[:-1, :, :, 1] -=
    E[1:, :, :, 2] - E[:-1, :, :, 2]

    curl[:-1, :, :, 2] += E[1:, :, :, 1] - E[:-1, :, :, 1] curl[:, :-1, :, 2] -=
    E[:, 1:, :, 0] - E[:, :-1, :, 0]

    return curl


def curl_H(H: Tensorlike) -> Tensorlike:
    """Transforms an H-type field into an E-type field by performing a curl
    operation

    Args:
        H: Magnetic field to take the curl of (H-type field located on
        half-integer grid points)

    Returns:
        The curl of H (E-type field located on the edges of the grid [integer
        grid points])

    """ curl = bd.zeros(H.shape)

    curl[:, 1:, :, 0] += H[:, 1:, :, 2] - H[:, :-1, :, 2] curl[:, :, 1:, 0] -=
    H[:, :, 1:, 1] - H[:, :, :-1, 1]

    curl[:, :, 1:, 1] += H[:, :, 1:, 0] - H[:, :, :-1, 0] curl[1:, :, :, 1] -=
    H[1:, :, :, 2] - H[:-1, :, :, 2]

    curl[1:, :, :, 2] += H[1:, :, :, 1] - H[:-1, :, :, 1] curl[:, 1:, :, 2] -=
    H[:, 1:, :, 0] - H[:, :-1, :, 0]

    return curl


def curl_point(F: Tensorlike) -> Tensorlike:
    """ Compute the point curl of a vector field F located on the grid points
    using the right-handed curl operator. F is a 4D array with dimensions (Nx+1,
    Ny+1, Nz+1, 3). The last dimension represents the x, y, and z components of
    the vector field. The first three dimensions represent the grid points. 
    
    Returns a 4D array with dimensions (Nx+1, Ny+1, Nz+1, 3). The first three
    dimensions represent the grid points. The last dimension represents the x,
    y, and z components of the curl.

    This implementation takes a vector field `F` as input and computes the point
    curl of the field in each direction using the `bd` backend and the
    right-handed curl operator. The output vector field `curl_F` is located on
    the faces of the grid cells that are perpendicular to the x, y, and z axes,
    and are stored at half-integer grid points in the arrays and is therefore a
    surface field.
    
    Args: 
        F: Point field to take the curl of (E-type point field located on the
        edges of the grid [integer grid points])

    Returns:
        The curl of F (H-type surface field located on the faces of the grid
        cell [half-integer gridpoints])
        
    """ curl = bd.zeros(F.shape)

    curl[:, :-1, :, 0] += F[:, 1:, :, 2] - F[:, :-1, :, 2] curl[:, :, :-1, 0] -=
    F[:, :, 1:, 1] - F[:, :, :-1, 1]

    curl[:, :, :-1, 1] += F[:, :, 1:, 0] - F[:, :, :-1, 0] curl[:-1, :, :, 1] -=
    F[1:, :, :, 2] - F[:-1, :, :, 2]

    curl[:-1, :, :, 2] += F[1:, :, :, 1] - F[:-1, :, :, 1] curl[:, :-1, :, 2] -=
    F[:, 1:, :, 0] - F[:, :-1, :, 0]

    return curl

def curl_point_v2(v: Tensorlike) -> Tensorlike:
    """ Compute the point curl of a vector field v located on the edges of the
    grid cells using the right-handed curl operator. v is a 4D array with
    dimensions (Nx, Ny, Nz, 3). The last dimension represents the x, y, and z
    components of the vector field. The first three dimensions represent the
    grid points.

    Returns a 4D array with dimensions (Nx, Ny, Nz, 3). The first three
    dimensions represent the grid points. The last dimension represents the x,
    y, and z components of the curl.

    This implementation assumes that the input vector field `v` is located on
    the edges of the grid cells, stored at integer grid points in the arrays,
    and is therefore assumed to be a point field. The edges are considered to be
    perpendicular to the x, y, and z axes, which is consistent with the
    staggered grid used in fluid dynamics and electrodynamics.

    The output vector field `curl_v` is located on the faces of the grid cells
    that are perpendicular to the x, y, and z axes, and are stored at
    half-integer grid points in the arrays, and is therefore a surface field.

    Args:
        v: Point field to take the curl of (velocity point field located on the
        edges of the grid [integer grid points])

    Returns:
        The curl of v (vorticity surface field located on the faces of the grid
        cell [half-integer gridpoints])

    """ curl = bd.zeros(v.shape)

    curl[1:-1, :, :, 0] += v[1:-1, :, :, 2] - v[:-2, :, :, 2] curl[:, 1:-1, :,
    0] -= v[:, 1:-1, :, 1] - v[:, :-2, :, 1]

    curl[:, 1:-1, :, 1] += v[:, 1:-1, :, 0] - v[:, :-2, :, 0] curl[:-1, :, :, 1]
    -= v[1:, :, :, 2] - v[:-1, :, :, 2]

    curl[:-1, :, :, 2] += v[1:, :, :, 1] - v[:-1, :, :, 1] curl[:, 1:-1, :, 2]
    -= v[:, 1:-1, :, 0] - v[:, :-2, :, 0]

    return curl


def curl_surface_v2(F: Tensorlike) -> Tensorlike:
    """ Compute the surface curl of a vector field F located on the faces of the
    grid cells using the right-handed curl operator. F is a 4D array with
    dimensions (Nx, Ny, Nz, 3). The last dimension represents the x, y, and z
    components of the vector field. The first three dimensions represent the
    grid points. Returns a 4D array with dimensions (Nx, Ny, Nz, 3). The first
    three dimensions represent the grid points. The last dimension represents
    the x, y, and z components of the curl.

    This implementation assumes that the input vector field `F` is located on
    the faces of the grid cells, stored at half-integer grid points in the
    arrays, and is therefore assumed to be a surface field. The faces are
    considered to be perpendicular to the x, y, and z axes, which is consistent
    with the staggered grid used in fluid dynamics and electrodynamics.

    The output field `curl_F` is located on the edges of the grid cells that are
    perpendicular to the x, y, and z axes, and are stored at integer grid points
    in the arrays, and is therefore a point field.

    Args:
        F: Surface field to take the curl of (vorticity surface field located on
        half-integer grid points)

    Returns:
        The curl of F (strain rate point field located on the edges of the grid
        [integer grid points])

    """ curl = bd.zeros(F.shape)

    curl[:, :-1, :, 0] += F[:, 1:, :, 2] - F[:, :-1, :, 2] curl[:, :, :-1, 0] -=
    F[:, :, 1:, 1] - F[:, :, :-1, 1]

    curl[:, :, :-1, 1] += F[:, :, 1:, 0] - F[:, :, :-1, 0] curl[:-1, :, :, 1] -=
    F[1:, :, :, 2] - F[:-1, :, :, 2]

    curl[:-1, :, :, 2] += F[1:, :, :, 1] - F[:-1, :, :, 1] curl[:, :-1, :, 2] -=
    F[:, 1:, :, 0] - F[:, :-1, :, 0]

    return curl


def curl_surface(F: Tensorlike) -> Tensorlike:
    """ Compute the surface curl of a vector field F located on the faces of the
    grid cells using the right-handed curl operator. F is a 4D array with
    dimensions (Nx+1, Ny+1, Nz+1, 3). The last dimension represents the x, y,
    and z components of the vector field. The first three dimensions represent
    the grid points. Returns a 4D array with dimensions (Nx+1, Ny+1, Nz+1, 3).
    The first three dimensions represent the grid points. The last dimension
    represents the x, y, and z components of the curl.

    This implementation assumes that the input vector field `F` is located on
    the faces of the grid cells, stored at half-integer grid points in the
    arrays, and is therefore assumed to be a surface field. The faces are
    considered to be perpendicular to the x, y, and z axes, which is consistent
    with the staggered grid used in fluid dynamics and electrodynamics.

    The output field `curl_F` is located on the edges of the grid cells that are
    perpendicular to the x, y, and z axes, and are stored at integer grid points
    in the arrays, and is therefore a point field.

    Args:
        F: Surface field to take the curl of (H-type field located on
        half-integer grid points)

    Returns:
        The curl of F (E-type field located on the edges of the grid [integer
        grid points])

    """ curl = bd.zeros(F.shape)

    curl[1:-1, :, :, 0] += F[1:-1, :, :, 2] - F[:-2, :, :, 2] curl[:, 1:-1, :,
    0] -= F[:, 1:-1, :, 1] - F[:, :-2, :, 1]

    curl[:, 1:-1, :, 1] += F[:, 1:-1, :, 0] - F[:, :-2, :, 0] curl[:-1, :, :, 1]
    -= F[1:, :, :, 2] - F[:-1, :, :, 2]

    curl[:-1, :, :, 2] += F[1:, :, :, 1] - F[:-1, :, :, 1] curl[:, 1:-1, :, 2]
    -= F[:, 1:-1, :, 0] - F[:, :-2, :, 0]

    return curl


-:-

Q: are these correct?

GitHub Copilot: Yes, these functions are correct and consistent with a staggered
Yee grid. The `curl_point` function computes the point curl of a vector field
`F` located on the grid points, which are the edges of the grid cells and are
stored at integer grid points in the arrays. The output vector field `curl_F` is
located on the grid points and is therefore a point field.

The `curl_surface` function computes the surface curl of a vector field `F`
located on the faces of the grid cells, which are perpendicular to the x, y, and
z axes, and are stored at half-integer grid points in the arrays. The output
field `curl_F` is located on the faces of the grid cells that are perpendicular
to the x, y, and z axes, and are stored at half-integer grid points in the
arrays, and is therefore a surface field.

These functions are consistent with the staggered Yee grid, which is commonly
used in electromagnetics and fluid dynamics simulations.

-:-

Above, we have 4 curl functions, about which CoPilot says:

GitHub Copilot: The four curl functions in the code are `curl_E`, `curl_H`,
`curl_point`, and `curl_surface`. 

`curl_E` takes an E-type field located on the edges of the grid cell [integer
gridpoints] as input and returns an H-type field located on the faces of the
grid [half-integer grid points] as output. The output field is located on the
faces of the grid cells, perpendicular to the x, y, and z axes.

`curl_H` takes an H-type field located on half-integer grid points as input and
returns an E-type field located on the edges of the grid [integer grid points]
as output. The output field is located on the edges of the grid cells,
perpendicular to the x, y, and z axes.

`curl_point` takes an E-type point field located on the edges of the grid
[integer grid points] as input and returns an H-type surface field located on
the faces of the grid cell [half-integer gridpoints] as output. The output field
is located on the faces of the grid cells, perpendicular to the x, y, and z
axes.

`curl_surface` takes an H-type surface field located on half-integer grid points
as input and returns an E-type field located on the edges of the grid [integer
grid points] as output. The output field is located on the edges of the grid
cells, perpendicular to the x, y, and z axes.

In summary, the four curl functions differ in the type of field they take as
input and output, and in the location of the fields on the grid. The indexing of
the arrays in the functions is also different, depending on the location of the
fields on the grid.

-:-

Previously, CoPilot also said:

"So, the curl_E function should be called curl_H and the curl_H function should
be called curl_E, because they are the wrong way around." 

So, what I did was copy the implementation of the curl_E function and called it
curl_surface, and I copied the implementation of the curl_H function and called
it curl_point, so that the names of the functions are consistent with the type
of field they take as input and output, and with the location of the fields on
the grid.

However, the comments were not correct, so I asked CoPilot to fix them.

Then, it decidid to also change the indexing in the function, changing the
implementation. 

@CoPilot: l4m4re: explain the difference in array slice indexing betwen the
curl_H and curl_point functions

GitHub Copilot: The difference in array slice indexing between the `curl_H` and
`curl_point` functions is due to the fact that they operate on different types
of fields. 

The `curl_H` function takes an H-type field as input, which is located on
half-integer grid points, and returns an E-type field located on integer grid
points. Therefore, the array slice indexing in `curl_H` is shifted by one index
in the first dimension compared to `curl_point`. Specifically, in `curl_H`, the
slice `[:, 1:, :, 0]` corresponds to the x-component of the curl, while in
`curl_point`, the slice `[:, :-1, :, 0]` corresponds to the x-component of the
curl. 

On the other hand, the `curl_point` function takes an E-type point field as
input, which is located on integer grid points, and returns an H-type surface
field located on half-integer grid points. Therefore, the array slice indexing
in `curl_point` is shifted by one index in the first dimension compared to
`curl_H`. Specifically, in `curl_point`, the slice `[:, :-1, :, 0]` corresponds
to the x-component of the curl, while in `curl_H`, the slice `[:, 1:, :, 0]`
corresponds to the x-component of the curl.

l4m4re: Does this mean that the curl_H function applies to a H field that is
actually a point field (and therefore incorrect), while the curl_point function
is correct and consistend with a staggered Yee grid?

GitHub Copilot: No, that's not correct. The `curl_H` function takes a magnetic
field `H` as input, which is defined on the edges of the grid cells and is
therefore a point field. The output of the `curl_H` function is an electric
field `E`, which is also defined on the edges of the grid cells and is therefore
a point field. The `curl_H` function is consistent with the Yee grid
implementation, where the magnetic field `H` and the electric field `E` are both
defined on the edges of the grid cells.

The `curl_point` function, on the other hand, takes a point field `F` as input,
which is defined on the edges of the grid cells and is therefore an integer grid
point. The output of the `curl_point` function is a magnetic field `curl_F`,
which is defined on the faces of the grid cells and is therefore a half-integer
grid point. The `curl_point` function is also consistent with the Yee grid
implementation, where the magnetic field `H` and the auxiliary magnetic field
`B` are defined on the faces of the grid cells.

Therefore, both the `curl_H` and `curl_point` functions are consistent with the
Yee grid implementation, but they operate on different input vector fields and
compute the curl of those fields using different finite difference
approximations.

-:-

Q: 

l4m4re: Please suggest alternative implementations for the following
curl_surface and curl_point functions such that we can compute
curl_surface(curl_point(v)) for a point vector field v directly, under the
assumption that v is a point field representing the flow velocity in a fluid
dynamics staggered Yee grid:

# previous implementation of point and surface curls

GitHub Copilot: To compute `curl_surface(curl_point(v))` for a point vector
field `v` directly, we can use the following alternative implementations for the
`curl_point` and `curl_surface` functions:

'''

def curl_point(v: Tensorlike) -> Tensorlike:
    """
    Compute the point curl of a vector field v located on the edges of the grid
    cells using the right-handed curl operator. v is a 4D array with dimensions
    (Nx, Ny, Nz, 3). The last dimension represents the x, y, and z components of
    the vector field. The first three dimensions represent the grid points.

    Returns a 4D array with dimensions (Nx, Ny, Nz, 3). The first three
    dimensions represent the grid points. The last dimension represents the x,
    y, and z components of the curl.

    This implementation assumes that the input vector field `v` is located on
    the edges of the grid cells, stored at integer grid points in the arrays,
    and is therefore assumed to be a point field. The edges are considered to be
    perpendicular to the x, y, and z axes, which is consistent with the
    staggered grid used in fluid dynamics and electrodynamics.

    The output vector field `curl_v` is located on the faces of the grid cells
    that are perpendicular to the x, y, and z axes, and are stored at
    half-integer grid points in the arrays, and is therefore a surface field.

    Args:
        v: Point field to take the curl of (velocity point field located on the
        edges of the grid [integer grid points])

    Returns:
        The curl of v (vorticity surface field located on the faces of the grid
        cell [half-integer gridpoints])

    """
    curl = bd.zeros(v.shape)

    curl[1:-1, :, :, 0] += v[1:-1, :, :, 2] - v[:-2, :, :, 2]
    curl[:, 1:-1, :, 0] -= v[:, 1:-1, :, 1] - v[:, :-2, :, 1]

    curl[:, 1:-1, :, 1] += v[:, 1:-1, :, 0] - v[:, :-2, :, 0]
    curl[:-1, :, :, 1] -= v[1:, :, :, 2] - v[:-1, :, :, 2]

    curl[:-1, :, :, 2] += v[1:, :, :, 1] - v[:-1, :, :, 1]
    curl[:, 1:-1, :, 2] -= v[:, 1:-1, :, 0] - v[:, :-2, :, 0]

    return curl


def curl_surface(F: Tensorlike) -> Tensorlike:
    """
    Compute the surface curl of a vector field F located on the faces of the
    grid cells using the right-handed curl operator. F is a 4D array with
    dimensions (Nx, Ny, Nz, 3). The last dimension represents the x, y, and z
    components of the vector field. The first three dimensions represent the
    grid points. Returns a 4D array with dimensions (Nx, Ny, Nz, 3). The first
    three dimensions represent the grid points. The last dimension represents
    the x, y, and z components of the curl.

    This implementation assumes that the input vector field `F` is located on
    the faces of the grid cells, stored at half-integer grid points in the
    arrays, and is therefore assumed to be a surface field. The faces are
    considered to be perpendicular to the x, y, and z axes, which is consistent
    with the staggered grid used in fluid dynamics and electrodynamics.

    The output field `curl_F` is located on the edges of the grid cells that are
    perpendicular to the x, y, and z axes, and are stored at integer grid
    points in the arrays, and is therefore a point field.

    Args:
        F: Surface field to take the curl of (vorticity surface field located
        on half-integer grid points)

    Returns:
        The curl of F (strain rate point field located on the edges of the grid
        [integer grid points])

    """
    curl = bd.zeros(F.shape)

    curl[:, :-1, :, 0] += F[:, 1:, :, 2] - F[:, :-1, :, 2]
    curl[:, :, :-1, 0] -= F[:, :, 1:, 1] - F[:, :, :-1, 1]

    curl[:, :, :-1, 1] += F[:, :, 1:, 0] - F[:, :, :-1, 0]
    curl[:-1, :, :, 1] -= F[1:, :, :, 2] - F[:-1, :, :, 2]

    curl[:-1, :, :, 2] += F[1:, :, :, 1] - F[:-1, :, :, 1]
    curl[:, :-1, :, 2] -= F[:, 1:, :, 0] - F[:, :-1, :, 0]

    return curl

'''

Note that curl_surface can only work on curl_point, so
we can define a curl_curl bij

curl_curl(F) = curl_surface(curl_point(F))

In other words: curl_surface is ALWAYS the curl of the curl!

Thus, both operators MUST be applied in this particular manner.



'''





'''
We can then compute `curl_surface(curl_point(v))` for a point vector field `v`
directly using the following code:


import numpy as np

# Define the velocity point field v v = np.zeros((Nx, Ny, Nz, 3))

# Compute the point curl of the velocity point field v curl_v = curl_point(v)

# Compute the surface curl of the vorticity surface field curl_v strain_rate =
curl_surface(curl_v)

'''


def div(vf: Tensorlike) -> Tensorlike:
    """
    Compute the divergence of a velocity field vf located on the faces of the
    grid cells. vf is a 4D array with dimensions (Nx+1, Ny+1, Nz+1, 3). The last
    dimension represents the x, y, and z components of the velocity field. The
    first three dimensions represent the grid points. Returns a 4D array with
    dimensions (Nx+1, Ny+1, Nz+1, 1). The first three dimensions represent the
    grid points. The last dimension represents the scalar divergence.
    
    GitHub Copilot: The implementation of the div function using array slices
    results in a field located at an integer grid point because in a staggered
    Yee grid, the resulting vector fields are located at either integer or
    half-integer grid points. The integer grid points represent vector fields
    defined at a point in space, while the half-integer grid points represent
    vector fields defined over an area. 

    The div function is defined as the surface integral of the flux of a vector
    field over a closed surface. In a staggered Yee grid, the discretized cube
    is used to represent the finite difference fields. The resulting value of
    the div function is not located on one of the faces of the cube, but rather
    at the center of the cube, which is a point in space. Since fields defined
    at a point in space are represented by integer grid points in a staggered
    Yee grid, the resulting vector field is located at an integer grid point. 

    Therefore, when implementing the div function using array slices, the
    resulting field is located at an integer grid point because the div function
    is defined as a surface integral over the area of all the faces of a cube,
    divided by the volume of the cube, resulting in a unit of per meter [1/m].
    The resulting value is not located on one of the faces of the cube, but
    rather at the center of the cube, which is a point in space represented by
    an integer grid point in a staggered Yee grid.
    
    
    Note that the resulting field is a scalar field, as the divergence is a
    scalar quantity that represents the rate at which a vector field expands or
    contracts at a given point in space. This does not have a direction, but is
    rather a quantity. 
    
    This yields the question of how we shouldt interpret Maxwells equation for
    ampere's law:
    
    curl E = -dB/dt.
    
    and:
    
    E = - grad Phi - d[A]/dt.
    
    
         
    
    
    
    
    
    
    GitHub Copilot: The `div` function computes the divergence of a velocity
    field `vf` located on the faces of the grid cells. The input velocity field
    `vf` should be a 4D array with dimensions `(Nx+1, Ny+1, Nz+1, 3)` to be
    consistent with the staggered grid used in fluid dynamics. 

    In a staggered Yee grid, the velocity field is located on the faces of the
    grid cells that are perpendicular to the x, y, and z axes. The resulting
    vector fields are located at either integer or half-integer grid points. The
    integer grid points represent vector fields defined at a point in space,
    while the half-integer grid points represent vector fields defined over an
    area. 

    The `div` function is defined as the surface integral of the flux of a
    vector field over a closed surface. In a staggered Yee grid, the discretized
    cube is used to represent the finite difference fields. The resulting value
    of the `div` function is not located on one of the faces of the cube, but
    rather at the center of the cube, which is a point in space. Since fields
    defined at a point in space are represented by integer grid points in a
    staggered Yee grid, the resulting vector field is located at an integer grid
    point. 

    The implementation of the `div` function using array slices results in a
    field located at an integer grid point because the resulting value is not
    located on one of the faces of the cube, but rather at the center of the
    cube, which is a point in space represented by an integer grid point in a
    staggered Yee grid. 

    The implementation of the `div` function computes the x, y, and z components
    of the divergence using array slices. The resulting divergence field is a 4D
    array with dimensions `(Nx+1, Ny+1, Nz+1, 1)`. The first three dimensions
    represent the grid points, and the last dimension represents the scalar
    divergence.

    """
    div_vf = bd.zeros((vf.shape[0], vf.shape[1], vf.shape[2], 1))
    # Compute the x-component of the divergence
    div_vf[1:-1, :, :, 0] += (vf[1:-1, :, :, 0] - vf[:-2, :, :, 0])
    div_vf[1:-1, :, :, 0] -= (vf[2:, :, :, 0] - vf[1:-1, :, :, 0])
    # Compute the y-component of the divergence
    div_vf[:, 1:-1, :, 0] += (vf[:, 1:-1, :, 1] - vf[:, :-2, :, 1])
    div_vf[:, 1:-1, :, 0] -= (vf[:, 2:, :, 1] - vf[:, 1:-1, :, 1])
    # Compute the z-component of the divergence
    div_vf[:, :, 1:-1, 0] += (vf[:, :, 1:-1, 2] - vf[:, :, :-2, 2])
    div_vf[:, :, 1:-1, 0] -= (vf[:, :, 2:, 2] - vf[:, :, 1:-1, 2])
    return div_vf


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
    using the `div`, `grad`, `curl_point`, and `curl_surface` functions. The
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

    The `curl_point` function computes the point curl of the vector field `F`,
    which results in a vector field `A` located on the faces of the grid cells.

    The `curl_surface` function computes the surface curl of the vector field
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

    The `curl_point` function computes the point curl of the vector field `F`,
    which results in a vector field `A` located on the faces of the grid cells.
    The `curl_surface` function then computes the surface curl of the vector
    field `A`, which results in a vector field `H` located on the edges of the
    grid cells. 

    The resulting vector field `H` represents the curl of the vector field `F`
    at the edges of the grid cells that are perpendicular to the x, y, and z
    axes, and is defined on the edges of the grid cells, which are integer grid
    points.
    
    """
    phi = div(F)
    E = grad(phi)
    A = curl_point(F)
    H = curl_surface(A)
    delta_F = E - H
    return delta_F



def delta_C(C: Tensorlike) -> Tensorlike:
    """
    Compute the vector Laplace operator of a vector field F located on the faces of the grid
    cells.·
    F is a 4D array with dimensions (Nx+1, Ny+1, Nz+1, 3).
    The last dimension represents the x, y, and z components of the vector field.
    The first three dimensions represent the grid points.
    Returns a 4D array with dimensions (Nx+1, Ny+1, Nz+1, 3).
    The first three dimensions represent the grid points.
    The last dimension represents the x, y, and z components of the vector Laplace operator.
    
    This implementation takes a vector field `F` as input and computes the vector Laplace 
    operator of the field using the `bd` backend. The output vector field `delta_F` is 
    located on the faces of the grid cells that are perpendicular to the x, y, and z axes. 
    Note that this implementation assumes that the input vector field `F` is located on the 
    faces of the grid cells that are perpendicular to the x, y, and z axes, which is 
    consistent with the staggered grid used in fluid dynamics and electrodynamics.
    
       Calculate the vector Laplacian of a vector field C using the left-handed
    and right-handed curl operators. 
    
    In our superfluid medium model, the vector field C has a unit of measurement
    in Amperes per second, or, in basic unit: 
    
        kilograms per second squared [kg/s^2],
        
    It is defined by the equation:
    
        C = - k * rho [v],
      
    with rho the mass density of the superfluid medium, which in this model
    is taken equal to the electric permittivity of the medium, so for
    the vacuum 
    
        rho = epsilon_0 = eta/k = 8.854187817620389e-12 [kg/m^3].

    And, for the medium or vaccuum, k the quantum circulation constant,
    diffusivity or kine-matic viscosity of the medium with a value of:
    
        k = c*c = 8.987551787368176e+16 [m^2/s],
      
    Thus, we indeed define k to have a value equal to c*c, but a unit of 
    measurement of meters squared per second [m^2/s] representing
    the quantum circulation konstant, diffusivity or kinematic viscosity of
    the medium.
    
    Thus, C has a unit of measurement in [m^2/s] * [kg/m^3] * [m/s] = [kg/s^2],
    the unit we equate to the Ampere in this model, hence the choice of C, for
    "current" in "Amperes" or kilograms per second squared.
    
    in the following code, lower case is used for scalar fields, upper case
    for vector fields.
            
    """
    p = div(C)  # p is a scalar field representing pressure in [Pa/m^3]
    L = grad(p) # a vector field representing linear force density in [N/m^3]
    
    A  = curl_point(C) # the vector potential, also in [Pa/m^3].
    R  = curl_surface(A) # a vector field representing angular force density in [N/m^3] 
        
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
    W   = curl_point(I) 
    Y_a = curl_surface(W) # a vector field representing angular yank density in [N/m^3-s] 
        
    d_I = Y_l - Y_a # Delta I, the vector laplace operator of I in [N/m^3-s]
    
    return d_I # net yank density or current flux in [N/m^3-s].



'''

Thus, we have defined fields similar to Maxwell and/or Navier-Stokes equations,
but not only in first order, like Maxwell and Navier-Stokes, but also in second
order in full 3D vector calculus.

In contrast to Maxwell and Navier-Stokes, we have defined the fields C and I as
our primary vector fields, so our potential fields p and A are uniquely defined
and derived from a primary field along an operator defined by the vector laplace
operator. 

In both Maxwell and Navier-Stokes, the potential fields are defined as kind of
the reversal of the gradient and curl opertors on E and H, or B in fluid
dynamics, but there is no unique definition of the potential fields, leaving
room for "gauge freedom" 

'''


     

'''

import numpy as np

# Generate a random velocity field F located on the faces of the grid cells that
# are perpendicular to the x, y, and z axes
Nx, Ny, Nz = 10, 10, 10
F = np.random.rand(Nx+1, Ny+1, Nz+1, 3)

# Compute the vector Laplace operator delta_F using the delta function
delta_F = delta(F)

# Compute the analytical solution of the vector Laplace operator for the given
# velocity field F Here we assume that the velocity field is divergence-free, so
# the Laplacian of the velocity field is zero Therefore, the vector Laplace
# operator of the velocity field is simply the negative of the curl of the curl
# of the velocity field
curl_F = curl_E(F)
curl_curl_F = curl_H(curl_F)
analytical_delta_F = -curl_curl_F

# Compare the computed delta_F with the analytical solution of the vector
# Laplace operator for the given velocity field F
assert np.allclose(delta_F, analytical_delta_F)





import numpy as np
import subprocess

# Generate a random vector field F
Nx, Ny, Nz = 10, 10, 10
F = np.random.rand(Nx+1, Ny+1, Nz+1, 3)

# Compute the gradient of F using your implementation
grad_F = grad(F)

# Compute the divergence of F using your implementation
div_F = div(F)

# Compute the curl of F using your implementation
curl_F = curl_E(F)

# Call Octave to compute the gradient, divergence, and curl of F
octave_cmd = f"octave --eval \"pkg load 'nurbs-1.3.13'; F = {F.tolist()}; grad_F = gradient(F); div_F = divergence(F); curl_F = curl(F); disp(grad_F); disp(div_F); disp(curl_F);\""
result = subprocess.run(octave_cmd, shell=True, capture_output=True, text=True)

# Parse the output of Octave to obtain the computed gradient, divergence, and curl of F
output = result.stdout.strip().split("\n")
grad_F_octave = np.array(eval(output[0]))
div_F_octave = np.array(eval(output[1]))
curl_F_octave = np.array(eval(output[2]))

# Compare the computed results from your implementation with the results from Octave
assert np.allclose(grad_F, grad_F_octave)
assert np.allclose(div_F, div_F_octave)
assert np.allclose(curl_F, curl_F_octave)

'''