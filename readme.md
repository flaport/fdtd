# Python 3D FDTD Simulator

## [Under Construction]

## Update Equations
An as quick as possible explanation of the FDTD discretization of the maxwell equations.


An electromagnetic FDTD solver solves the time-dependent Maxwell Equations

```
    curl(H) = ε*ε0*dE/dt
    curl(E) = -µ*µ0*dH/dt
```

These two equations are called *Ampere's Law* and *Faraday's Law* respectively.

Where ε and µ are the relative permittivity and permeability tensors
respectively. ε0 and µ0 are the vacuum permittivity and permeability and their
square root can be absorbed into E and H respectively, such that `E := √ε0*E`
and `H := √µ0*H`.

Doing this, the Maxwell equations can be written as update equations:
```
    E  += c*dt*inv(ε)*curl(H)
    H  -= c*dt*inv(µ)*curl(E)
```
The electric and magnetic field can then be discretized on a grid with
interlaced Yee-coordinates, which in 3D looks like this:

![grid discretization in 3D](yee.svg)

According to the Yee discretization algorithm, there are inherently two types
of fields on the grid: `E`-type fields on integer grid locations and `H`-type
fields on half-integer grid locations.

The beauty of these interlaced coordinates is that they enable a very natural
way of writing the curl of the electric and magnetic fields: the curl of an
H-type field will be an E-type field and vice versa.

This way, the curl of E can be written as
```
    curl(E)[m,n,p] = (dEz/dy - dEy/dz, dEx/dz - dEz/dx, dEy/dx - dEx/dy)[m,n,p]
                   =( ((Ez[m,n+1,p]-Ez[m,n,p])/dy - (Ey[m,n,p+1]-Ey[m,n,p])/dz),
                      ((Ex[m,n,p+1]-Ex[m,n,p])/dz - (Ez[m+1,n,p]-Ez[m,n,p])/dx),
                      ((Ey[m+1,n,p]-Ey[m,n,p])/dx - (Ex[m,n+1,p]-Ex[m,n,p])/dy) )
                   =(1/du)*( ((Ez[m,n+1,p]-Ez[m,n,p]) - (Ey[m,n,p+1]-Ey[m,n,p])), [assume dx=dy=dz=du]
                             ((Ex[m,n,p+1]-Ex[m,n,p]) - (Ez[m+1,n,p]-Ez[m,n,p])),
                             ((Ey[m+1,n,p]-Ey[m,n,p]) - (Ex[m,n+1,p]-Ex[m,n,p])) )

```
this can be written efficiently with array slices (note that the factor
`(1/du)` was left out):

```python
def curl_E(E):
    curl_E = np.zeros(E.shape)
    curl_E[:,:-1,:,0] += E[:,1:,:,2] - E[:,:-1,:,2]
    curl_E[:,:,:-1,0] -= E[:,:,1:,1] - E[:,:,:-1,1]

    curl_E[:,:,:-1,1] += E[:,:,1:,0] - E[:,:,:-1,0]
    curl_E[:-1,:,:,1] -= E[1:,:,:,2] - E[:-1,:,:,2]

    curl_E[:-1,:,:,2] += E[1:,:,:,1] - E[:-1,:,:,1]
    curl_E[:,:-1,:,2] -= E[:,1:,:,0] - E[:,:-1,:,0]
    return curl_E
```

The curl for H can be obtained in a similar way (note again that the factor
`(1/du)` was left out):
```python
def curl_H(H):
    curl_H = np.zeros(H.shape)

    curl_H[:,1:,:,0] += H[:,1:,:,2] - H[:,:-1,:,2]
    curl_H[:,:,1:,0] -= H[:,:,1:,1] - H[:,:,:-1,1]

    curl_H[:,:,1:,1] += H[:,:,1:,0] - H[:,:,:-1,0]
    curl_H[1:,:,:,1] -= H[1:,:,:,2] - H[:-1,:,:,2]

    curl_H[1:,:,:,2] += H[1:,:,:,1] - H[:-1,:,:,1]
    curl_H[:,1:,:,2] -= H[:,1:,:,0] - H[:,:-1,:,0]
    return curl_H
```

The update equations can now be rewritten as
```
    E  += (c*dt/du)*inv(ε)*curl_H
    H  -= (c*dt/du)*inv(µ)*curl_E
```

The number `(c*dt/du)` is a dimensionless parameters called the *courant
number*.  For stability reasons, the courant number should always be smaller
than `1/√D`, with `D` the dimension of the simulation. This can be intuitively
understood as being the condition that the field energy may not transit
through more than one mesh cell in a single time step. This yields the final
update equations for the FDTD algorithm:

```
    E  += sc*inv(ε)*curl_H
    H  -= sc*inv(µ)*curl_E
```

```python
class Grid:
    # initialization...

    def step(self):
        self.update_E()
        self.update_H()
        self.timesteps_passed += 1

    def update_E(self):
        curl = curl_H(self.H)
        self.E += self.courant_number * self.inverse_permittivity * curl
        # source implementation here

    def update_H(self):
        curl = curl_E(self.E)
        self.H -= self.courant_number * self.inverse_permeability * curl
        # source implementation here
```

## Sources

Ampere's Law can be updated to incorporate a current density:
```
    curl(H) = J + ε*ε0*dE/dt
```
Making again the usual substitutions `sc:=c*dt/du`, `E := √ε0*E` and `H := √µ0*H`, the
update equations can be modified:

```
    E += sc*inv(ε)*curl_H - dt*inv(ε)*J/√ε0
```

Making one final substitution `Es := -dt*inv(ε)*J/√ε0` allows us to write this in a
very clean way:
```
    E += sc*inv(ε)*curl_H + Es
```

Where we defined Es as the *electric field source term*.

In FDTD simulations, it is often useful to also define a *magnetic field source term*
`Hs`, which would be derived from the *magnetic current density* if it were to exist.
In the same way, Faraday's update equation can be rewritten as
```
    H  -= sc*inv(µ)*curl_E + Hs
```
