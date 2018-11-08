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

The number `(c*dt/du)` is a dimensionless parameter called the *courant number* `sc`. For
stability reasons, the courant number should always be smaller than `1/√D`, with `D`
the dimension of the simulation. This can be intuitively be understood as the condition
that information should always travel slower than the speed of light through the grid.
in the FDTD method described here, information can only travel to the neighbouring grid
cells (through application of the curl). It would therefore take `D` timesteps to
travel over the diagonal of a `D`-dimensional cube (square in `2D`, cube in `3D`), the
courant condition follows then automatically from the fact that the length of this
diagonal is `1/√D`.

This yields the final update equations for the FDTD algorithm:


```
    E  += sc*inv(ε)*curl_H
    H  -= sc*inv(µ)*curl_E
```

This is also how it is implemented:

```python
class Grid:
    # ... [initialization]

    def step(self):
        self.update_E()
        self.update_H()

    def update_E(self):
        self.E += self.courant_number * self.inverse_permittivity * curl_H(self.H)

    def update_H(self):
        self.H -= self.courant_number * self.inverse_permeability * curl_E(self.E)
```

## Sources


Ampere's Law can be updated to incorporate a current density:
```
    curl(H) = J + ε*ε0*dE/dt
```
Making again the usual substitutions `sc := c*dt/du`, `E := √ε0*E` and `H := √µ0*H`, the
update equations can be modified to include the current density:

```
    E += sc*inv(ε)*curl_H - dt*inv(ε)*J/√ε0
```

Making one final substitution `Es := -dt*inv(ε)*J/√ε0` allows us to write this in a
very clean way:
```
    E += sc*inv(ε)*curl_H + Es
```

Where we defined Es as the *electric field source term*.

It is often useful to also define a *magnetic field source term* `Hs`, which would be
derived from the *magnetic current density* if it were to exist. In the same way,
Faraday's update equation can be rewritten as
```
    H  -= sc*inv(µ)*curl_E + Hs
```

```python
class Source:
    # ... [initialization]
    def update_E(self):
        # electric source function here

    def update_H(self):
        # magnetic source function here

class Grid:
    # ... [initialization]
    def update_E(self):
        # ... [electric field update equation]
        for source in self._sources:
            source.update_E()

    def update_H(self):
        # ... [magnetic field update equation]
        for source in self._sources:
            source.update_H()

```

## Lossy Medium

When a material has a conductivity σ, a conduction-current will ensure that the medium is lossy. Ampere's law with a conduction current becomes
```
    curl(H) = σE + ε*ε0*dE/dt
```

making the usual subsitutions, this becomes:
```
    E(t+dt) - E(t) = sc*inv(ε)*curl_H(t+dt/2) - dt*inv(ε)*σE(t+dt/2)/√ε0
```

This update equation depends on the electric field on a half-integer timestep (a *magnetic field timestep*). We need to make a substitution to interpolate the electric field to this timestep:
```
    (1 + 0.5*dt*inv(ε)*σ/√ε0)*E(t+dt) = sc*inv(ε)*curl_H(t+dt/2) + (1 - 0.5*inv(ε)*σ/√ε0)*E(t)
```

Which, after substitution `σ := σ/√ε0` yield the new update equations:
```
    f = 0.5*inv(ε)*σ
    E *= inv(1 + f) * (1 - f)
    E += inv(1 + f)*sc*inv(ε)*curl_H
```

If we want to keep track of the absorption:
```
    f = 0.5*inv(ε)*σ
    Enoabs = E + sc*inv(ε)*curl_H
    E *= inv(1 + f) * (1 - f)
    E += inv(1 + f)*sc*inv(ε)*curl_H
    dE = Enoabs - E
    abs += ε*E*dE + 0.5*ε*dE**2
```

Note that the more complicated the permittivity tensor ε is, the more time consuming this
algorithm will be. It is therefore sometimes the right decision to transfer the absorption to the magnetic domain by introducing a (*nonphysical*) magnetic conductivity, because
the permeability µ usually has a much easier form.

Which, after substitution `σm := σm/√µ0`, we get the magnetic field update equations:
```
    f = 0.5*inv(µ)*σm
    H *= inv(1 + f) * (1 - f)
    H += inv(1 + f)*sc*inv(µ)*curl_E
```

or if we want to keep track of the absorption:
```
    f = 0.5*inv(µ)*σm
    Hnoabs = E + sc*inv(µ)*curl_E
    H *= inv(1 + f) * (1 - f)
    H += inv(1 + f)*sc*inv(µ)*curl_E
    dH = Hnoabs - H
    abs += µ*H*dH + 0.5*µ*dH**2
```


## Boundary Conditions

### Periodic Boundary Conditions

Assuming we want periodic boundary conditions along the `X`-direction, then we have to
make sure that the fields at `Xlow` and `Xhigh` are the same. This has to be enforced
after performing the update equations:

Note that the electric field `E` is dependent on `curl_H`, which means that
the first indices of E will not be updated through the update equations.
It's those indices that need to be set through the periodic boundary condition.
Concretely: `E[0]` needs to be set to equal `E[-1]`. For the magnetic field, the
inverse is true: `H` is dependent on `curl_E`, which means that its last indices
will not be set. This has to be done by the boundary condition: `H[-1]` needs to
be set equal to `H[-1]`:


```python
class PeriodicBoundaryX:
    # ... [initialization]
    def update_E(self):
        self.grid.E[0, :, :, :] = self.grid.E[-1, :, :, :]

    def update_H(self):
        self.grid.H[-1, :, :, :] = self.grid.H[0, :, :, :]

class Grid:
    # ... [initialization]
    def update_E(self):
        # ... [electric field update equation]
        # ... [electric field source functions]
        for boundary in self._boundaries:
            boundary.update_E()

    def update_H(self):
        # ... [magnetic field update equation]
        # ... [magnetic field source functions]
        for boundary in self._boundaries:
            boundary.update_H()
```

