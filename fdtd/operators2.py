
'''GitHub Copilot: Sure, here are some possible implementations of the gradient
And curl operators in a staggered Yee grid:

**Gradient operator:**
'''

def gradient(field, dx):
    """
    Compute the gradient of a scalar field using finite differences.

    Parameters
    ----------
    field : ndarray
        Scalar field.
    dx : float
        Grid spacing.

    Returns
    -------
    ndarray
        Gradient of the scalar field.
    """
    grad_x = np.gradient(field, dx, axis=0)
    grad_y = np.gradient(field, dx, axis=1)
    grad_z = np.gradient(field, dx, axis=2)
    return np.stack((grad_x, grad_y, grad_z), axis=-1)


'''
**Point curl operator:**
'''

def curl_point(field, dx):
    """
    Compute the point curl of a vector field using finite differences.

    Parameters
    ----------
    field : ndarray
        Vector field.
    dx : float
        Grid spacing.

    Returns
    -------
    ndarray
        Point curl of the vector field.
    """
    curl_x = np.gradient(field[..., 2], dx, axis=1) - np.gradient(field[..., 1], dx, axis=2)
    curl_y = np.gradient(field[..., 0], dx, axis=2) - np.gradient(field[..., 2], dx, axis=0)
    curl_z = np.gradient(field[..., 1], dx, axis=0) - np.gradient(field[..., 0], dx, axis=1)
    return np.stack((curl_x, curl_y, curl_z), axis=-1)

'''
**Surface curl operator:**
'''
def curl_surface(field, dx):
    """
    Compute the surface curl of a vector field using finite differences.

    Parameters
    ----------
    field : ndarray
        Vector field.
    dx : float
        Grid spacing.

    Returns
    -------
    ndarray
        Surface curl of the vector field.
    """
    curl_x = np.gradient(field[..., 2], dx, axis=1) - np.gradient(field[..., 1], dx, axis=2)
    curl_y = np.gradient(field[..., 0], dx, axis=2) - np.gradient(field[..., 2], dx, axis=0)
    curl_z = np.gradient(field[..., 1], dx, axis=0) - np.gradient(field[..., 0], dx, axis=1)
    return np.stack((curl_x[..., None], curl_y[..., None], curl_z[..., None]), axis=-1)

'''

Note that these implementations use the `np.gradient` function from NumPy to
compute the finite differences. The `dx` parameter is the grid spacing, which is
assumed to be constant in a staggered Yee grid implementation.

'''
