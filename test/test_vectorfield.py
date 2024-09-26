import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from synthrouting.vectorfields import (
    Circular,
    ConstantCurrent,
    NoCurrent,
    Sink,
    Source,
    VectorfieldReal,
)
from synthrouting.vectorfields.base import Vectorfield


def test_no_current_vectorfield():
    vectorfield = NoCurrent()
    assert (vectorfield.get_current(0, 0) == jnp.asarray([0, 0])).all()
    assert vectorfield.du(0, 0)[0] == 0
    assert vectorfield.dv(0, 0)[0] == 0
    assert vectorfield.du(0, 0)[1] == 0
    assert vectorfield.dv(0, 0)[1] == 0


def test_circular_vectorfield():
    vectorfield = Circular()
    assert (vectorfield.get_current(0, 0) == jnp.asarray([0.05, -0.15])).all()
    assert vectorfield.du(0, 0)[0] == 0
    assert vectorfield.dv(0, 0)[0] == -0.05
    assert vectorfield.du(0, 0)[1] == 0.05
    assert vectorfield.dv(0, 0)[1] == 0


@pytest.mark.parametrize("x", [-2, 0, 2])
@pytest.mark.parametrize("theta", [0, 1])
@pytest.mark.parametrize("vel", [5, 10])
def test_ode_zermelo(x: float, theta: float, vel: float):
    vf_euclidean = Circular(spherical=False)
    vf_spherical = Circular(spherical=True)
    # We need the conversion factor because spherical uses radians while
    # euclidean uses meters
    rad2m = vf_spherical.rad2m

    p = (x, 0, theta)
    t = [0, 5, 10]
    dx_euc, dy_euc, dt_euc = vf_euclidean.ode_zermelo(p, t, vel=vel)
    dx_sph, dy_sph, dt_sph = vf_spherical.ode_zermelo(p, t, vel=vel)
    np.testing.assert_allclose(dx_euc, dx_sph * rad2m, rtol=1e-5)
    np.testing.assert_allclose(dy_euc, dy_sph * rad2m, rtol=1e-5)
    np.testing.assert_allclose(dt_euc, dt_sph * rad2m, rtol=1e-5)


@pytest.mark.parametrize(
    "vf", [Circular(), Sink(), Source(), ConstantCurrent(), NoCurrent()]
)
def test_jacobian(vf: Vectorfield):
    vf_dis = vf.discretize(x_min=-2, x_max=2, y_min=-2, y_max=2, step=0.01)
    np.testing.assert_allclose(vf.du(0.0, 0.0), vf_dis.du(0.0, 0.0), rtol=1e-3)
    np.testing.assert_allclose(vf.dv(0.0, 0.0), vf_dis.dv(0.0, 0.0), rtol=1e-3)


def test_real_vectorfield():
    u = pd.DataFrame(
        np.random.uniform(low=0, high=5, size=(10, 10)),
        index=np.arange(start=-5, stop=5, step=1),
        columns=np.arange(start=-5, stop=5, step=1),
    )
    v = pd.DataFrame(
        np.random.uniform(low=-5, high=0, size=(10, 10)),
        index=np.arange(start=-5, stop=5, step=1),
        columns=np.arange(start=-5, stop=5, step=1),
    )
    vf = VectorfieldReal(u, v, radians=False)

    # Test one point
    x = jnp.array(vf.arr_x[5])
    y = jnp.array(vf.arr_y[5])
    u, v = vf.get_current(x, y)
    np.testing.assert_allclose(u, vf.u[5, 5], rtol=1e-4)
    np.testing.assert_allclose(v, vf.v[5, 5], rtol=1e-4)

    # Test several points
    x = vf.arr_x
    y = vf.arr_y
    u, v = vf.get_current(x, y)
    np.testing.assert_allclose(u, np.diag(vf.u), rtol=1e-4)
    np.testing.assert_allclose(v, np.diag(vf.v), rtol=1e-4)


def test_land():
    m = np.random.uniform(low=-5, high=5, size=(10, 10))
    m[-2, -2] = np.nan  # In point (3, 3)
    df = pd.DataFrame(
        m,
        index=np.arange(start=-5, stop=5, step=1),
        columns=np.arange(start=-5, stop=5, step=1),
    )
    vf = VectorfieldReal(df, df, radians=False)

    x = jnp.array([2, 2.5, 3, 3, 3, 3.5, 4])
    y = jnp.array([2, 2.5, 2.5, 3, 3.5, 3.5, 4])

    land = vf.is_land(x, y)
    expect = jnp.array([False, True, True, True, True, True, False])
    assert (land == expect).all()

    x = jnp.array([2, 2.5, 3, 3, 3, 3.5, 4])
    y = jnp.array([2, 2.5, 2.5, 3, 3.5, 3.5, 4])

    land = vf.is_land(x, y)
    expect = jnp.array([False, True, True, True, True, True, False])
    assert (land == expect).all()
