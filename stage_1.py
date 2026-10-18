# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.25.1",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    <h1><b>STAGE 1: JOUKOWSKY AIRFOILS</h1>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - Author: Pascal Bouda - CEA Research Engineer ([pascal.bouda@cea.fr](mailto:pascal.bouda@cea.fr))
    - This notebook develops the physics first. **No particular knowledge is required.**
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.interpolate import CubicSpline

    return mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # 1. THE JOUKOWSKY TRANSFORMATION

    ## 1.1 Motivation

    Understanding how a fluid flows around a solid body is a fundamental
    problem in fluid mechanics. Examples include air flowing around an
    aircraft wing and water flowing around a hydrofoil.

    The mathematical description of real fluid motion can be challenging
    because of viscosity, turbulence, flow separation, and complex
    geometries. A useful first step is to introduce an idealised model:
    **potential flow theory**.

    Potential flow allows us to simplify the governing equations and obtain
    analytical solutions for selected flow configurations. The Joukowsky
    transformation then provides a mathematical technique for connecting
    a simple geometry, such as a circle, to an airfoil-like shape.

    ## 1.2 What is potential flow?

    A velocity field describes the velocity of the fluid at every point.
    In two dimensions, it is usually represented by

    $$\mathbf{V}=(u,v)$$

    If the flow is irrotational, we can introduce a scalar function
    called the **velocity potential**, $\phi$, such that

    $$\mathbf{V}=\nabla \phi$$

    Therefore,

    $$
    u=\frac{\partial\phi}{\partial x},
    \qquad
    v=\frac{\partial\phi}{\partial y}.
    $$

    Instead of determining two velocity components independently, we can
    determine one scalar function and obtain the velocity by differentiation.

    This simplification is powerful, but it requires assumptions about the
    physical behaviour of the fluid.

    ## 1.3 Hypotheses of the potential-flow model

    ### Continuum hypothesis

    We treat the fluid as a continuous medium, with properties such as
    velocity, pressure, and density defined at every point.

    This allows us to describe fluid motion using differential equations
    rather than tracking individual molecules.

    ### Inviscid flow

    We neglect viscous stresses in the region being modelled. The dynamic
    viscosity is denoted by $\mu$.

    This assumption allows us to use the Euler equations instead of the
    full Navier–Stokes equations.

    Real fluids always have viscosity. The approximation is most useful
    in regions where viscous effects are relatively small, away from
    important boundary layers and wakes.

    ### Incompressible flow

    We assume that density is constant, so conservation of mass becomes

    $$
    \nabla\cdot\mathbf{V}=0.
    $$

    In two dimensions,

    $$
    \frac{\partial u}{\partial x}
    +\frac{\partial v}{\partial y}=0.
    $$

    This is often a reasonable approximation for liquid flows and
    low-speed aerodynamic flows. It becomes less appropriate when
    compressibility effects are significant.

    ### Irrotational flow

    We assume that the local vorticity vanishes:

    $$
    \boldsymbol{\omega}
    =\nabla\times\mathbf{V}=\mathbf{0}.
    $$

    In two dimensions, this means

    $$
    \frac{\partial v}{\partial x}
    -\frac{\partial u}{\partial y}=0.
    $$

    Under this condition, a velocity potential can be defined in a
    simply connected region.

    An important subtlety is that irrotational flow does not necessarily
    imply zero circulation around a body. In a region surrounding an
    airfoil, the fluid domain is multiply connected, so nonzero
    circulation can coexist with zero local vorticity in the fluid.

    ### Steady flow

    We assume that the velocity and pressure at a fixed point do not
    change with time:

    $$
    \frac{\partial\mathbf{V}}{\partial t}=0.
    $$

    ### Two-dimensional flow

    We assume that the flow does not vary in the spanwise direction.
    The velocity is considered to lie entirely in the $x$–$y$ plane.

    This is an idealisation of an airfoil with infinite span. A real
    finite wing has three-dimensional effects, including wingtip vortices.

    ## 1.4 From the hypotheses to Laplace's equation

    We now combine two of our assumptions.

    **Step 1 — Conservation of mass**

    For an incompressible fluid,

    $$
    \nabla\cdot\mathbf{V}=0.
    $$

    **Step 2 — Introduce the velocity potential**

    For irrotational flow,

    $$
    \mathbf{V}=\nabla\phi.
    $$

    **Step 3 — Substitute into continuity**

    $$
    \nabla\cdot(\nabla\phi)=0.
    $$

    Since the divergence of the gradient is the Laplacian,

    $$
    \boxed{\nabla^2\phi=0.}
    $$

    For two-dimensional flow, this becomes

    $$
    \boxed{
    \frac{\partial^2\phi}{\partial x^2}
    +\frac{\partial^2\phi}{\partial y^2}=0.
    }
    $$

    This is **Laplace's equation**, the fundamental equation of
    incompressible, irrotational potential flow.

    Laplace's equation is linear. Consequently, under appropriate
    conditions, known potential-flow solutions can be combined to
    construct more complicated flows.

    This property will be useful when studying flow around cylinders
    and, later, airfoils.

    ## 1.5 Boundary conditions

    A governing equation alone is not enough to determine a flow.
    We must also specify how the fluid behaves at the boundaries.

    **Impermeability of the body**

    For a stationary, impermeable surface,

    $$
    \boxed{\mathbf{V}\cdot\mathbf{n}=0}
    $$

    where $\mathbf{n}$ is the unit normal to the body surface.

    This means that fluid cannot pass through the solid. In inviscid
    theory, tangential slip along the surface is allowed.

    **Freestream condition**

    Far from the body, the velocity approaches the prescribed uniform
    freestream velocity:

    $$
    \mathbf{V}\rightarrow\mathbf{V}_\infty.
    $$

    These conditions help define the potential-flow solution around
    the body.

    ## 1.6 Why introduce the Joukowsky transformation?

    We use complex coordinates
    $$
    \zeta=\xi+i\eta,\qquad z=x+iy,
    $$

    The Joukowsky transformation is a conformal mapping between two
    complex planes. A common form is

    $$
    \boxed{z=\zeta+\frac{a^2}{\zeta}}
    $$

    where $\zeta$ is the complex coordinate in a generating plane,
    $z$ is the coordinate in the physical plane, and $a$ is a
    characteristic length.

    We also consider a generating circle with its centre ($\zeta_c=\mu+i\nu$) and radius ($R$). To make the circle pass through the critical point ($\zeta=a$), $R$ is finally such as
    $$
    R=|a-\zeta_c|=\sqrt{(a-\mu)^2+\nu^2}
    $$

    **Sign convention:** the complex potential uses ($dW/d\zeta=u-iv$), positive ($\Gamma$) is counterclockwise, and positive angle of attack is \(\alpha\). The Kutta condition below gives the signed circulation; positive lift for a conventional airfoil at positive incidence is then \(L'=-\rho U_\infty\Gamma\).


    The transformation can map a circle into an airfoil-like shape.
    This offers an important advantage: instead of solving the
    governing equations directly around a complicated geometry,
    we can start from a known potential-flow solution around a circle
    and transform it into the airfoil plane.

    However, the transformation does not remove the assumptions of
    potential-flow theory. Its predictions remain subject to the
    limitations of the idealised model.


    ## 1.7 Solution for a uniform flow around an obstacle

    For a circle, the complex potential is the sum of uniform flow, a doublet contribution that enforces no penetration, and circulation:
    $$
    W(\zeta)=U_\infty e^{-i\alpha}(\zeta-\zeta_c)
    +\frac{U_\infty R^2e^{i\alpha}}{\zeta-\zeta_c}
    +\frac{\Gamma}{2\pi i}\log\left(\frac{\zeta-\zeta_c}{R}\right)
    $$

    Differentiating gives the complex velocity:
    $$
    \frac{dW}{d\zeta}=U_\infty e^{-i\alpha}
    -\frac{U_\infty R^2e^{i\alpha}}{(\zeta-\zeta_c)^2}
    +\frac{\Gamma}{2\pi i(\zeta-\zeta_c)}.
    $$

    Where $\Gamma$ is deduced from a finite trailing-edge velocity condition (i.e. $dW/d\zeta=0$):

    $$
    \boxed{\Gamma=-4\pi R U_\infty\sin(\alpha-\beta)}
    $$

    where $\beta=\arg(a-\zeta_c)$

    Based on the latter, the pressure coefficient can be deduced from the physical surface speed ($V_s$). Indeed, Bernoulli's equation gives:

    $$
    \boxed{C_p=1-\left(\frac{V_s}{U_\infty}\right)^2}
    $$

    where $V_s = \left|\dfrac{dW}{dz}\right|=\left|\dfrac{dW/d\zeta}{1-a^2/\zeta^2}\right|$

    Besides, the lift coefficient is such as,

    $$
    \boxed{C_L=\frac{L'}{\frac12\rho U_\infty^2c}
    =\frac{-2\Gamma}{U_\infty c}}.
    $$

    where $L'=-\rho U_\infty\Gamma$. This is referred as the **Kutta-Joukowsky theorem**.



    ## 1.8 Online references

    1. **NASA Glenn Research Center — Conformal Mapping**
       An accessible introduction to the mapping of circles into
       airfoils and the role of circulation and the Kutta condition.

       <https://www.grc.nasa.gov/WWW/K-12/airplane/map.html>

    2. **MIT OpenCourseWare — Fluid Mechanics**
       Undergraduate material on fluid motion, potential flow, and
       the governing equations.

       <https://ocw.mit.edu/courses/16-01-unified-engineering-i-ii-iii-iv-fall-2005-spring-2006/pages/fluid-mechanics/>

    3. **MIT OpenCourseWare — Aerodynamics**
       Further background on airfoils and aerodynamic theory.

       <https://ocw.mit.edu/courses/16-100-aerodynamics-fall-2005/>

    4. **MIT OpenCourseWare — Potential Flow Theory**
       A more advanced resource for students who want to explore the
       mathematical foundations in greater depth.

       <https://ocw.mit.edu/courses/2-25-advanced-fluid-mechanics-fall-2013/pages/potential-flow-theory/>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. GLOBAL PARAMETERS
    """)
    return


@app.cell
def _():
    # figure
    figsize = (12, 5)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # 3. EXERCICES
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 1 — A tale of a circle with NumPy
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Given a centre $\zeta_c = \mu + i\nu$ , a radius $R(a)$ as defined in the previous section, and a number of points $N$, try to sample and draw a full circle with two different methodologies:

    * Using carterisan coordinates
    * Using complex numbers

    *Tip : You might start by performing a `np.linspace` for an angle $\theta$*
    """)
    return


@app.cell
def _():
    N_0 = 4
    R_0 = 2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To assess our circle appoximation, we want to compare its perimeter. Try to figure out a way to do it for a various number of points. In particular, you might take advantage of [`np.diff`](https://numpy.org/doc/stable/reference/generated/numpy.diff.html) to compute segment lengths. Plot the gap between the exact circle length and its approximation with the polygon. What do you observe ?
    """)
    return


@app.cell
def _(np):
    N_sampling = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])

    L_polygon = []
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Aiming for a better accuracy, we will use [cubic splines](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html) to draw the circle. As done for the segments, try to increase the number of analytical sampling points and draw the circumfrerence error of the both methods used up to now. What do you observe ?

    *Tip : Numerical integration through `np.trapezoid` might be invoked to compute spline length.*
    """)
    return


@app.cell
def _():
    SPLINE_SUPERSAMPLING = 20
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise 2: Back to the Joukowsky transform


    Try to implement the Joukowsky transform based on the variables provided but the notebook for the input parameters. The goal is to get a reactive matplotlib plot of the airfoil that can vary with the parameters.
    """)
    return


@app.cell
def _(mo):
    a_slider = mo.ui.slider(start=0, stop=2, label="a", value=1, step=0.01, full_width=True)
    mu_slider = mo.ui.slider(start=-0.5, stop=0., label="$\mu$", value=-0.08, step=0.01, full_width=True)
    nu_slider = mo.ui.slider(start=0., stop=0.5, label="$\\nu$", value=0.12, step=0.01, full_width=True)

    mo.vstack((a_slider, mu_slider, nu_slider))
    return a_slider, mu_slider, nu_slider


@app.cell(hide_code=True)
def _(a_slider, mu_slider, nu_slider):
    a_j = a_slider.value
    mu_j = mu_slider.value
    nu_j = nu_slider.value
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercice 3: Streamlines: sample a flow field

    The complex potential ($W=\phi+i\psi$) has streamfunction ($\psi=\operatorname{Im}(W)$). Streamlines are contours of constant ($\psi$).

    To visualize the flow, sample a rectangular grid in the auxiliary plane, evaluate ($W$), and contour its imaginary part. Mask points inside the generating circle. Then map the grid points into the physical plane for a second visualization.

    **Numerical caution:** the map has a pole at ($\zeta=0$), so avoid grid points too close to zero. For a clear first exercise, plot the circle-plane streamlines and the airfoil contour together as separate figures. A full mapped curvilinear grid is more involved and is a useful extension exercise.

    *Tips*

    1. Create `xi_grid, eta_grid = np.meshgrid(...)`.
    2. Construct `zeta_grid = xi_grid + 1j*eta_grid`.
    3. Evaluate the complex potential on the grid.
    4. Mask the circle interior before calling `contour`.
    """)
    return


@app.cell
def _():
    N = 500
    a = 1.0
    mu = -0.08
    nu = 0.12
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise 4: Pressure on the physical-plane

    Use NumPy array expressions to evaluate $dW/d\zeta$, $dz/d\zeta$, and their ratio at all boundary samples. The ratio is formally a \(0/0\) limit at the trailing edge; avoid dividing directly at that point. Mask near-singular samples or estimate the limit from nearby points.

    Then plot $C_p$ against $x/c$. Conventionally, the vertical axis is often inverted so lower pressure appears higher on the page.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise 5: Model validation

    Pressure force per unit span on a closed body is
    $$
    \mathbf F_p'=-\oint p\mathbf n\,ds.
    $$
    For a counterclockwise contour, this can be written
    $$
    F_x'=\oint p\,dy,\qquad F_y'=-\oint p\,dx.
    $$

    Use `np.roll` to obtain the next point and approximate \(dx,dy\). Sum the pressure-force contributions around the contour. Compare the integrated lift with $-\rho U_\infty\Gamma$.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise 6: Lift curve

    A parameter sweep is a natural use of NumPy arrays and pandas. We can evaluate the analytical circulation relation for a sequence of angles, calculate lift and $C_L(\alpha)$. Here are the steps:

    1. Create an array of angles from $-8^\circ$ to $12^\circ$ (example).
    2. Convert it to radians using `np.deg2rad`.
    3. Plot $C_L$ versus angle of attack.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
