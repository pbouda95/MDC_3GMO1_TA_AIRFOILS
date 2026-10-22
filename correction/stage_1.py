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

    return CubicSpline, mo, np, pd, plt


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
    return (figsize,)


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
def _(figsize, np, plt):
    N_0 = 4
    R_0 = 2
    _theta = np.linspace(0, 2*np.pi, N_0 + 1)

    # Method 1: trigonometric parameterization
    x1 = R_0 * np.cos(_theta)
    y1 = R_0 * np.sin(_theta)

    # Method 2: complex exponential
    z2 = R_0 * np.exp(1j * _theta)
    x2 = z2.real
    y2 = z2.imag

    fig_1, ax_1 = plt.subplots(1, 2, figsize=figsize)
    ax_1[0].plot(x1, y1, "o-", label="Trigonometric")
    ax_1[0].set_title("Method 1: Trigonometric")
    ax_1[0].axis("equal")
    ax_1[0].grid()
    ax_1[0].legend()

    ax_1[1].plot(x2, y2, "o-", label="Complex exponential")
    ax_1[1].set_title("Method 2: Complex exponential")
    ax_1[1].axis("equal")
    ax_1[1].grid()
    ax_1[1].legend()

    plt.tight_layout()
    plt.show()
    return (R_0,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To assess our circle appoximation, we want to compare its perimeter. Try to figure out a way to do it for a various number of points. In particular, you might take advantage of [`np.diff`](https://numpy.org/doc/stable/reference/generated/numpy.diff.html) to compute segment lengths. Plot the gap between the exact circle length and its approximation with the polygon. What do you observe ?
    """)
    return


@app.cell
def _(R_0, figsize, np, plt):
    N_sampling = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])

    L_polygon = []

    for _N in N_sampling:

        _theta = np.linspace(0, 2*np.pi, _N + 1)

        # Sample the circle at n + 1 parameter values
        t = np.linspace(0, 2 * np.pi, _N + 1)

        _z = R_0 * np.exp(1j * _theta)
        _x = _z.real
        _y = _z.imag

        _dx = np.diff(_x)
        _dy = np.diff(_y)

        _length = np.sum(np.sqrt(_dx**2 + _dy**2))

        L_polygon.append(_length)


    L_exact = 2 * np.pi * R_0
    error_polygon = np.abs(L_exact - np.array(L_polygon))

    plt.figure(figsize=figsize)
    plt.loglog(N_sampling, error_polygon, "o-", label="Straight segments")
    plt.xlabel("Number of segments N")
    plt.ylabel("Absolute circumference error")
    plt.title("Convergence towards the exact circumference")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return L_exact, N_sampling, error_polygon


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Aiming for a better accuracy, we will use [cubic splines](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html) to draw the circle. As done for the segments, try to increase the number of analytical sampling points and draw the circumfrerence error of the both methods used up to now. What do you observe ?

    *Tip : Numerical integration through `np.trapezoid` might be invoked to compute spline length.*
    """)
    return


@app.cell
def _(CubicSpline, L_exact, N_sampling, R_0, error_polygon, figsize, np, plt):
    SPLINE_SUPERSAMPLING = 20

    L_spline = []


    for _N in N_sampling:


        _t = np.linspace(0, 2 * np.pi, _N + 1)

        _z = R_0 * np.exp(1j * _t)
        _x = _z.real
        _y = _z.imag

        _sx = CubicSpline(_t, _x, bc_type="periodic")
        _sy = CubicSpline(_t, _y, bc_type="periodic")

        t_spline = np.linspace(0, 2 * np.pi, SPLINE_SUPERSAMPLING * _N + 1)

        _dx_dt = _sx(t_spline, 1)
        _dy_dt = _sy(t_spline, 1)

        _speed = np.sqrt(_dx_dt**2 + _dy_dt**2)

        L_spline.append(np.trapezoid(_speed, t_spline))


    error_spline = np.abs(L_exact - np.array(L_spline))

    plt.figure(figsize=figsize)
    plt.loglog(N_sampling, error_polygon, "o-", color='red',  label="Straight segments")
    plt.loglog(N_sampling, error_spline, "o-", color='blue', label="Splines")
    plt.xlabel("Sampling")
    plt.ylabel("Absolute circumference error")
    plt.title("Convergence towards the exact circumference")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()
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
def _(R_0, a_slider, figsize, mo, mu_slider, np, nu_slider, plt):
    a_j = a_slider.value
    mu_j = mu_slider.value
    nu_j = nu_slider.value

    R_j = np.sqrt((a_j - mu_j)**2 + nu_j**2)
    N_j = 500
    theta_j = np.linspace(0, 2*np.pi, N_j, endpoint=False)
    zeta_c_j = mu_j + 1j * nu_j
    zeta_j = zeta_c_j + R_0 * np.exp(1j * theta_j)

    # Sample the circle at n + 1 parameter values
    t_j = np.linspace(0, 2 * np.pi, N_j + 1)

    zeta_j = (mu_j + 1j * nu_j) + R_0 * np.exp(1j * theta_j)

    z_j = zeta_j + a_j**2/zeta_j
    chord = z_j.real.max() - z_j.real.min()
    x_le = z_j.real.min()
    x_te = z_j.real.max()
    fig_airfoil, ax_airfoil = plt.subplots(figsize=figsize)
    ax_airfoil.plot(z_j.real, z_j.imag, lw=2)
    ax_airfoil.axhline(0, color="grey", lw=0.8)
    ax_airfoil.set_aspect("equal", adjustable="box")
    ax_airfoil.set_xlabel("$x$")
    ax_airfoil.set_ylabel("$y$")
    ax_airfoil.set_title("Mapped contour in the physical plane")
    ax_airfoil.grid(True, alpha=0.3)
    mo.md(f"Approximate chord from sampled extrema: **c = {chord:.4f}**. Leading-edge x ≈ {x_le:.4f}; trailing-edge x ≈ {x_te:.4f}.")

    fig_airfoil
    return (chord,)


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
def _(figsize, np, plt):
    N = 500
    a = 1.0
    mu = -0.08
    nu = 0.12
    zeta_c  = mu + 1j * nu
    R = np.sqrt((a - mu)**2 + nu**2)
    beta = np.angle(a - zeta_c)
    theta = np.linspace(0, 2 * np.pi, N + 1)
    zeta = (mu + 1j * nu) + R * np.exp(1j * theta)
    z = zeta + a**2/zeta

    # A grid in the auxiliary plane. Keep it away from zeta=0.
    gx = np.linspace(-2.5, 2.5, 300)
    gy = np.linspace(-2.0, 2.0, 240)
    XI, ETA = np.meshgrid(gx, gy)
    ZG = XI + 1j*ETA
    relative = ZG - zeta_c

    # Numerical application
    U_inf = 30.0  # m/s
    alpha_deg = 5.0
    alpha = np.deg2rad(alpha_deg)
    rho = 1.225  # kg/m^3
    Gamma = -4 * np.pi * R * U_inf * np.sin(alpha - beta)

    WG = (U_inf * np.exp(-1j*alpha) * relative + U_inf * R**2 * np.exp(1j*alpha)/relative + Gamma/(2j*np.pi)*np.log(relative/R))
    mask = np.logical_or(np.abs(relative) <= R, np.logical_not(np.isfinite(WG.imag)))
    psi = np.ma.array(WG.imag, mask=mask)

    circle_phi = np.linspace(0, 2*np.pi, N)
    circle = zeta_c + R*np.exp(1j*circle_phi)


    fig_stream, ax_stream = plt.subplots(figsize=figsize)
    ax_stream.contour(XI, ETA, psi, levels=45, linewidths=0.7)
    ax_stream.plot(circle.real, circle.imag, color="black", lw=2)
    ax_stream.set_aspect("equal")
    ax_stream.set_xlabel(r"$\xi$")
    ax_stream.set_ylabel(r"$\eta$")
    ax_stream.set_title("Streamlines in the generating-circle plane")
    ax_stream.set_xlim(-2.5, 2.5)
    ax_stream.set_ylim(-2, 2)
    ax_stream.grid(True, alpha=0.2)
    fig_stream
    return Gamma, R, U_inf, a, alpha, beta, rho, z, zeta, zeta_c


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise 4: Pressure on the physical-plane

    Use NumPy array expressions to evaluate $dW/d\zeta$, $dz/d\zeta$, and their ratio at all boundary samples. The ratio is formally a \(0/0\) limit at the trailing edge; avoid dividing directly at that point. Mask near-singular samples or estimate the limit from nearby points.

    Then plot $C_p$ against $x/c$. Conventionally, the vertical axis is often inverted so lower pressure appears higher on the page.
    """)
    return


@app.cell
def _(Gamma, R, U_inf, a, alpha, figsize, np, plt, z, zeta, zeta_c):
    rel = zeta - zeta_c

    dW_dzeta = (U_inf*np.exp(-1j*alpha)
                - U_inf*R**2*np.exp(1j*alpha)/(rel**2)
                + Gamma/(2j*np.pi*rel))

    dz_dzeta = 1 - a**2/zeta**2
    physical_velocity = np.full(zeta.shape, np.nan + 1j*np.nan, dtype=complex)
    safe = np.abs(dz_dzeta) > 1e-7

    physical_velocity[safe] = dW_dzeta[safe]/dz_dzeta[safe]
    V_surface = np.abs(physical_velocity)
    Cp = 1 - (V_surface/U_inf)**2

    x_over_c = (z.real-z.real.min())/(z.real.max() - z.real.min())


    fig_cp, ax_cp = plt.subplots(figsize=figsize)
    valid_cp = np.isfinite(Cp)
    ax_cp.plot(x_over_c[valid_cp], Cp[valid_cp], ".", ms=3)
    ax_cp.invert_yaxis()
    ax_cp.set_xlabel("$x/c$")
    ax_cp.set_ylabel("$C_p$")
    ax_cp.set_title("Surface pressure coefficient (trailing-edge singular samples masked)")
    ax_cp.grid(True, alpha=0.3)

    fig_cp
    return (Cp,)


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
def _(Cp, Gamma, U_inf, chord, mo, np, rho, z):
    # Pressure force from successive contour segments. Use finite Cp samples;
    # this is a diagnostic approximation, not a substitute for a TE-limit treatment.
    lift_per_span = -rho*U_inf*Gamma
    q_inf = 0.5*rho*U_inf**2
    CL = lift_per_span/(q_inf*chord)

    z_next = np.roll(z, -1)
    dx = z_next.real - z.real
    dy = z_next.imag - z.imag
    cp_next = np.roll(Cp, -1)
    cp_mid = 0.5*(Cp + cp_next)
    valid_segments = np.isfinite(cp_mid)
    Fx_pressure = np.sum(q_inf*cp_mid[valid_segments]*dy[valid_segments])
    Fy_pressure = -np.sum(q_inf*cp_mid[valid_segments]*dx[valid_segments])
    lift_kj = -rho*U_inf*Gamma
    mo.md(f"""
    **Diagnostic pressure integral**
    - Pressure-force x component: `{Fx_pressure:.5f} N/m`
    - Pressure-force y component: `{Fy_pressure:.5f} N/m`
    - Kutta–Joukowski lift (signed convention): `{lift_kj:.5f} N/m`
    """)
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
def _(R, U_inf, beta, chord, np, pd, plt, rho):
    alpha_sweep_deg = np.linspace(-6, 12, 81)
    alpha_sweep = np.deg2rad(alpha_sweep_deg)
    Gamma_sweep = -4*np.pi*R*U_inf*np.sin(alpha_sweep-beta)
    lift_sweep = -rho*U_inf*Gamma_sweep
    CL_sweep = lift_sweep/(0.5*rho*U_inf**2*chord)
    lift_df = pd.DataFrame({
        "alpha_deg": alpha_sweep_deg,
        "alpha_rad": alpha_sweep,
        "CL": CL_sweep,
    })
    fig_lift, ax_lift = plt.subplots(figsize=(7, 4.5))
    ax_lift.plot(lift_df["alpha_deg"], lift_df["CL"], lw=2)
    ax_lift.axhline(0, color="grey", lw=0.8)
    ax_lift.axvline(0, color="grey", lw=0.8)
    ax_lift.set_xlabel("Angle of attack (degrees)")
    ax_lift.set_ylabel("$C_L$")
    ax_lift.set_title("Ideal Joukowski-airfoil lift curve")
    ax_lift.grid(True, alpha=0.3)

    fig_lift
    return (lift_df,)


@app.cell
def _(lift_df, mo):
    mo.md("**Lift-curve DataFrame preview**")
    lift_df
    return


if __name__ == "__main__":
    app.run()
