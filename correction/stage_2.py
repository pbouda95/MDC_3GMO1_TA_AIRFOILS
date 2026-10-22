import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <h1><b>STAGE 2: PANEL METHOD</h1>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Author: Pascal Bouda - CEA Research Engineer ([pascal.bouda@cea.fr](mailto:pascal.bouda@cea.fr))
    - This notebook develops the physics first. **No particular knowledge is required.**
    """)
    return


@app.cell
def _():
    import time
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    return mo, np, pd, plt, time


@app.cell
def _(mo):
    mo.md(r"""
    # 1. THEORETICAL BACKGROUND

    ## 1.1 Model

    The airfoil contour is replaced by a *vortex sheet* of density
    $\gamma(s)$ (circulation per unit length). The velocity is
    $\mathbf V=\mathbf V_\infty+\mathbf V_\gamma$ with
    $\mathbf V_\infty = V_\infty(\cos\alpha,\sin\alpha)$ and $\mathbf V_\gamma$ the
    velocity induced by the sheet (Biot–Savart). Since each vortex is a potential flow
    solution, the only condition left is the boundary condition on the body:

    $$\mathbf V\cdot\mathbf n = 0 \quad\text{on the contour.}$$

    ## 1.2 Discretization

    The contour is split into $N$ straight panels with $N+1$ nodes.

    Let us define the following geometrical features for a panel $j$ going from node $j$ to node $j+1$:

    $$\ell_j=\|\mathbf x_{j+1}-\mathbf x_j\|,\quad
      \mathbf t_j=\frac{\mathbf x_{j+1}-\mathbf x_j}{\ell_j},\quad
      \mathbf n_j=(-t_{y,j},\,t_{x,j}),\quad
      \mathbf x_{c,j}=\tfrac12(\mathbf x_j+\mathbf x_{j+1}).$$

    With the clockwise ordering, the **left-hand normal points out of the body**.

    The unknowns are the vortex densities $\gamma_j$ **at the nodes** ($N+1$ unknowns);
    on panel $j$ the density varies *linearly* and is continuous between panels:

    $$\gamma(s)=\gamma_j(1-s/\ell_j)+\gamma_{j+1}\,s/\ell_j$$

    Then, boundary condition can be enforced at the **control point** (mid-point) $i$ of each
    panel with the following equations:

    $$\boxed{\sum_{j=1}^{N+1} A_{ij}\,\gamma_j = -\mathbf V_\infty\cdot\mathbf n_i,
    \qquad A_{ij} = \mathbf n_i\cdot\mathbf u_{ij}}$$

    where $\mathbf u_{ij}$ is the velocity induced at control point $i$ by the vortex
    distribution attached to node $j$ ($\gamma_j=1$, all other nodal values 0).

    This gives $N$ equations for $N+1$ unknowns.

    ## 1.3 Kutta condition


    For a *closed* body the $N$ equations are not enough: a constant
    $\gamma$ on the whole contour produces zero normal velocity (it is a pure vortex
    outside, nothing inside), so the circulation is free. The Kutta condition fixes
    it: the flow leaves the sharp trailing edge smoothly, i.e. the two trailing-edge
    nodes carry opposite densities, $\gamma_1+\gamma_{N+1}=0$. This is the $(N+1)$-th
    equation.

    ## 1.3 Expression of $A_{ij}$

    $A$ can be seen as the encoding of the velocity induced by one panel. Its tems can be computed through an integration of the point-vortex Biot–Savart law along a panel with linear density.

    Panel $k$ goes from node $k$ to node $k+1$ (length $\ell$, tangent $\mathbf t_k$, normal $\mathbf n_k$).
    For a control point $\mathbf p_i$, its coordinates and geometric features can be expressed in the panel frame:

    $$\xi=(\mathbf p_i-\mathbf x_k)\cdot\mathbf t_k,\qquad \eta=(\mathbf p_i-\mathbf x_k)\cdot\mathbf n_k$$

    $$\Theta=\operatorname{atan2}\big(\ell\eta,\ \xi(\xi-\ell)+\eta^2\big),\qquad
      L=\tfrac12\ln\frac{\xi^2+\eta^2}{(\xi-\ell)^2+\eta^2}$$

    $$I_1=\xi\Theta-\eta L,\qquad J_1=\xi L-\ell+\eta\Theta$$

    Then, the velocity contributions in the panel frame, per unit $\gamma$, for the **start node** $k$ ($a$) and the
    **end node** $k+1$ ($b$) can be expressed:

    $$\begin{aligned}
    u_\xi^{a}&=-\tfrac{1}{2\pi}\big(\Theta-\tfrac{I_1}{\ell}\big), & u_\eta^{a}&=\tfrac{1}{2\pi}\big(L-\tfrac{J_1}{\ell}\big),\\
    u_\xi^{b}&=-\tfrac{1}{2\pi}\tfrac{I_1}{\ell}, & u_\eta^{b}&=\tfrac{1}{2\pi}\tfrac{J_1}{\ell}.
    \end{aligned}$$

    Global velocity components are then such as:

    $$
    \boxed{(U,V)^{a,b}=u_\xi^{a,b}\,\mathbf t_k+u_\eta^{a,b}\,\mathbf n_k}
    $$

    The matrix terms can then be derived, where node $j$ belongs to panel $j$ (as start) and panel $j-1$ (as end):

    $$\boxed{A_{ij}=\mathbf n_i\cdot\Big[(U,V)^{a}_{i,\,\text{panel }j}+(U,V)^{b}_{i,\,\text{panel }j-1}\Big],\qquad
      \text{rhs}_i=-\mathbf V_\infty\cdot\mathbf n_i}$$


    NB:
     - Special case : For the control point $i$ on its own panel $i$, the local coordinates are such as $\Theta=\pi,\ \eta=0,\ L=0$ (limit)
     - Kutta row derivation (last row of the system): $(1,0,\dots,0,1)$, and its rhs is $0$.



    ## 1.5 Post-processing

    Several quantities can be computed from the problem solution:

    - Global circulation: $\Gamma=\sum_j\tfrac12(\gamma_j+\gamma_{j+1})\ell_j$
    - Lift coefficient $C_L=-2\Gamma/(V_\infty c)$ (sign: our $\gamma>0$ is counter-clockwise, lift needs clockwise circulation).
    - Pressure: $C_p = 1-(V_t/V_\infty)^2$ with $V_t=\mathbf V\cdot\mathbf t_i$ at the
    control points.


    ## 1.6 References

    - **M. Drela**, *XFOIL: An Analysis and Design System for Low Reynolds Number Airfoils*,
    in *Low Reynolds Number Aerodynamics*, Lecture Notes in Engineering, vol. 54, Springer, 1989.
    DOI: [10.1007/978-3-642-84010-4_1](https://doi.org/10.1007/978-3-642-84010-4_1)
    — inviscid linear-vorticity panel method with nodal unknowns, the scheme used in this notebook
    (full text may require institutional access).

    - **W. H. Mason**, *Applied Computational Aerodynamics*, Virginia Tech, Chapter 4 (panel methods).
    [CAtxtChap4.pdf](http://www.dept.aoe.vt.edu/~mason/Mason_f/CAtxtChap4.pdf)
    — free course text on source and vortex panel methods with airfoil examples
    (the files may now be hosted under `archive.aoe.vt.edu/mason/Mason_f/`).

    - **Cambridge University Engineering Department**, *Aerofoil Section Analysis using 2D panel methods*.
    [panel2d.html](https://www-mdp.eng.cam.ac.uk/web/library/enginfo/aerothermal_dvd_only/aero/panel2d/panel2d.html)
    — short introduction to a 2D inviscid panel method with the Kutta condition
    (constant-strength source and vortex distributions, i.e. a simpler variant than ours).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. EXERCISES
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise 1: Airfoil coordinates from a CSV file

    The airfoil is stored as a CSV file with columns `x`, `y` (chord $c\approx1$),
    ordered from the **lower trailing edge, along the lower surface to the leading edge,
    then along the upper surface back to the upper trailing edge** (clockwise).
    For a sharp trailing edge, the first and last points coincide.

    Load this data with the `pandas` library and plot it.
    """)
    return


@app.cell
def _(np, pd, plt):
    CSV_PATH = "naca2412.csv"

    df = pd.read_csv(CSV_PATH)
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    # equivalent with numpy:
    # x, y = np.loadtxt(CSV_PATH, delimiter=",", skiprows=1, unpack=True)

    assert x.shape == y.shape
    assert np.isclose(x[0], x[-1]) and np.isclose(y[0], y[-1]), "contour not closed"
    N_base = len(x) - 1

    fig1, ax1 = plt.subplots(figsize=(8, 2.5))
    ax1.plot(x, y, "o-", ms=3, lw=1)
    ax1.set_aspect("equal")
    ax1.set_title(f"NACA 2412 loaded from CSV: {N_base} panels")
    ax1.grid(alpha=0.3)

    fig1
    return x, y


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise 2:  Defining panel geometry features

    Write `panel_geometry(x, y)` returning the control points, lengths,
    tangents and normals using only `np.diff` and array arithmetic. Add a check
    that the normals point outward, and make the function robust to a counter-clockwise
    file (shoelace formula: signed area $\tfrac12\sum (x_jy_{j+1}-x_{j+1}y_j)$ is
    negative for a clockwise contour).
    """)
    return


@app.cell
def _(np):
    def ensure_clockwise(x, y):
        area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
        return (x[::-1], y[::-1]) if area > 0 else (x, y)

    def panel_geometry(x, y):
        x, y = ensure_clockwise(x, y)
        dx, dy = np.diff(x), np.diff(y)
        length = np.hypot(dx, dy)
        tx, ty = dx / length, dy / length
        nx, ny = -ty, tx
        xc, yc = 0.5 * (x[:-1] + x[1:]), 0.5 * (y[:-1] + y[1:])
        return dict(x=x, y=y, xc=xc, yc=yc, length=length, tx=tx, ty=ty, nx=nx, ny=ny)

    return (panel_geometry,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Execute the following cell. It should check if the function is robust to the airfoil orientation.
    """)
    return


@app.cell
def _(mo, np, panel_geometry, x, y):

    _g = panel_geometry(x, y)
    # outward normal: pointing away from the (interior) mid-chord point
    _out = (_g["xc"] - 0.4) * _g["nx"] + (_g["yc"] - 0.0) * _g["ny"]
    assert np.all(_out > 0), "some normals point inward"
    # closed contour: sum of panel vectors = 0
    assert np.isclose(np.sum(_g["length"] * _g["tx"]), 0, atol=1e-12)
    assert np.isclose(np.sum(_g["length"] * _g["ty"]), 0, atol=1e-12)
    # CCW input gives the same geometry
    _g2 = panel_geometry(x[::-1], y[::-1])
    assert np.allclose(_g["nx"], _g2["nx"]) and np.allclose(_g["ny"], _g2["ny"])
    mo.md("✅ **Test 2 passed:** outward normals, closed contour, orientation handling.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise 3: Influence coefficients


    ### Reference implementation with loops

    Try to implement the influence matrix $A$ base and velocity using python loops. A particular attention is need for self influence with the following analytical results: $u_\xi=-\tfrac14$ for each of the two nodal weights, $u_\eta=\pm\tfrac{1}{2\pi}$.
    """)
    return


@app.cell
def _(np, panel_geometry):
    def influence_with_loops(px, py, x, y, same_body=False):

        g = panel_geometry(x, y)

        N, M = len(g["length"]), len(px)
        U, V = np.zeros((M, N + 1)), np.zeros((M, N + 1))
        for i in range(M):          # points
            for k in range(N):      # panels
                x0, y0 = g["x"][k], g["y"][k]
                tx, ty = g["tx"][k], g["ty"][k]
                nx, ny = g["nx"][k], g["ny"][k]
                l = g["length"][k]

                dx, dy = px[i] - x0, py[i] - y0
                xi = dx * tx + dy * ty
                eta = dx * nx + dy * ny
                theta = np.arctan2(l * eta, xi * (xi - l) + eta**2)
                L = 0.5 * np.log((xi**2 + eta**2) / ((xi - l) ** 2 + eta**2))
                if same_body and i == k:  # control point i lies on panel i
                    theta, eta, L = np.pi, 0.0, 0.0

                I1 = xi * theta - eta * L
                J1 = xi * L - l + eta * theta
                u_xi_a, u_xi_b = -(theta - I1 / l) / (2 * np.pi), -(I1 / l) / (2 * np.pi)
                u_eta_a, u_eta_b = (L - J1 / l) / (2 * np.pi), (J1 / l) / (2 * np.pi)

                U[i, k] += u_xi_a * tx + u_eta_a * nx
                U[i, k + 1] += u_xi_b * tx + u_eta_b * nx
                V[i, k] += u_xi_a * ty + u_eta_a * ny
                V[i, k + 1] += u_xi_b * ty + u_eta_b * ny
        return U, V

    return (influence_with_loops,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Vectorized implementation (broadcasting)

    Write `influence(px, py, x, y, same_body=False)` without any python loop, with the same output as `influence_loops`.

    **Idea:** use *column vectors* for points (index $i$, shape `(M, 1)`) and *row
    vectors* for panel quantities (index $k$, shape `(1, N)`). Every arithmetic
    operation then produces an `(M, N)` array: this replaces the double loop over
    points and panels. The $(M,N)$ contributions "to the start node" and "to the end node" of
    each panel are then placed into the $(M,N+1)$ matrix by slicing:
    `U[:, :-1] += ...; U[:, 1:] += ...` (no loop either).

    Two details:
    - the angle is computed with a single `arctan2`:
      $\Theta=\operatorname{atan2}\!\big(\ell\eta,\ \xi(\xi-\ell)+\eta^2\big)$;
    - on the panel itself ($\eta=0$) the formula is ambiguous (the sign of a rounding
      error would flip $\Theta$ between $\pm\pi$): we overwrite the diagonal values
      $\Theta=\pi$, $\eta=0$, $L=0$.
    """)
    return


@app.cell
def _(np, panel_geometry):
    def influence(px, py, x, y, same_body=True):

        g = panel_geometry(x, y)

        # points as columns (M,1), panel quantities as rows (1,N)
        px, py = np.asarray(px)[:, None], np.asarray(py)[:, None]
        x0, y0 = g["x"][None, :-1], g["y"][None, :-1]
        tx, ty = g["tx"][None, :], g["ty"][None, :]
        nx, ny = g["nx"][None, :], g["ny"][None, :]
        l = g["length"][None, :]
        N = l.shape[1]

        dx, dy = px - x0, py - y0
        xi = dx * tx + dy * ty
        eta = dx * nx + dy * ny
        theta = np.arctan2(l * eta, xi * (xi - l) + eta**2)        # angle seen
        L = 0.5 * np.log((xi**2 + eta**2) / ((xi - l) ** 2 + eta**2))  # ln(r1/r2)
        if same_body:  # control point i lies on panel i
            idx = np.arange(N)
            theta[idx, idx], eta[idx, idx], L[idx, idx] = np.pi, 0.0, 0.0

        I1 = xi * theta - eta * L
        J1 = xi * L - l + eta * theta
        # contribution of the panel to its start node (a) and end node (b)
        u_xi_a, u_xi_b = -(theta - I1 / l) / (2 * np.pi), -(I1 / l) / (2 * np.pi)
        u_eta_a, u_eta_b = (L - J1 / l) / (2 * np.pi), (J1 / l) / (2 * np.pi)

        M = px.shape[0]
        U, V = np.zeros((M, N + 1)), np.zeros((M, N + 1))
        U[:, :-1] += u_xi_a * tx + u_eta_a * nx
        U[:, 1:] += u_xi_b * tx + u_eta_b * nx
        V[:, :-1] += u_xi_a * ty + u_eta_a * ny
        V[:, 1:] += u_xi_b * ty + u_eta_b * ny

        return U, V

    return (influence,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Validation and Speed test

    Prove that your implementations gives the same result. Then perform a speed test on them.
    """)
    return


@app.cell
def _(influence, influence_with_loops, mo, np, panel_geometry, time, x, y):
    _g = panel_geometry(x, y)

    # (1) vectorized vs loops + quadrature
    _Uv, _Vv = influence(_g["xc"], _g["yc"], x, y, same_body=True)
    _Ul, _Vl = influence_with_loops(_g["xc"], _g["yc"], x, y, same_body=True)
    _err = max(np.abs(_Uv - _Ul).max(), np.abs(_Vv - _Vl).max())
    assert _err < 1e-8, _err


    # timing
    _t0 = time.perf_counter(); influence_with_loops(_g["xc"], _g["yc"], x, y, True)
    _t_loop = time.perf_counter() - _t0
    _t0 = time.perf_counter(); influence(_g["xc"], _g["yc"], x, y, True)
    _t_vec = time.perf_counter() - _t0

    mo.md(f"""
    **{_t_loop*1e3:.0f} ms** vs vectorized **{_t_vec*1e3:.2f} ms**
    (speed-up ≈ {_t_loop/_t_vec:.0f}×).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercice 4: Solving the problem

    Write `solve_panels(x, y, alpha_deg, Vinf=1.0)` which

    - builds $A$ and $B$ using `influence`
    - builds the right-hand side
    - appends the Kutta row
    - solves the problem owing to `np.linalg.solve`
    - returns
        - $\gamma$
        - $\Gamma$
        - $C_L$ owing to the Kutta–Joukowski theorem
        - $C_p$ at control points
        - $C_L$ (lift), $C_D$ (drag) obtained by integrating the pressure on the panels


    *Reminder - Pressure force per unit span on a closed body is*
    $$
    \mathbf F_p'=-\oint p\mathbf n\,ds.
    $$
    """)
    return


@app.cell
def _(influence, np, panel_geometry):
    def assemble_system(x, y, alpha_deg, Vinf=1.0):
        """Return A ((N+1)x(N+1), last row = Kutta), rhs (N+1), B (N x (N+1)), geometry."""

        g = panel_geometry(x, y)

        U, V = influence(g["xc"], g["yc"], g["x"], g["y"], same_body=True)
        A = U * g["nx"][:, None] + V * g["ny"][:, None]
        B = U * g["tx"][:, None] + V * g["ty"][:, None]
        a = np.deg2rad(alpha_deg)
        rhs = -Vinf * (np.cos(a) * g["nx"] + np.sin(a) * g["ny"])
        # Kutta condition as the last row: gamma_1 + gamma_{N+1} = 0
        kutta = np.zeros((1, A.shape[1]))
        kutta[0, 0] = kutta[0, -1] = 1.0

        return np.vstack([A, kutta]), np.append(rhs, 0.0), B, g

    def solve_panels(x, y, alpha_deg, Vinf=1.0):

        A, rhs, B, g = assemble_system(x, y, alpha_deg, Vinf)
        gamma = np.linalg.solve(A, rhs)

        chord = g["x"].max() - g["x"].min()
        Gamma = np.sum(0.5 * (gamma[:-1] + gamma[1:]) * g["length"])
        Cl_kj = -2.0 * Gamma / (Vinf * chord)  # Kutta-Joukowski

        a = np.deg2rad(alpha_deg)
        Vt = Vinf * (np.cos(a) * g["tx"] + np.sin(a) * g["ty"]) + B @ gamma
        Cp = 1.0 - (Vt / Vinf) ** 2

        # force on the body: dF = -p n dl  ->  coefficients
        Cx = np.sum(-Cp * g["length"] * g["nx"]) / chord
        Cy = np.sum(-Cp * g["length"] * g["ny"]) / chord
        Cl_cp = Cy * np.cos(a) - Cx * np.sin(a)
        Cd_cp = Cx * np.cos(a) + Cy * np.sin(a)

        return dict(gamma=gamma, 
                    Gamma=Gamma, 
                    Cl=Cl_kj, 
                    Cp=Cp, 
                    Vt=Vt,
                    Cl_cp=Cl_cp, 
                    Cd_cp=Cd_cp, 
                    geom=g, 
                    chord=chord)

    return (solve_panels,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercice 5: Resolving a case

    Plot the result for a given angle of attack.
    """)
    return


@app.cell
def _(mo, plt, solve_panels, x, y):
    sol5 = solve_panels(x, y, 5.0)
    _g = sol5["geom"]
    fig5, (axa, axb) = plt.subplots(1, 2, figsize=(11, 3.8))
    axa.plot(_g["xc"], -sol5["Cp"], ".-", lw=1)
    axa.set_xlabel("x/c"); axa.set_ylabel("$-C_p$")
    axa.set_title(r"Pressure coefficient, $\alpha=5^\circ$"); axa.grid(alpha=0.3)
    axb.plot(_g["x"], sol5["gamma"], ".-", lw=1)
    axb.set_xlabel("x/c"); axb.set_ylabel(r"$\gamma$")
    axb.set_title(r"Nodal vortex density (lower to upper surface)"); axb.grid(alpha=0.3)
    fig5.tight_layout()
    mo.vstack([
        mo.md(f"**alpha = 5°, N = {len(_g['xc'])}:**  "
              f"C_L (Kutta–Joukowski) = {sol5['Cl']:.4f},  "
              f"C_L (pressure integral) = {sol5['Cl_cp']:.4f},  "
              f"C_D (pressure integral) = {sol5['Cd_cp']:.1e}  (d'Alembert: 0)"),
        fig5,
    ])
    return (sol5,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercice 6 — Validation of the global solution

    Try to validate your implementation with the following checks:
    - The two independent lift evaluations agree (circulation vs pressure integral)
    - The drag is (almost) zero — d'Alembert's paradox
    """)
    return


@app.cell
def _(mo, sol5):
    assert abs(sol5["Cl"] - sol5["Cl_cp"]) < 0.02
    assert abs(sol5["Cd_cp"]) < 1e-2
    mo.md(f"✅ **Test passed:** |C_Lkj − C_Lp| = {abs(sol5['Cl'] - sol5['Cl_cp']):.1e}   "
          f"""|C_d| = {abs(sol5["Cd_cp"]):.1e} """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercise 7: Lift curve $C_L(\alpha)$

    Using the airfoil loaded from the CSV file, compute $C_L$ for a varying $\alpha$. Fit a straight line, give the lift slope
    $dC_L/d\alpha$ (per radian) and the zero-lift angle $\alpha_{L0}$, and compare
    with thin-airfoil theory ($2\pi$).
    """)
    return


@app.cell
def _(mo, np, plt, solve_panels, x, y):
    alphas = np.arange(-8.0, 14.5, 1.0)
    Cl_alpha = np.array([solve_panels(x, y, a)["Cl"] for a in alphas])

    slope, intercept = np.polyfit(np.deg2rad(alphas), Cl_alpha, 1)
    alpha_L0 = -np.rad2deg(intercept / slope)

    fig7, ax7 = plt.subplots(figsize=(12, 5))
    ax7.plot(alphas, Cl_alpha, "o-", label="panel method (N = %d)" % (len(x) - 1))
    ax7.axhline(0, color="gray", lw=0.5); ax7.axvline(0, color="gray", lw=0.5)
    ax7.set_xlabel(r"$\alpha$ [deg]"); ax7.set_ylabel("$C_\\ell$")
    ax7.legend()
    ax7.grid(alpha=0.3)

    mo.vstack([mo.md(f"**Test:** slope = {slope:.3f} /rad (thin airfoil 2π = {2*np.pi:.3f})"), fig7])
    return


if __name__ == "__main__":
    app.run()
