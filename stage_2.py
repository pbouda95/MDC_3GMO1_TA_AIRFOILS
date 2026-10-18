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

    return (mo,)


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
def _():
    CSV_PATH = "naca2412.csv"

    return


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
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Execute the following cell. It should check if the function is robust to the airfoil orientation.
    """)
    return


@app.cell
def _():
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
def _():
    return


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
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Validation and Speed test

    Prove that your implementations gives the same result. Then perform a speed test on them.
    """)
    return


@app.cell
def _():
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
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercice 5: Resolving a case

    Plot the result for a given angle of attack.
    """)
    return


@app.cell
def _():
    return


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
def _():
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
def _():
    return


if __name__ == "__main__":
    app.run()
