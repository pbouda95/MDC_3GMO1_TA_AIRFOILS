# MDC_3GMO1_TA: Exploring Airfoils Aerodynamics and Mechanics with Scientific Python

- **Author:** Pascal Bouda - CEA Research Engineer in Structural Dynamics ([pascal.bouda@cea.fr](mailto:pascal.bouda@cea.fr))
- **Prerequisites:** Lab 1 (Joukowsky transform) and Lab 2 (panel method).
- **Format:** groups of 2 students
- **Deliverables:** Project code, oral defence.

---

## 0. Marking criteria

 - **Final code structure**: A particular attention should be provided on the code structure. All efforts to come up with a Object-Oriented Programming (OOP) code will be rewarded (classes, methods, attributes, inheritance). This is similar for all efforts to use the mode advanced functions of the NumPy library.
 By contrast, you should avoid unuseful code lines (duplicated code, loops instead of NumPy broadcasting ...)

 - **Physical interpretations**: The project is about fight mechanics and fluid mechanics. However, this is not the most critical part. No *a priori* knowledge was expected, so any effort to try to better understand
 the underlying physics will be awarded

 - **Oral defence**: A advanced presentation of the code structure is expected. In particular, the choice of the number of files/folders, the code hierarchy, etc...

---

## 1. Objectives

You will extrapolate what you learned in the two labs to **two real airfoils** and confront your models with **wind-tunnel data**. The project has three parts. Depenging on the group progress, the third part will be made mandatory or optional.

1. **Models vs. experiment:** compute the lift curve $C_L(\alpha)$ of two airfoils with (a) the panel method and (b) a fitted Joukowsky transform, and compare both with the experimental reference.
2. **Critical analysis:** explain every discrepancy between the three data sets (Joukowsky, panels, experiment).
3. **Application:** use the experimental $C_L(\alpha)$ in a very simple flight-mechanics model to find an **optimal trajectory** to a target altitude. You will observe that *the highest angle of attack is not the best solution*, because of drag and stall.

---

## 2. Choosing the airfoils and the data

Experimental data: [UIUC Low-Speed Airfoil Tests (LSAT) database](https://m-selig.ae.illinois.edu/uiuc_lsat.html>). For each airfoil you will need (i) the **coordinates** (`.dat` file) and (ii) the **lift (and drag) polars** at one Reynolds number.

Our models are **inviscid, incompressible, 2D flow**. Choose data that are as close as possible to these hypotheses.

### Selection criteria

| Criterion | Why | What to look for |
|---|---|---|
| **Highest Reynolds number available** | Viscous effects are strongest at low Re and are not captured by the models | Prefer $Re \gtrsim 2\times10^5$. Avoid the lowest Re of the database. |
| **Thin to moderate thickness** (≈ 6–15 %) | Thick airfoils separate earlier at the trailing edge; Joukowsky fits thin profiles better | $t/c \le 12\%$ is a good target |
| **Small to moderate camber** | Joukowsky profiles are only mildly cambered; strongly cambered/reflexed sections fit poorly | Camber ≲ 4 % |
| **Sharp (thin) trailing edge** | Joukowsky has a cusped trailing edge, and the Kutta condition is applied at a sharp point | Check the coordinates: avoid blunt trailing edges and flat-bottom sections |
| **Smooth, "classical" geometry** | Fit quality and panel convergence | Avoid exotic shapes (S-shaped, reflexed, very flat undersides such as Clark-Y type) for the Joukowsky fit |
| **Clean lift curve with a visible linear range** | You need a slope and a zero-lift angle to compare | Check that the data cover negative and positive $\alpha$, ideally up to and past stall |


### Recommended strategy

* **Airfoil A (easy case, "model-friendly"):** a symmetric or slightly cambered thin profile (e.g. a NACA 4-digit such as 0009/0012/2412, *if available in the database at a sufficiently high Re*). Expected: good agreement in the linear range.
* **Airfoil B (more demanding case):** a more cambered or thicker profile, different enough from A to be informative Expected: larger discrepancies, richer discussion.
* **Justify** your two choices in a short table (thickness, camber, Re, trailing-edge type, quality of data). A justified choice is a plus.
* Use the **same Reynolds number** for all comparisons of a given airfoil, and state it.


### Candidate airfoils (Examples)

| Airfoil | Type | Why it is interesting for this project | Expected difficulty |
|---|---|---|---|
| **NACA 0009** | Symmetric, 9 % thick | Best "model-friendly" case: thin, symmetric, sharp trailing edge. $\alpha_{L=0}\approx 0$ gives a clean test of slope. Joukowsky fit needs only $x_0$ and $a$ ($y_0 \approx 0$), a good check of the fit | Easy |
| **NACA 64A010** | Symmetric, 10 % thick, laminar-flow section | Thin with a sharp trailing edge. Comparison with NACA 0009 shows the effect of the thickness distribution. The Joukowsky fit of its shape is less accurate (different thickness distribution) | Easy to medium |
| **NACA 2414** / **NACA 2415** | 2 % camber, 14–15 % thick | Slightly cambered, classical 4-digit shape, smooth geometry. Good for fitting $y_0$ (camber). The thickness is larger, so the discrepancy with the inviscid slope is larger | Medium |
| **NACA 6409** | 6 % camber, 9 % thick | Strongly cambered thin section: tests the capacity of the Joukowsky family to represent camber (large $y_0$) and the nonzero $\alpha_{L=0}$ | Medium to hard |

### Where to find the data online

* [UIUC LSAT main page](https://m-selig.ae.illinois.edu/uiuc_lsat.html)
* [Airfoils tested (figures and alphabetical list)](https://m-selig.ae.illinois.edu/uiuc_lsat_airfoilsTested.html)
* [Data books in PDF](https://m-selig.ae.illinois.edu/uiuc_lsat.html) (with plots of lift and drag polars and the run/Re tables)"
* [Airfoil coordinates (`.dat`, Selig format) UIUC Airfoil Coordinates Database](https://m-selig.ae.illinois.edu/ads.html)


### Reynolds number: definition and retrieving the velocity

The experimental polars are labelled by **Reynolds number, not by velocity**. The Reynolds number compares inertial and viscous effects:

$$
Re = \frac{\rho\, V\, c}{\mu} = \frac{V\, c}{\nu}, \qquad \nu = \frac{\mu}{\rho}
$$

with $\rho$ the air density (kg/m³), $V$ the freestream velocity (m/s), $c$ the **chord** (m, the reference length), $\mu$ the dynamic viscosity (Pa·s) and $\nu$ the kinematic viscosity (m²/s).

**Retrieving the velocity** corresponding to a given polar:

$$
V = \frac{Re\,\nu}{c} = \frac{Re\,\mu}{\rho\, c}
$$

Procedure:

1. Read the Reynolds number of the polar (e.g. $Re = 200\,000$).
2. Take the **chord $c$** of the tested model from the data book (check the units: inches vs. metres).
3. Take $\nu$ at the test conditions. At sea level, standard atmosphere ($15\,°C$): $\rho \approx 1.225$ kg/m³, $\mu \approx 1.79\times10^{-5}$ Pa·s, hence $\nu \approx 1.46\times10^{-5}$ m²/s.
4. Compute $V$, then check the **Mach number** $M = V/a$ ($a \approx 340$ m/s at sea level) to verify that the incompressible hypothesis holds (here $M \ll 0.3$).

*Worked example (illustrative values):* $Re = 2\times10^5$, $c = 0.30$ m, $\nu = 1.46\times10^{-5}$ m²/s:

$$
V = \frac{2\times10^{5}\times 1.46\times10^{-5}}{0.30} \approx 9.7\ \text{m/s}, \qquad M \approx 0.03
$$

---

## 3. Part 1 – Joukowsky transform with profile fitting

A Joukowsky airfoil is obtained from a circle in the $\zeta$-plane through

$$z = \zeta + \frac{a^2}{\zeta}$$

The circle must pass through the point $\zeta = a$ (trailing edge). The free parameters are:

* the circle centre $(x_0, y_0)$: $x_0$ controls thickness, $y_0$ controls camber,
* the transform parameter $a$ (sets the size, roughly $c \approx 4a$).

### Fitting procedure

1. Define the optimisation variables, for example $\mathbf{p} = (x_0, y_0, a)$. You will most likely also need **geometric alignment variables** (translation of the leading edge, a small rotation $\delta\alpha$, a scale factor), since the real profile and the Joukowsky profile are not positioned identically. Justify which ones you include.
2. Define a **cost function**: e.g. mean squared distance between the experimental points and the Joukowsky contour (resample both with the same parametrisation, for instance the normalised abscissa $x/c$ for upper and lower surfaces separately), possibly with a weight emphasising the leading edge.
3. Minimise it with a method of your choice (Nelder–Mead, `scipy.optimize.least_squares`, differential evolution…). **Test several starting points** and check that the minimum is robust.
4. Report the fit error, overlay the fitted and real profiles, and comment on where they differ (leading edge, trailing edge angle, lower surface…).
5. Compute the lift curve of the fitted profile.

## 4. Part 2 – Panel method

1. Import the coordinates. Check the ordering and orientation of the panels.
2. Compute $C_L(\alpha)$ for $\alpha \in [-6°, +10°]$ (for a dozen of values)
3. **Convergence study:** try to vary the number of panels $N$ (e.g. 40, 80, 160, 320, 640). Plot $C_L$ (at a fixed $\alpha$, versus $N$. Also test the panel distribution (uniform vs. cosine spacing) and discuss. Report the $N$ you retain and why.
4. Compute the lift curve of the fitted profile.

---

## 5. Part 3 – Comparison and discussion

Plot, for each airfoil, on **one figure**: Joukowsky, panel method, experiment. Trying to explain the result might be a plus.

---

## 6. Part 4 – Optimal climb trajectory (Maybe optional)

### 6.1 Goal

Given an aircraft and the **experimental** lift curve $C_L(\alpha)$ (and drag polar), find the angle-of-attack command $\alpha(t)$ that brings the aircraft to a **target altitude** $h_T$ in in a minimum time.

### 6.2 Model

Point-mass aircraft in a vertical plane, constant mass, constant thrust $T$ (aligned with the flight path to simplify). **The dependency of the air density $\rho$ and of the viscosity $\mu$ on the altitude is ignored**: both are kept constant at their sea-level values ($\rho = 1.225$ kg/m³, $\mu = 1.79\times10^{-5}$ Pa·s). The flight mechanics equations are such as:

$$
\begin{aligned}
m\,\dot V &= T\cos\alpha - D - mg\sin\gamma \\
m V\,\dot\gamma &= T\sin\alpha + L - mg\cos\gamma \\
\dot h &= V\sin\gamma, \qquad \dot x = V\cos\gamma
\end{aligned}
$$

with $L = \tfrac12\rho V^2 S\,C_L(\alpha)$ and $D = \tfrac12\rho V^2 S\,C_D(\alpha)$.

* $C_L(\alpha)$: your choice among the experimental curve (reference), with a smooth interpolation including the **post-stall drop**.
* $C_D$: use the experimental airfoil drag if available, plus an induced-drag term $C_D = C_{D0} + k\,C_L^2$.
* Choose a simple aircraft (e.g. a light aircraft or an RC aircraft: $m$, $S$, $AR$, $T$) and give its data.

### 6.3 Possible aircraft and environment

The values below describe a small electric UAV / large RC aircraft. They are chosen so that the Reynolds number stays in the range of the UIUC data (≈ $1.5$–$4\times10^5$) during the whole climb.
| Quantity | Value |
|---|---|
| Mass $m$ | 2.0 kg (weight $W \approx 19.6$ N) |
| Wing area $S$ | 0.375 m² |
| Chord $c$ | 0.25 m |
| Span $b$ | 1.5 m, aspect ratio $AR = b^2/S = 6$ |
| Wing loading $W/S$ | ≈ 52 N/m² |
| Constant thrust $T$ | 8 N ($T/W \approx 0.41$) |
| Air density / viscosity | $\rho = 1.225$ kg/m³, $\mu = 1.79\times10^{-5}$ Pa·s, hence $\nu = 1.46\times10^{-5}$ m²/s |
| Gravity | $g = 9.81$ m/s² |

**Mission**

| Quantity | Value |
|---|---|
| Initial conditions | $h_0 = 0$ m, $V_0 = 12$ m/s, $\gamma_0 = 0$ |
| Target | $h_T = 100$ m |
| Stall speed | Derived from the equilibrium between lift and gravity |
| Induced drag coefficient $k$ | 0.066 |
| Minimum speed | $V \ge 1.2\,V_{stall}$ |


**Order of magnitude to check your results** (quasi-steady climb, from $\sin\gamma \approx (T-D)/W$, $L \approx W$):

| $\alpha$ (deg) | $V$ (m/s) | $L/D$ | climb rate (m/s) |
|---|---|---|---|
| 0 | ≈ 21 | ≈ 6 | ≈ 5.0 |
| 2 | ≈ 15 | ≈ 10 | ≈ 4.5 |
| 6 | ≈ 10 | ≈ 11 | ≈ 3.3 |
| 12 (stall) | ≈ 8.4 | ≈ 9.5 | ≈ 2.5 |

The maximum $\alpha$ is clearly **not** the best: it gives the lowest climb rate (and no margin to stall), whereas a small $\alpha$ makes the aircraft faster. With these values the optimal quasi-steady $\alpha$ is expected at a few degrees, so the flight time to 100 m is of the order of 25–30 s including the initial acceleration. The best-$L/D$ angle ($C_L \approx 0.67$, $\alpha \approx 5$°) gives the steepest climb **angle** $\gamma$ (minimum drag) but not the best climb **rate** $\dot h = V\sin\gamma$, which depends on the excess **power** and favours a higher speed (smaller $\alpha$).

### 6.4 Command parametrisation

A linear variation of $\alpha$ with steps can be considered (flexible).

Add the constraints you consider necessary: $V \ge 1.2\,V_{stall}$, maximum load factor, $\alpha \le \alpha_{stall}$ (or not, to study its effect), $h \ge 0$.

### 6.5 Optimisation

1. Integrate the equation (use `scipy.integrate.solve_ivp`).
2. Minimise the time to reach $h_T$ over the command parameters (`scipy.optimize`, or a simple parametric sweep first to visualise the landscape).
3. Plot $h(t)$, $V(t)$, $\gamma(t)$, $\alpha(t)$, $C_L$, $L/D$.

### 6.6 Results analysis

Try to analyze the results. Any physical interpreration attempt will be a plus.