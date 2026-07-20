"""
geometry.py — Module 1: NACA 4-digit aerofoil geometry generator

What this does:
  Takes a NACA 4-digit code (e.g. "2412") and returns the (x, y)
  coordinates of the aerofoil surface as two arrays: upper and lower.

NACA 4-digit breakdown (example: NACA 2412):
  2  → max camber = 2% of chord
  4  → max camber is at 40% of chord
  12 → max thickness = 12% of chord

The maths used here is substitution into closed-form equations —
no calculus needed, just plug-and-chug at each x position.
"""

import numpy as np
import matplotlib.pyplot as plt


def naca4_geometry(code: str, n_panels: int = 100) -> dict:
    """
    Generate NACA 4-digit aerofoil coordinates.

    Parameters
    ----------
    code     : 4-character string, e.g. "2412"
    n_panels : number of panels per surface (more = smoother, slower solver)

    Returns
    -------
    dict with keys:
        'x_upper', 'y_upper'  — upper surface coordinates
        'x_lower', 'y_lower'  — lower surface coordinates
        'x_camber', 'y_camber'— mean camber line (useful for visualisation)
        'params'              — dict of the decoded NACA parameters
    """

    # --- Decode the 4-digit code ---
    if len(code) != 4 or not code.isdigit():
        raise ValueError(f"Expected a 4-digit string like '2412', got '{code}'")

    m = int(code[0]) / 100      # max camber as fraction of chord (e.g. 0.02)
    p = int(code[1]) / 10       # location of max camber as fraction (e.g. 0.4)
    t = int(code[2:]) / 100     # max thickness as fraction of chord (e.g. 0.12)

    # --- x positions: cosine spacing clusters points near leading/trailing edge ---
    # This is important for the solver — more panels where curvature is highest.
    # Cosine spacing: x = 0.5 * (1 - cos(theta)) as theta goes 0 → pi
    beta = np.linspace(0, np.pi, n_panels + 1)
    x = 0.5 * (1 - np.cos(beta))   # x from 0 to 1 (normalised chord)

    # --- Thickness distribution (symmetric about camber line) ---
    # NACA standard formula — five coefficients, fixed by definition
    yt = (t / 0.2) * (
          0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4     # use -0.1036 for a closed trailing edge
    )

    # --- Mean camber line and its gradient ---
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)

    if m == 0 or p == 0:
        # Symmetric aerofoil — camber line is flat (NACA 00xx series)
        pass
    else:
        # Forward of max camber location
        fwd = x <= p
        yc[fwd]     = (m / p**2) * (2 * p * x[fwd] - x[fwd]**2)
        dyc_dx[fwd] = (2 * m / p**2) * (p - x[fwd])

        # Aft of max camber location
        aft = x > p
        yc[aft]     = (m / (1 - p)**2) * (1 - 2 * p + 2 * p * x[aft] - x[aft]**2)
        dyc_dx[aft] = (2 * m / (1 - p)**2) * (p - x[aft])

    # --- Surface angle (theta) at each x position ---
    theta = np.arctan(dyc_dx)

    # --- Upper and lower surface coordinates ---
    # Upper: camber line + thickness component, rotated by theta
    x_upper = x  - yt * np.sin(theta)
    y_upper = yc + yt * np.cos(theta)

    # Lower: camber line - thickness component
    x_lower = x  + yt * np.sin(theta)
    y_lower = yc - yt * np.cos(theta)

    return {
        'x_upper':  x_upper,
        'y_upper':  y_upper,
        'x_lower':  x_lower,
        'y_lower':  y_lower,
        'x_camber': x,
        'y_camber': yc,
        'params': {
            'code': code,
            'max_camber': m,
            'camber_location': p,
            'max_thickness': t,
            'n_panels': n_panels
        }
    }


def plot_aerofoil(geo: dict, show_camber: bool = True) -> None:
    """Quick visualisation of the generated aerofoil."""
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(geo['x_upper'], geo['y_upper'], 'b-', linewidth=2, label='Upper surface')
    ax.plot(geo['x_lower'], geo['y_lower'], 'b-', linewidth=2, label='Lower surface')

    if show_camber:
        ax.plot(geo['x_camber'], geo['y_camber'], 'r--', linewidth=1, label='Camber line')

    ax.set_aspect('equal')
    ax.set_xlabel('x/c  (chord fraction)')
    ax.set_ylabel('y/c  (chord fraction)')
    ax.set_title(f"NACA {geo['params']['code']} aerofoil  "
                 f"(t={geo['params']['max_thickness']*100:.0f}%,  "
                 f"m={geo['params']['max_camber']*100:.0f}%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Try a cambered aerofoil (NACA 2412) and a symmetric one (NACA 0012)
    for code in ["2412", "0012", "4415"]:
        geo = naca4_geometry(code, n_panels=100)
        p = geo['params']
        print(f"NACA {code}: "
              f"max thickness {p['max_thickness']*100:.0f}% chord, "
              f"max camber {p['max_camber']*100:.0f}% at {p['camber_location']*100:.0f}% chord")
        plot_aerofoil(geo)





#-------------------------------------------------------------------------

"""
Addition to geometry.py — Module 2: Flap deflection

Takes the plain NACA 4-digit geometry from naca4_geometry() and deflects
everything aft of a hinge point by a given angle, rotating rigidly about
a hinge located ON THE CAMBER LINE at the given chordwise location.

Why rotate about the camber line and not the chord line:
  For a cambered aerofoil, y=0 (the chord line) does not lie on the
  aerofoil surface at the hinge x-location — the camber line does.
  Hinging on the chord line would leave a gap/kink between the fixed
  forward section and the rotated aft section. Hinging on the camber
  line keeps the surface continuous.
"""



def deflect_flap(geo: dict, hinge_x: float = 0.7, deflection_deg: float = 0.0) -> dict:
    """
    Apply a rigid flap deflection to aerofoil geometry produced by
    naca4_geometry().

    Parameters
    ----------
    geo            : dict returned by naca4_geometry()
    hinge_x        : chordwise location of the flap hinge, as a fraction
                     of chord (e.g. 0.7 = hinge at 70% chord)
    deflection_deg : flap deflection angle in degrees.
                     Positive = trailing edge down (increases camber,
                     increases lift and drag) — the sign convention
                     used for a typical landing/lift-augmenting flap.

    Returns
    -------
    dict with the SAME keys as naca4_geometry()'s output
    ('x_upper', 'y_upper', 'x_lower', 'y_lower', 'x_camber', 'y_camber',
    'params'), so it's a drop-in replacement anywhere the plain geometry
    dict is used (including the future .dat writer in dataset.py).
    """

    delta = np.radians(deflection_deg)

    # --- Find the hinge point: interpolate the camber line at hinge_x ---
    x_camber = geo['x_camber']
    y_camber = geo['y_camber']
    hinge_y = np.interp(hinge_x, x_camber, y_camber)

    def rotate_aft_section(x, y):
        """
        Rotate points with x >= hinge_x rigidly about (hinge_x, hinge_y).
        Points forward of the hinge are left untouched.
        """
        x_new = x.copy()
        y_new = y.copy()

        aft_mask = x >= hinge_x

        # Shift so hinge is at the origin, rotate, shift back.
        # Standard 2D rotation matrix. Positive delta = trailing edge
        # rotates downward (negative y direction) — that's why we rotate
        # by +delta here but the y-axis convention makes it deflect down;
        # check the sign against a known case (e.g. deflection_deg=10
        # should visibly droop the trailing edge downward when plotted).
        dx = x[aft_mask] - hinge_x
        dy = y[aft_mask] - hinge_y

        x_new[aft_mask] = hinge_x + dx * np.cos(delta) + dy * np.sin(delta)
        y_new[aft_mask] = hinge_y - dx * np.sin(delta) + dy * np.cos(delta)

        return x_new, y_new

    x_upper_new, y_upper_new = rotate_aft_section(geo['x_upper'], geo['y_upper'])
    x_lower_new, y_lower_new = rotate_aft_section(geo['x_lower'], geo['y_lower'])
    x_camber_new, y_camber_new = rotate_aft_section(x_camber, y_camber)

    new_params = dict(geo['params'])
    new_params['hinge_x'] = hinge_x
    new_params['deflection_deg'] = deflection_deg

    return {
        'x_upper':  x_upper_new,
        'y_upper':  y_upper_new,
        'x_lower':  x_lower_new,
        'y_lower':  y_lower_new,
        'x_camber': x_camber_new,
        'y_camber': y_camber_new,
        'params':   new_params,
    }


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from geometry import naca4_geometry  # adjust import to match your file layout

    base = naca4_geometry("2412", n_panels=100)
    flapped = deflect_flap(base, hinge_x=0.7, deflection_deg=15)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(base['x_upper'], base['y_upper'], 'b-', label='Undeflected upper')
    ax.plot(base['x_lower'], base['y_lower'], 'b-', label='Undeflected lower')
    ax.plot(flapped['x_upper'], flapped['y_upper'], 'r--', label='Flap +15° upper')
    ax.plot(flapped['x_lower'], flapped['y_lower'], 'r--', label='Flap +15° lower')
    ax.axvline(0.7, color='gray', linestyle=':', label='Hinge (70% chord)')
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title('NACA 2412 — flap deflection check')
    plt.tight_layout()
    plt.show()
