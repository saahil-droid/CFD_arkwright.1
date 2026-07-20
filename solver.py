import numpy as np
import matplotlib.pyplot as plt
from panel import make_panels
from geometry import naca4_geometry


#-------------------------------------------------------------
# PIECE 1 — make panels flat at 0,0

def to_local_frame(panel_j, point_x, point_y):
    dx = point_x - panel_j.start_x # distance in global x-direction
    dy = point_y - panel_j.start_y # distance in global y-direction

    theta = panel_j.panel_angle #find the angle of the panel

    local_x = dx * np.cos(theta) + dy * np.sin(theta) # transform to local x-direction to make the panel flat at 0,0
    local_y = -dx * np.sin(theta) + dy * np.cos(theta) # transform to local y-direction to make the panel flat at 0,0

    return local_x, local_y


# ============================================================
# PIECE 2 — flat-panel source solution
# ============================================================

def flat_panel_source_velocity(local_x, local_y, panel_length):

    r1_sq = local_x**2 + local_y**2 #pythagus to find the distance from the point to the start of the panel
    r2_sq = (local_x - panel_length)**2 + local_y**2 # to find the distance from the point to the end of the panel

    theta1 = np.arctan2(local_y, local_x) # find the angle from the point to the start of the panel
    theta2 = np.arctan2(local_y, local_x - panel_length) # find the angle from the point to the end of the panel

    if r1_sq > 0 and r2_sq > 0:    
        u = (1 / (4 * np.pi)) * np.log(r1_sq / r2_sq) # compute the x-component of the velocity
    else:
        u = 0.0

    v = (1 / (2 * np.pi)) * (theta2 - theta1) # compute the y-component of the velocity
    return u, v


# ============================================================
# PIECE 3 — influence coefficient
# ============================================================

def source_influence(panel_i, panel_j):

    if panel_i is panel_j:
        return 0.5  # self-influence approximation

    local_x, local_y = to_local_frame( #this part will transform the coordinates of the midpoint of panel_i to the local frame of panel_j
        panel_j,
        panel_i.midpoint_x, 
        panel_i.midpoint_y
    )


    u_local, v_local = flat_panel_source_velocity( #this part l59-63 looks at the local coor of the midpoint of panel_i in the local frame of panel_j and computes the velocity induced by panel_j at that point
        local_x,
        local_y,
        panel_j.length
    )

    theta = panel_j.panel_angle

    global_u = ( #this part l67-70 transforms the local velocity components back to the global frame
        u_local * np.cos(theta)
        - v_local * np.sin(theta)
    )

    global_v = ( #this part l72-75 will transform the local velocity components back to the global frame
        u_local * np.sin(theta)
        + v_local * np.cos(theta)
    )

    influence = ( #this part l77-80 will compute the influence coefficient by taking the dot product of the induced velocity vector with the normal vector of panel_i
        global_u * panel_i.normal_x +
        global_v * panel_i.normal_y
    )

    return influence


# ============================================================
# PIECE 4 — assemble linear system
# Enforces no-penetration condition at each panel midpoint.
# ============================================================

def build_system(panels, v_inf, alpha_deg):

    n = len(panels)
    A = np.zeros((n, n))
    b = np.zeros(n)

    alpha = np.radians(alpha_deg)

    for i, panel_i in enumerate(panels):

        for j, panel_j in enumerate(panels):
            A[i, j] = source_influence(panel_i, panel_j)

        freestream_normal = (
            np.cos(alpha) * panel_i.normal_x +
            np.sin(alpha) * panel_i.normal_y
        )

        b[i] = -v_inf * freestream_normal

    return A, b


def solve_strengths(panels, v_inf, alpha_deg):
    A, b = build_system(panels, v_inf, alpha_deg)
    return np.linalg.solve(A, b)


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    # --- Geometry ---
    geo = naca4_geometry("0012", n_panels=50)

    x = np.concatenate([
        geo["x_upper"],
        geo["x_lower"][::-1][1:]   # removes duplicate TE point
    ])

    y = np.concatenate([
        geo["y_upper"],
        geo["y_lower"][::-1][1:]
    ])

    panels = make_panels(x, y)

    print(f"Number of panels: {len(panels)}")

    # --- Solve ---
    sigma = solve_strengths(panels, v_inf=1.0, alpha_deg=0)

    # --- check ---
    print("\nSigma (first 10):", sigma[:10])
    print("Sigma (last 10):", sigma[-10:])

    # --- Mass conservation (correct weighted form) ---
    mass_balance = sum(
        sigma[i] * panels[i].length
        for i in range(len(panels))
    )

    print("\nMass balance (should be ~0):", mass_balance)




