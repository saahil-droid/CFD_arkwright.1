import numpy as np
from panel import make_panels


def to_local_frame(source_panel, point_x, point_y):
    """Re-describe a point as if source_panel were lying flat along the
    x-axis, starting at the origin."""
    shifted_x = point_x - source_panel.start_x
    shifted_y = point_y - source_panel.start_y

    tilt_angle = source_panel.panel_angle

    local_x = shifted_x * np.cos(tilt_angle) + shifted_y * np.sin(tilt_angle)
    local_y = -shifted_x * np.sin(tilt_angle) + shifted_y * np.cos(tilt_angle)

    return local_x, local_y


def flat_panel_source_push(local_x, local_y, panel_length):
    """Velocity induced by a unit-strength flat source panel of the given
    length, lying on the local x-axis, evaluated at (local_x, local_y).
    Returns (push_along_panel, push_away_from_panel), still in the local frame."""
    distance_to_start_sq = local_x**2 + local_y**2
    distance_to_end_sq = (local_x - panel_length)**2 + local_y**2

    angle_from_start = np.arctan2(local_y, local_x)
    angle_from_end = np.arctan2(local_y, local_x - panel_length)

    if distance_to_start_sq > 0 and distance_to_end_sq > 0:
        push_along_panel = (1 / (4 * np.pi)) * np.log(distance_to_start_sq / distance_to_end_sq)
    else:
        push_along_panel = 0.0

    push_away_from_panel = (1 / (2 * np.pi)) * (angle_from_end - angle_from_start)

    return push_along_panel, push_away_from_panel


def source_influence(checking_panel, source_panel):
    """How much a unit-strength source on source_panel contributes to the
    NORMAL velocity at checking_panel's control point. One cell of the matrix."""
    if checking_panel is source_panel:
        return 0.5  # a panel's own source has a fixed, known self-influence

    local_x, local_y = to_local_frame(
        source_panel, checking_panel.midpoint_x, checking_panel.midpoint_y
    )
    push_along_panel, push_away_from_panel = flat_panel_source_push(
        local_x, local_y, source_panel.length
    )

    tilt_angle = source_panel.panel_angle
    real_world_push_x = push_along_panel * np.cos(tilt_angle) - push_away_from_panel * np.sin(tilt_angle)
    real_world_push_y = push_along_panel * np.sin(tilt_angle) + push_away_from_panel * np.cos(tilt_angle)

    return real_world_push_x * checking_panel.normal_x + real_world_push_y * checking_panel.normal_y


def vortex_influence(checking_panel, source_panel):
    """How much a unit-strength vortex on source_panel contributes to the
    NORMAL velocity at checking_panel's control point."""
    if checking_panel is source_panel:
        return 0.0  # a panel's own vortex has zero self-induced normal velocity

    local_x, local_y = to_local_frame(
        source_panel, checking_panel.midpoint_x, checking_panel.midpoint_y
    )
    source_push_along, source_push_away = flat_panel_source_push(
        local_x, local_y, source_panel.length
    )

    # a vortex pushes perpendicular to what a source pushes: swap the two, flip one sign
    #vortex_push_along = source_push_away
    #vortex_push_away = -source_push_along

    vortex_push_along = -source_push_away
    vortex_push_away = source_push_along

    tilt_angle = source_panel.panel_angle
    real_world_push_x = vortex_push_along * np.cos(tilt_angle) - vortex_push_away * np.sin(tilt_angle)
    real_world_push_y = vortex_push_along * np.sin(tilt_angle) + vortex_push_away * np.cos(tilt_angle)

    return real_world_push_x * checking_panel.normal_x + real_world_push_y * checking_panel.normal_y


def tangential_influence(checking_panel, source_panel, influence_type):
    """How much source_panel's unit-strength source or vortex contributes to
    the TANGENTIAL velocity at checking_panel's control point.
    influence_type is 'source' or 'vortex'."""
    tangent_x = np.cos(checking_panel.panel_angle)
    tangent_y = np.sin(checking_panel.panel_angle)

    if checking_panel is source_panel:
        # source induces no tangential velocity on itself;
        # a vortex's self tangential influence mirrors a source's self normal influence (0.5)
        return 0.0 if influence_type == "source" else 0.5

    local_x, local_y = to_local_frame(
        source_panel, checking_panel.midpoint_x, checking_panel.midpoint_y
    )
    source_push_along, source_push_away = flat_panel_source_push(
        local_x, local_y, source_panel.length
    )

    
    #if influence_type == "source":
    #    push_along, push_away = source_push_along, source_push_away
    #else:
     #   push_along, push_away = source_push_away, -source_push_along


    if influence_type == "source":
        push_along, push_away = source_push_along, source_push_away

    else:
        push_along, push_away = -source_push_away, source_push_along

    tilt_angle = source_panel.panel_angle
    real_world_push_x = push_along * np.cos(tilt_angle) - push_away * np.sin(tilt_angle)
    real_world_push_y = push_along * np.sin(tilt_angle) + push_away * np.cos(tilt_angle)

    

    return real_world_push_x * tangent_x + real_world_push_y * tangent_y


def build_full_system(panels, freestream_speed, angle_of_attack_deg):
    """Builds the (panel_count+1) x (panel_count+1) linear system:
    one no-flow-through-surface equation per panel, plus the Kutta condition."""
    panel_count = len(panels)
    influence_matrix = np.zeros((panel_count + 1, panel_count + 1))
    required_normal_velocity = np.zeros(panel_count + 1)

    angle_of_attack_rad = np.radians(angle_of_attack_deg)

    for i, checking_panel in enumerate(panels):
        for j, source_panel in enumerate(panels):
            influence_matrix[i, j] = source_influence(checking_panel, source_panel)

        # shared vortex column: every panel's no-penetration equation also depends on gamma
        influence_matrix[i, panel_count] = sum(
            vortex_influence(checking_panel, source_panel) for source_panel in panels
        )

        freestream_normal_component = (
            np.cos(angle_of_attack_rad) * checking_panel.normal_x
            + np.sin(angle_of_attack_rad) * checking_panel.normal_y
        )
        required_normal_velocity[i] = -freestream_speed * freestream_normal_component

    # Kutta condition: tangential velocity at the two trailing-edge panels must sum to zero
    upper_te_panel = panels[0]
    lower_te_panel = panels[-1]

    #upper_te_panel = panels[-1]
    #lower_te_panel = panels[0]

    for j, source_panel in enumerate(panels):
        influence_matrix[panel_count, j] = (
            tangential_influence(upper_te_panel, source_panel, "source")
            + tangential_influence(lower_te_panel, source_panel, "source")
        )

    influence_matrix[panel_count, panel_count] = sum(
        tangential_influence(upper_te_panel, source_panel, "vortex")
        + tangential_influence(lower_te_panel, source_panel, "vortex")
        for source_panel in panels
    )

    upper_tangent = (np.cos(upper_te_panel.panel_angle), np.sin(upper_te_panel.panel_angle))
    lower_tangent = (np.cos(lower_te_panel.panel_angle), np.sin(lower_te_panel.panel_angle))
    freestream_tangential_total = freestream_speed * (
        np.cos(angle_of_attack_rad) * upper_tangent[0] + np.sin(angle_of_attack_rad) * upper_tangent[1]
        + np.cos(angle_of_attack_rad) * lower_tangent[0] + np.sin(angle_of_attack_rad) * lower_tangent[1]
    )
    required_normal_velocity[panel_count] = -freestream_tangential_total

    




    print("TE panels:")
    print("upper angle:", upper_te_panel.panel_angle)
    print("lower angle:", lower_te_panel.panel_angle)

    print("upper tangent:",
        upper_tangent)

    print("lower tangent:",
        lower_tangent)


    return influence_matrix, required_normal_velocity

def solve_for_strengths(panels, freestream_speed, angle_of_attack_deg):
    """Solves the full system and returns (source_strengths, vortex_strength)."""
    influence_matrix, required_normal_velocity = build_full_system(
        panels, freestream_speed, angle_of_attack_deg
    )
    solution = np.linalg.solve(influence_matrix, required_normal_velocity)

    source_strengths = solution[:-1]   # one per panel
    vortex_strength = solution[-1]     # single shared value

    return source_strengths, vortex_strength


if __name__ == "__main__":
    from geometry import naca4_geometry

    shape = naca4_geometry("0012", n_panels=50)
    x_coords = np.concatenate([shape["x_upper"], shape["x_lower"][::-1][1:]])
    y_coords = np.concatenate([shape["y_upper"], shape["y_lower"][::-1][1:]])
    panels = make_panels(x_coords, y_coords)

    source_strengths, vortex_strength = solve_for_strengths(panels, freestream_speed=1.0, angle_of_attack_deg=0)
    print("Symmetric NACA 0012 @ 0 deg AoA")
    print("Vortex strength (should be ~0):", vortex_strength)

    source_strengths_5, vortex_strength_5 = solve_for_strengths(panels, freestream_speed=1.0, angle_of_attack_deg=5)
    print("\nSymmetric NACA 0012 @ 5 deg AoA")
    print("Vortex strength (should be clearly nonzero):", vortex_strength_5)

    total_circulation = vortex_strength_5 * sum(p.length for p in panels)
    lift_coefficient = 2 * total_circulation  # v_inf = 1, chord = 1
    print("Approx lift coefficient @ 5 deg:", lift_coefficient)



import numpy as np


def tangential_velocity_at_panel(checking_panel, panels, source_strengths, vortex_strength,
                                    freestream_speed, angle_of_attack_deg):
    """Total tangential velocity at checking_panel's control point: freestream
    contribution + every panel's source and vortex contribution."""
    angle_of_attack_rad = np.radians(angle_of_attack_deg)

    tangent_x = np.cos(checking_panel.panel_angle)
    tangent_y = np.sin(checking_panel.panel_angle)
    freestream_tangential = freestream_speed * (
        np.cos(angle_of_attack_rad) * tangent_x + np.sin(angle_of_attack_rad) * tangent_y
    )

    induced_tangential = 0.0
    for source_panel, sigma in zip(panels, source_strengths):
        induced_tangential += sigma * tangential_influence(checking_panel, source_panel, "source")

    induced_tangential += vortex_strength * sum(
        tangential_influence(checking_panel, source_panel, "vortex") for source_panel in panels
    )

    return freestream_tangential + induced_tangential


def compute_pressure_coefficients(panels, source_strengths, vortex_strength, freestream_speed,
                                     angle_of_attack_deg):
    """Cp at every panel's control point, from Bernoulli's equation."""
    cp_values = []
    for panel in panels:
        v_tangential = tangential_velocity_at_panel(
            panel, panels, source_strengths, vortex_strength, freestream_speed, angle_of_attack_deg
        )
        cp = 1 - (v_tangential / freestream_speed) ** 2
        cp_values.append(cp)
    return np.array(cp_values)


def integrate_pressure_forces(panels, cp_values, angle_of_attack_deg):
    """Integrate Cp around the surface to get Cl and Cd from pressure alone
    (this should come out near-zero for Cd — d'Alembert's paradox)."""
    angle_of_attack_rad = np.radians(angle_of_attack_deg)

    normal_force_coefficient = -sum(
        cp * panel.normal_y * panel.length for cp, panel in zip(cp_values, panels)
    )
    axial_force_coefficient = -sum(
        cp * panel.normal_x * panel.length for cp, panel in zip(cp_values, panels)
    )

    cl_pressure = (
        normal_force_coefficient * np.cos(angle_of_attack_rad)
        - axial_force_coefficient * np.sin(angle_of_attack_rad)
    )
    cd_pressure = (
        normal_force_coefficient * np.sin(angle_of_attack_rad)
        + axial_force_coefficient * np.cos(angle_of_attack_rad)
    )

    return cl_pressure, cd_pressure


def skin_friction_drag_coefficient(freestream_speed, chord_length, kinematic_viscosity=1.5e-5):
    """Flat-plate turbulent skin-friction estimate (simplification — treats
    the whole chord as turbulent from the leading edge)."""
    reynolds_number = (freestream_speed * chord_length) / kinematic_viscosity
    cf_turbulent = 0.074 / reynolds_number ** 0.2
    return cf_turbulent  # applied over wetted area ≈ 2 × chord for a thin aerofoil, per unit span


def compute_total_drag(panels, source_strengths, vortex_strength, freestream_speed,
                          angle_of_attack_deg, chord_length, kinematic_viscosity=1.5e-5):
    """Full Cl, Cd pipeline: pressure integration + skin friction correction."""
    cp_values = compute_pressure_coefficients(
        panels, source_strengths, vortex_strength, freestream_speed, angle_of_attack_deg
    )
    cl_pressure, cd_pressure = integrate_pressure_forces(panels, cp_values, angle_of_attack_deg)

    cd_skin_friction = skin_friction_drag_coefficient(
        freestream_speed, chord_length, kinematic_viscosity
    ) * 2  # rough factor of 2 for upper + lower surface wetted area

    cd_total = cd_pressure + cd_skin_friction

    return cp_values, cl_pressure, cd_total
