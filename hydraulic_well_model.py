# %%
"""
Pressure drop in a vertical well with an oil-water (liquid-liquid) mixture.
Homogeneous (no-slip) model with a single-phase friction correlation.

The script estimates the pressure drop in a vertical well when the circulating
fluid is a mixture of oil and water. The total pressure gradient dP/dz is the sum
of two terms:

    1) Hydrostatic component (weight of the fluid column).
    2) Friction component (losses against the pipe wall).

Governing equation
-------------------
    dP/dz = rho_m * g  ±  f * rho_m * v^2 / (2*D)
    friction term -> + for injection, - for production (with z pointing down)

    rho_m : mixture density [kg/m3]
    g     : gravity [m/s2]
    f     : Darcy friction factor [-]
    v     : average velocity in the pipe [m/s]
    D     : inner diameter [m]

Assumptions
-----------
- Vertical well (z positive downward).
- Steady state.
- Incompressible, constant properties (fixed rho and mu).
- Homogeneous (no-slip) model: oil and water move at the same average velocity;
  slip / separated phases are not modelled.
- Acceleration term is neglected.
- Friction is computed as an equivalent single-phase flow using Re (with effective
  properties) and a simple Darcy f.

Suitable for quick estimates and for comparing the effect of more water or more
oil on the pressure drop. It is not appropriate for clearly two-phase flow with
slip, changing flow patterns with depth, gas, or strong property variations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Tuple, List
import matplotlib.pyplot as plt


# %%
# ---------------------------------------------------------------------------
# 1) Data structures (to group parameters)
# ---------------------------------------------------------------------------

@dataclass
class Well:
    """
    Basic well / pipe geometry.

    depth_m     : vertical depth [m]
    diameter_m  : tubing inner diameter [m]
    roughness_m : absolute roughness [m]; 0 => hydraulically smooth
    """
    depth_m: float
    diameter_m: float
    roughness_m: float = 0.0


@dataclass
class FluidProps:
    """
    Properties of the two liquid phases (oil and water), assumed constant.

    rho_oil, rho_wat : densities [kg/m3]
    mu_oil,  mu_wat  : dynamic viscosities [Pa*s]  (1 cP = 1e-3 Pa*s)
    """
    rho_oil: float
    mu_oil: float
    rho_wat: float
    mu_wat: float


# Mixing rule for viscosity, flow direction, and location of the known pressure.
MixRule = Literal["log", "linear"]
FlowDirection = Literal["down", "up"]        # down = injection, up = production (z down)
PressureRef = Literal["wellhead", "bottom"]  # where the known pressure is located

# %%
# ---------------------------------------------------------------------------
# 2) Helper functions (geometry and mixtures)
# ---------------------------------------------------------------------------


def area(diameter_m: float) -> float:
    """Cross-sectional area of a circular pipe: A = pi * D^2 / 4  [m2]."""
    return math.pi * (diameter_m ** 2) / 4.0


def mixture_from_massflows(
    m_oil: float,
    m_wat: float,
    props: FluidProps,
    mix_rule: MixRule = "log",
) -> Tuple[float, float, float, float, float]:
    """
    Compute effective mixture properties from oil and water mass flow rates.

    Steps:
        1) Convert mass flows to volumetric flows:  q = m / rho
        2) Volume fractions:  phi_oil = q_oil / (q_oil + q_wat)
        3) Mixture density (volume-weighted):
               rho_mix = phi_oil*rho_oil + phi_wat*rho_wat
        4) Mixture viscosity (two options):
               "log"    (Arrhenius, common for viscosity):
                   ln(mu_mix) = phi_oil*ln(mu_oil) + phi_wat*ln(mu_wat)
               "linear":
                   mu_mix = phi_oil*mu_oil + phi_wat*mu_wat

    Parameters
    ----------
    m_oil, m_wat : oil and water mass flow rates [kg/s]
    props        : oil/water densities and viscosities
    mix_rule     : "log" or "linear"

    Returns
    -------
    q_oil   : oil volumetric flow [m3/s]
    q_wat   : water volumetric flow [m3/s]
    phi_oil : oil volume fraction [-]
    rho_mix : effective mixture density [kg/m3]
    mu_mix  : effective mixture viscosity [Pa*s]

    Note
    ----
    The homogeneous model assumes both phases move together (no-slip). In reality
    oil and water can show slip, holdup, emulsification, etc.
    """
    # Reject non-physical inputs.
    if m_oil < 0 or m_wat < 0:
        raise ValueError("Mass flow rates must be non-negative.")
    if props.rho_oil <= 0 or props.rho_wat <= 0:
        raise ValueError("Densities must be positive.")
    if props.mu_oil <= 0 or props.mu_wat <= 0:
        raise ValueError("Viscosities must be positive.")

    # 1) Mass flow (kg/s) -> volumetric flow (m3/s)
    q_oil = m_oil / props.rho_oil
    q_wat = m_wat / props.rho_wat
    q_tot = q_oil + q_wat

    if q_tot <= 0:
        raise ValueError("Total volumetric flow is zero; there is no flow to compute.")

    # 2) Volume fractions
    phi_oil = q_oil / q_tot
    phi_wat = 1.0 - phi_oil

    # 3) Mixture density (volume-weighted average)
    rho_mix = phi_oil * props.rho_oil + phi_wat * props.rho_wat

    # 4) Mixture viscosity
    if mix_rule == "log":
        # Arrhenius-type mixing, common for viscosity.
        mu_mix = math.exp(phi_oil * math.log(props.mu_oil) + phi_wat * math.log(props.mu_wat))
    elif mix_rule == "linear":
        # Simple linear mixing.
        mu_mix = phi_oil * props.mu_oil + phi_wat * props.mu_wat
    else:
        raise ValueError("mix_rule must be 'log' or 'linear'.")

    return q_oil, q_wat, phi_oil, rho_mix, mu_mix


# ---------------------------------------------------------------------------
# 3) Friction factor (Darcy) and Reynolds number
# ---------------------------------------------------------------------------


def friction_factor_swamee_jain(Re: float, rel_rough: float) -> float:
    """
    Darcy friction factor in turbulent flow (explicit Swamee-Jain approximation).

        f = 0.25 / [ log10( e/(3.7D) + 5.74/Re^0.9 ) ]^2

    Applies mainly to turbulent flow (Re > ~5000).

    Parameters
    ----------
    Re        : Reynolds number [-], must be > 0
    rel_rough : relative roughness e/D [-], >= 0
    """
    if Re <= 0:
        raise ValueError("Re must be positive.")
    if rel_rough < 0:
        raise ValueError("Relative roughness must be non-negative.")

    term = rel_rough / 3.7 + 5.74 / (Re ** 0.9)
    return 0.25 / (math.log10(term) ** 2)


def friction_factor(Re: float, rel_rough: float) -> float:
    """
    Darcy friction factor with simple regime handling:

        laminar     : f = 64/Re              for Re < 2300
        transition  : linear blend           2300..4000
        turbulent   : Swamee-Jain            for Re > 4000

    The transition blend is not a refined model; it only provides numerical
    continuity between the laminar and turbulent branches.
    """
    if Re <= 0:
        raise ValueError("Re must be positive.")

    if Re < 2300:
        return 64.0 / Re

    if Re > 4000:
        return friction_factor_swamee_jain(Re, rel_rough)

    # Linear blend across the transition region.
    f_lam = 64.0 / 2300.0
    f_turb = friction_factor_swamee_jain(4000.0, rel_rough)
    w = (Re - 2300.0) / (4000.0 - 2300.0)
    return (1.0 - w) * f_lam + w * f_turb


# ---------------------------------------------------------------------------
# 4) Pressure gradient in a vertical well
# ---------------------------------------------------------------------------


def pressure_gradient_vertical(
    well: Well,
    q_total_m3s: float,
    rho_mix: float,
    mu_mix: float,
    g: float = 9.81,
    flow_direction: FlowDirection = "down",
) -> Tuple[float, float, float, float, float]:
    """
    Pressure gradient [Pa/m] in a vertical well.

    Coordinate convention: z positive downward, z = 0 at the wellhead.

    Magnitudes:
        dP/dz_hydro     = rho_mix * g                    (always + with z down)
        dP/dz_fric_mag  = f * rho_mix * v^2 / (2*D)      (positive magnitude)

    Friction always opposes motion. With z pointing down:
        flow_direction="down" (injection, flow toward +z):
            dP/dz_total = + rho g + dP/dz_fric_mag
        flow_direction="up"   (production, flow toward -z):
            dP/dz_total = + rho g - dP/dz_fric_mag

    Parameters
    ----------
    q_total_m3s    : total volumetric flow [m3/s], used as a magnitude (> 0)
    flow_direction : "down" or "up"

    Returns
    -------
    dPdz_total       : total gradient [Pa/m] (signed, per convention)
    dPdz_hydro       : hydrostatic gradient [Pa/m]
    dPdz_fric_signed : friction gradient [Pa/m] (signed)
    Re               : Reynolds number [-]
    f                : Darcy friction factor [-]
    """
    if q_total_m3s <= 0:
        raise ValueError("q_total_m3s must be positive (flow magnitude).")
    if rho_mix <= 0 or mu_mix <= 0:
        raise ValueError("rho_mix and mu_mix must be positive.")
    if well.diameter_m <= 0 or well.depth_m <= 0:
        raise ValueError("Well diameter and depth must be positive.")
    if flow_direction not in ("down", "up"):
        raise ValueError("flow_direction must be 'down' or 'up'.")

    # Cross-sectional area.
    A = area(well.diameter_m)

    # Average velocity (magnitude).
    v = q_total_m3s / A

    # Reynolds number (equivalent mixture).
    Re = rho_mix * v * well.diameter_m / mu_mix

    # Relative roughness e/D.
    rel_rough = (well.roughness_m / well.diameter_m) if well.roughness_m > 0 else 0.0

    # Darcy friction factor.
    f = friction_factor(Re, rel_rough)

    # Hydrostatic component (always positive with z down).
    dPdz_h = rho_mix * g

    # Friction magnitude (positive).
    dPdz_f_mag = f * rho_mix * v * v / (2.0 * well.diameter_m)

    # Sign from the flow direction.
    sign = +1.0 if flow_direction == "down" else -1.0
    dPdz_f_signed = sign * dPdz_f_mag

    # Total gradient for P(z) with z positive downward.
    dPdz_total = dPdz_h + dPdz_f_signed

    return dPdz_total, dPdz_h, dPdz_f_signed, Re, f


# ---------------------------------------------------------------------------
# 5) Total pressure drop over the full depth
# ---------------------------------------------------------------------------


def pressure_drop(
    well: Well,
    m_oil: float,
    m_wat: float,
    props: FluidProps,
    mix_rule: MixRule = "log",
    g: float = 9.81,
    flow_direction: FlowDirection = "down",
) -> dict:

    q_oil, q_wat, phi_oil, rho_mix, mu_mix = mixture_from_massflows(
        m_oil, m_wat, props, mix_rule=mix_rule
    )
    q_total = q_oil + q_wat

    dPdz_total, dPdz_h, dPdz_f_signed, Re, f = pressure_gradient_vertical(
        well, q_total, rho_mix, mu_mix, g=g, flow_direction=flow_direction
    )

    dP_total = dPdz_total * well.depth_m
    dP_h = dPdz_h * well.depth_m
    dP_f = dPdz_f_signed * well.depth_m

    return {
        "mix_rule": mix_rule,
        "flow_direction": flow_direction,
        "q_oil_m3s": q_oil,
        "q_wat_m3s": q_wat,
        "q_total_m3s": q_total,
        "phi_oil": phi_oil,
        "rho_mix_kgm3": rho_mix,
        "mu_mix_Pas": mu_mix,
        "velocity_ms": q_total / area(well.diameter_m),
        "Re": Re,
        "friction_factor_Darcy": f,
        "dPdz_total_Pam": dPdz_total,
        "dPdz_hydro_Pam": dPdz_h,
        "dPdz_fric_Pam": dPdz_f_signed,
        "deltaP_total_bar": dP_total / 1e5,
        "deltaP_hydro_bar": dP_h / 1e5,
        "deltaP_fric_bar": dP_f / 1e5,
    }


# ---------------------------------------------------------------------------
# 6) Pressure profile P(z) (discretised)
# ---------------------------------------------------------------------------


def pressure_profile(
    well: Well,
    p_ref_bar: float,
    p_ref_at: PressureRef,
    m_oil: float,
    m_wat: float,
    props: FluidProps,
    mix_rule: MixRule = "log",
    n_steps: int = 50,
    g: float = 9.81,
    flow_direction: FlowDirection = "down",
) -> List[Tuple[float, float]]:
    """
    Pressure profile P(z) for a vertical well, assuming a constant gradient.

    Convention:
        z = 0 at the wellhead, z positive downward, z = H (= well.depth_m) at the
        bottomhole.

    Reference pressure:
        p_ref_at="wellhead":  P(0) = p_ref_bar   (WHP)
        p_ref_at="bottom":    P(H) = p_ref_bar   (BHP)

    General form:
        P(z) = P(z_ref) + (dP/dz) * (z - z_ref)
        with z_ref = 0 (wellhead) or z_ref = H (bottom).

    With z pointing down the hydrostatic term is +rho*g and the friction term
    changes sign with flow_direction:
        "down": dP/dz = +rho*g + friction
        "up"  : dP/dz = +rho*g - friction
    """
    if p_ref_at not in ("wellhead", "bottom"):
        raise ValueError("p_ref_at must be 'wellhead' or 'bottom'.")
    if n_steps <= 0:
        raise ValueError("n_steps must be a positive integer.")

    result = pressure_drop(
        well, m_oil, m_wat, props,
        mix_rule=mix_rule, g=g, flow_direction=flow_direction
    )
    dPdz = result["dPdz_total_Pam"]  # Pa/m

    H = well.depth_m
    z_ref = 0.0 if p_ref_at == "wellhead" else H

    profile: List[Tuple[float, float]] = []
    for i in range(n_steps + 1):
        z = H * i / n_steps
        p_bar = p_ref_bar + (dPdz * (z - z_ref)) / 1e5
        profile.append((z, p_bar))

    return profile


def pressure_at_wellhead_or_bottom(
    well: Well,
    p_ref_bar: float,
    p_ref_at: PressureRef,
    m_oil: float,
    m_wat: float,
    props: FluidProps,
    mix_rule: MixRule = "log",
    g: float = 9.81,
    flow_direction: FlowDirection = "down",
) -> dict:
    """
    Return WHP and BHP from a single known pressure (at the wellhead or bottom).
    Useful for a quick report without discretising a full profile.
    """
    if p_ref_at not in ("wellhead", "bottom"):
        raise ValueError("p_ref_at must be 'wellhead' or 'bottom'.")

    result = pressure_drop(
        well, m_oil, m_wat, props,
        mix_rule=mix_rule, g=g, flow_direction=flow_direction
    )
    dPdz = result["dPdz_total_Pam"]
    H = well.depth_m

    if p_ref_at == "wellhead":
        WHP = p_ref_bar
        BHP = p_ref_bar + (dPdz * H) / 1e5
    elif p_ref_at == "bottom":
        BHP = p_ref_bar
        WHP = p_ref_bar - (dPdz * H) / 1e5
    else:
        raise ValueError("p_ref_at must be 'wellhead' or 'bottom'.")

    return {
        **result,
        "WHP_bar": WHP,
        "BHP_bar": BHP,
    }


# %%
# ===========================================================================
# Example 1: known wellhead pressure (WHP), upward flow (production)
# ===========================================================================
if __name__ == "__main__":

    # Well geometry
    well = Well(
        depth_m=2500.0,
        diameter_m=0.0762,
        roughness_m=1e-5,
    )

    # Fluid properties
    props = FluidProps(
        rho_oil=800.0,   # kg/m3
        mu_oil=3e-3,     # Pa*s (3 cP)
        rho_wat=1000.0,  # kg/m3
        mu_wat=0.7e-3,   # Pa*s (0.7 cP)
    )

    # Operating conditions
    flow_direction = "up"   # "up" = production
    p_ref_bar = 50.0        # known WHP
    p_ref_at = "wellhead"

    m_oil = 5.0   # kg/s
    m_wat = 10.0  # kg/s

    # WHP / BHP
    summary = pressure_at_wellhead_or_bottom(
        well=well,
        p_ref_bar=p_ref_bar,
        p_ref_at=p_ref_at,
        m_oil=m_oil,
        m_wat=m_wat,
        props=props,
        mix_rule="log",
        flow_direction=flow_direction,
    )

    print("\n=== CASE: KNOWN WELLHEAD PRESSURE (WHP) ===")
    print(f"Flow direction : {summary['flow_direction']}")
    print(f"WHP = {summary['WHP_bar']:.2f} bar")
    print(f"BHP = {summary['BHP_bar']:.2f} bar")
    print(f"dP total = {summary['deltaP_total_bar']:.2f} bar")
    print(f"dP hydro = {summary['deltaP_hydro_bar']:.2f} bar")
    print(f"dP fric  = {summary['deltaP_fric_bar']:.2f} bar")

    # Pressure profile
    prof = pressure_profile(
        well=well,
        p_ref_bar=p_ref_bar,
        p_ref_at=p_ref_at,
        m_oil=m_oil,
        m_wat=m_wat,
        props=props,
        mix_rule="log",
        n_steps=10,
        flow_direction=flow_direction,
    )

    print("\nPressure profile (z [m], P [bar]):")
    for z, p in prof:
        print(f"{z:7.1f} m : {p:8.2f} bar")


# %%
# ===========================================================================
# Example 2: known bottomhole pressure (BHP), downward flow (injection)
# ===========================================================================
if __name__ == "__main__":

    # Well geometry
    well = Well(
        depth_m=2500.0,
        diameter_m=0.0762,
        roughness_m=1e-5,
    )

    # Fluid properties
    props = FluidProps(
        rho_oil=800.0,   # kg/m3
        mu_oil=3e-3,     # Pa*s (3 cP)
        rho_wat=1000.0,  # kg/m3
        mu_wat=0.7e-3,   # Pa*s (0.7 cP)
    )

    # Operating conditions
    flow_direction = "down"  # "down" = injection
    p_ref_bar = 350.0        # known BHP
    p_ref_at = "bottom"

    m_oil = 5.0   # kg/s
    m_wat = 20.0  # kg/s

    # WHP / BHP
    summary = pressure_at_wellhead_or_bottom(
        well=well,
        p_ref_bar=p_ref_bar,
        p_ref_at=p_ref_at,
        m_oil=m_oil,
        m_wat=m_wat,
        props=props,
        mix_rule="log",
        flow_direction=flow_direction,
    )

    print("\n=== CASE: KNOWN BOTTOMHOLE PRESSURE (BHP) ===")
    print(f"Flow direction : {summary['flow_direction']}")
    print(f"BHP = {summary['BHP_bar']:.2f} bar")
    print(f"WHP = {summary['WHP_bar']:.2f} bar")
    print(f"dP total = {summary['deltaP_total_bar']:.2f} bar")
    print(f"dP hydro = {summary['deltaP_hydro_bar']:.2f} bar")
    print(f"dP fric  = {summary['deltaP_fric_bar']:.2f} bar")

    # Pressure profile
    prof2 = pressure_profile(
        well=well,
        p_ref_bar=p_ref_bar,
        p_ref_at=p_ref_at,
        m_oil=m_oil,
        m_wat=m_wat,
        props=props,
        mix_rule="log",
        n_steps=10,
        flow_direction=flow_direction,
    )

    print("\nPressure profile (z [m], P [bar]):")
    for z, p in prof2:
        print(f"{z:7.1f} m : {p:8.2f} bar")


# %%
# --- Plot 1: pressure profile P(z) ---
# Case 1 (production) is stored in "prof"; case 2 (injection) in "prof2".
z1 = [z for z, _ in prof]
p1 = [p for _, p in prof]

z2 = [z for z, _ in prof2]
p2 = [p for _, p in prof2]

plt.figure()
plt.plot(p1, z1, label="Case 1: known WHP, production (up)")
plt.plot(p2, z2, label="Case 2: known BHP, injection (down)")
plt.gca().invert_yaxis()  # depth increases downward
plt.xlabel("Pressure [bar]")
plt.ylabel("Depth z [m]")
plt.title("Pressure profile P(z)")
plt.legend()
plt.grid(True)
plt.show()


# %%
# --- Plot 2: gradient decomposition (hydrostatic vs friction) ---
# Uses "summary" from the last computed case (case 2 above).

def pam_to_bar_per_100m(x_pam: float) -> float:
    return x_pam * 100.0 / 1e5  # Pa/m -> bar/100m

labels = ["Case (last computed)"]

hydro = [pam_to_bar_per_100m(summary["dPdz_hydro_Pam"])]
fric = [pam_to_bar_per_100m(summary["dPdz_fric_Pam"])]
total = [pam_to_bar_per_100m(summary["dPdz_total_Pam"])]

x = list(range(len(labels)))
w = 0.25

plt.figure()
plt.bar([i - w for i in x], hydro, width=w, label="Hydrostatic", zorder=3)
plt.bar(x, fric, width=w, label="Friction", zorder=3)
plt.bar([i + w for i in x], total, width=w, label="Total", zorder=3)
plt.xticks(x, labels)
plt.ylabel("Gradient [bar/100 m]")
plt.title("Pressure gradient: hydrostatic vs friction")
plt.grid(True, axis="y", zorder=0)
plt.legend()
plt.show()
