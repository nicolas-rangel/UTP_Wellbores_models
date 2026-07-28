# %%
"""
Transient thermal model of a vertical well (Ramey / line-source style).

The script estimates the temperature of a fluid circulating through a vertical
well as it flows up or down, exchanging heat with the surrounding rock.

It computes:
    1) The fluid temperature profile along the well, T(z).
    2) The outlet temperature as a function of time, Tout(t).

Physical basis
--------------
- The rock around the well has a temperature that increases with depth
  (geothermal gradient).
- The fluid enters the well at a temperature Tin.
- As it moves, the fluid relaxes toward the local rock temperature.
- How fast it relaxes depends on:
    a) how much heat it can exchange with the rock (UA' = 1 / resistance), and
    b) the thermal inertia carried by the flow (m_dot * cp).

Main assumptions
----------------
- Formation temperature is linear with depth.
- Heat exchange is modelled only through a transient conduction resistance in the
  rock (logarithmic, Ramey-type). No tubing/cement resistance or explicit internal
  convection is included.
- Constant properties (rock and fluid).

Notes
-----
The model is not restricted to water. The example uses cp = 4180 J/kg/K (water),
but any fluid or effective mixture cp can be supplied (e.g. a water-oil mixture).
"""

import numpy as np
import matplotlib.pyplot as plt

# Seconds in one year
SEC_IN_YEAR = 365 * 24 * 3600.0


def formation_resistance_per_length(
    t_seconds: float,
    r: float,
    krock: float,
    rhorock: float,
    cprock: float,
    Kfac: float = 1.4986,
) -> float:
    """
    Transient thermal resistance of the formation per metre of well [K·m/W].

    The resistance is time dependent because the rock does not heat up (or cool
    down) instantaneously: heat diffuses radially into the rock over time, and the
    affected radius grows roughly as sqrt(alpha * t), where alpha is the thermal
    diffusivity.

    Formula (Ramey / line-source):
        Rf'(t) = ln( Kfac * sqrt(alpha*t) / r ) / (2*pi*krock)

    Parameters
    ----------
    t_seconds : elapsed time [s]
    r         : effective well radius (characteristic exchange radius) [m]
    krock     : rock thermal conductivity [W/m/K]
    rhorock, cprock : rock density [kg/m3] and specific heat [J/kg/K], used for alpha
    Kfac      : classic model constant

    Returns
    -------
    Rf_prime : thermal resistance per metre of well [K·m/W]
    """
    # Rock thermal diffusivity [m^2/s]: larger alpha => heat spreads faster.
    alpha = krock / (rhorock * cprock)

    # Avoid t = 0 (would give sqrt(alpha*t) = 0 and an invalid logarithm).
    t_seconds = max(float(t_seconds), 1.0)

    # sqrt(alpha*t) is a measure of the thermal radius reached by diffusion.
    arg = Kfac * np.sqrt(alpha * t_seconds) / r

    # Keep the argument > 1 so the logarithm stays positive.
    arg = max(float(arg), 1.000001)

    return float(np.log(arg) / (2.0 * np.pi * krock))


def solve_well_profile_at_time(
    *,
    D: float,
    npts: int,
    direction: str,
    Tin: float,
    T_surface: float,
    geothermal_gradient: float,
    t_years: float,
    # Rock properties
    krock: float,
    rhorock: float,
    cprock: float,
    # Geometry
    r: float,
    # Fluid / flow
    mrate_kg_per_s: float,
    cpfluid: float,
    # Constant inside the logarithm
    Kfac: float = 1.4986,
):
    """
    Solve the fluid temperature profile along the well for a given time.

    Depth z is measured from the surface: z = 0 at surface, z = D at the bottom.
    The formation temperature increases linearly with depth:
        Tf(z) = T_surface + G*z,   with G = geothermal_gradient [°C/m]

    The fluid temperature T(s) relaxes toward the rock temperature along the flow
    path s:
        dT/ds = a * (Tf(s) - T),   with a = UA' / (m_dot * cp)

    where UA' [W/K/m] is the conductance per metre and m_dot*cp [W/K] is the
    thermal inertia of the flow. Here R_total' is approximated by Rf'(t) alone.

    Parameters
    ----------
    D                   : total depth [m]
    npts                : number of discrete points in z
    direction           : "down" (inlet at wellhead) or "up" (inlet at bottomhole)
    Tin                 : inlet temperature [°C]
    T_surface           : formation temperature at surface [°C]
    geothermal_gradient : geothermal gradient [°C/m]
    t_years             : time at which the transient resistance is evaluated [years]
    krock, rhorock, cprock : rock properties
    r                   : characteristic radius [m]
    mrate_kg_per_s      : fluid mass flow rate [kg/s]
    cpfluid             : fluid specific heat [J/kg/K]

    Returns
    -------
    z         : depth array [m]
    Tfluid    : fluid temperature at each z [°C]
    Tout      : outlet temperature (surface or bottom, per direction) [°C]
    Tf        : formation temperature at each z [°C]
    a         : spatial coefficient [1/m]
    Rf_prime  : transient resistance per metre evaluated at t [K·m/W]
    """
    # Convert time from years to seconds (for the transient formula).
    t_seconds = float(t_years) * SEC_IN_YEAR

    mdot = float(mrate_kg_per_s)

    # m_dot * cp = thermal capacity of the flow [W/K]. Larger => the fluid changes
    # its temperature less per metre.
    mcp = mdot * float(cpfluid)

    # Transient formation resistance per metre of well at this time.
    Rf_prime = formation_resistance_per_length(
        t_seconds=t_seconds,
        r=r,
        krock=krock,
        rhorock=rhorock,
        cprock=cprock,
        Kfac=Kfac,
    )

    # Conductance per metre [W/K/m]: larger UA' => easier heat transfer.
    UA_prime = 1.0 / Rf_prime

    # Spatial coefficient a [1/m]:
    #   large a  => T approaches Tf quickly
    #   small a  => T changes slowly
    a = UA_prime / mcp

    # Depth mesh from 0 to D.
    z = np.linspace(0.0, float(D), int(npts))

    G = float(geothermal_gradient)

    # Formation temperature at each depth.
    Tf = float(T_surface) + G * z

    direction = direction.lower().strip()

    if direction == "down":
        # Fluid enters at the surface (z=0) and leaves at the bottom (z=D).
        # Path coordinate s coincides with z: s = 0 at surface, s = D at bottom.
        s = z

        # Formation as a function of s (identical to Tf(z) here).
        Tf_s = float(T_surface) + G * s

        # Analytical solution for Tf(s) = Tf0 + G*s (linear, increasing):
        #   dT/ds = a (Tf(s) - T)
        #   T(s)  = Tf0 + G*s - (G/a) + (Tin - Tf0 + G/a) * exp(-a*s)
        #
        # The exp(-a*s) term shows how the memory of Tin fades with distance; for
        # large s, T(s) tracks Tf(s) with an offset of about G/a.
        Tfluid = Tf_s - (G / a) + (float(Tin) - float(T_surface) + (G / a)) * np.exp(-a * s)

        # For "down", the outlet is the last point (bottom).
        Tout = float(Tfluid[-1])

    elif direction == "up":
        # Fluid enters at the bottom (z=D) and leaves at the surface (z=0).
        # Path coordinate s is the distance travelled from the bottom:
        #   s = 0 at the bottom (z=D), s = D at the surface (z=0).
        s = float(D) - z

        # Formation temperature at the bottom.
        Tf_bottom = float(T_surface) + G * float(D)

        # Formation as a function of s: going up (s increases), z decreases, so Tf
        # decreases linearly: Tf(s) = Tf_bottom - G*s.
        Tf_s = Tf_bottom - G * s

        # Analytical solution for Tf(s) = Tf_bottom - G*s (linear, decreasing):
        #   T(s) = Tf_bottom - G*s + (G/a) + (Tin - Tf_bottom - G/a) * exp(-a*s)
        Tfluid = Tf_s + (G / a) + (float(Tin) - Tf_bottom - (G / a)) * np.exp(-a * s)

        # For "up", the outlet is at the surface (z=0), which is index 0.
        Tout = float(Tfluid[0])

    else:
        raise ValueError("direction must be 'down' (inlet at wellhead) or 'up' (inlet at bottomhole).")

    return z, Tfluid, Tout, Tf, a, Rf_prime


def run_time_series(
    *,
    t_years: np.ndarray,
    Tin: np.ndarray,
    D: float = 5000.0,
    npts: int = 501,
    direction: str = "up",
    # Formation
    T_surface: float = 15.0,
    geothermal_gradient: float = 0.03,  # °C/m (30 °C/km)
    # Rock
    krock: float = 3.0,
    rhorock: float = 2663.0,
    cprock: float = 1112.0,
    # Geometry
    r: float = 0.078,
    # Fluid / flow
    rho_w: float = 1000.0,   # water density [kg/m3]
    rho_o: float = 850.0,    # oil density [kg/m3]
    cp_w: float = 4180.0,    # water specific heat [J/kg/K]
    cp_o: float = 2200.0,    # oil specific heat [J/kg/K]
    phi_w: float = 0.9,      # water cut (volume fraction)
    mrate_kg_per_s: float = 100,
    # Model constant
    Kfac: float = 1.4986,
    store_profiles: bool = True,
):
    """
    Run the model for a time series Tin(t).

    For each time in t_years the function:
        1) solves the profile T(z) with solve_well_profile_at_time(...),
        2) stores the outlet temperature Tout(t),
        3) optionally stores the full profile T(z, t) in a matrix Tzt.

    Parameters
    ----------
    t_years        : array of times [years], must be > 0
    Tin            : array of inlet temperatures [°C], same length as t_years
    store_profiles : if True, keep all T(z) profiles over time

    Returns
    -------
    z        : depth mesh [m]
    Tout_ts  : outlet temperature vs time [°C]
    Tf       : formation temperature profile (constant in time here) [°C]
    Tzt      : (nt x nz) matrix of T(z, t) if store_profiles=True, else None
    """
    t_years = np.asarray(t_years, dtype=float)
    Tin = np.asarray(Tin, dtype=float)

    if t_years.shape != Tin.shape:
        raise ValueError("t_years and Tin must have the same shape/length.")

    # t must be > 0 because of the logarithm in the transient resistance.
    if np.any(t_years <= 0.0):
        raise ValueError("All t_years must be > 0 (e.g. start at 0.01 years) to avoid the log singularity.")

    Tout_ts = np.zeros_like(t_years)

    # Store the z mesh and Tf only once (they do not change in time here).
    z_out = None
    Tf_out = None

    Tzt = None
    if store_profiles:
        Tzt = np.zeros((len(t_years), int(npts)), dtype=float)

    for i, (ty, ti) in enumerate(zip(t_years, Tin)):
        # Convert the water cut (volume fraction) to a mass fraction.
        w_w_mass = (phi_w * rho_w) / (phi_w * rho_w + (1.0 - phi_w) * rho_o)

        # Effective mixture cp (mass-weighted average).
        cp_mix = w_w_mass * cp_w + (1.0 - w_w_mass) * cp_o

        z, Tfluid, Tout, Tf, a, Rf_prime = solve_well_profile_at_time(
            D=D,
            npts=npts,
            direction=direction,
            Tin=ti,
            T_surface=T_surface,
            geothermal_gradient=geothermal_gradient,
            t_years=ty,
            krock=krock,
            rhorock=rhorock,
            cprock=cprock,
            r=r,
            mrate_kg_per_s=mrate_kg_per_s,
            cpfluid=cp_mix,
            Kfac=Kfac,
        )

        Tout_ts[i] = Tout

        if store_profiles:
            Tzt[i, :] = Tfluid

        if z_out is None:
            z_out = z
            Tf_out = Tf

    return z_out, Tout_ts, Tf_out, Tzt


# %%
# ---------------------------------------------------------------------------
# Example (runs only when the file is executed directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Time vector in years. Must start > 0 to avoid the log singularity at t = 0.
    t_years = np.linspace(0.1, 30, 100)

    # Example inlet history Tin(t) [°C].
    # Option 1: constant
    # Tin = np.full_like(t_years, 60.0)
    #
    # Option 2: decreases linearly from 100 to 85 °C over the window
    Tin = 100.0 - 15.0 * (t_years - t_years.min()) / (t_years.max() - t_years.min())

    # Flow direction:
    #   "up":   inlet at bottom (z=D), outlet at surface (z=0)
    #   "down": inlet at surface (z=0), outlet at bottom (z=D)
    direction = "up"

    z, Tout_ts, Tf, Tzt = run_time_series(
        t_years=t_years,
        Tin=Tin,
        D=2000.0,                  # well depth [m]
        npts=501,                  # spatial points (more points => smoother profile)
        direction=direction,
        T_surface=15.0,            # formation temperature at surface [°C]
        geothermal_gradient=0.03,  # 30 °C/km
        krock=3.0,                 # rock conductivity [W/m/K]
        rhorock=2663.0,            # rock density [kg/m^3]
        cprock=1112.0,             # rock specific heat [J/kg/K]
        r=0.078,                   # effective radius [m]
        mrate_kg_per_s=25,         # mass flow rate [kg/s]
        store_profiles=True,       # keep T(z, t) profiles
    )

    outlet_label = "Outlet at wellhead (z=0)" if direction == "up" else "Outlet at bottomhole (z=D)"

    # -------------------------------------------------------------------------
    # 1) Plot: Tin(t) and Tout(t)
    # -------------------------------------------------------------------------
    plt.figure()
    plt.plot(t_years, Tin, "k--", label="Tin(t) (inlet)")
    plt.plot(t_years, Tout_ts, "ro-", markersize=3, label="Tout(t) (outlet)")
    plt.xlabel("time (years)")
    plt.ylabel("Temperature (°C)")
    plt.title(f"Inlet and outlet temperature vs time\n({outlet_label})")
    plt.grid(True)
    plt.legend()
    plt.show()

    # -------------------------------------------------------------------------
    # 2) Plot: formation profile Tf(z) and a few fluid profiles T(z)
    # -------------------------------------------------------------------------
    plt.figure()
    plt.plot(Tf, z, "--", label="Formation temperature Tf(z)")

    # Three times: start, middle, end.
    for idx in [0, len(t_years) // 2, -1]:
        plt.plot(Tzt[idx, :], z, label=f"Fluid T(z) at t={t_years[idx]:.2f} years")

    # Depth increases downward.
    plt.gca().invert_yaxis()
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Depth z (m)")
    plt.title("Temperature profiles along the well")
    plt.grid(True)
    plt.legend()
    plt.show()
