# UTP Wellbore Models

A small toolkit of vertical-well models and a thermodynamic exergy calculator,
developed for the Minciencias–UTP (Universidad Tecnológica de Pereira) project.
The scripts support quick estimates of well thermal behaviour, hydraulic pressure
drop, and the exergy content of a working fluid.

## Contents

| File | Description |
|------|-------------|
| `thermal_well_model.py` | Transient thermal model of a vertical well (Ramey / line-source). Computes the fluid temperature profile `T(z)` along the well and the outlet temperature over time `Tout(t)`, exchanging heat with the surrounding rock. |
| `hydraulic_well_model.py` | Pressure drop in a vertical well for an oil–water mixture, using a homogeneous (no-slip) model with a single-phase Darcy friction correlation. Returns the pressure profile `P(z)` and wellhead/bottomhole pressures. |
| `exergy_lookup_table.py` | Generates a self-contained Excel workbook with an interactive exergy calculator and enthalpy/entropy lookup tables computed with CoolProp. |

## Requirements

- Python 3.9+
- `numpy`
- `matplotlib` (thermal and hydraulic models)
- `CoolProp` and `openpyxl` (exergy table generator)

```bash
pip install numpy matplotlib CoolProp openpyxl
```

## Usage

Each script is self-contained and can be run directly. They also expose reusable
functions if you prefer to import them.

```bash
python thermal_well_model.py     # temperature profiles and outlet-temperature plots
python hydraulic_well_model.py   # pressure profiles and gradient decomposition
python exergy_lookup_table.py    # writes Exergy_Lookup_Table.xlsx
```

The `.py` files are organised as `# %%` cells, so they can also be run
cell-by-cell in the VS Code / Jupyter interactive window.

### Thermal well model

Solves a relaxation equation for the fluid temperature along the flow path:

```
dT/ds = a · (Tf(s) − T),   a = UA' / (ṁ·cp)
```

where `Tf(z) = T_surface + G·z` is the formation temperature (linear geothermal
gradient) and the formation heat exchange is modelled with a transient,
Ramey-type conduction resistance. Flow can be `"up"` (inlet at bottomhole) or
`"down"` (inlet at wellhead). An effective mixture `cp` is supported for
water–oil flows.

### Hydraulic well model

Total pressure gradient as the sum of a hydrostatic and a friction component:

```
dP/dz = ρ_m·g  ±  f·ρ_m·v² / (2·D)
```

with a volume-weighted mixture density and a log (Arrhenius) or linear mixing
rule for viscosity. The friction term is signed by flow direction (`"down"` =
injection, `"up"` = production, with `z` positive downward).

### Exergy lookup table

Computes specific physical exergy relative to a dead state `(T0, P0)`:

```
ex = (h − h0) − T0·(s − s0)
```

The generated workbook has three sheets: an exergy calculator (enter T, P and
mass flow rate; read back h, s, exergy, gross heat rate and exergy rate) plus
enthalpy and entropy lookup tables over a temperature–pressure grid. The
calculator interpolates from the lookup sheets with `INDEX/MATCH`, so it works in
Excel without Python once generated.
