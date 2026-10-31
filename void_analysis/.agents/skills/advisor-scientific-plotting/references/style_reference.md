# Figure Styling & Visual Palette Reference

This reference contains the exact rcParams, color palettes, and layout standards for figures in this project.

## 1. Unified rcParams Configuration

Call `plt.rcParams.update(UNIFIED_RCPARAMS)` or apply this dictionary in your scripts:

```python
UNIFIED_RCPARAMS = {
    'font.size': 12,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
    'axes.titlesize': 12.5,
    'axes.titleweight': 'bold',
    'axes.linewidth': 1.1,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
}
```

## 2. Color Palettes

### Halo Subsamples
| Sample | Hex Color | Name |
| :--- | :--- | :--- |
| `20p` | `#8e44ad` | Wisteria Purple |
| `10p` | `#2980b9` | Belize Hole Blue |
| `5p` | `#27ae60` | Nephritis Green |
| `2_5p`| `#d4ac0d` | Sun Yellow |

### Comparison Variants & Reference Models
| Series | Marker | Color | Hex |
| :--- | :--- | :--- | :--- |
| Simone / Reference | `o` | Near Black | `#1a1a1a` |
| Vdn Fiducial Truth | solid line | Near Black | `#1a1a1a` |
| BitVF TOUCH/OV00 | `o` | Blue | `#2980b9` |
| BitVF CENTER/OV05 | `s` | Orange | `#e67e22` |
| REVOLVER | `^` | Crimson | `#c0392b` |

## 3. Exclusion Bands and Cutoffs
- **Band Fill**: `#f0f0f0` (Light grey)
- **Cutoff Line**: `#aaaaaa` (Grey, dashed, `linestyle='--'`, `linewidth=1.0`)
- **Residual Shading**:
  - $1\sigma$: `#27ae60` (or grey) with `alpha=0.15`
  - $2\sigma$: `#27ae60` (or grey) with `alpha=0.07`

## 4. Axis Labels & KaTeX / LaTeX Syntax
Always use raw strings `r"..."` for LaTeX labels:
- Radial scale: $R \; [h^{-1}\mathrm{Mpc}]$ $\rightarrow$ `r"$R \quad [h^{-1}\mathrm{Mpc}]$"`
- Differential number density: $\mathrm{d}n/\mathrm{d}\ln R \; [h^3\mathrm{Mpc}^{-3}]$ $\rightarrow$ `r"$\mathrm{d}n / \mathrm{d}\ln R \quad [h^3\,\mathrm{Mpc}^{-3}]$"`
- Standardized pull: `r"$(d - m) / \sigma_{\mathrm{JK}}$"`
- Cosmological parameters: `r"$\Omega_m$"`, `r"$\sigma_8$"`, `r"$b_{\mathrm{void}}$"`, `r"$\delta_v$"`
