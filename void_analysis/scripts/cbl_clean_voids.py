#!/usr/bin/env python3
"""
cbl_clean_voids.py

Replace our hand-rolled Phase-1 cleaning with CBL's native
`Catalogue::clean_void_catalogue`. Closes the chain of trust to a fully
CBL-native end-to-end pipeline: raw BitVF → CBL cleaning → CBL theory →
sampler (verified emcee ≡ CBL Sampler) → cosmological inference.

Parameter choice mirrors the Contarini et al. (2023) recipe encoded in
Simone's filename convention `SM0p00_CS0p79_TH-0p80_OV00`:

  - threshold (CBL) = 1 + δ_v_NL = 1 + (-0.80) = 0.20        (TH-0p80)
  - rescale = True                                            (Eulerian rescaling)
  - ratio   = (parameter sweep)                               (CS0p79 → 0.79 ?)
  - overlap = True, criterion = CentralDensity                (OV00 ≡ strict)
  - initial_radius = False                                    (no input radius cut)
  - no smoothing post-process                                 (SM0p00)

CLI:
  python scripts/cbl_clean_voids.py --frac 5p
  python scripts/cbl_clean_voids.py --frac 5p --ratio 0.79 --tag _r0p79
  python scripts/cbl_clean_voids.py --frac 5p --ratio 0.79 --ol-crit contrast
  python scripts/cbl_clean_voids.py --all
"""
import os, sys, time, argparse, tempfile
import numpy as np
import pandas as pd

CBL_HOME = os.environ.get('CBL_HOME', '/Users/arun/projects/CosmoBolognaLib')
sys.path.insert(0, os.path.join(CBL_HOME, 'Python'))
os.environ.setdefault('DYLD_LIBRARY_PATH', CBL_HOME)
import CosmoBolognaLib as cbl
cat   = cbl.catalogue
chain = cbl.chainmesh

HERE      = os.path.dirname(os.path.abspath(__file__))
RAW       = os.path.join(HERE, '..', 'data', 'raw')
SUBS      = os.path.join(HERE, '..', 'data', 'subsamples')
DEFAULT_OUT = os.path.join(HERE, '..', 'data', 'cleaned_cbl')

BOX = 250.0   # Aletheia comoving box side, h^-1 Mpc

# Fixed cleaning conventions (not swept; set to Simone-encoded values)
INITIAL_R   = False
DELTA_R     = [-1.0, 1000.0]   # no input radius cut
DO_RESCALE  = True
DO_OVERLAP  = True
DC_INNER_RATIO = 0.1            # inner-sphere ratio for compute_densityContrast

RAW_FILES = {
    'NREC50': 'voids_recon_50_bitvf_aletheia_{frac}',
    'BIT':    '0_voids_bitvf_aletheia_{frac}.txt',
}
SUB_FILE = 'DM_particles_ss{frac}_250cube.dat'


def load_raw_voids(frac, field):
    """Read raw BitVF void list. Returns DataFrame (positions in the original
    [375, 625] frame — CBL is offset-agnostic so long as voids and tracers
    share a coordinate system)."""
    path = os.path.join(RAW, RAW_FILES[field].format(frac=frac))
    cols = ['x', 'y', 'z', 'radius', 'r_ext', 'delta_2MPS', 'delta_avg']
    df = pd.read_csv(path, sep=r'\s+', comment='#', names=cols)
    df = df[df['radius'] > 0].reset_index(drop=True)
    return df


def load_tracers(frac):
    path = os.path.join(SUBS, SUB_FILE.format(frac=frac))
    return np.loadtxt(path, usecols=(0, 1, 2))


def write_temp_void_file(df, tmpdir, radius_col='r_ext'):
    """Write voids as X Y Z R, using radius_col as R."""
    path = os.path.join(tmpdir, '_voids_for_cbl.dat')
    out = df[['x', 'y', 'z', radius_col]].copy()
    out.to_csv(path, sep=' ', index=False, header=False, float_format='%.6f')
    return path


def write_temp_tracer_file(parts, tmpdir):
    path = os.path.join(tmpdir, '_tracers_for_cbl.dat')
    np.savetxt(path, parts, fmt='%.6f')
    return path


def clean_one(frac, field='NREC50', *,
              ratio=1.5,
              threshold=0.20,
              ol_crit='central',
              radius_col='r_ext',
              out_dir=None,
              tag='',
              verbose=True):
    """Run CBL cleaning on one (field, frac) subsample with explicit parameters.

    Parameters
    ----------
    ratio : float
        Outer-sphere ratio (units of R_v) for central-density evaluation
        during clean_void_catalogue. Simone's `CS0p79` likely → 0.79.
    threshold : float
        Density threshold = 1 + δ_v_NL. Simone's `TH-0p80` → 0.20.
    ol_crit : {'central', 'contrast'}
        Overlap criterion. 'central' → CentralDensity (matches OV00),
        'contrast' → DensityContrast.
    radius_col : {'r_ext', 'radius'}
        Which BitVF radius to pass to CBL as the initial Radius.
        r_ext is the physical radius; radius is the watershed grid radius.
    tag : str
        Suffix on the output filename (e.g. '_r0p79').
    """
    out_dir = out_dir or DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[CBL CLEAN] {field} {frac}")
    print(f"  params: ratio={ratio}  threshold={threshold:.4f}  "
          f"ol_crit={ol_crit}  radius_col={radius_col}  tag={tag!r}")
    print(f"{'='*60}")
    t0 = time.time()

    df    = load_raw_voids(frac, field)
    parts = load_tracers(frac)
    n_voids_raw = len(df)
    n_parts     = len(parts)
    print(f"  Raw voids:    {n_voids_raw}")
    print(f"  DM tracers:   {n_parts}")
    print(f"  Particle box: [{parts[:,0].min():.1f}, {parts[:,0].max():.1f}] × ...")
    print(f"  Void box:     [{df['x'].min():.1f}, {df['x'].max():.1f}] × ...")

    # Build via file-based constructor (robust)
    tmpdir      = tempfile.mkdtemp(prefix='cbl_cleaning_')
    void_file   = write_temp_void_file(df, tmpdir, radius_col=radius_col)
    tracer_file = write_temp_tracer_file(parts, tmpdir)
    if verbose:
        print(f"  Temp dir:     {tmpdir}")

    # Void Catalogue (X Y Z R)
    void_attr = cat.VarCast(
        [cat.Var__X_, cat.Var__Y_, cat.Var__Z_, cat.Var__Radius_])
    vdcat = cat.Catalogue(
        cat.ObjectType__Void_, cat.CoordinateType__comoving_,
        void_attr, [1, 2, 3, 4], [void_file],
        0, 1.1, 1.0,
    )
    if verbose:
        print(f"  Void Catalogue:   {vdcat.nObjects()} objects "
              f"(R range {vdcat.Min(cat.Var__Radius_):.2f}–"
              f"{vdcat.Max(cat.Var__Radius_):.2f})")

    # Tracer Catalogue (X Y Z)
    tr_attr = cat.VarCast([cat.Var__X_, cat.Var__Y_, cat.Var__Z_])
    trcat = cat.Catalogue(
        cat.ObjectType__Halo_, cat.CoordinateType__comoving_,
        tr_attr, [1, 2, 3], [tracer_file],
        0, 1.1, 1.0,
    )
    if verbose:
        print(f"  Tracer Catalogue: {trcat.nObjects()} particles "
              f"(MPS = {trcat.mps():.3f})")

    # ChainMesh3D for spatial lookups
    ChM = chain.ChainMesh3D(
        2.0 * trcat.mps(),
        trcat.var(cat.Var__X_), trcat.var(cat.Var__Y_), trcat.var(cat.Var__Z_),
        vdcat.Max(cat.Var__Radius_),
    )

    ol_crit_var = (cat.Var__CentralDensity_ if ol_crit == 'central'
                   else cat.Var__DensityContrast_)

    # Pre-compute density attributes
    print(f"  Computing centralDensity (threshold={threshold:.4f}) and "
          f"densityContrast (inner ratio={DC_INNER_RATIO}) ...")
    t1 = time.time()
    vdcat.compute_centralDensity(trcat, ChM, threshold)
    vdcat.compute_densityContrast(trcat, ChM, DC_INNER_RATIO)
    print(f"    done in {time.time() - t1:.1f}s")

    # Clean
    t2 = time.time()
    vdcat.clean_void_catalogue(
        INITIAL_R, DELTA_R, threshold, DO_RESCALE,
        trcat, ChM, ratio, DO_OVERLAP, ol_crit_var)
    print(f"  clean_void_catalogue: {time.time() - t2:.1f}s")

    n_clean = vdcat.nObjects()
    print(f"  Retained: {n_clean} voids ({100*n_clean/n_voids_raw:.1f}%)")

    out_df = pd.DataFrame({
        'x':                np.array(vdcat.var(cat.Var__X_)),
        'y':                np.array(vdcat.var(cat.Var__Y_)),
        'z':                np.array(vdcat.var(cat.Var__Z_)),
        'R':                np.array(vdcat.var(cat.Var__Radius_)),
        'central_density':  np.array(vdcat.var(cat.Var__CentralDensity_)),
        'density_contrast': np.array(vdcat.var(cat.Var__DensityContrast_)),
    })

    out_path = os.path.join(
        out_dir, f'cleaned_cbl_{field.lower()}_cat_{frac}{tag}.csv')
    out_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    r_arr = out_df['R'].values
    print(f"  R range:  [{r_arr.min():.2f}, {r_arr.max():.2f}]  "
          f"median {np.median(r_arr):.2f}  N(R≥7) = {int((r_arr >= 7).sum())}")
    print(f"  Total wall time: {time.time() - t0:.1f}s")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frac', default='5p',
                    choices=['20p', '10p', '5p', '2_5p'],
                    help='Single subsample to clean (ignored if --all).')
    ap.add_argument('--field', default='NREC50', choices=['NREC50', 'BIT'])
    ap.add_argument('--all', action='store_true',
                    help='Clean all four NREC50 subsamples.')
    ap.add_argument('--ratio', type=float, default=1.5,
                    help='Outer-sphere ratio (units of R_v) for '
                         'clean_void_catalogue. Simone\'s `CS0p79` likely → 0.79.')
    ap.add_argument('--threshold', type=float, default=0.20,
                    help='CBL density threshold = 1 + δ_v_NL. '
                         'Simone\'s `TH-0p80` → 0.20.')
    ap.add_argument('--ol-crit', default='central',
                    choices=['central', 'contrast'],
                    help='Overlap criterion. CentralDensity matches OV00.')
    ap.add_argument('--radius-col', default='r_ext',
                    choices=['r_ext', 'radius'],
                    help='Which BitVF radius column to pass as initial R.')
    ap.add_argument('--out-dir', default=None,
                    help='Output dir (default: data/cleaned_cbl).')
    ap.add_argument('--tag', default='',
                    help='Suffix on output filename. E.g. "_r0p79".')
    args = ap.parse_args()

    kw = dict(ratio=args.ratio, threshold=args.threshold,
              ol_crit=args.ol_crit, radius_col=args.radius_col,
              out_dir=args.out_dir, tag=args.tag)

    if args.all:
        for f in ['20p', '10p', '5p', '2_5p']:
            try:
                clean_one(f, field=args.field, **kw)
            except Exception as exc:
                print(f"\n[ERROR] {args.field} {f}: {type(exc).__name__}: {exc}")
    else:
        clean_one(args.frac, field=args.field, **kw)


if __name__ == '__main__':
    main()
