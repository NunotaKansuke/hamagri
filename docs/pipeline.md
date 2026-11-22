# Event pipeline

`hamagri.pipeline` takes one event from photometry to binary-lens solutions:

1. **Scan.** Every atlas map is scanned over the alpha FFT at 16 point-lens
   geometries around a PSPL fit (`geometry_stencil`). Nothing is ranked or
   discarded: the whole `(map, geometry)` table is kept (`GeometryProfiles`).
2. **Hills and regions.** Strict downhill basins of the `(log s, log q, log rho)`
   grid give the hills; around each one, a tempered weighted covariance of
   the neighbouring rows gives the search region (`Landscape`).
3. **Battle royale.** A population of particles descends with damped
   Gauss-Newton steps on direct VBMicrolensing light curves. Particles in
   the same valley merge, while close/wide and other light-curve degeneracies
   are kept. Each round culls the population and refills it with random
   jumps (`Battle`, `Settings`).
4. **Polish.** LM polishes the finalists (up to 100 evaluations each). The
   best solutions are then profiled in rho (x0.7, x0.35, x1.5) and in u0
   (+-rho), and each profile point is re-polished.
5. **Free parallax.** With a parallax set-up, steps 1-4 hold piE fixed at
   the value you supply and the last step frees it
   (see [Parallax](#parallax-pie)).

Solutions within `solution_dchi2` (25) of the best are reported as distinct.
Two fits count as the same solution when `|d ln s| < 0.1`, `|d ln q| < 0.7`,
`|d alpha| < 0.2` and u0 has the same sign, with `q > 1` mapped to
`(1/q, alpha + pi)` first.

## Roman

The pipeline was developed and tuned on simulated light curves from the
Roman Galactic Bulge Time Domain Survey (15-min cadence, W146 photometry,
events with tE of days to months), searched over the Roman-extended atlas
([Atlas data](atlas-data.md#roman-additions)). The `Settings` defaults, the
16-point stencil and the m_max = 128 scan are those choices. Other surveys
should work with the same code, but the search budget (`wall_budget`,
`population`, `hills`) may need adjusting for very different cadence or
noise.

## Parallax (piE)

Roman observes from a halo orbit around L2, and annual parallax is often
measurable over its 72-day seasons. The pipeline handles piE in two stages:

- **Fixed piE during the scan and the search.** Supply one value
  `piE = (piEN, piEE)`, for example from a host-star estimate (lens
  brightness and the source/lens proper motion) or from a point-lens fit
  with parallax. Use `(0, 0)` if nothing is known. At fixed piE, parallax is
  a per-epoch shift `(dtau, dbeta)` of the source trajectory that does not
  depend on the binary parameters. It is applied to the data before the FFT
  scan, so the alpha FFT remains exact, and the battle and the polish use the
  same fixed value. Because parallax breaks the (u0, alpha) -> (-u0, -alpha)
  symmetry, each hill is searched together with its mirror image.
- **Free piE at the end.** Every distinct polished and profiled fit is
  refit with piEN and piEE free (up to 400 evaluations). The fixed-piE fit is
  kept wherever it is still better, so a poor starting piE cannot make the
  result worse. Fits with opposite u0 signs (the ecliptic degeneracy) are
  reported as separate solutions; fits that differ only in piE are not.

For Roman, use a satellite table for the observer:
`Parallax(ra_deg, dec_deg, t_ref, satellite=1, satellite_dir=...)` with
`satellite1.txt` in VBMicrolensing's format (a JPL Horizons ephemeris of
Roman). `t_ref` is VBMicrolensing's `t0_par`, held fixed, usually near the
peak.

## Python

```python
from hamagri.pipeline import Parallax, solve_event
from hamagri import Dataset

datasets = [Dataset(t, flux, error, "W146")]
result = solve_event(
    "hamagri-atlas", datasets, pspl=(t0, u0, tE),
    parallax=Parallax(ra_deg, dec_deg, t_ref),   # optional
    piE=(piEN, piEE),                            # held fixed until the last step
    workers=8, seed=0, profiles="scan.npz",      # reuse or save the FFT scan
)
print(result["best"]["chi2"], result["best"]["parameters"])
for solution in result["solutions"]:
    print(solution["label"], solution["chi2"], solution["parameters"])
```

Parameters are reported in VBMicrolensing's frame: `t0` and `u0` refer to the
centre of mass, `alpha` follows the VBM convention, and fluxes are fitted
linearly per dataset. `result` also records the finalists, the profiles, the
fixed-piE solutions, the search log and the timings, and it is JSON-serialisable.

Each worker is a separate process with its own VBMicrolensing instance.
A job that exceeds its timeout is killed and the worker restarted; the
count is in `result["killed_jobs"]`. `Settings` holds every search constant.
The defaults are those used for Roman-like simulated light curves
(about 7 min of search and 8 workers per event).

## Command line

```bash
hamagri solve hamagri-atlas event.json solutions.json --workers 8 --profiles scan.npz
```

`event.json`:

```json
{
  "pspl": [t0, u0, tE],
  "datasets": [{"path": "w146.txt", "name": "W146"}],
  "parallax": {"ra_deg": 268.0, "dec_deg": -29.0, "t_ref": 10500.0},
  "piE": [0.0, 0.0]
}
```

Dataset files hold the columns `time flux error`; set `"columns"` to choose
other columns. Relative paths, including `parallax.satellite_dir`, are
resolved from the event file's directory. `parallax` and `piE` are optional.
For a space observer, set `satellite` to VBMicrolensing's satellite index and
`satellite_dir` to the directory that holds `satellite<N>.txt`.

## Seeds and repeatability

The search is stochastic: the random draws in the seed and refill steps
depend on `seed`. On hard events, two seeds can end in different local
minima, so compare A/B changes over several seeds rather than over one run.

## Growing an atlas

A search that ends at the edge of the atlas, or between its grid points,
can be rerun on an extended atlas without rebuilding the existing maps:

```bash
hamagri add-maps build-config.json hamagri-atlas rows.txt additions
hamagri combine-maps hamagri-atlas-v2 hamagri-atlas additions
```

`rows.txt` lists `map_id log10_s log10_q log10_rho` per line, with ids that
are not yet in the atlas. A row inside the atlas's `(log s, log q)` hull joins
the nearest existing bucket and reuses that bucket's radial nodes. A row
outside it gets a new bucket, with its own pilot layout. `add-maps` accepts
`--part-index/--part-count`, so the additions can be built as a scheduler
array and joined with `merge-maps`. `combine-maps` hard-links the shards of
its sources into a new collection and leaves both sources unchanged. The
Python equivalents are `plan_additions`, `build_additions` and `combine_maps`.
