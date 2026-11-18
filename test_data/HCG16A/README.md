# Test data: HCG 16A

The neutral hydrogen of the spiral galaxy HCG 16A, a member of the compact
group HCG 16, observed with MeerKAT (Ianjamasimanana et al. 2025, A&A 696,
A176) at a resolution of 16.8 x 12.9 arcsec.

| File | Content |
|---|---|
| `HCG16A_cube.fits` | data cube, 83 x 83 pixels of 3 arcsec, 105 channels of 5.7 km/s, Jy/beam. The emission of the other members of the group is replaced by Gaussian noise with the rms of the cube (0.27 mJy/beam) |
| `HCG16A_galaxies.csv` | name and optical position of the galaxy |
| `sofia_HCG16A.par` | parameter file of SoFiA-2 |
| `HCG16A.yml` | configuration file |

## Running the test

SoFiA-2, TiRiFiC, pyFAT and Kinemetry have to be installed. Run in a copy of
this directory, because the output is written next to the input and AGE
rewrites the parameter file of SoFiA-2:

```
cp -r test_data/HCG16A /path/to/work/HCG16A
cd /path/to/work/HCG16A
python -m anomalous_gas_extractor.anomalous_gas_extractor configuration_file=HCG16A.yml \
    kinematics.kinemetry_path=/path/to/the/directory/with/kinemetry.py
```

Programs that are not on the search path are given with their directory, in
`HCG16A.yml` or on the command line: `internal.sofia=/path/to/sofia`,
`internal.tirific=/path/to/tirific`, `internal.pyfat=/path/to/pyFAT`.

The cube holds one galaxy, so the step `deblend` is left out. The steps
`sofia`, `components` and `rmax` take less than a minute, the tilted-ring
model of pyFAT some tens of minutes.

## Expected result

The main results are in `component_summary.csv`; the figures are in `figures/`.
With SoFiA-2 2.6, pyFAT 1.0.1 and a distance of 49 Mpc:

| Quantity | Value |
|---|---|
| rotating disc | yes: resolved and dominated by rotation (v/sigma = 3.6) |
| `r_max` (Kinemetry) | 38.7 arcsec, 4.6 beams along the major axis |
| tilted-ring model (pyFAT) | converged; radius 37.0 arcsec, inclination 53 deg, position angle 10 deg |
| HI mass | 1.01e9 solar masses |
| in the rotating disc | 8.2e8 solar masses (81 per cent) |
| anomalous gas | 1.9e8 solar masses (19 per cent) |

The disc spans more than three beams, so it is modelled with pyFAT. With
`pyfat.minimum_beams_across=5` it is not: the disc is then the ellipse of
Kinemetry with semi-major axis `r_max` and holds 86 per cent of the HI.
