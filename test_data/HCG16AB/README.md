# Test data: HCG 16A and HCG 16B

The neutral hydrogen of the spiral galaxies HCG 16A and HCG 16B, two members of
the compact group HCG 16, observed with MeerKAT (Ianjamasimanana et al. 2025,
A&A 696, A176) at a resolution of 16.8 x 12.9 arcsec. The HI of the two
galaxies is connected, so SoFiA-2 finds them as one detection. This is a fast
test of all steps, including the deblending.

| File | Content |
|---|---|
| `HCG16AB_cube.fits` | data cube, 80 x 87 pixels of 3 arcsec, 138 channels of 5.7 km/s, Jy/beam. The emission of the other members of the group is replaced by Gaussian noise with the rms of the cube (0.27 mJy/beam) |
| `HCG16AB_optical.fits` | r-band image of the DESI Legacy Imaging Surveys (DECaLS), pixels of 1.25 arcsec, on which the deblending finds the two galaxies |
| `HCG16AB_galaxies.csv` | names and optical positions of the two galaxies |
| `sofia_HCG16AB.par` | parameter file of SoFiA-2 |
| `HCG16AB.yml` | configuration file |

## Running the test

SoFiA-2 and Kinemetry have to be installed. No internet access is needed: the
deblending downloads no catalogue (`general.internet_repository: ['None']`).
Run in a copy of this directory, because the output is written next to the
input and AGE rewrites the parameter file of SoFiA-2:

```
cp -r test_data/HCG16AB /path/to/work/HCG16AB
cd /path/to/work/HCG16AB
AGE configuration_file=HCG16AB.yml \
    kinematics.kinemetry_path=/path/to/the/directory/with/kinemetry.py
```

Programs that are not on the search path are given with their directory, in
`HCG16AB.yml` or on the command line, e.g. `internal.sofia=/path/to/sofia`.

The test takes one to two minutes. The tilted-ring models take most of the time
of a full run and are left out: the step `pyfat` is not in `input.steps_to_run`,
so the disc of each galaxy is the ellipse of Kinemetry with semi-major axis
`r_max`. To include the models, give
`input.steps_to_run=[sofia,deblend,components,rmax,pyfat,hi_mass,figures]`;
TiRiFiC and pyFAT then have to be installed too.

The same test is part of the test suite:

```
KINEMETRY_PATH=/path/to/the/directory/with/kinemetry.py pytest tests/test_end_to_end.py
```

## Expected result

The main results are in `component_summary.csv`; the figures are in `figures/`.
With SoFiA-2 2.6, deblend_sofia_detections 1.1.1 and a distance of 49 Mpc:

| Quantity | HCG 16A | HCG 16B |
|---|---|---|
| voxels of the component | 18762 | 17016 |
| rotating disc | yes (v/sigma = 3.1) | yes (v/sigma = 1.5) |
| `r_max` (Kinemetry) | 34.1 arcsec | 26.4 arcsec |
| HI mass | 1.02e9 solar masses | 5.97e8 solar masses |
| in the rotating disc | 8.1e8 solar masses (80 per cent) | 2.8e8 solar masses (47 per cent) |
| anomalous gas | 2.0e8 solar masses (20 per cent) | 3.1e8 solar masses (53 per cent) |
