"""Rotating disc and anomalous gas of deblended HI components.

The steps are run by ``anomalous_gas_extractor.anomalous_gas_extractor``. The modules
follow the order of the analysis:

components      mask of the components from the output of SoFiA-2 and of the deblending
datacube        read the data cube and the mask of the components, moment maps
kinemetry_rings harmonic expansion of a velocity field along ellipses (Kinemetry)
rmax            outer radius of the regularly rotating disc (r_max)
centre          kinematic centre of the rotating disc
coverage_test   minimum fraction of a ring that has to be sampled
rotation        does a component host a rotating disc at all?
fat             tilted-ring modelling of the rotating discs with pyFAT
hi_mass         HI mass inside and outside the rotating disc
pv              position-velocity diagrams along a line on the sky
residuals       moment maps of data minus model, made with one mask
pipeline        the steps components, rmax, pyfat and hi_mass
configuration   the settings of all steps and the file with their defaults

figures         the step figures: all figures, one directory per kind
figure_data     what the figures need of every component
sky_axes        maps on the sky in the style of the figures
rmax_figures    velocity fields with r_max, Kinemetry profiles
plots           radial profiles of the tilted-ring models
disc_figures    column density and velocity field of rotating disc and anomalous gas
slices, pv_figures  position-velocity slices and diagrams
model_figures   model compared with the data
overview        map of all components
"""
