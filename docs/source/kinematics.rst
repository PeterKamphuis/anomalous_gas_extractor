Rotating discs and anomalous gas
================================

``anomalous_gas_extractor/anomalous_gas_extractor.py`` separates the gas of every
galaxy in a data cube into a regularly rotating disc and anomalous gas. It is
run with

.. code-block:: text

   python -m anomalous_gas_extractor.anomalous_gas_extractor configuration_file=run.yml

After the source finding (step ``sofia``, run by AGE) and the deblending (step
``deblend``) every component is tested for a rotating disc, the discs are
modelled and the HI masses of the discs and of the anomalous gas are measured.
The steps are run in this order; each one writes its results to disk and the
next one reads them, so they can also be run one at a time:

=============  ==============================================================
``sofia``      source finding with SoFiA-2
``deblend``    deblending of the detections with deblend_sofia_detections
``components`` one mask with a label for every component
``rmax``       kinematic centre and outer radius of the regular rotation
               (``r_max``) of every component with Kinemetry; which components
               host a rotating disc
``pyfat``      tilted-ring models with pyFAT of the rotating discs that span
               at least three beams
``hi_mass``    HI mass inside and outside the rotating discs
``figures``    figures of all steps
=============  ==============================================================

SoFiA-2, deblend_sofia_detections, Kinemetry and pyFAT are used as they are
installed; the program only writes their input and reads their output. All
settings, the input data and the limits of the tests are given in one
configuration file:

.. code-block:: yaml

   cube_name: cube.fits                      # the cube that is searched and deblended
   input:
     steps_to_run: [sofia, deblend, components, rmax, pyfat, hi_mass, figures]
     sofia_parameter_file: sofia.par
     known_source_table: galaxies.csv        # name, RA, Dec (degrees), ...
   deblend:
     options:                                # any setting of deblend_sofia_detections
       input:
         manual_optical_image: [optical.fits]
         use_peak_deblending: true
   components:
     only_named: true
   kinematics:
     kinemetry_path: /path/to/the/directory/with/kinemetry.py
     ring_tests: [coverage, k5k1, vsys]
     k5k1_max: 0.10
   pyfat:
     minimum_beams_across: 3.0               # smaller rotating discs are not modelled
     flat_disc: true
   hi_mass:
     distance_mpc: 49.0
   figures:
     font_file: /path/to/a/font.otf

.. code-block:: text

   python -m anomalous_gas_extractor.anomalous_gas_extractor print_examples=true

writes the file ``AGE_defaults.yml`` in the current directory, with all
settings, their defaults and a line of explanation above every setting. Change
what is needed and give the file as ``configuration_file``. Every setting can
also be given on the command line, e.g. ``hi_mass.distance_mpc=49``.

The settings are those of AGE (``input``, ``directories``, ``internal``, ...)
and one section for every step that is added: ``deblend``, ``components``,
``kinematics``, ``pyfat``, ``hi_mass`` and ``figures``. The modules of these
steps are in ``anomalous_gas_extractor/kinematics``.

Test data
---------

``test_data/HCG16A`` holds a small cube with the HI of the galaxy HCG 16A, the
table with its position, a parameter file for SoFiA-2 and the configuration
file ``HCG16A.yml``; see the ``README.md`` in that directory.

Output
------

Everything is written below the run directory (``directories.run_directory``):

.. code-block:: text

   <output directory of SoFiA-2>/      catalogue, mask and cubelets of SoFiA-2
       Watershed_Output/               output of the deblending
   components/                         mask of the components
   kinematics/                         r_max tables
   pyfat/                              input, output and tables of the tilted-ring models
   hi_mass/                            HI mass table
   component_summary.csv               the main results of all steps, one line per component
   figures/                            all figures, one directory per kind (see step figures)
   Logs/                               log files

Step deblend
------------

The step starts deblend_sofia_detections on the output of SoFiA-2. The settings
under ``deblend.options`` are passed on to it unchanged, in the layout of its own
configuration file; this is the place for the markers, the optical image, the
region in which counterparts are searched, and so on. The names of these
settings differ between versions of deblend_sofia_detections: a setting that
the installed version does not have is refused with the list of the settings it
does have. The program itself only sets the SoFiA-2 parameter file, the run
directory, the log file and the table of known sources
(``input.known_source_table``).

Step components
---------------

SoFiA-2 writes a mask in which every detection has its own number, and the
deblending writes the components of a detection as ``final_mask.fits`` on the
grid of the cubelet of that detection. The step combines them into one mask on
the grid of the cube, ``components/component_labels.fits``:

* a detection that was deblended contributes the components of its final mask,
  any other detection is one component. ``components.sources`` limits the mask
  to the listed detections (their numbers in the SoFiA-2 mask);
* a component is named after the object of ``input.known_source_table`` that it
  holds: the object belongs to the component with the largest flux on the line
  of sight through its position. The labels follow the order of the table; the
  other components are named ``source<number>``. With ``only_named: true`` only
  the components with a known object are kept;
* the names are stored in the header of the mask (``LABEL<n>``) and, with the
  number of voxels of every component, in ``components/components.csv``.

There are two ways of working, chosen in the configuration file:

* **Deblend the cube on which the kinematics are measured.** Leave
  ``kinematics.cube`` unset. The mask is on the grid of the deblended cube and
  the following steps use that cube.
* **Deblend one cube and measure the kinematics on another** of the same field,
  for example deblend at low resolution, where the extended gas is detected,
  and model the discs at high resolution. Give the second cube as
  ``kinematics.cube``. The labels are carried to it: every voxel takes the
  label of the voxel of the deblended cube in which it falls (nearest pixel on
  the sky, nearest channel in velocity).

  The second cube is cut to the part of the sky with components plus
  ``cutout_margin_beams`` beams, or to the part of the sky covered by the FITS
  file ``cutout_region`` if one is given, and written to
  ``components/kinematics_cube.fits`` (``cutout_margin_beams: null`` without a
  region keeps the cube as it is). Because the mask of the deblended cube can
  be wider than the emission of the second cube, a label is kept only where
  that cube has emission if a mask of its emission is given
  (``emission_mask``) or made with SoFiA-2 (``emission_sofia_parameters``, a
  SoFiA-2 parameter file; the program sets its input and output).

The following steps use this mask and cube. A mask made in another way can be
given with ``kinematics.mask`` (and its cube with ``kinematics.cube``); the step
is then not needed.

Requirements
------------

* Kinemetry (Krajnovic et al. 2006), the Python version distributed by its
  author, and its dependency ``plotbin``. It is not on PyPI; give its directory
  with ``kinematics.kinemetry_path`` unless it is installed in the environment.
* pyFAT 1.0.1 or later for the catalogue columns ``r_max``, ``xpos_min/max``
  and ``ypos_min/max``, with TiRiFiC and SoFiA-2. Until 1.0.1 is released on
  PyPI it is installed from the Beta branch of the pyFAT repository. The
  commands are set with ``internal.pyfat``, ``internal.tirific`` and
  ``internal.sofia``.

Step rmax
---------

For every component the step

#. makes the velocity field: the intensity-weighted mean velocity of the voxels
   of the component, blanked where the integrated intensity is below
   ``velocity_field_snr`` times the noise;
#. runs Kinemetry on ellipses with free position angle and flattening around a
   first guess of the centre (``centre_table``, else the centroid of the
   brightest emission) to find a first estimate of the disc;
#. fits the kinematic centre: Kinemetry with a free centre on the rings of that
   disc, the centre is the median of the ring centres (``fit_centre``);
#. repeats the ring fits around the kinematic centre and tests every ring with
   the tests listed in ``ring_tests`` (default ``[coverage, k5k1, vsys]``):

   * ``coverage``: the fraction of the ellipse on the velocity field is at least
     the minimum coverage,
   * ``k5k1``: ``k5/k1 <= k5k1_max``, the fifth-order term is small compared with
     the rotation,
   * ``vsys``: ``|a0 - v_sys| <= max(vsys_drift_kms, vsys_drift_fraction * k1)``,
     the mean velocity of the ring stays at the systemic velocity of the
     innermost rings,
   * ``amplitude``: the fit is constrained by the data. a0 lies within the
     velocities measured on the ring, and the fitted velocity curve deviates
     from a0 by at most ``amplitude_ratio_max`` times the largest measured
     deviation. Where part of an ellipse has no data the fit is free there; a
     fit that the data do not constrain runs far beyond the measured velocities;

   ``coverage`` and ``amplitude`` both reject rings that are not constrained by
   the data, the first from the number of sampled points and the second from
   the result of the fit. With ``ring_tests: [amplitude, k5k1]`` a ring is
   judged on its fit alone;
#. if ``coverage`` is one of the ring tests, measures the minimum coverage on the
   data (``measure_coverage``): wedges of the velocity field are blanked in the
   completely sampled regular rings and the fits repeated; the minimum coverage
   is the lowest coverage at which the gap alone changes the verdict of the
   tests on k5/k1 and on the systemic velocity in at most 5 per cent of the
   trials. Components without a completely sampled regular ring use
   ``default_coverage``;
#. takes as rotating disc the first uninterrupted run of rings that pass all
   selected tests and starts within ``start_within_beams`` of the centre.
   ``r_max`` is the radius where the first test is crossed, interpolated between
   the last ring of the run and the first failing ring. The amplitude ratio is
   not interpolated: if ``amplitude`` is the only test that fails, ``r_max`` is
   the radius of the last ring of the run. ``r_max`` is a lower limit if the run
   ends only because the data end (no ring left, or only ``coverage`` or
   ``amplitude`` fail);
#. decides whether the run is a rotating disc. A smooth velocity gradient, for
   example along a tidal stream, also passes the ring tests. The run is accepted
   only if its radial extent is at least ``minimum_disc_beams`` beams, if k1
   does not fall from its first to its last ring and if the largest k1 exceeds
   ``minimum_v_over_sigma`` times the velocity dispersion inside ``r_max``
   (second moment, corrected for beam smearing). In short: the disc has to be
   resolved and dominated by rotation. The column ``failed_rotation_tests``
   lists what a component without rotating disc fails.

The table also gives ``disc_diameter_beams``, the extent of the disc along its
major axis in beams: ``2 r_max`` divided by the major axis of the beam.

Output in ``kinematics/``: ``rmax_table.csv`` (one line per component),
``rmax_results.json`` (the same with all rings and the coverage test) and
``rings_<component>.csv``.

Step pyfat
----------

A tilted-ring model needs a resolved disc. A rotating disc is modelled only if
it spans at least ``pyfat.minimum_beams_across`` beams along its major axis
(``disc_diameter_beams``). The default is 3, which is also the smallest value
that is accepted. A smaller rotating disc is not given to pyFAT: its disc is
the ellipse of Kinemetry with semi-major axis ``r_max`` (see the step
``hi_mass``). Components without a rotating disc are not modelled either.

Each disc that is modelled is written to a
cube of its own in which the other components are replaced by noise, and so is
its own gas beyond ``r_max + cut_beyond_rmax_beams`` beams (set this to ``null``
to keep all gas of the component). The catalogue of pyFAT carries ``r_max``,
beyond which pyFAT adds no rings, and limits that keep the centre within
``centre_tolerance_beams`` of the kinematic centre. By default a flat disc is
fitted (``flat_disc``). Any other setting of pyFAT can be given under
``pyfat.overrides`` in the layout of the pyFAT configuration file.

The radius of the disc is the radius at which the surface brightness of the
final model reaches zero on both sides: the first ring without emission beyond
the last ring with emission (TiRiFiC interpolates linearly between the rings),
or the last ring of the model if it still has emission. Because the rings lie
on a fixed grid this radius can differ slightly from ``r_max``.

Output in ``pyfat/``:

* the input and output of pyFAT in one directory per component,
* ``pyfat_table.csv`` and ``pyfat_results.json``: centre, systemic velocity,
  inclination, position angle, largest rotation velocity and radius of every disc,
* ``model_rings_<component>.csv``: the rings of the final model with their
  uncertainties; surface brightness (Jy km/s arcsec^-2), rotation velocity,
  inclination, position angle and dispersion for the approaching and the
  receding side.

pyFAT can take hours. With ``pyfat.reuse_fits: true`` it is not run again if its
results already list every disc: the existing models are read and the tables
are written again.

Step hi_mass
------------

The rotating disc of a component is the ellipse on the sky with the centre,
radius, inclination and position angle of the pyFAT model. The rings of the
model start at radius zero and the model ends at its outermost ring, so the
ellipse is used as it is; ``beam_margin`` (default 0) widens it by that
fraction of the beam if the emission smeared beyond the edge of the disc should
be counted as well. The gas of the component inside the ellipse is
counted as rotating disc, the rest as anomalous gas; all gas of a component
without rotating disc is anomalous. If a rotating disc has no tilted-ring
model, because it spans fewer than ``pyfat.minimum_beams_across`` beams or
because pyFAT was not run or did not converge, the ellipse is taken from
Kinemetry: the kinematic centre, ``r_max`` as semi-major axis and the mean
position angle and flattening of the disc rings. The column
``disc_radius_source`` says which ellipse was used (``pyFAT`` or
``r_max (Kinemetry)``) and the column ``tilted_ring_model`` why (``pyFAT
model``, ``disc smaller than 3 beams``, ``no usable pyFAT model``, ``pyFAT not
run`` or ``no rotating disc``).

HI masses are ``49.7 D_L^2 S`` solar masses with the flux integrated over
frequency in Jy Hz (Meyer et al. 2017). The distance is ``distance_mpc`` or,
if not set, the Hubble flow distance of each component for
``hubble_constant``. The column ``model_hi_mass_msun`` gives the mass of the
pyFAT model cube for comparison.

Output: ``hi_mass/hi_mass_table.csv``. Besides the fluxes and masses it lists the
ellipse that was used for every disc (``disc_ra_deg``, ``disc_dec_deg``,
``disc_radius_arcsec``, ``disc_inclination_deg``, ``disc_position_angle_deg``
and the margin ``disc_margin_arcsec`` added to both semi-axes).

The step also writes ``component_summary.csv`` in the run directory, one line
per component with the main results of all steps: whether it has a rotating
disc, the HI mass in total, in the disc and in anomalous gas with the two
fractions, the radius of the disc in arcsec and kpc with its centre,
inclination, position angle, systemic and largest rotation velocity, ``r_max``
and the kinematic centre from Kinemetry, and the fluxes.

Step figures
------------

The figures of all steps are written to ``figures/`` (``figures.output_directory``),
one directory for every kind of figure:

.. code-block:: text

   figures/
       rmax/               velocity field of every component with r_max; Kinemetry profiles
       models/             radial profiles of the tilted-ring models, together and one by one
       disc_anomalous/     column density and velocity field of the rotating disc and of the
                           anomalous gas; the velocity fields with the position-velocity slices
       position_velocity/  position-velocity diagrams along the major axis, together and
                           one by one
       model_comparison/   velocity field of the model on that of the data; residual moment
                           maps (data minus model) and the table of their statistics

``figures.products`` selects the kinds; a kind whose input is missing (for
example the model figures before pyFAT has run) is skipped.
``figures.font_file`` and ``figures.format`` set the font and the file format,
``figures.display_names`` the names written in the figures. The
position-velocity slice of a component follows the position angle of its
tilted-ring model and is ``figures.pv_width_beams`` wide; for a component
without model the angle is measured from the velocity gradient of its gas or
taken from ``figures.position_angles``. ``figures.enable: false`` switches the
step off.

Position-velocity diagrams
--------------------------

``anomalous_gas_extractor.kinematics.pv.position_velocity`` extracts a
position-velocity diagram from a cube along a straight line on the sky, given
its centre, position angle, offsets and width. It is not a step of the program;
it is meant for inspecting the discs along the major axis of their models.

Residual moment maps
--------------------

``anomalous_gas_extractor.kinematics.residuals.residual_moments`` compares a
disc with its model. The model cube is placed on the grid of the data cube, the
voxels of the component are selected in both cubes with the same mask, and the
moment maps of the two masked cubes are returned together with their
differences (integrated intensity, mean velocity and velocity dispersion).
``hi_mass.surface_density`` converts an integrated intensity to solar masses
per square parsec.

Tests
-----

``pytest tests`` runs the unit tests on synthetic data. The tests that need
Kinemetry are skipped if it is not installed; its directory can be given in the
environment variable ``KINEMETRY_PATH``.
