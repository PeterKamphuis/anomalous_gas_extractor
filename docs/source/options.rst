Configuration options
=====================

Introduction
------------

``anomalous-gas-extractor`` can use a YAML configuration file to control
source finding, deblending, component creation, kinematics, tilted-ring
modelling, HI masses and figures. The defaults are in
``anomalous_gas_extractor/kinematics/configuration.py``.

Run the program with

.. code-block:: text

   AGE configuration_file=run.yml

Print a complete configuration file containing every setting and its default
with

.. code-block:: text

   AGE print_examples=true

The command-line form of a setting overrides the YAML file, for example
``hi_mass.distance_mpc=49``. The YAML file does not need to contain settings
that should retain their defaults.

Standalone keywords
-------------------

**print_examples**

  *bool, optional, default = False*

  Write ``AGE.yml`` in the current directory, with the defaults and a short
  explanation of every setting, and exit.

**configuration_file**

  *str or null, optional, default = null*

  Name or path of the YAML configuration file. A relative name is interpreted
  from the run directory.

**cube_name**

  *str or null, optional, default = null*

  Data cube to search and deblend when ``input.sofia_parameter_file`` is
  ``Template`` or is not otherwise supplying a cube name.

Input keywords
--------------

*Specified with ``input``.*

**known_source_table**

  *list of str, optional, default = ['None']*

  Comma-separated table with the known galaxies: name, right ascension and
  declination in degrees, followed by any other columns. The objects name the
  components and provide the first guess for their centres.

**sofia_parameter_file**

  *str, optional, default = 'Template'*

  SoFiA-2 parameter file. Set it to ``Template`` to use the default SoFiA-2
  settings, or give the name or path of a parameter file.

**steps_to_run**

  *list of str, optional*

  Steps to run, in execution order. The default is

  .. code-block:: text

     sofia, deblend, components, rmax, pyfat, hi_mass, figures

  The possible steps are ``sofia``, ``deblend``, ``asymmetry``,
  ``components``, ``rmax``, ``pyfat``, ``hi_mass`` and ``figures``. Unknown
  step names are rejected.

**verbose**

  *bool, optional, default = True*

  Print progress and informational messages.

**ncpu**

  *int, optional*

  Number of processes to use. The default is the number of CPUs available to
  the current process, with one CPU reserved when this is the full CPU count.

**multiprocessing**

  *bool, optional, default = True*

  Use multiprocessing while processing sources.

Directory keywords
------------------

*Specified with ``directories``.*

**run_directory**

  *str, optional, default = current working directory*

  Directory below which output is written. The directory must exist.

**data_directory**

  *str, optional, default = ''*

  Directory containing input data. An empty value means that input data is
  searched for in the run directory.

**sofia_run_directory**

  *str, optional, default = ''*

  Directory containing the SoFiA-2 run. It is normally filled automatically.

Logging keywords
----------------

*Specified with ``logging``.*

**enable**

  *bool, optional, default = True*

  Enable logging.

**enable_log**

  *bool, optional, default = True*

  Write messages to log files.

**verbose**

  *bool, optional, default = False*

  Include detailed diagnostic messages.

**log_directory**

  *str, optional, default = 'Logs/<day-month-year>'*

  Directory in which log files are written.

**log_file**

  *str, optional, default = 'Log.txt'*

  Name of the main log file.

**debug_functions**

  *list of str, optional, default = ['NONE']*

  Function names for which debug messages are printed. Use ``ALL`` for every
  function or ``NONE`` to disable function-specific debug messages.

Internal keywords
-----------------

*Specified with ``internal``.*

**sofia**

  *str, optional, default = 'sofia'*

  Name or path of the SoFiA-2 executable.

**tirific**

  *str, optional, default = 'tirific'*

  Name or path of the TiRiFiC executable.

**pyfat**

  *str, optional, default = 'pyFAT'*

  Name or path of the pyFAT executable.

Step deblend
------------

*Specified with ``deblend``.*

**options**

  *dict, optional, default = {}*

  Settings for ``deblend_sofia_detections``, in the layout of that package's
  configuration file. These settings are passed on unchanged and must exist in
  the installed version of the package. For example:

  .. code-block:: yaml

     deblend:
       options:
         input:
           manual_optical_image: [image.fits]
           use_peak_deblending: true

Step components
---------------

*Specified with ``components``.*

**output_directory**

  *str, optional, default = 'components'*

  Directory of the component mask and tables.

**sources**

  *list of int or null, optional, default = null*

  SoFiA-2 detection numbers to include. If unset, all detections are used.

**only_named**

  *bool, optional, default = False*

  Keep only components that contain an object from
  ``input.known_source_table``.

**watershed_directory**

  *str, optional, default = 'Watershed_Output'*

  Directory containing the deblending output below the SoFiA-2 output directory.

**cutout_margin_beams**

  *float or null, optional, default = 3.0*

  If ``kinematics.cube`` is different from the deblended cube, carry the labels
  to that cube and cut it to the component region plus this margin in beams.
  Set to ``null`` to keep the whole cube.

**cutout_region**

  *str or null, optional, default = null*

  Alternatively, cut the kinematics cube to the sky region covered by this FITS
  image or cube.

**emission_mask**

  *str or null, optional, default = null*

  Keep a label only where this mask on the kinematics grid has emission
  (positive values).

**emission_sofia_parameters**

  *str or null, optional, default = null*

  Alternatively, use a SoFiA-2 parameter file to create the emission mask.

Step rmax
---------

*Specified with ``kinematics``.*

**cube**

  *str or null, optional, default = null*

  Cube used for the kinematics. Relative paths start at the run directory. If
  unset, the cube that was deblended is used.

**mask**

  *str or null, optional, default = null*

  Integer component-label mask on the grid of ``kinematics.cube``. If unset,
  the mask produced by the ``components`` step is used.

**components**

  *dict[str, int] or null, optional, default = null*

  Component names and their integer labels, for example
  ``{HCG16A: 1}``. If unset, all labels are used and named from mask header
  keywords ``LABEL<n>`` or ``component_<n>``.

**centre_table**

  *str or null, optional, default = null*

  Comma-separated table containing component names, right ascensions and
  declinations in degrees, and optionally other columns, for the first guesses
  of the centres. If unset, ``input.known_source_table`` is used; otherwise the
  centroid of the brightest emission is used.

**kinemetry_path**

  *str or null, optional, default = null*

  Directory containing ``kinemetry.py`` when Kinemetry is not installed in the
  Python environment.

**output_directory**

  *str, optional, default = 'kinematics'*

  Directory for the rmax tables.

**velocity_field_snr**

  *float, optional, default = 3.0*

  A pixel belongs to the velocity field if its integrated intensity exceeds this
  signal-to-noise ratio.

**ring_step_pixels**

  *float, optional, default = 2.0*

  Step in semi-major axis between fitted rings, in pixels.

**maximum_radius_arcsec**

  *float or null, optional, default = null*

  Maximum radius for fitted rings, in arcseconds. If unset, rings run to the
  edge of the velocity field.

**ring_tests**

  *list of str, optional, default = ['coverage', 'k5k1', 'vsys']*

  Tests that a ring must pass to count as regular:

  ``coverage``
    The fraction of the ellipse covered by the velocity field is at least
    ``default_coverage``.

  ``k5k1``
    The harmonic ratio k5/k1 is at most ``k5k1_max``.

  ``vsys``
    The systemic velocity drift is at most
    ``max(vsys_drift_kms, vsys_drift_fraction * k1)``.

  ``amplitude``
    The fitted velocity curve stays within ``amplitude_ratio_max`` times the
    velocities measured on the ring. This fit is constrained by the data.

**k5k1_max**

  *float, optional, default = 0.10*

  Maximum k5/k1 ratio for the ``k5k1`` test.

**vsys_drift_kms**

  *float, optional, default = 15.0*

  Maximum allowed systemic-velocity drift in km/s.

**vsys_drift_fraction**

  *float, optional, default = 0.15*

  Maximum systemic-velocity drift relative to k1.

**amplitude_ratio_max**

  *float, optional, default = 1.25*

  Maximum allowed ratio for the ``amplitude`` test.

**start_within_beams**

  *float, optional, default = 1.5*

  The regular-ring run must start within this number of beams from the first
  kinematic centre guess.

**fit_centre**

  *bool, optional, default = True*

  Fit the kinematic centre. If false, retain the first guess.

**measure_coverage**

  *bool, optional, default = True*

  Measure the minimum ring coverage with the masking test. For components with
  no completely sampled regular ring, or if false, use ``default_coverage``.
  This is used only when ``coverage`` is in ``ring_tests``.

**default_coverage**

  *float, optional, default = 0.60*

  Default ring coverage used when coverage cannot be measured directly.

**minimum_disc_beams**

  *float, optional, default = 1.0*

  A component is resolved as a rotating disc if its radial extent is at least
  this number of beams.

**minimum_v_over_sigma**

  *float, optional, default = 1.0*

  A component is dominated by rotation if its largest k1 exceeds this factor
  times the velocity dispersion inside rmax, and k1 does not fall from the
  first to the last ring.

Step pyfat
----------

*Specified with ``pyfat``.*

**output_directory**

  *str, optional, default = 'pyfat'*

  Directory in which pyFAT is run.

**minimum_beams_across**

  *float, optional, default = 3.0*

  Minimum major-axis extent, in beams, for a rotating disc to be modelled with
  pyFAT. The value must be at least 3.0. Smaller rotating discs are represented
  by an ellipse with semi-major axis rmax.

**flat_disc**

  *bool, optional, default = True*

  Fit a flat disc: inclination, position angle and dispersion do not vary with
  radius.

**force_fit**

  *bool, optional, default = True*

  Skip pyFAT's checks on source quality.

**cut_beyond_rmax_beams**

  *float or null, optional, default = 0.5*

  Replace gas beyond rmax plus this number of beams with noise in the cube
  given to pyFAT. Set to ``null`` to keep all gas of the component.

**spatial_margin_beams**

  *float, optional, default = 3.0*

  Margin around the emission, in beams, in the cube given to pyFAT.

**velocity_half_width_kms**

  *float, optional, default = 300.0*

  Half width in velocity of the cube given to pyFAT, in km/s.

**centre_tolerance_beams**

  *float, optional, default = 0.5*

  Maximum displacement, in beams, between the pyFAT and kinematic centres.

**random_seed**

  *int, optional, default = 1*

  Seed for noise that replaces emission from other components.

**reuse_fits**

  *bool, optional, default = False*

  If true, do not rerun pyFAT when the output directory already lists all discs;
  read the existing models and make the tables again.

**overrides**

  *dict, optional, default = {}*

  Any other pyFAT setting in the layout of its configuration file. These take
  precedence over the settings above. For example:

  .. code-block:: yaml

     pyfat:
       overrides:
         fitting:
           ring_size: 1.1

Step hi_mass
------------

*Specified with ``hi_mass``.*

**output_directory**

  *str, optional, default = 'hi_mass'*

  Directory of the HI-mass table.

**distance_mpc**

  *float or null, optional, default = null*

  Distance of all components in Mpc. If unset, use each component's Hubble-flow
  distance.

**hubble_constant**

  *float, optional, default = 70.0*

  Hubble constant used to calculate component distances, in km/s/Mpc.

**beam_margin**

  *float, optional, default = 0.0*

  Widen the rotating-disc ellipse by this fraction of the beam FWHM. The rings
  of the tilted-ring model start at radius zero and end at the outermost ring,
  so by default the ellipse is not widened.

Step figures
------------

*Specified with ``figures``.*

**enable**

  *bool, optional, default = True*

  Make figures. If false, the figures step does nothing.

**output_directory**

  *str, optional, default = 'figures'*

  Directory below which figures are written, grouped into one subdirectory per
  figure kind.

**products**

  *list of str, optional*

  Figure kinds to make. The default is:

  ``rmax``
    Velocity field of every component with rmax and the Kinemetry profiles.

  ``models``
    Radial profiles of the tilted-ring models.

  ``disc_anomalous``
    Column density and velocity field of the rotating disc and anomalous gas.

  ``position_velocity``
    Position-velocity diagrams along the major axis.

  ``model_comparison``
    Velocity field of the model over the data and residual moment maps.

**font_file**

  *str or null, optional, default = null*

  TTF or OTF font file for the figures. If unset, use the default matplotlib
  font.

**format**

  *str, optional, default = 'pdf'*

  Figure file format supported by matplotlib.

**pv_width_beams**

  *float, optional, default = 1.0*

  Width of position-velocity slices in beams.

**position_angles**

  *dict[str, float], optional, default = {}*

  Position angle of the major-axis slice for components without a tilted-ring
  model, in degrees from north through east, keyed by component name. If unset,
  the angle is measured from the velocity gradient.

**display_names**

  *dict[str, str], optional, default = {}*

  Names used for components in figures, for example
  ``{HCG16A: 'HCG 16A'}``.
