"""The steps of the program that follow the deblending: components, r_max, pyFAT and HI masses.

Each step reads the results of the previous one from disk, so the steps can be
run one at a time:

components  components/component_labels.fits, components.csv, components.json and, if the
            kinematics use another cube than the one that was deblended, kinematics_cube.fits
rmax        kinematics/rmax_table.csv, rmax_results.json, rings_<component>.csv
pyfat       pyfat/<component>/ (input and output of pyFAT), pyfat_table.csv, pyfat_results.json,
            model_rings_<component>.csv
hi_mass     hi_mass/hi_mass_table.csv and component_summary.csv, the summary of all steps

The figures of all steps are made by the step figures (module figures).
"""

from __future__ import annotations

import csv
import functools
import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits
from omegaconf import OmegaConf

from anomalous_gas_extractor.kinematics import (components, coverage_test, fat, hi_mass,
                                                kinemetry_rings)
from anomalous_gas_extractor.kinematics.centre import kinematic_centre
from anomalous_gas_extractor.kinematics.datacube import (DataCube, MomentMaps, normalise_name,
                                                         read_data_cube, safe_name)
from anomalous_gas_extractor.kinematics.rmax import (DiscExtent, RingCriteria, find_rmax,
                                                     measure_rings)
from anomalous_gas_extractor.kinematics.rotation import (RotationCriteria, RotationVerdict,
                                                         classify_rotation)

logger = logging.getLogger(__name__)

RMAX_TABLE = "rmax_table.csv"
RMAX_RESULTS = "rmax_results.json"
PYFAT_TABLE = "pyfat_table.csv"
PYFAT_RESULTS = "pyfat_results.json"
HI_MASS_TABLE = "hi_mass_table.csv"
SUMMARY_TABLE = "component_summary.csv"
COMPONENT_LABELS = "component_labels.fits"
COMPONENT_TABLE = "components.csv"
COMPONENT_RECORD = "components.json"
KINEMATICS_CUBE = "kinematics_cube.fits"
# significant digits of the numbers in the csv tables (the json files keep full precision)
TABLE_DIGITS = 9
# margin added to the radius of the first-pass disc when the centre is fitted (beams)
CENTRE_FIT_MARGIN_BEAMS = 0.5
# radius used for the centre fit when the first pass finds no disc (beams)
CENTRE_FIT_FALLBACK_BEAMS = 1.5


@dataclass
class ComponentResult:
    """Everything the r_max step derives for one component."""

    name: str
    label: int
    ra_deg: float
    dec_deg: float
    centre_source: str
    centre_scatter_arcsec: float
    extent: DiscExtent
    verdict: RotationVerdict
    coverage: Optional[coverage_test.CoverageResult] = None
    coverage_source: str = "default"
    beam_fwhm_arcsec: float = np.nan
    beam_major_arcsec: float = np.nan
    ring_tests: Tuple[str, ...] = ()

    def summary(self) -> Dict[str, object]:
        """One row of the r_max table."""
        extent, verdict = self.extent, self.verdict
        r_max = extent.r_max_arcsec
        return {
            "component": self.name,
            "label": self.label,
            "rotating_disc": verdict.is_rotating_disc,
            "r_max_arcsec": r_max,
            "r_max_is_lower_limit": extent.is_lower_limit,
            "r_max_beams": None if r_max is None else r_max / self.beam_fwhm_arcsec,
            # extent of the disc along its major axis in units of the major axis of the beam
            "disc_diameter_beams": None if r_max is None else 2.0 * r_max / self.beam_major_arcsec,
            "stopped_by": extent.stopped_by,
            "centre_ra_deg": self.ra_deg,
            "centre_dec_deg": self.dec_deg,
            "centre_source": self.centre_source,
            "centre_scatter_arcsec": self.centre_scatter_arcsec,
            "vsys_kms": extent.vsys_reference_kms,
            "position_angle_deg": extent.position_angle_deg,
            "axis_ratio": extent.axis_ratio,
            "k1_max_kms": extent.k1_max_kms,
            "minimum_coverage": extent.minimum_coverage,
            "minimum_coverage_source": self.coverage_source,
            "disc_extent_beams": verdict.disc_extent_beams,
            "k1_first_kms": verdict.k1_first_kms,
            "k1_last_kms": verdict.k1_last_kms,
            "dispersion_kms": verdict.dispersion_kms,
            "v_over_sigma": verdict.v_over_sigma,
            "failed_rotation_tests": "; ".join(verdict.failed_tests),
            "ring_tests": " ".join(self.ring_tests),
        }


# ---------------------------------------------------------------------------
# input and output
# ---------------------------------------------------------------------------

def _absolute(path: str, cfg) -> str:
    """A path of the configuration, relative to the run directory if not absolute."""
    return path if os.path.isabs(path) else os.path.join(cfg.directories.run_directory, path)


def _output_directory(cfg, name: str) -> str:
    directory = _absolute(name, cfg)
    os.makedirs(directory, exist_ok=True)
    return directory


def _cube_file(cfg) -> str:
    if cfg.kinematics.cube is not None:
        return _absolute(cfg.kinematics.cube, cfg)
    if cfg.cube_name is None:
        raise ValueError("give the data cube with kinematics.cube or cube_name")
    return os.path.join(cfg.directories.data_directory, cfg.cube_name)


@functools.lru_cache(maxsize=1)
def _read_cube(cube_file: str, mask_file: str, components: Tuple[Tuple[str, int], ...]) -> DataCube:
    return read_data_cube(cube_file, mask_file, dict(components) or None)


def component_files(cfg) -> Tuple[str, str]:
    """The cube and the mask of the components that the kinematic steps use.

    A mask given with ``kinematics.mask`` takes precedence over the one made by
    the step components.
    """
    if cfg.kinematics.mask is not None:
        return _cube_file(cfg), _absolute(cfg.kinematics.mask, cfg)
    record = os.path.join(_absolute(cfg.components.output_directory, cfg), COMPONENT_RECORD)
    if not os.path.isfile(record):
        raise FileNotFoundError("there is no mask of the components: run the step 'components' "
                                "or give a mask with kinematics.mask")
    with open(record) as saved:
        files = json.load(saved)
    return files["cube"], files["labels"]


def load_cube(cfg) -> DataCube:
    """The data cube and the mask of the components."""
    cube_file, mask_file = component_files(cfg)
    names = tuple((cfg.kinematics.components or {}).items())
    cube = _read_cube(cube_file, mask_file, names)
    logger.info("cube %s: %d components, rms %.3f mJy/beam, beam %.1f x %.1f arcsec",
                cube_file, len(cube.components), cube.rms * 1e3,
                cube.beam_major_arcsec, cube.beam_minor_arcsec)
    return cube


def read_source_table(table_file: str) -> Dict[str, Tuple[float, float]]:
    """Name and position of the objects in a comma separated table: name, RA, Dec (degrees), ...

    Lines that start with # are comments. The order of the table is kept.
    """
    positions = {}
    with open(table_file) as table:
        for row in csv.reader(line for line in table if line.strip() and not line.startswith("#")):
            positions[row[0].strip()] = (float(row[1]), float(row[2]))
    return positions


def read_centre_table(table_file: str) -> Dict[str, Tuple[float, float]]:
    """Starting centres from a table of objects, by normalised name."""
    return {normalise_name(name): position
            for name, position in read_source_table(table_file).items()}


def _known_source_table(cfg) -> Optional[str]:
    known = cfg.input.known_source_table
    if known[0].lower() == "none":
        return None
    return _absolute(known if isinstance(known, str) else known[0], cfg)


def _centre_table(cfg) -> Dict[str, Tuple[float, float]]:
    table = cfg.kinematics.centre_table
    table = _absolute(table, cfg) if table is not None else _known_source_table(cfg)
    return read_centre_table(table) if table else {}


def _plain(value):
    """Values that json and csv can store: numpy types converted, NaN as None."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _table_value(value):
    """A value as written to a csv table: floats with nine significant digits."""
    value = _plain(value)
    return float(f"{value:.{TABLE_DIGITS}g}") if isinstance(value, float) else value


def write_table(path: str, rows: List[Dict[str, object]]) -> None:
    """Write rows that share the same keys as a csv table."""
    if not rows:
        return
    with open(path, "w", newline="") as table:
        writer = csv.DictWriter(table, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([{key: _table_value(value) for key, value in row.items()}
                          for row in rows])


def _read_json(path: str, step: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} is missing: run the step '{step}' first")
    with open(path) as results:
        return json.load(results)


# ---------------------------------------------------------------------------
# step components
# ---------------------------------------------------------------------------

def _read_cube_file(file_name: str, memory_map: bool = False):
    """Data (three axes) and header of a FITS cube."""
    with fits.open(file_name, memmap=memory_map) as hdul:
        data = hdul[0].data
        data = data.reshape(data.shape[-3:])
        return (data if memory_map else np.array(data)), hdul[0].header.copy()


def _emission(cfg, directory: str, cube_file: str, header: fits.Header,
              shape: Tuple[int, ...]) -> Optional[np.ndarray]:
    """Voxels of the cube of the kinematics that hold emission, if a mask of them was asked for."""
    settings = cfg.components
    mask_file = None if settings.emission_mask is None else _absolute(settings.emission_mask, cfg)
    if mask_file is None and settings.emission_sofia_parameters is not None:
        logger.info("running SoFiA-2 on %s for the mask of its emission", cube_file)
        mask_file = components.run_sofia_mask(
            cfg.internal.sofia, _absolute(settings.emission_sofia_parameters, cfg), cube_file,
            os.path.join(directory, "emission_mask"))
    if mask_file is None:
        return None
    values, mask_header = _read_cube_file(mask_file)
    offset = components.grid_offset(mask_header, header)
    return components.place(np.nan_to_num(values) > 0, offset, shape)


def _labels_on_kinematics_cube(cfg, directory: str, labels: np.ndarray, header: fits.Header,
                               target_file: str):
    """Carry the labels to the cube of the kinematics; returns labels, data, header and cube file."""
    target, target_header = _read_cube_file(target_file, memory_map=True)
    margin, region = cfg.components.cutout_margin_beams, cfg.components.cutout_region
    cube_file = os.path.join(directory, KINEMATICS_CUBE)
    if region is not None:
        rows, columns = components.region_footprint(fits.getheader(_absolute(region, cfg)),
                                                    target_header, target.shape)
    elif margin is not None:
        beam_pixels = target_header["BMAJ"] / abs(target_header["CDELT2"])
        rows, columns = components.footprint(labels, header, target_header, target.shape,
                                             margin * beam_pixels)
    else:
        rows, columns = slice(0, target.shape[1]), slice(0, target.shape[2])
        cube_file = target_file
    data = np.array(target[:, rows, columns], dtype=np.float32)
    cut_header = components.cutout_header(target_header, rows, columns)
    if cube_file != target_file:
        fits.writeto(cube_file, data, cut_header, overwrite=True)
    transferred = components.transfer_labels(labels, header, cut_header, data.shape)
    emission = _emission(cfg, directory, cube_file, cut_header, data.shape)
    if emission is not None:
        transferred[~emission] = 0
    return transferred, data, cut_header, cube_file


def run_components(cfg) -> List[components.Component]:
    """Make the mask of the components from the output of SoFiA-2 and of the deblending."""
    parameter_file = os.path.join(cfg.directories.sofia_run_directory,
                                  cfg.input.sofia_parameter_file)
    products = components.sofia_products(parameter_file)
    directory = _output_directory(cfg, cfg.components.output_directory)
    sofia_mask, mask_header = _read_cube_file(products.mask_file)
    data, header = _read_cube_file(products.cube_file)
    final_masks = components.read_final_masks(
        os.path.join(products.output_directory, cfg.components.watershed_directory),
        mask_header, sofia_mask.shape)
    labels, parts = components.assemble_labels(np.nan_to_num(sofia_mask).astype(np.int32),
                                               final_masks, cfg.components.sources)
    table = _known_source_table(cfg)
    known = read_source_table(table) if table else {}
    components.name_components(labels, data, components.celestial_wcs(header), parts, known)
    labels, parts = components.order_components(labels, parts, list(known),
                                                cfg.components.only_named)

    cube_file = products.cube_file
    target = None if cfg.kinematics.cube is None else _absolute(cfg.kinematics.cube, cfg)
    if target is not None and not os.path.samefile(target, cube_file):
        labels, data, header, cube_file = _labels_on_kinematics_cube(cfg, directory, labels,
                                                                     header, target)
    components.measure_components(labels, data, parts)
    parts = [part for part in parts if part.n_voxels > 0]

    labels_file = os.path.join(directory, COMPONENT_LABELS)
    fits.writeto(labels_file, labels.astype(np.int16), components.label_header(header, parts),
                 overwrite=True)
    write_table(os.path.join(directory, COMPONENT_TABLE), [asdict(part) for part in parts])
    with open(os.path.join(directory, COMPONENT_RECORD), "w") as record:
        json.dump({"cube": cube_file, "labels": labels_file,
                   "components": _plain([asdict(part) for part in parts])}, record, indent=2)
    _read_cube.cache_clear()
    for part in parts:
        logger.info("component %d %s: detection %d, %d voxels", part.label, part.name,
                    part.source_id, part.n_voxels)
    logger.info("mask of the components written to %s", labels_file)
    return parts


# ---------------------------------------------------------------------------
# step rmax
# ---------------------------------------------------------------------------

def ring_criteria(cfg) -> RingCriteria:
    settings = cfg.kinematics
    return RingCriteria(k5k1_max=settings.k5k1_max, vsys_drift_kms=settings.vsys_drift_kms,
                        vsys_drift_fraction=settings.vsys_drift_fraction,
                        amplitude_ratio_max=settings.amplitude_ratio_max,
                        tests=tuple(settings.ring_tests),
                        start_within_beams=settings.start_within_beams,
                        ring_step_pixels=settings.ring_step_pixels,
                        maximum_radius_arcsec=settings.maximum_radius_arcsec)


def intensity_centroid(maps: MomentMaps) -> Tuple[float, float]:
    """Centroid (x, y) of the pixels brighter than half the peak intensity."""
    bright = maps.intensity >= 0.5 * maps.intensity.max()
    rows, columns = np.nonzero(bright)
    weights = maps.intensity[bright]
    return (float(np.sum(columns * weights) / weights.sum()),
            float(np.sum(rows * weights) / weights.sum()))


def analyse_component(cube: DataCube, name: str, label: int,
                      start: Optional[Tuple[float, float]], cfg) -> ComponentResult:
    """Kinematic centre, r_max and classification of one component.

    1. rings around the starting centre give a first estimate of the disc,
    2. the kinematic centre is fitted on the rings of that disc,
    3. rings around the kinematic centre are tested with the default coverage,
    4. if the coverage is one of the ring tests, the minimum coverage is measured
       on the complete regular rings and the ring tests are repeated with it,
    5. the run of regular rings is classified as rotating disc or not.
    """
    settings = cfg.kinematics
    criteria = ring_criteria(cfg)
    pixel, beam = cube.pixel_arcsec, cube.beam_major_arcsec
    maps = cube.moment_maps(label, settings.velocity_field_snr)

    if start is not None:
        x_centre, y_centre = cube.world_to_pixel(*start)
        centre_source = "centre table"
    else:
        x_centre, y_centre = intensity_centroid(maps)
        centre_source = "intensity centroid"
    rings = measure_rings(maps.velocity, x_centre, y_centre, pixel, criteria)
    extent = find_rmax(rings, settings.default_coverage, beam, criteria)

    centre_scatter = np.nan
    if settings.fit_centre:
        if extent.found:
            fit_radius = extent.disc_rings[-1].radius_arcsec
        else:
            fit_radius = CENTRE_FIT_FALLBACK_BEAMS * beam
        fit_radius += CENTRE_FIT_MARGIN_BEAMS * beam
        estimate = kinematic_centre(maps.velocity, x_centre, y_centre, fit_radius / pixel,
                                    beam / pixel, criteria.ring_step_pixels)
        if estimate is None:
            logger.warning("%s: the centre could not be fitted; keeping the %s", name, centre_source)
        else:
            x_centre, y_centre, centre_source = estimate.x, estimate.y, "kinematic"
            centre_scatter = estimate.scatter_pixels * pixel
            rings = measure_rings(maps.velocity, x_centre, y_centre, pixel, criteria)
            extent = find_rmax(rings, settings.default_coverage, beam, criteria)

    coverage, coverage_source = None, "default"
    if settings.measure_coverage and "coverage" in criteria.tests:
        coverage = coverage_test.minimum_coverage(maps.velocity, x_centre, y_centre, rings, pixel,
                                                  criteria, n_processes=cfg.input.ncpu)
        if coverage is not None:
            coverage_source = "masking test"
            extent = find_rmax(rings, coverage.minimum_coverage, beam, criteria)

    verdict = classify_rotation(
        extent, maps, x_centre, y_centre, pixel, cube.beam_fwhm_arcsec,
        criteria.ring_step_pixels * pixel,
        RotationCriteria(minimum_disc_beams=settings.minimum_disc_beams,
                         minimum_v_over_sigma=settings.minimum_v_over_sigma))
    ra, dec = cube.pixel_to_world(x_centre, y_centre)
    return ComponentResult(name=name, label=label, ra_deg=ra, dec_deg=dec,
                           centre_source=centre_source, centre_scatter_arcsec=centre_scatter,
                           extent=extent, verdict=verdict, coverage=coverage,
                           coverage_source=coverage_source,
                           beam_fwhm_arcsec=cube.beam_fwhm_arcsec,
                           beam_major_arcsec=cube.beam_major_arcsec, ring_tests=criteria.tests)


def _log_component(result: ComponentResult) -> None:
    extent = result.extent
    if not extent.found:
        logger.info("%s: no rotating disc (%s)", result.name, extent.stopped_by)
        return
    logger.info("%s: r_max %s %.1f arcsec (%s); k1_max %.0f km/s, v/sigma %.1f, extent %.1f "
                "beams -> %s", result.name, ">=" if extent.is_lower_limit else "=",
                extent.r_max_arcsec, extent.stopped_by, extent.k1_max_kms,
                result.verdict.v_over_sigma, result.verdict.disc_extent_beams,
                "rotating disc" if result.verdict.is_rotating_disc
                else "not a rotating disc: " + ", ".join(result.verdict.failed_tests))


def run_rmax(cfg) -> List[ComponentResult]:
    """Derive r_max of all components and classify them; writes the r_max tables."""
    
    kinemetry_rings.set_kinemetry_path(cfg.kinematics.kinemetry_path)
    kinemetry_rings.load_kinemetry()
    cube = load_cube(cfg)
    centres = _centre_table(cfg)
    directory = _output_directory(cfg, cfg.kinematics.output_directory)

    results = []
    for name, label in cube.components.items():
        result = analyse_component(cube, name, label, centres.get(normalise_name(name)), cfg)
        _log_component(result)
        write_table(os.path.join(directory, f"rings_{safe_name(name)}.csv"),
                    [asdict(ring) for ring in result.extent.rings])
        results.append(result)

    write_table(os.path.join(directory, RMAX_TABLE), [result.summary() for result in results])
    document = {"components": [
        dict(result.summary(), rings=[asdict(ring) for ring in result.extent.rings],
             coverage_test=None if result.coverage is None else asdict(result.coverage))
        for result in results]}
    with open(os.path.join(directory, RMAX_RESULTS), "w") as output:
        json.dump(_plain(document), output, indent=2)
    logger.info("r_max table written to %s", os.path.join(directory, RMAX_TABLE))
    return results


def read_rmax_results(cfg) -> List[dict]:
    """The rows of the r_max table written by :func:`run_rmax`."""
    directory = _absolute(cfg.kinematics.output_directory, cfg)
    return _read_json(os.path.join(directory, RMAX_RESULTS), "rmax")["components"]


# ---------------------------------------------------------------------------
# step pyfat
# ---------------------------------------------------------------------------

def fat_settings(cfg) -> fat.FatSettings:
    settings = cfg.pyfat
    return fat.FatSettings(
        executable=cfg.internal.pyfat, tirific=cfg.internal.tirific, sofia=cfg.internal.sofia,
        ncpu=cfg.input.ncpu, distance_mpc=cfg.hi_mass.distance_mpc,
        flat_disc=settings.flat_disc, force_fit=settings.force_fit,
        cut_beyond_rmax_beams=settings.cut_beyond_rmax_beams,
        spatial_margin_beams=settings.spatial_margin_beams,
        velocity_half_width_kms=settings.velocity_half_width_kms,
        centre_tolerance_beams=settings.centre_tolerance_beams,
        random_seed=settings.random_seed,
        overrides=OmegaConf.to_container(settings.overrides, resolve=True))


def _model_row(name: str, model: fat.FatModel) -> Dict[str, object]:
    return {"component": name, "converged": model.converged, "message": model.message,
            "disc_radius_arcsec": model.disc_radius_arcsec,
            "ra_deg": model.ra_deg, "dec_deg": model.dec_deg, "vsys_kms": model.vsys_kms,
            "inclination_deg": model.inclination_deg,
            "position_angle_deg": model.position_angle_deg,
            "vrot_max_kms": model.vrot_max_kms,
            "definition_file": model.definition_file, "model_cube": model.model_cube}


def large_enough_for_model(row: dict, cfg) -> bool:
    """Whether a rotating disc spans enough beams along its major axis for a tilted-ring model."""
    return bool(row["rotating_disc"]
                and row["disc_diameter_beams"] >= cfg.pyfat.minimum_beams_across)


def _fat_targets(cfg) -> Dict[str, fat.FatTarget]:
    """The rotating discs that are large enough as targets for pyFAT."""
    return {row["component"]: fat.FatTarget(
                name=safe_name(row["component"]), label=row["label"],
                ra_deg=row["centre_ra_deg"], dec_deg=row["centre_dec_deg"],
                vsys_kms=row["vsys_kms"], r_max_arcsec=row["r_max_arcsec"])
            for row in read_rmax_results(cfg) if large_enough_for_model(row, cfg)}


def _log_small_discs(cfg) -> None:
    for row in read_rmax_results(cfg):
        if row["rotating_disc"] and not large_enough_for_model(row, cfg):
            logger.info("%s: the rotating disc spans %.1f beams, fewer than the %g needed for a "
                        "tilted-ring model; its disc is the ellipse of Kinemetry with r_max",
                        row["component"], row["disc_diameter_beams"],
                        cfg.pyfat.minimum_beams_across)


def _write_ring_table(cfg, directory: str, target: fat.FatTarget, model: fat.FatModel) -> None:
    """Table with the rings of one tilted-ring model."""
    distance = cfg.hi_mass.distance_mpc
    if distance is None:
        distance = hi_mass.hubble_distance(model.vsys_kms, cfg.hi_mass.hubble_constant)
    rings = fat.model_rings(fat.read_tirific_def(model.definition_file), distance)
    write_table(os.path.join(directory, f"model_rings_{target.name}.csv"), rings)


def collect_pyfat_models(cfg) -> Dict[str, fat.FatModel]:
    """Read the final models in the output of pyFAT and write their tables."""
    directory = _output_directory(cfg, cfg.pyfat.output_directory)
    targets = _fat_targets(cfg)
    models = {component: fat.read_model(directory, target.name)
              for component, target in targets.items()}
    for name, model in models.items():
        if model.usable:
            logger.info("%s: pyFAT disc radius %.1f arcsec, inclination %.0f deg, PA %.0f deg",
                        name, model.disc_radius_arcsec, model.inclination_deg,
                        model.position_angle_deg)
            _write_ring_table(cfg, directory, targets[name], model)
        else:
            logger.warning("%s: no usable pyFAT model (%s)", name, model.message)
    table = [_model_row(name, model) for name, model in models.items()]
    write_table(os.path.join(directory, PYFAT_TABLE), table)
    with open(os.path.join(directory, PYFAT_RESULTS), "w") as output:
        json.dump(_plain({"models": table}), output, indent=2)
    return models


def _fits_exist(directory: str, targets: List[fat.FatTarget]) -> bool:
    """Whether the results of pyFAT in a directory list every target."""
    fitted = fat.read_fit_status(os.path.join(directory, fat.RESULTS_CATALOGUE))
    return all(target.name in fitted for target in targets)


def run_pyfat(cfg) -> Dict[str, fat.FatModel]:
    """Model the rotating discs with pyFAT, limited to r_max around the kinematic centre.

    Only rotating discs of at least ``pyfat.minimum_beams_across`` beams along
    the major axis are modelled.
    """
    targets = list(_fat_targets(cfg).values())
    directory = _output_directory(cfg, cfg.pyfat.output_directory)
    _log_small_discs(cfg)
    if not targets:
        logger.info("no component hosts a rotating disc that is large enough: nothing for "
                    "pyFAT to fit")
        return collect_pyfat_models(cfg)
    if cfg.pyfat.reuse_fits and _fits_exist(directory, targets):
        logger.info("reading the existing pyFAT models in %s", directory)
        return collect_pyfat_models(cfg)
    cube = load_cube(cfg)
    settings = fat_settings(cfg)
    for target in targets:
        fat.write_input_cube(cube, target, settings,
                             os.path.join(directory, target.name, f"{target.name}.fits"))
    fat.write_catalogue(os.path.join(directory, fat.CATALOGUE), targets, cube, settings)
    fat.write_configuration(os.path.join(directory, fat.CONFIGURATION), directory, settings)
    fat.run_pyfat(directory, settings)
    return collect_pyfat_models(cfg)


def read_pyfat_models(cfg) -> Dict[str, dict]:
    """The rows of the pyFAT table by component; empty if pyFAT was not run.

    A model left by an earlier run for a disc that is too small for one is not returned.
    """
    results = os.path.join(_absolute(cfg.pyfat.output_directory, cfg), PYFAT_RESULTS)
    if not os.path.isfile(results):
        return {}
    modelled = {row["component"] for row in read_rmax_results(cfg)
                if large_enough_for_model(row, cfg)}
    with open(results) as models:
        return {row["component"]: row for row in json.load(models)["models"]
                if row["component"] in modelled}


# ---------------------------------------------------------------------------
# step hi_mass
# ---------------------------------------------------------------------------

def _disc_geometry(row: dict, model: Optional[dict]) -> Tuple[Optional[hi_mass.DiscGeometry], str]:
    """Geometry of the rotating disc of a component and where it comes from.

    The pyFAT model is used if there is one and it converged. Otherwise (disc
    too small for a model, or no usable model) the disc is taken from Kinemetry:
    r_max with the mean position angle and flattening of its rings.
    """
    if not row["rotating_disc"]:
        return None, "no rotating disc"
    if model is not None and model["converged"] and model["disc_radius_arcsec"]:
        return hi_mass.DiscGeometry(
            ra_deg=model["ra_deg"], dec_deg=model["dec_deg"],
            radius_arcsec=model["disc_radius_arcsec"],
            inclination_deg=model["inclination_deg"],
            position_angle_deg=model["position_angle_deg"]), "pyFAT"
    inclination = float(np.degrees(np.arccos(np.clip(row["axis_ratio"], 0.0, 1.0))))
    return hi_mass.DiscGeometry(
        ra_deg=row["centre_ra_deg"], dec_deg=row["centre_dec_deg"],
        radius_arcsec=row["r_max_arcsec"], inclination_deg=inclination,
        position_angle_deg=row["position_angle_deg"]), "r_max (Kinemetry)"


def _model_status(row: dict, model: Optional[dict], cfg) -> str:
    """Why a component has, or does not have, a tilted-ring model."""
    if not row["rotating_disc"]:
        return "no rotating disc"
    if not large_enough_for_model(row, cfg):
        return f"disc smaller than {cfg.pyfat.minimum_beams_across:g} beams"
    if model is None:
        return "pyFAT not run"
    if model["converged"] and model["disc_radius_arcsec"]:
        return "pyFAT model"
    return "no usable pyFAT model"


def _component_distance(cube: DataCube, row: dict, cfg) -> float:
    """The configured distance or, if none is given, the Hubble flow distance of a component."""
    if cfg.hi_mass.distance_mpc is not None:
        return cfg.hi_mass.distance_mpc
    vsys = row["vsys_kms"]
    if vsys is None:
        maps = cube.moment_maps(row["label"], cfg.kinematics.velocity_field_snr)
        vsys = float(np.nanmedian(maps.velocity))
    return hi_mass.hubble_distance(vsys, cfg.hi_mass.hubble_constant)


def _kpc(radius_arcsec: Optional[float], distance_mpc: float) -> Optional[float]:
    if radius_arcsec is None:
        return None
    return radius_arcsec * fat.ARCSEC_TO_RADIAN * distance_mpc * 1e3


def _summary_row(row: dict, model: Optional[dict], masses: Dict[str, object]) -> Dict[str, object]:
    """One line of the summary: what the three steps found for a component."""
    distance = masses["distance_mpc"]
    fitted = model is not None and masses["disc_radius_source"] == "pyFAT"
    return {
        "component": row["component"],
        "rotating_disc": row["rotating_disc"],
        "distance_mpc": distance,
        "total_hi_mass_msun": masses["total_hi_mass_msun"],
        "disc_hi_mass_msun": masses["disc_hi_mass_msun"],
        "anomalous_hi_mass_msun": masses["anomalous_hi_mass_msun"],
        "disc_fraction": masses["disc_fraction"],
        "anomalous_fraction": 1.0 - masses["disc_fraction"],
        "disc_radius_arcsec": masses["disc_radius_arcsec"],
        "disc_radius_kpc": _kpc(masses["disc_radius_arcsec"], distance),
        "disc_radius_source": masses["disc_radius_source"],
        "tilted_ring_model": masses["tilted_ring_model"],
        "disc_diameter_beams": row["disc_diameter_beams"],
        "disc_ra_deg": masses["disc_ra_deg"],
        "disc_dec_deg": masses["disc_dec_deg"],
        "disc_inclination_deg": masses["disc_inclination_deg"],
        "disc_position_angle_deg": masses["disc_position_angle_deg"],
        "disc_vsys_kms": model["vsys_kms"] if fitted else None,
        "disc_vrot_max_kms": model["vrot_max_kms"] if fitted else None,
        "r_max_arcsec": row["r_max_arcsec"],
        "r_max_kpc": _kpc(row["r_max_arcsec"], distance),
        "r_max_is_lower_limit": row["r_max_is_lower_limit"],
        "kinematic_centre_ra_deg": row["centre_ra_deg"],
        "kinematic_centre_dec_deg": row["centre_dec_deg"],
        "kinematic_vsys_kms": row["vsys_kms"],
        "kinematic_position_angle_deg": row["position_angle_deg"],
        "kinematic_axis_ratio": row["axis_ratio"],
        "k1_max_kms": row["k1_max_kms"],
        "v_over_sigma": row["v_over_sigma"],
        "total_flux_jy_kms": masses["total_flux_jy_kms"],
        "disc_flux_jy_kms": masses["disc_flux_jy_kms"],
        "anomalous_flux_jy_kms": masses["anomalous_flux_jy_kms"],
    }


def run_hi_mass(cfg) -> List[Dict[str, object]]:
    """HI mass of every component inside and outside its rotating disc."""
    rows = read_rmax_results(cfg)
    models = read_pyfat_models(cfg)
    cube = load_cube(cfg)
    settings = cfg.hi_mass
    margin = settings.beam_margin * cube.beam_fwhm_arcsec

    table = []
    for row in rows:
        name = row["component"]
        model = models.get(name)
        geometry, source = _disc_geometry(row, model)
        distance = _component_distance(cube, row, cfg)
        region = None if geometry is None else hi_mass.disc_region(cube, geometry, margin)
        budget = hi_mass.mass_budget(cube, row["label"], distance, region)
        model_mass = None
        if source == "pyFAT" and model["model_cube"] and os.path.isfile(model["model_cube"]):
            model_mass = hi_mass.model_cube_mass(model["model_cube"], distance)
        table.append({
            "component": name,
            "rotating_disc": row["rotating_disc"],
            "disc_radius_arcsec": None if geometry is None else geometry.radius_arcsec,
            "disc_radius_source": source,
            "tilted_ring_model": _model_status(row, model, cfg),
            "disc_ra_deg": None if geometry is None else geometry.ra_deg,
            "disc_dec_deg": None if geometry is None else geometry.dec_deg,
            "disc_inclination_deg": None if geometry is None else geometry.inclination_deg,
            "disc_position_angle_deg": None if geometry is None else geometry.position_angle_deg,
            "disc_margin_arcsec": None if geometry is None else margin,
            "r_max_arcsec": row["r_max_arcsec"],
            "distance_mpc": distance,
            "total_flux_jy_kms": budget.total_flux_jy_kms,
            "disc_flux_jy_kms": budget.disc_flux_jy_kms,
            "anomalous_flux_jy_kms": budget.anomalous_flux_jy_kms,
            "total_hi_mass_msun": budget.total_mass,
            "disc_hi_mass_msun": budget.disc_mass,
            "anomalous_hi_mass_msun": budget.anomalous_mass,
            "disc_fraction": budget.disc_fraction,
            "model_hi_mass_msun": model_mass,
        })
        logger.info("%s: M_HI total %.2e, rotating disc %.2e, anomalous %.2e solar masses (%s)",
                    name, budget.total_mass, budget.disc_mass, budget.anomalous_mass, source)
    directory = _output_directory(cfg, settings.output_directory)
    write_table(os.path.join(directory, HI_MASS_TABLE), table)
    summary = os.path.join(cfg.directories.run_directory, SUMMARY_TABLE)
    write_table(summary, [_summary_row(row, models.get(row["component"]), masses)
                          for row, masses in zip(rows, table)])
    logger.info("HI mass table written to %s, summary of all steps to %s",
                os.path.join(directory, HI_MASS_TABLE), summary)
    return table
