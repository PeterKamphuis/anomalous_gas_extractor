"""The step figures: figures of the results of the steps rmax, pyfat and hi_mass.

:func:`make_figures` reads the results of the earlier steps and writes the
kinds of figure listed in ``figures.products``, each in its own directory below
``figures.output_directory``:

rmax               velocity_fields_rmax, kinemetry_profiles
models             models_all, model_<component>
disc_anomalous     column_density, velocity_fields, velocity_fields_pv_slices
position_velocity  pv_all, pv_<component>
model_comparison   model_on_data, residual_moments, residual_statistics.csv

A kind of figure whose input does not exist yet is skipped with a warning.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List

from matplotlib import rc_context

from anomalous_gas_extractor.kinematics import (disc_figures, figure_data, model_figures, pipeline,
                                                plots, pv_figures, rmax_figures, sky_axes, slices)
from anomalous_gas_extractor.kinematics.datacube import DataCube, normalise_name
from anomalous_gas_extractor.kinematics.figure_data import Component

logger = logging.getLogger(__name__)


@dataclass
class FigureRun:
    """The configuration, the cube and the components of one call of the step figures."""

    cfg: object
    cube: DataCube
    components: List[Component]

    def file(self, product: str, name: str, extension: str = "") -> str:
        """Path of an output file of a kind of figure; its directory is created."""
        directory = pipeline._output_directory(
            self.cfg, os.path.join(self.cfg.figures.output_directory, product))
        return os.path.join(directory, f"{name}.{extension or self.cfg.figures.format}")

    def component_slices(self) -> Dict[str, slices.Slice]:
        """The position-velocity slice of every component by name."""
        settings = self.cfg.figures
        width = settings.pv_width_beams * self.cube.beam_fwhm_arcsec
        return {component.name: slices.component_slice(
                    self.cube, component, width, settings.position_angles.get(component.name))
                for component in self.components}


def _rmax(run: FigureRun) -> List[str]:
    """Velocity fields with r_max and the Kinemetry profiles."""
    cfg, cube = run.cfg, run.cube
    written = [run.file("rmax", "velocity_fields_rmax")]
    centres = pipeline._centre_table(cfg)
    rmax_figures.plot_velocity_fields(
        cube, run.components, written[0],
        {component.name: centres.get(normalise_name(component.name))
         for component in run.components})
    profiles = [rmax_figures.RingProfiles(
                    component.title, figure_data.kinemetry_rings(cfg, component),
                    component.row["r_max_arcsec"], component.row["r_max_is_lower_limit"])
                for component in run.components]
    profiles = [profile for profile in profiles if profile.rings]
    if not profiles:
        logger.warning("no Kinemetry profiles: the ring tables of the step rmax are missing")
        return written
    written.append(run.file("rmax", "kinemetry_profiles"))
    rmax_figures.plot_kinemetry_profiles(profiles, written[-1], cfg.kinematics.k5k1_max,
                                         cube.beam_fwhm_arcsec)
    return written


def _models(run: FigureRun) -> List[str]:
    """Radial profiles of the tilted-ring models: all together and one figure each."""
    cfg, written, profiles = run.cfg, [], []
    for component in run.components:
        rings = figure_data.model_rings(cfg, component) if component.has_model else []
        if not rings:
            continue
        profile = plots.ModelProfiles(component.title, rings, component.geometry.radius_arcsec,
                                      component.row["r_max_arcsec"])
        profiles.append(profile)
        written.append(run.file("models", f"model_{component.file_name}"))
        plots.plot_model_profiles(rings, written[-1], f"{component.title}: tilted-ring model",
                                  profile.disc_radius_arcsec, profile.r_max_arcsec,
                                  component.distance_mpc, cfg.figures.font_file)
    if not profiles:
        logger.warning("no figures of the models: run the step 'pyfat' first")
        return written
    written.append(run.file("models", "models_all"))
    # a common distance gives the radius in kpc as well
    plots.plot_model_grid(profiles, written[-1], cfg.hi_mass.distance_mpc, cfg.figures.font_file)
    return written


def _disc_anomalous(run: FigureRun) -> List[str]:
    """Column density and velocity field of the rotating discs and the anomalous gas."""
    written = [run.file("disc_anomalous", name)
               for name in ("column_density", "velocity_fields", "velocity_fields_pv_slices")]
    disc_figures.plot_column_density(run.cube, run.components, written[0])
    disc_figures.plot_velocity_fields(run.cube, run.components, written[1])
    disc_figures.plot_velocity_fields(run.cube, run.components, written[2], run.component_slices())
    return written


def _position_velocity(run: FigureRun) -> List[str]:
    """Position-velocity diagrams: all components together and one figure each."""
    strips = run.component_slices()
    diagrams = [pv_figures.extract_diagram(run.cube, component, strips[component.name],
                                           figure_data.model_rings(run.cfg, component))
                for component in run.components]
    written = [run.file("position_velocity", "pv_all")]
    pv_figures.plot_all_diagrams(run.cube, diagrams, written[0])
    for diagram in diagrams:
        written.append(run.file("position_velocity", f"pv_{diagram.component.file_name}"))
        pv_figures.plot_diagram(run.cube, diagram, written[-1])
    return written


def _model_comparison(run: FigureRun) -> List[str]:
    """Velocity field of the models on the data, residual moment maps and their statistics."""
    cube = run.cube
    comparisons = [model_figures.compare_with_model(cube, component,
                                                    run.cfg.kinematics.velocity_field_snr)
                   for component in run.components if component.model_cube is not None]
    for comparison in comparisons:
        if not comparison.compared.any():
            logger.warning("%s: data and model do not overlap above the noise; not compared",
                           comparison.component.name)
    comparisons = [comparison for comparison in comparisons if comparison.compared.any()]
    if not comparisons:
        logger.warning("no comparison with the models: run the step 'pyfat' first")
        return []
    written = [run.file("model_comparison", "model_on_data"),
               run.file("model_comparison", "residual_moments"),
               run.file("model_comparison", "residual_statistics", "csv")]
    model_figures.plot_model_on_data(cube, comparisons, written[0])
    model_figures.plot_residual_moments(cube, comparisons, written[1])
    pipeline.write_table(written[2], model_figures.residual_statistics(comparisons))
    return written


PRODUCTS: Dict[str, Callable[[FigureRun], List[str]]] = {
    "rmax": _rmax, "models": _models, "disc_anomalous": _disc_anomalous,
    "position_velocity": _position_velocity, "model_comparison": _model_comparison}


def make_figures(cfg) -> List[str]:
    """Write the figures listed in ``figures.products``; returns the files written.

    Nothing is written if ``figures.enable`` is false or if the step rmax has
    not been run.
    """
    if not cfg.figures.enable:
        return []
    unknown = [product for product in cfg.figures.products if product not in PRODUCTS]
    if unknown:
        raise ValueError(f"figures.products {unknown}: choose from {list(PRODUCTS)}")
    try:
        rows = pipeline.read_rmax_results(cfg)
    except FileNotFoundError as error:
        logger.warning("no figures: %s", error)
        return []
    cube = pipeline.load_cube(cfg)
    run = FigureRun(cfg, cube, figure_data.load_components(cfg, cube, rows))
    if not run.components:
        logger.warning("no figures: no component has a velocity field")
        return []
    written = []
    with rc_context(sky_axes.house_style(cfg.figures.font_file)):
        for product in cfg.figures.products:
            written += PRODUCTS[product](run)
    logger.info("%d figures and tables written below %s", len(written),
                pipeline._absolute(cfg.figures.output_directory, cfg))
    return written
