"""What the figures show of every component, collected from the results of the earlier steps.

The rotating disc of a component is the one of the HI mass budget: the ellipse
of the tilted-ring model, of the Kinemetry rings if there is no model, widened
by ``hi_mass.beam_margin``. The figures and the mass table therefore show the
same separation into rotating disc and anomalous gas.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from anomalous_gas_extractor.kinematics import hi_mass, pipeline
from anomalous_gas_extractor.kinematics.datacube import DataCube, MomentMaps, safe_name

logger = logging.getLogger(__name__)

MODEL = "pyFAT"


@dataclass
class Component:
    """One component of the mask with the results of the steps rmax and pyfat.

    Attributes
    ----------
    name, title : str
        Name in the mask and name written in the figures.
    label : int
        Label in the mask.
    row : dict
        Results of the step rmax.
    model : dict, optional
        Tilted-ring model of the step pyfat, if the disc was modelled.
    geometry : DiscGeometry, optional
        The rotating disc on the sky; None without rotating disc.
    disc_source : str
        Where the geometry comes from: pyFAT or Kinemetry.
    margin_arcsec : float
        Margin added to the ellipse of the disc.
    distance_mpc : float
        Distance used for lengths and masses.
    maps : MomentMaps
        Moment maps of the component.
    footprint, region : 2-D boolean arrays
        Pixels with voxels of the component and pixels inside the ellipse of
        its rotating disc.
    """

    name: str
    title: str
    label: int
    row: dict
    model: Optional[dict]
    geometry: Optional[hi_mass.DiscGeometry]
    disc_source: str
    margin_arcsec: float
    distance_mpc: float
    maps: MomentMaps
    footprint: np.ndarray
    region: np.ndarray

    @property
    def file_name(self) -> str:
        """Name of the component in file names."""
        return safe_name(self.name)

    @property
    def has_disc(self) -> bool:
        return self.geometry is not None

    @property
    def has_model(self) -> bool:
        """Whether the disc is the one of a tilted-ring model."""
        return self.has_disc and self.disc_source == MODEL

    @property
    def model_cube(self) -> Optional[str]:
        """Model cube of the tilted-ring model, if it exists."""
        cube_file = self.model["model_cube"] if self.has_model else None
        return cube_file if cube_file and os.path.isfile(cube_file) else None

    @property
    def kinematic_vsys_kms(self) -> float:
        """Systemic velocity of Kinemetry; the median velocity if it found no regular ring."""
        if self.row["vsys_kms"] is not None:
            return self.row["vsys_kms"]
        return float(np.nanmedian(self.maps.velocity))

    @property
    def vsys_kms(self) -> float:
        """Systemic velocity of the tilted-ring model, of Kinemetry without a model."""
        return self.model["vsys_kms"] if self.has_model else self.kinematic_vsys_kms

    @property
    def disc_pixels(self) -> np.ndarray:
        """Pixels with emission of the component inside its rotating disc."""
        return self.footprint & (self.maps.intensity > 0) & self.region

    @property
    def anomalous_pixels(self) -> np.ndarray:
        """Pixels with emission of the component outside its rotating disc."""
        return self.footprint & (self.maps.intensity > 0) & ~self.region


def _footprint(cube: DataCube, label: int) -> np.ndarray:
    """Pixels with voxels of a component."""
    box = cube.bounding_box(label)
    footprint = np.zeros(cube.data.shape[1:], dtype=bool)
    footprint[box[1:]] = (cube.labels[box] == label).any(axis=0)
    return footprint


def inside_rmax(cube: DataCube, component: Component) -> np.ndarray:
    """Pixels within r_max of the kinematic centre; none if no r_max was found."""
    row = component.row
    if row["r_max_arcsec"] is None:
        return np.zeros(component.footprint.shape, dtype=bool)
    return hi_mass.disc_region(cube, hi_mass.DiscGeometry(
        row["centre_ra_deg"], row["centre_dec_deg"], row["r_max_arcsec"], 0.0, 0.0))


def load_components(cfg, cube: DataCube, rows: List[dict]) -> List[Component]:
    """The components of the step rmax with their moment maps and rotating discs.

    A component too faint for a velocity field is left out.
    """
    models = pipeline.read_pyfat_models(cfg)
    margin = cfg.hi_mass.beam_margin * cube.beam_fwhm_arcsec
    components = []
    for row in rows:
        name = row["component"]
        maps = cube.moment_maps(row["label"], cfg.kinematics.velocity_field_snr)
        if not maps.valid.any():
            logger.warning("%s: no velocity field, the component is left out of the figures", name)
            continue
        geometry, source = pipeline._disc_geometry(row, models.get(name))
        region = np.zeros(maps.intensity.shape, dtype=bool)
        if geometry is not None:
            region = hi_mass.disc_region(cube, geometry, margin)
        components.append(Component(
            name=name, title=cfg.figures.display_names.get(name, name), label=row["label"],
            row=row, model=models.get(name) if source == MODEL else None, geometry=geometry,
            disc_source=MODEL if source == MODEL else "Kinemetry", margin_arcsec=margin,
            distance_mpc=pipeline._component_distance(cube, row, cfg), maps=maps,
            footprint=_footprint(cube, row["label"]), region=region))
    return components


def _table_value(text: str):
    """A cell of a csv table: a number, a truth value, NaN if blank, else the text."""
    if text in ("True", "False"):
        return text == "True"
    try:
        return float(text) if text else np.nan
    except ValueError:
        return text


def read_table(path: str) -> List[Dict[str, object]]:
    """Rows of a csv table written by the pipeline; an empty list if the file is missing."""
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as table:
        return [{key: _table_value(value) for key, value in row.items()}
                for row in csv.DictReader(table)]


def kinemetry_rings(cfg, component: Component) -> List[Dict[str, object]]:
    """Ring table of the step rmax of a component."""
    directory = pipeline._absolute(cfg.kinematics.output_directory, cfg)
    return read_table(os.path.join(directory, f"rings_{component.file_name}.csv"))


def model_rings(cfg, component: Component) -> List[Dict[str, object]]:
    """Ring table of the tilted-ring model of a component."""
    directory = pipeline._absolute(cfg.pyfat.output_directory, cfg)
    return read_table(os.path.join(directory, f"model_rings_{component.file_name}.csv"))
