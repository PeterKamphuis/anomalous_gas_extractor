"""Mask of the components from the output of the source finding and the deblending.

SoFiA-2 writes a mask in which every detection carries its own number.
deblend_sofia_detections divides a detection into components and writes them as
``final_mask.fits`` on the grid of the cubelet it cut around that detection.
This module combines the two into a single mask on the grid of the data cube, in
which every component has its own label, and names the components after the
objects of a table with known sources.

The kinematics can be measured on another cube of the same field than the one
that was deblended, for example one with a higher resolution. The labels are
then carried to that cube: every voxel takes the label of the voxel of the
deblended cube in which it falls.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

from anomalous_gas_extractor.kinematics.datacube import normalise_name, spectral_axis

logger = logging.getLogger(__name__)

SOFIA_MASK_SUFFIX = "_mask.fits"
FINAL_MASK = "final_mask.fits"
SOURCE_DIRECTORY = "watershed_source_{}"
# a component is dropped if less than this fraction of its voxels is still free
MINIMUM_FREE_FRACTION = 0.5
# two grids are the same if their reference values and steps agree to this fraction of a pixel
GRID_TOLERANCE = 1e-3
FOURTH_AXIS = re.compile(r"(NAXIS|CTYPE|CRVAL|CRPIX|CDELT|CUNIT|CROTA)4|PC0?4_0?\d|PC0?\d_0?4")


@dataclass
class Component:
    """One component of the mask."""

    label: int
    name: str
    source_id: int
    deblend_label: Optional[int]
    n_voxels: int = 0
    summed_intensity: float = 0.0


@dataclass(frozen=True)
class SofiaProducts:
    """Files of a SoFiA-2 run that the deblending started from."""

    cube_file: str
    mask_file: str
    output_directory: str


def read_sofia_parameters(file_name: str) -> Dict[str, str]:
    """The settings of a SoFiA-2 parameter file."""
    settings = {}
    with open(file_name) as parameters:
        for line in parameters:
            line = line.split("#")[0]
            if "=" in line:
                key, value = line.split("=", 1)
                settings[key.strip()] = value.strip()
    return settings


def sofia_products(parameter_file: str) -> SofiaProducts:
    """The cube, the mask and the output directory named in a SoFiA-2 parameter file."""
    settings = read_sofia_parameters(parameter_file)
    cube_file = settings.get("input.data", "")
    if not cube_file:
        raise ValueError(f"{parameter_file} does not set input.data")
    directory = settings.get("output.directory") or os.path.dirname(cube_file)
    base = settings.get("output.filename") or os.path.splitext(os.path.basename(cube_file))[0]
    return SofiaProducts(cube_file, os.path.join(directory, base + SOFIA_MASK_SUFFIX), directory)


def grid_offset(header: fits.Header, reference: fits.Header) -> Tuple[int, int, int]:
    """Offset (z, y, x) in pixels of a cut-out in the cube it was cut from.

    Raises
    ------
    ValueError
        If the two grids differ by more than a shift of a whole number of pixels.
    """
    offset = []
    for axis in (3, 2, 1):
        step = header[f"CDELT{axis}"]
        same_step = abs(step - reference[f"CDELT{axis}"]) <= GRID_TOLERANCE * abs(step)
        same_value = abs(header[f"CRVAL{axis}"] - reference[f"CRVAL{axis}"]) <= (
            GRID_TOLERANCE * abs(step))
        shift = reference[f"CRPIX{axis}"] - header[f"CRPIX{axis}"]
        if not (same_step and same_value and abs(shift - round(shift)) <= GRID_TOLERANCE):
            raise ValueError(f"axis {axis}: the two grids are not shifted by whole pixels")
        offset.append(int(round(shift)))
    return offset[0], offset[1], offset[2]


def _overlap(shape: Sequence[int], cut_shape: Sequence[int], offset: Sequence[int]):
    """Slices of a cube and of a cut-out placed in it at an offset, where the two overlap."""
    inside, of_cut = [], []
    for size, cut_size, shift in zip(shape, cut_shape, offset):
        first, last = max(shift, 0), min(shift + cut_size, size)
        if last <= first:
            return None
        inside.append(slice(first, last))
        of_cut.append(slice(first - shift, last - shift))
    return tuple(inside), tuple(of_cut)


def place(values: np.ndarray, offset: Sequence[int], shape: Sequence[int]) -> np.ndarray:
    """A cut-out on the grid of the cube it belongs to; zero outside the cut-out."""
    placed = np.zeros(shape, dtype=values.dtype)
    overlap = _overlap(shape, values.shape, offset)
    if overlap is not None:
        placed[overlap[0]] = values[overlap[1]]
    return placed


def celestial_wcs(header: fits.Header) -> WCS:
    """The two sky axes of the coordinate system of a cube."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        return WCS(header).celestial


def read_final_masks(directory: str, reference: fits.Header,
                     shape: Sequence[int]) -> Dict[int, np.ndarray]:
    """The deblended masks of all detections, each placed on the grid of the cube."""
    masks = {}
    if not os.path.isdir(directory):
        return masks
    for entry in sorted(os.listdir(directory)):
        prefix = SOURCE_DIRECTORY.format("")
        file_name = os.path.join(directory, entry, FINAL_MASK)
        if not (entry.startswith(prefix) and entry[len(prefix):].isdigit()
                and os.path.isfile(file_name)):
            continue
        with fits.open(file_name) as hdul:
            values = np.nan_to_num(np.squeeze(hdul[0].data)).astype(np.int32)
            offset = grid_offset(hdul[0].header, reference)
        masks[int(entry[len(prefix):])] = place(values, offset, shape)
    return masks


def assemble_labels(sofia_mask: np.ndarray, final_masks: Dict[int, np.ndarray],
                    sources: Optional[Sequence[int]] = None
                    ) -> Tuple[np.ndarray, List[Component]]:
    """One label for every component of the selected detections.

    A detection that was deblended contributes the components of its final
    mask, any other detection is one component. The detections are treated in
    the order of their numbers and a voxel keeps the first label it receives; a
    component that loses more than half of its voxels in this way repeats gas
    that is already labelled and is left out.
    """
    present = [int(value) for value in np.unique(sofia_mask) if value > 0]
    selected = present if sources is None else [int(source) for source in sources]
    missing = [source for source in selected if source not in present]
    if missing:
        raise ValueError(f"detections {missing} are not in the SoFiA-2 mask (present: {present})")

    labels = np.zeros(sofia_mask.shape, dtype=np.int32)
    components: List[Component] = []
    for source in selected:
        if source in final_masks:
            parts = [(int(value), final_masks[source] == value)
                     for value in np.unique(final_masks[source]) if value > 0]
        else:
            parts = [(None, sofia_mask == source)]
        for deblend_label, voxels in parts:
            free = voxels & (labels == 0)
            if free.sum() < MINIMUM_FREE_FRACTION * voxels.sum():
                logger.info("detection %d, component %s: mostly labelled already, left out",
                            source, deblend_label)
                continue
            label = len(components) + 1
            labels[free] = label
            components.append(Component(label, "", source, deblend_label))
    return labels, components


def name_components(labels: np.ndarray, data: np.ndarray, celestial: WCS,
                    components: List[Component],
                    known_sources: Dict[str, Tuple[float, float]]) -> None:
    """Name each component after the known source it holds.

    A known source belongs to the component with the largest flux on the line of
    sight through its position. Components without a known source are named
    after their detection.
    """
    by_label = {component.label: component for component in components}
    for name, (ra_deg, dec_deg) in known_sources.items():
        x, y = celestial.world_to_pixel_values(ra_deg, dec_deg)
        column, row = int(round(float(x))), int(round(float(y)))
        if not (0 <= row < labels.shape[1] and 0 <= column < labels.shape[2]):
            logger.warning("%s lies outside the cube", name)
            continue
        on_sight = labels[:, row, column]
        flux = np.clip(np.nan_to_num(data[:, row, column]), 0.0, None)
        candidates = [int(value) for value in np.unique(on_sight) if value > 0]
        if not candidates:
            logger.warning("%s: no detected gas at its position", name)
            continue
        component = by_label[max(candidates, key=lambda value: flux[on_sight == value].sum())]
        if component.name:
            logger.warning("%s and %s are in the same component; it keeps the name %s",
                           name, component.name, component.name)
        else:
            component.name = name
    per_source: Dict[int, int] = {}
    for component in components:
        per_source[component.source_id] = per_source.get(component.source_id, 0) + 1
    counter: Dict[int, int] = {}
    for component in components:
        if component.name:
            continue
        counter[component.source_id] = counter.get(component.source_id, 0) + 1
        component.name = f"source{component.source_id}"
        if per_source[component.source_id] > 1:
            component.name += f"_{counter[component.source_id]}"


def order_components(labels: np.ndarray, components: List[Component],
                     known_sources: Sequence[str], only_named: bool = False
                     ) -> Tuple[np.ndarray, List[Component]]:
    """Number the components: the known sources first, in the order of their table."""
    rank = {normalise_name(name): index for index, name in enumerate(known_sources)}
    named = sorted((component for component in components
                    if normalise_name(component.name) in rank),
                   key=lambda component: rank[normalise_name(component.name)])
    others = [] if only_named else [component for component in components
                                    if normalise_name(component.name) not in rank]
    ordered = np.zeros_like(labels)
    for label, component in enumerate(named + others, start=1):
        ordered[labels == component.label] = label
        component.label = label
    return ordered, named + others


def measure_components(labels: np.ndarray, data: np.ndarray,
                       components: List[Component]) -> None:
    """Fill in the number of voxels and the summed intensity of every component."""
    for component in components:
        voxels = labels == component.label
        component.n_voxels = int(voxels.sum())
        component.summed_intensity = float(np.nansum(data[voxels]))


def _nearest_channels(velocity_kms: np.ndarray, reference_kms: np.ndarray) -> np.ndarray:
    """Channel of the reference axis nearest to each velocity; -1 beyond the axis."""
    order = np.argsort(reference_kms)
    position = np.clip(np.searchsorted(reference_kms[order], velocity_kms), 1, order.size - 1)
    lower, upper = order[position - 1], order[position]
    nearest = np.where(np.abs(velocity_kms - reference_kms[lower])
                       <= np.abs(velocity_kms - reference_kms[upper]), lower, upper)
    width = np.median(np.abs(np.diff(reference_kms[order])))
    return np.where(np.abs(velocity_kms - reference_kms[nearest]) <= 0.5 * width, nearest, -1)


def footprint(labels: np.ndarray, header: fits.Header, target: fits.Header,
              shape: Sequence[int], margin_pixels: float) -> Tuple[slice, slice]:
    """Box (rows, columns) of another cube that holds the labelled part of the sky plus a margin.

    ``shape`` is the shape (channels, rows, columns) of the other cube and the margin
    is given in its pixels.
    """
    rows, columns = np.nonzero(labels.any(axis=0))
    if rows.size == 0:
        raise ValueError("the mask of the components is empty")
    corners_x = np.concatenate([columns - 0.5, columns + 0.5, columns - 0.5, columns + 0.5])
    corners_y = np.concatenate([rows - 0.5, rows - 0.5, rows + 0.5, rows + 0.5])
    ra, dec = celestial_wcs(header).pixel_to_world_values(corners_x, corners_y)
    x, y = celestial_wcs(target).world_to_pixel_values(ra, dec)
    first_x = max(int(np.floor(x.min() - margin_pixels)), 0)
    last_x = min(int(np.ceil(x.max() + margin_pixels)) + 1, shape[2])
    first_y = max(int(np.floor(y.min() - margin_pixels)), 0)
    last_y = min(int(np.ceil(y.max() + margin_pixels)) + 1, shape[1])
    if last_x <= first_x or last_y <= first_y:
        raise ValueError("the two cubes do not cover the same part of the sky")
    return slice(first_y, last_y), slice(first_x, last_x)


def region_footprint(region: fits.Header, target: fits.Header,
                     shape: Sequence[int]) -> Tuple[slice, slice]:
    """Box (rows, columns) of a cube that lies within the part of the sky covered by another file.

    ``region`` is the header of an image or cube that covers the wanted part of the
    sky; only the pixels of the cube that fall completely inside it are kept.
    """
    columns, rows = region["NAXIS1"], region["NAXIS2"]
    corners_x = np.array([-0.5, columns - 0.5, -0.5, columns - 0.5])
    corners_y = np.array([-0.5, -0.5, rows - 0.5, rows - 0.5])
    ra, dec = celestial_wcs(region).pixel_to_world_values(corners_x, corners_y)
    x, y = celestial_wcs(target).world_to_pixel_values(ra, dec)
    first_x, last_x = max(int(np.ceil(x.min() + 0.5)), 0), min(int(np.floor(x.max() - 0.5)) + 1,
                                                                  shape[2])
    first_y, last_y = max(int(np.ceil(y.min() + 0.5)), 0), min(int(np.floor(y.max() - 0.5)) + 1,
                                                                  shape[1])
    if last_x <= first_x or last_y <= first_y:
        raise ValueError("the region does not overlap the cube")
    return slice(first_y, last_y), slice(first_x, last_x)


def transfer_labels(labels: np.ndarray, header: fits.Header, target: fits.Header,
                    shape: Sequence[int]) -> np.ndarray:
    """Labels on the grid of another cube: the label of the nearest voxel of the labelled cube."""
    rows, columns = np.indices(shape[1:])
    ra, dec = celestial_wcs(target).pixel_to_world_values(columns, rows)
    x, y = celestial_wcs(header).world_to_pixel_values(ra, dec)
    column, row = np.rint(x).astype(int), np.rint(y).astype(int)
    on_map = (row >= 0) & (row < labels.shape[1]) & (column >= 0) & (column < labels.shape[2])
    velocity, _ = spectral_axis(target, shape[0])
    reference, _ = spectral_axis(header, labels.shape[0])
    channel = _nearest_channels(velocity, reference)

    transferred = np.zeros(shape, dtype=labels.dtype)
    planes = labels[:, np.where(on_map, row, 0), np.where(on_map, column, 0)]
    planes[:, ~on_map] = 0
    covered = channel >= 0
    transferred[covered] = planes[channel[covered]]
    return transferred


def cube_header(header: fits.Header) -> fits.Header:
    """Header of a cube without the keywords of a fourth (Stokes) axis."""
    cube = header.copy()
    for key in [key for key in cube if FOURTH_AXIS.fullmatch(key)]:
        del cube[key]
    cube["NAXIS"] = 3
    return cube


def cutout_header(header: fits.Header, rows: slice, columns: slice) -> fits.Header:
    """Header of the part of a cube inside a box on the sky."""
    cut = cube_header(header)
    cut["CRPIX1"] = header["CRPIX1"] - columns.start
    cut["CRPIX2"] = header["CRPIX2"] - rows.start
    return cut


def label_header(header: fits.Header, components: List[Component]) -> fits.Header:
    """Header of a label mask: the grid of the cube and the name of every label."""
    labelled = cube_header(header)
    labelled["BUNIT"] = ""
    for component in components:
        labelled[f"LABEL{component.label}"] = (component.name, "name of the component")
    return labelled


def run_sofia_mask(sofia: str, parameter_file: str, cube_file: str, directory: str) -> str:
    """Run SoFiA-2 on a cube with the settings of a parameter file; returns the mask it writes."""
    settings = read_sofia_parameters(parameter_file)
    base = os.path.splitext(os.path.basename(cube_file))[0] + "_emission"
    settings.update({"input.data": cube_file, "output.directory": directory,
                     "output.filename": base, "output.writeMask": "true",
                     "output.overwrite": "true"})
    os.makedirs(directory, exist_ok=True)
    written = os.path.join(directory, base + ".par")
    with open(written, "w") as parameters:
        parameters.writelines(f"{key} = {value}\n" for key, value in settings.items())
    with open(os.path.join(directory, base + ".log"), "w") as log:
        subprocess.run([sofia, written], check=True, stdout=log, stderr=subprocess.STDOUT)
    return os.path.join(directory, base + SOFIA_MASK_SUFFIX)
