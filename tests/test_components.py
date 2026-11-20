import json

import numpy as np
import pytest
from astropy.io import fits
from omegaconf import OmegaConf

from anomalous_gas_extractor.config.config import defaults
from anomalous_gas_extractor.kinematics import components, pipeline
from anomalous_gas_extractor.run_scripts.run_scripts import (deblend_arguments, deblend_settings,
                                                            flatten_options)

from conftest import PIXEL_DEG, cube_header

SHAPE = (20, 24, 30)


def sofia_mask():
    """Two detections: number 1 on the left half, number 2 in the lower right corner."""
    mask = np.zeros(SHAPE, dtype=np.int32)
    mask[5:15, 4:20, 2:14] = 1
    mask[5:15, 2:8, 20:28] = 2
    return mask


def final_mask_of_detection_1():
    """Detection 1 divided into an upper (label 7) and a lower (label 3) component."""
    final = np.zeros(SHAPE, dtype=np.int32)
    final[5:15, 4:12, 2:14] = 3
    final[5:15, 12:20, 2:14] = 7
    return final


def sky_position(header, column, row):
    ra, dec = components.celestial_wcs(header).pixel_to_world_values(column, row)
    return float(ra), float(dec)


def test_sofia_products_are_read_from_the_parameter_file(tmp_path):
    parameters = tmp_path / "sofia.par"
    parameters.write_text("# comment\ninput.data = /data/cube.fits\noutput.directory = /out\n"
                          "output.filename = \npipeline.threads = 4   # inline comment\n")
    assert components.read_sofia_parameters(str(parameters))["pipeline.threads"] == "4"
    products = components.sofia_products(str(parameters))
    assert products.cube_file == "/data/cube.fits"
    assert products.mask_file == "/out/cube_mask.fits"
    # relative names start at the directory of the parameter file, where SoFiA-2 is run
    parameters.write_text("input.data = cube.fits\noutput.directory = found\n"
                          "output.filename = group\n")
    products = components.sofia_products(str(parameters))
    assert products.cube_file == f"{tmp_path}/cube.fits"
    assert products.mask_file == f"{tmp_path}/found/group_mask.fits"
    parameters.write_text("output.directory = /out\n")
    with pytest.raises(ValueError):
        components.sofia_products(str(parameters))


def test_grid_offset_and_place():
    reference = cube_header(SHAPE)
    cut = reference.copy()
    cut["CRPIX1"], cut["CRPIX2"], cut["CRPIX3"] = (reference["CRPIX1"] - 3, reference["CRPIX2"] - 2,
                                                   reference["CRPIX3"] - 1)
    assert components.grid_offset(cut, reference) == (1, 2, 3)
    placed = components.place(np.ones((4, 5, 6), dtype=np.int32), (1, 2, 3), SHAPE)
    assert placed.sum() == 120 and placed[1:5, 2:7, 3:9].all()
    # a cut-out that sticks out of the cube keeps the part inside
    assert components.place(np.ones((4, 5, 6), dtype=np.int32), (18, 22, -2), SHAPE).sum() == 16
    cut["CRPIX1"] += 0.4
    with pytest.raises(ValueError):
        components.grid_offset(cut, reference)
    cut["CRPIX1"] -= 0.4
    cut["CDELT1"] *= 2
    with pytest.raises(ValueError):
        components.grid_offset(cut, reference)


def test_deblended_and_other_detections_become_components():
    labels, parts = components.assemble_labels(sofia_mask(), {1: final_mask_of_detection_1()})
    assert [(part.source_id, part.deblend_label) for part in parts] == [(1, 3), (1, 7), (2, None)]
    assert np.array_equal(np.unique(labels), [0, 1, 2, 3])
    assert (labels == 3).sum() == (sofia_mask() == 2).sum()

    labels, parts = components.assemble_labels(sofia_mask(), {1: final_mask_of_detection_1()},
                                               sources=[2])
    assert [part.source_id for part in parts] == [2] and labels.max() == 1
    with pytest.raises(ValueError):
        components.assemble_labels(sofia_mask(), {}, sources=[5])


def test_component_that_repeats_labelled_gas_is_left_out():
    # the final mask of detection 1 also covers most of detection 2
    final = final_mask_of_detection_1()
    final[5:15, 2:8, 20:27] = 9
    labels, parts = components.assemble_labels(sofia_mask(), {1: final})
    assert [(part.source_id, part.deblend_label) for part in parts] == [(1, 3), (1, 7), (1, 9)]
    assert labels[10, 4, 27] == 0


def test_components_are_named_after_the_known_sources():
    header = cube_header(SHAPE)
    labels, parts = components.assemble_labels(sofia_mask(), {1: final_mask_of_detection_1()})
    data = np.ones(SHAPE, dtype=np.float32)
    known = {"Upper": sky_position(header, 8, 16), "Lower": sky_position(header, 8, 8),
             "Nowhere": sky_position(header, 16, 22)}
    components.name_components(labels, data, components.celestial_wcs(header), parts, known)
    assert [part.name for part in parts] == ["Lower", "Upper", "source2"]

    ordered, kept = components.order_components(labels, parts, list(known))
    assert [(part.label, part.name) for part in kept] == [(1, "Upper"), (2, "Lower"), (3, "source2")]
    assert ordered[10, 16, 8] == 1 and ordered[10, 8, 8] == 2 and ordered[10, 4, 24] == 3
    ordered, kept = components.order_components(ordered, kept, list(known), only_named=True)
    assert [part.name for part in kept] == ["Upper", "Lower"] and ordered.max() == 2

    components.measure_components(ordered, data, kept)
    assert kept[0].n_voxels == 960 and kept[0].summed_intensity == pytest.approx(960.0)


def test_separate_pieces_with_their_own_known_source_become_components():
    header = cube_header(SHAPE)
    celestial = components.celestial_wcs(header)
    data = np.ones(SHAPE, dtype=np.float32)
    known = {"Upper": sky_position(header, 8, 16), "Lower": sky_position(header, 8, 8)}
    # the deblending gave one label to two pieces that do not touch
    labels = np.zeros(SHAPE, dtype=np.int32)
    labels[5:15, 14:19, 5:12] = 1
    labels[5:15, 6:11, 5:12] = 1
    labels[5:15, 2:4, 20:24] = 1          # a third piece without a known source
    parts = [components.Component(1, "", 1, 7)]
    components.separate_pieces(labels, data, celestial, parts, known)
    components.name_components(labels, data, celestial, parts, known)
    assert [(part.label, part.name, part.deblend_label) for part in parts] == [
        (1, "Upper", 7), (2, "Lower", 7)]
    assert labels[10, 16, 8] == 1 and labels[10, 8, 8] == 2 and labels[10, 3, 22] == 1

    # two known sources in one connected piece stay one component
    joined = np.zeros(SHAPE, dtype=np.int32)
    joined[5:15, 6:19, 5:12] = 1
    parts = [components.Component(1, "", 1, 7)]
    components.separate_pieces(joined, data, celestial, parts, known)
    assert len(parts) == 1 and joined.max() == 1


def test_unnamed_components_of_one_detection_are_numbered():
    labels, parts = components.assemble_labels(sofia_mask(), {1: final_mask_of_detection_1()})
    components.name_components(labels, np.ones(SHAPE), components.celestial_wcs(cube_header(SHAPE)),
                               parts, {})
    assert [part.name for part in parts] == ["source1_1", "source1_2", "source2"]


def finer_header(factor=2):
    """Header of a cube of the same field with pixels that are ``factor`` times smaller."""
    header = cube_header((SHAPE[0], SHAPE[1] * factor, SHAPE[2] * factor))
    header["CDELT1"], header["CDELT2"] = -PIXEL_DEG / factor, PIXEL_DEG / factor
    # the same position on the sky under the centre of the reference pixel
    coarse = cube_header(SHAPE)
    header["CRPIX1"] = (coarse["CRPIX1"] - 0.5) * factor + 0.5
    header["CRPIX2"] = (coarse["CRPIX2"] - 0.5) * factor + 0.5
    return header


def test_labels_are_carried_to_a_finer_cube():
    header, fine = cube_header(SHAPE), finer_header()
    labels, _ = components.assemble_labels(sofia_mask(), {1: final_mask_of_detection_1()})
    shape = (SHAPE[0], SHAPE[1] * 2, SHAPE[2] * 2)
    carried = components.transfer_labels(labels, header, fine, shape)
    assert np.array_equal(carried[:, ::2, ::2], labels)
    assert np.array_equal(carried[:, 1::2, 1::2], labels)

    # the labels cover the rows 2 to 19 and the columns 2 to 27 of the coarse cube, that is,
    # the edges 3.5 to 39.5 and 3.5 to 55.5 in pixels of the finer cube
    rows, columns = components.footprint(labels, header, fine, shape, margin_pixels=2.0)
    assert (rows.start, rows.stop, columns.start, columns.stop) == (1, 43, 1, 59)
    cut = components.cutout_header(fine, rows, columns)
    cut_shape = (SHAPE[0], rows.stop - rows.start, columns.stop - columns.start)
    assert np.array_equal(components.transfer_labels(labels, header, cut, cut_shape),
                          carried[:, rows, columns])


def test_cube_is_cut_to_the_sky_area_of_another_file():
    fine = finer_header()
    shape = (SHAPE[0], SHAPE[1] * 2, SHAPE[2] * 2)
    region = cube_header((1, 6, 10))
    region["NAXIS1"], region["NAXIS2"] = 10, 6
    # the region covers the columns 5 to 14 and the rows 6 to 11 of the coarse cube
    coarse = cube_header(SHAPE)
    region["CRPIX1"], region["CRPIX2"] = coarse["CRPIX1"] - 5, coarse["CRPIX2"] - 6
    rows, columns = components.region_footprint(region, fine, shape)
    assert (rows.start, rows.stop, columns.start, columns.stop) == (12, 24, 10, 30)
    region["CRPIX1"] = coarse["CRPIX1"] + 500
    with pytest.raises(ValueError):
        components.region_footprint(region, fine, shape)


def test_channels_are_matched_by_velocity():
    header = cube_header(SHAPE)
    shifted = header.copy()
    shifted["CRPIX3"] = header["CRPIX3"] + 4
    labels, _ = components.assemble_labels(sofia_mask(), {})
    carried = components.transfer_labels(labels, header, shifted, SHAPE)
    assert np.array_equal(carried[4:], labels[:-4]) and not carried[:4].any()


def test_fourth_axis_is_removed_from_the_header():
    header = cube_header(SHAPE)
    header["NAXIS"], header["CTYPE4"], header["CRPIX4"], header["PC4_4"] = 4, "STOKES", 1.0, 1.0
    cleaned = components.cube_header(header)
    assert cleaned["NAXIS"] == 3 and "CTYPE4" not in cleaned and "PC4_4" not in cleaned
    assert cleaned["CTYPE3"] == header["CTYPE3"]
    labelled = components.label_header(header, [components.Component(1, "Upper", 1, 7)])
    assert labelled["LABEL1"] == "Upper"


@pytest.fixture
def deblended_run(tmp_path):
    """A cube, the SoFiA-2 mask and the final mask of detection 1, as the two programs leave them."""
    header = cube_header(SHAPE)
    rng = np.random.default_rng(1)
    data = rng.normal(0.0, 1e-4, SHAPE).astype(np.float32)
    data[sofia_mask() > 0] += 0.01
    output = tmp_path / "sofia_output"
    (output / "Watershed_Output" / "watershed_source_1").mkdir(parents=True)
    fits.writeto(tmp_path / "cube.fits", data, header)
    fits.writeto(output / "cube_mask.fits", sofia_mask(), header)
    cut = header.copy()
    cut["CRPIX1"], cut["CRPIX2"], cut["CRPIX3"] = (header["CRPIX1"] - 1, header["CRPIX2"] - 3,
                                                   header["CRPIX3"] - 4)
    fits.writeto(output / "Watershed_Output" / "watershed_source_1" / "final_mask.fits",
                 final_mask_of_detection_1()[4:16, 3:21, 1:15], cut)
    (tmp_path / "sofia.par").write_text(f"input.data = {tmp_path}/cube.fits\n"
                                        f"output.directory = {output}\noutput.filename = cube\n")
    (tmp_path / "known.csv").write_text("#Name,RA,DEC\nUpper,%.6f,%.6f\nLower,%.6f,%.6f\n"
                                        % (sky_position(header, 8, 16) + sky_position(header, 8, 8)))
    return OmegaConf.merge(OmegaConf.structured(defaults), {
        "input": {"sofia_parameter_file": "sofia.par", "known_source_table": f"{tmp_path}/known.csv"},
        "directories": {"run_directory": f"{tmp_path}/", "sofia_run_directory": f"{tmp_path}/"}})


def test_step_components_writes_the_mask_for_the_kinematic_steps(deblended_run, tmp_path):
    cfg = deblended_run
    with pytest.raises(FileNotFoundError):
        pipeline.component_files(cfg)
    parts = pipeline.run_components(cfg)
    assert [(part.label, part.name) for part in parts] == [(1, "Upper"), (2, "Lower"), (3, "source2")]
    cube_file, labels_file = pipeline.component_files(cfg)
    assert cube_file == f"{tmp_path}/cube.fits"
    cube = pipeline.load_cube(cfg)
    assert cube.components == {"Upper": 1, "Lower": 2, "source2": 3}
    assert (cube.labels == 1).sum() == 960
    with open(tmp_path / "components" / "components.json") as record:
        assert json.load(record)["labels"] == labels_file

    cfg.components.sources = [1]
    cfg.components.only_named = True
    assert [part.name for part in pipeline.run_components(cfg)] == ["Upper", "Lower"]
    assert pipeline.load_cube(cfg).components == {"Upper": 1, "Lower": 2}


def test_step_components_carries_the_labels_to_another_cube(deblended_run, tmp_path):
    cfg = deblended_run
    fine = finer_header()
    fine["NAXIS"], fine["CTYPE4"], fine["CRPIX4"], fine["CRVAL4"], fine["CDELT4"] = 4, "STOKES", 1., 1., 1.
    shape = (SHAPE[0], SHAPE[1] * 2, SHAPE[2] * 2)
    data = np.zeros((1,) + shape, dtype=np.float32)
    data[0, :, 10:30, 6:20] = 0.01
    fits.writeto(tmp_path / "fine.fits", data, fine)
    emission = np.zeros(shape, dtype=np.int16)
    emission[:, 10:30, 6:20] = 1
    fits.writeto(tmp_path / "emission.fits", emission, components.cube_header(fine))
    cfg.kinematics.cube = "fine.fits"
    cfg.components.emission_mask = "emission.fits"
    cfg.components.cutout_margin_beams = 0.25

    pipeline.run_components(cfg)
    cube_file, _ = pipeline.component_files(cfg)
    assert cube_file == f"{tmp_path}/components/kinematics_cube.fits"
    cube = pipeline.load_cube(cfg)
    assert cube.data.shape[1] < shape[1] and cube.data.shape[2] < shape[2]
    assert set(cube.components) == {"Upper", "Lower"}
    # labels only where the finer cube has emission
    assert np.all(cube.data[cube.labels > 0] > 0.005)

    cfg.components.cutout_margin_beams = None
    pipeline.run_components(cfg)
    assert pipeline.component_files(cfg)[0] == f"{tmp_path}/fine.fits"
    assert pipeline.load_cube(cfg).labels.shape == shape


DEBLEND_SETTINGS = {"input.sofia_parameters", "input.tables", "input.markers",
                    "general.counterpart_region", "logging.log_directory", "logging.log_file",
                    "directories.run_directory", "directories.ancillary_directory",
                    "internal.sofia"}


def deblend_configuration(options, known_source_table="/tables/known.csv"):
    return OmegaConf.merge(OmegaConf.structured(defaults), {
        "input": {"sofia_parameter_file": "sofia.par", "known_source_table": known_source_table},
        "internal": {"sofia": "/opt/sofia"},
        "directories": {"run_directory": "/work/", "sofia_run_directory": "/run/"},
        "deblend": {"options": options}})


def test_flatten_options():
    assert flatten_options({"input": {"markers": ["a", "b"]}, "general": {"x": {"y": 1}}}) == {
        "input.markers": ["a", "b"], "general.x.y": 1}


def test_deblend_options_are_passed_on_unchanged():
    cfg = deblend_configuration({"input": {"markers": ["Optical Image", "8 HI peaks"]},
                                 "general": {"counterpart_region": "Full_Ellipse"}})
    parsed = OmegaConf.from_cli(deblend_arguments(cfg, "/run/Logs", DEBLEND_SETTINGS))
    assert parsed.input.sofia_parameters == "/run/sofia.par"
    assert parsed.input.tables == ["/tables/known.csv"]
    assert parsed.input.markers == ["Optical Image", "8 HI peaks"]
    assert parsed.general.counterpart_region == "Full_Ellipse"
    assert parsed.internal.sofia == "/opt/sofia"
    assert parsed.directories.run_directory == "/run/"
    # what the deblending downloads is kept below the run directory of AGE
    assert parsed.directories.ancillary_directory == "/work/ancillary_data"


def test_deblend_tables_are_found_from_the_run_directory_of_the_deblending():
    one = deblend_configuration({}, known_source_table="known.csv")
    assert OmegaConf.from_cli(
        deblend_arguments(one, "/run/Logs", DEBLEND_SETTINGS)).input.tables == ["/work/known.csv"]
    several = deblend_configuration({}, known_source_table=["known.csv", "/tables/more.csv"])
    assert OmegaConf.from_cli(deblend_arguments(several, "/run/Logs", DEBLEND_SETTINGS)
                              ).input.tables == ["/work/known.csv", "/tables/more.csv"]
    for nothing in (None, ["None"], "none"):
        cfg = deblend_configuration({}, known_source_table=nothing)
        assert OmegaConf.from_cli(
            deblend_arguments(cfg, "/run/Logs", DEBLEND_SETTINGS)).input.tables == ["None"]


def test_unknown_deblend_options_are_refused():
    cfg = deblend_configuration({"input": {"use_peak_deblending": True}})
    with pytest.raises(ValueError, match="not settings of the installed"):
        deblend_arguments(cfg, "/run/Logs", DEBLEND_SETTINGS)


def test_deblend_arguments_are_settings_of_the_installed_version():
    cfg = deblend_configuration({"input": {"markers": ["manual table", "5 HI peaks"]}})
    known = set(deblend_settings())
    assert {argument.split("=")[0] for argument in deblend_arguments(cfg, "/run/Logs")} <= known
