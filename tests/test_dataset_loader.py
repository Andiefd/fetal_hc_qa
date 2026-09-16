"""Unit tests for DatasetLoader (UC-1: Load Image Dataset, including its
skip-and-log exception path)."""
import os
import csv

import numpy as np
import cv2
import pytest

from fetal_hc_qa.dataset_loader import DatasetLoader, DatasetLoadError


def _write_case(tmp_path, filename, pixel_size=0.6, hc=250.0, with_annotation=True):
    img = np.zeros((64, 64), dtype=np.uint8)
    cv2.imwrite(os.path.join(tmp_path, filename), img)
    if with_annotation:
        ann_name = filename.replace(".png", "_Annotation.png")
        cv2.imwrite(os.path.join(tmp_path, ann_name), img)
    return {"filename": filename, "pixel size(mm)": pixel_size, "head circumference (mm)": hc}


def _write_csv(tmp_path, rows):
    csv_path = os.path.join(tmp_path, "training_set_pixel_size_and_HC.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "pixel size(mm)", "head circumference (mm)"])
        writer.writeheader()
        writer.writerows(rows)


def test_load_case_success(tmp_path):
    row = _write_case(tmp_path, "case1.png")
    _write_csv(tmp_path, [row])

    loader = DatasetLoader(str(tmp_path))
    case = loader.load_case("case1.png")
    assert case.case_id == "case1"
    assert case.pixel_spacing_mm == pytest.approx(0.6)
    assert case.sonographer_hc_mm == pytest.approx(250.0)
    assert case.annotation_mask is not None


def test_load_case_missing_metadata_csv_raises(tmp_path):
    loader = DatasetLoader(str(tmp_path))
    with pytest.raises(DatasetLoadError):
        loader.load_case("nope.png")


def test_load_case_missing_image_file_raises(tmp_path):
    row = {"filename": "ghost.png", "pixel size(mm)": 0.5, "head circumference (mm)": 200.0}
    _write_csv(tmp_path, [row])
    loader = DatasetLoader(str(tmp_path))
    with pytest.raises(DatasetLoadError):
        loader.load_case("ghost.png")


def test_load_case_no_metadata_row_raises(tmp_path):
    row = _write_case(tmp_path, "case1.png")
    _write_csv(tmp_path, [row])
    # image file exists but under a name absent from the CSV
    cv2.imwrite(os.path.join(tmp_path, "case2.png"), np.zeros((64, 64), dtype=np.uint8))

    loader = DatasetLoader(str(tmp_path))
    with pytest.raises(DatasetLoadError):
        loader.load_case("case2.png")


def test_load_dataset_skips_bad_case_and_keeps_good_ones(tmp_path):
    good_row = _write_case(tmp_path, "good.png")
    bad_row = {"filename": "missing.png", "pixel size(mm)": 0.5, "head circumference (mm)": 200.0}
    _write_csv(tmp_path, [good_row, bad_row])

    loader = DatasetLoader(str(tmp_path))
    cases = loader.load_dataset()
    assert len(cases) == 1
    assert cases[0].case_id == "good"


def test_list_case_ids_respects_limit(tmp_path):
    rows = [_write_case(tmp_path, f"case{i}.png") for i in range(5)]
    _write_csv(tmp_path, rows)
    loader = DatasetLoader(str(tmp_path))
    assert len(loader.list_case_ids()) == 5
    assert len(loader.list_case_ids(limit=2)) == 2
