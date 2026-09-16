"""
dataset_loader.py

DatasetLoader (Milestone 2 UML). Loads one case or a full dataset directory
written in the real HC18 file/column schema:
    <case_id>.png                 -- grayscale ultrasound image
    <case_id>_Annotation.png      -- ground-truth ellipse mask (evaluation only)
    training_set_pixel_size_and_HC.csv  -- columns: filename, pixel size(mm), head circumference (mm)

Implements UC-1 (Load Image Dataset) and its exception path exactly as
specified in Milestone 1: an invalid file format, missing metadata, or
corrupted image is logged and skipped rather than halting the batch.
"""
import os
import logging
from typing import List, Optional

import cv2
import pandas as pd

from .data_types import UltrasoundCase

logger = logging.getLogger(__name__)

CSV_FILENAME = "training_set_pixel_size_and_HC.csv"
VALID_EXTENSIONS = (".png",)


class DatasetLoadError(Exception):
    """Raised when a single case cannot be loaded; callers of load_dataset
    catch this per-case so one bad file does not halt the batch (UC-1 exception)."""


class DatasetLoader:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self._metadata: Optional[pd.DataFrame] = None

    def _metadata_df(self) -> pd.DataFrame:
        if self._metadata is None:
            csv_path = os.path.join(self.dataset_dir, CSV_FILENAME)
            if not os.path.isfile(csv_path):
                raise DatasetLoadError(f"Missing metadata CSV: {csv_path}")
            self._metadata = pd.read_csv(csv_path)
        return self._metadata

    def _validate_format(self, path: str) -> bool:
        return path.lower().endswith(VALID_EXTENSIONS) and os.path.isfile(path)

    def load_case(self, image_filename: str) -> UltrasoundCase:
        """Load a single case by its image filename (UC-1 main flow)."""
        image_path = os.path.join(self.dataset_dir, image_filename)
        if not self._validate_format(image_path):
            raise DatasetLoadError(f"Invalid or missing image file: {image_path}")

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise DatasetLoadError(f"Corrupted or unreadable image: {image_path}")

        df = self._metadata_df()
        row = df[df["filename"] == image_filename]
        if row.empty:
            raise DatasetLoadError(f"No metadata row found for {image_filename}")
        row = row.iloc[0]

        annotation_mask = None
        ann_path = os.path.join(
            self.dataset_dir, image_filename.replace(".png", "_Annotation.png")
        )
        if os.path.isfile(ann_path):
            annotation_mask = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)

        case_id = os.path.splitext(image_filename)[0]
        return UltrasoundCase(
            case_id=case_id,
            image=image,
            pixel_spacing_mm=float(row["pixel size(mm)"]),
            sonographer_hc_mm=float(row["head circumference (mm)"]),
            annotation_mask=annotation_mask,
        )

    def list_case_ids(self, limit: Optional[int] = None) -> List[str]:
        """Return the image filenames (usable directly as load_case()'s
        argument) for every case listed in the metadata CSV, in file order."""
        df = self._metadata_df()
        filenames = df["filename"].tolist()
        if limit is not None:
            filenames = filenames[:limit]
        return filenames

    def load_dataset(self, limit: Optional[int] = None) -> List[UltrasoundCase]:
        """Load every case in the dataset directory (UC-1 batch form).
        Per-case failures are logged and skipped, not raised, so a single
        corrupted file cannot halt the whole batch."""
        df = self._metadata_df()
        filenames = df["filename"].tolist()
        if limit is not None:
            filenames = filenames[:limit]

        cases = []
        for filename in filenames:
            try:
                cases.append(self.load_case(filename))
            except DatasetLoadError as exc:
                logger.warning("Skipping case %s: %s", filename, exc)
        return cases
