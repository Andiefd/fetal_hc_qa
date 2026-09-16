"""
pipeline_orchestrator.py

PipelineOrchestrator (Milestone 2 UML) -- top-level coordinator implementing
the `run_case` / `run_batch` pseudocode from Milestone 2's "Overview of
Design Concepts" section, verbatim in control flow:

    function run_case(image_path):
        case = dataset_loader.load_case(image_path)
        tensor = preprocessor.preprocess(case.image)
        mask = segmentation_model.predict_mask(tensor)
        ai_hc_mm = measurement_extractor.extract_hc(mask, case.pixel_spacing_mm)
        confidence = plane_quality_gate.assess_confidence(case.image, mask)
        result = comparison_engine.evaluate(case, ai_hc_mm, confidence)
        report = report_generator.generate_case_report(case, result)
        return report

This module composes every other service class exactly as designed in the
Milestone 2 UML class diagram and sequence diagrams, and is the single
entry point used by scripts/run_pipeline.py.
"""
import logging
from typing import List, Optional

from .dataset_loader import DatasetLoader
from .preprocessor import ImagePreprocessor
from .segmentation_model import HeadSegmentationModel
from .measurement_extractor import MeasurementExtractor
from .plane_quality_gate import PlaneQualityGate
from .flagging_engine import DiscrepancyFlaggingEngine
from .ga_day_converter import GADayConverter
from .comparison_engine import ComparisonEngine
from .report_generator import ReportGenerator
from .data_types import CaseReport, BatchSummaryReport

logger = logging.getLogger("fetal_hc_qa.pipeline")


class PipelineOrchestrator:
    def __init__(self, dataset_dir: str, output_dir: str):
        self.dataset_loader = DatasetLoader(dataset_dir)
        self.preprocessor = ImagePreprocessor()
        self.segmentation_model = HeadSegmentationModel()
        self.measurement_extractor = MeasurementExtractor()
        self.plane_quality_gate = PlaneQualityGate()
        self.comparison_engine = ComparisonEngine(
            gate=self.plane_quality_gate,
            flagger=DiscrepancyFlaggingEngine(),
            ga_converter=GADayConverter(),
        )
        self.report_generator = ReportGenerator(output_dir)

    def run_case(self, image_filename: str) -> Optional[CaseReport]:
        """Implements UC-1 through UC-5 for a single case, per the Milestone 2
        sequence diagram (sequence_main_pipeline.png). image_filename is the
        case's image filename within the dataset directory (e.g. "001_HC.png"),
        matching the DatasetLoader.load_case()/list_case_ids() contract.
        Returns None (and logs) if the case cannot be loaded at all -- the
        DatasetLoader's own skip-and-log exception path -- so a single corrupt
        case never halts a batch run (UC-1 exception path)."""
        try:
            case = self.dataset_loader.load_case(image_filename)
        except Exception as exc:  # DatasetLoadError, already logged by loader
            logger.warning("Skipping case %s: %s", image_filename, exc)
            return None

        tensor = self.preprocessor.preprocess(case.image)
        mask = self.segmentation_model.predict_mask(tensor)
        ai_hc_mm = None
        try:
            ai_hc_mm = self.measurement_extractor.extract_hc(mask, case.pixel_spacing_mm)
        except Exception as exc:
            logger.info("Case %s: measurement extraction failed (%s); "
                        "will be reported as UNMEASURABLE.", case.case_id, exc)

        confidence_image = tensor if tensor.shape == mask.shape else case.image
        result = self.comparison_engine.evaluate(case, ai_hc_mm, confidence_image, mask)
        report = self.report_generator.generate_case_report(case, result, mask, confidence_image)
        return report

    def run_batch(self, limit: Optional[int] = None) -> BatchSummaryReport:
        """Implements the batch-mode extension of UC-1..UC-5: iterates every
        case the DatasetLoader can enumerate, runs run_case() on each, and
        aggregates into a BatchSummaryReport (UC-5, Batch Evaluation Summary
        screen)."""
        case_ids = self.dataset_loader.list_case_ids(limit=limit)
        reports: List[CaseReport] = []
        for case_id in case_ids:
            report = self.run_case(case_id)
            if report is not None:
                reports.append(report)
        logger.info("Batch complete: %d/%d cases produced a report.",
                    len(reports), len(case_ids))
        return self.report_generator.generate_batch_summary(reports)
