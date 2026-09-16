"""
segmentation_model.py

HeadSegmentationModel (Milestone 2 UML).

*** KNOWN DEVIATION FROM MILESTONE 2 -- READ THIS DOCSTRING ***
Milestone 2 specified a PyTorch U-Net (Ronneberger et al., 2015) as the
segmentation backend. This development sandbox has no internet access to
install PyTorch/TensorFlow (pip and the CUDA/CPU wheel indexes are both
unreachable from this environment) or to download GPU compute, so a trained
U-Net could not be produced here. Milestone 1 always scoped model training
to Google Colab / Kaggle (free-tier GPU) for exactly this reason -- that
step has not changed, it just has not happened inside *this* sandboxed build
environment.

To keep every other component (ComparisonEngine, PlaneQualityGate,
DiscrepancyFlaggingEngine, ReportGenerator, PipelineOrchestrator) fully real,
tested, and runnable end to end for this milestone, HeadSegmentationModel's
predict_mask() is implemented here with a classical computer-vision
algorithm (Otsu thresholding + morphological cleanup + largest-contour
selection) instead of a trained neural network. This is not a cosmetic
stand-in: it is a real, functioning segmentation algorithm of the same kind
used in the field before deep-learning approaches (and structurally similar
to the Random-Forest + Hough-transform baseline van den Heuvel et al. (2018)
used for HC18 before benchmarking deep models against it).

The class's public interface -- predict_mask(tensor) -> Mask -- is exactly
the interface specified in Milestone 2, so a trained U-Net can be dropped in
as a second backend behind this same interface with no change to any other
module. See Implementation Plan, "Known Deviations and Reconciliation Plan".
"""
import numpy as np
import cv2


class HeadSegmentationModel:
    architecture = "classical-cv-otsu-contour (U-Net specified; see module docstring)"

    def __init__(self, weights_path: str = None):
        # No trained weights in this backend; kept for interface parity with
        # the Milestone 2 design (a future U-Net backend would load weights here).
        self.weights_path = weights_path

    # Morphological-close kernel: large enough to bridge the gaps that speckle
    # noise leaves in the thin bright calvarium ring after Otsu thresholding,
    # so the ring closes into one solid, contour-able region.
    CLOSE_KERNEL_SIZE = 17
    CLOSE_ITERATIONS = 2
    # A small erosion after filling compensates for the outward bias that
    # closing a *ring* (rather than a filled disc) introduces -- empirically
    # tuned against this build's synthetic dataset (see Implementation Plan,
    # "Known Deviations and Reconciliation Plan" / segmentation calibration).
    ERODE_KERNEL_SIZE = 5

    def predict_mask(self, tensor: np.ndarray) -> np.ndarray:
        """Predict a binary head-boundary mask for a preprocessed [0,1] float32
        image. Returns a uint8 ndarray (0/255) the same size as the input."""
        img8 = (np.clip(tensor, 0, 1) * 255).astype(np.uint8)

        _, thresh = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.CLOSE_KERNEL_SIZE, self.CLOSE_KERNEL_SIZE)
        )
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel,
                                   iterations=self.CLOSE_ITERATIONS)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(img8)
        if not contours:
            return mask  # UC-3 exception path: empty mask -> "unmeasurable" (see MeasurementExtractor)

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 0.01 * img8.shape[0] * img8.shape[1]:
            return mask  # too small to be a plausible head region

        cv2.drawContours(mask, [largest], -1, 255, thickness=-1)

        erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.ERODE_KERNEL_SIZE, self.ERODE_KERNEL_SIZE)
        )
        mask = cv2.erode(mask, erode_kernel, iterations=1)
        return mask
