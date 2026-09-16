"""
generate_synthetic_dataset.py

Generates a small, synthetic stand-in dataset that mimics the *exact* file
format of the real HC18 Grand Challenge dataset (van den Heuvel et al., 2018):

    - one grayscale .png ultrasound-style image per case
    - one grayscale "_Annotation.png" ellipse-boundary mask per case
      (stand-in for the sonographer's hand-drawn ellipse)
    - one CSV file with columns: filename, pixel size(mm), head circumference (mm)
      (this is the literal column schema used by the real HC18 training CSV)

WHY THIS EXISTS (read this before assuming it is real HC18 data):
This development sandbox has no internet access to Zenodo/grand-challenge.org,
so the real HC18 images could not be downloaded for this Milestone 3 build.
Because DatasetLoader (see src/fetal_hc_qa/dataset_loader.py) is written
against the real HC18 file/column schema rather than against this generator,
dropping the real HC18 files into data/ instead of this synthetic set requires
NO code changes anywhere in the pipeline. This is documented as a known,
environment-driven deviation in the Milestone 3 Implementation Plan.

The synthetic generator does attempt to be a reasonable stand-in for
evaluation purposes: head circumference values are sampled around the real
INTERGROWTH-21st median HC-for-GA curve (20-36 weeks), and the "sonographer"
annotation is the true synthetic ellipse perturbed by human-like measurement
noise (matching the Sarris et al., 2012 intra-observer scale), so that the
tiered flagging engine has a realistic mix of No Flag / Review / Re-measure
cases to exercise end to end.
"""
import os
import csv
import math
import random
import numpy as np
import cv2

random.seed(42)
np.random.seed(42)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample_dataset")
IMG_SIZE = 320
N_CASES = 60

# INTERGROWTH-21st median HC (mm) at 20/24/28/32/36 weeks (International Fetal and
# Newborn Growth Consortium for the 21st Century, 2023) -- used only to make the
# synthetic HC distribution realistic, not to fabricate clinical claims.
GA_WEEKS = [20, 24, 28, 32, 36]
HC_MM_AT_GA = [172.5, 219.1, 260.4, 294.4, 319.4]


def hc_for_ga(ga_weeks: float) -> float:
    """Linear interpolation over the INTERGROWTH-21st median HC curve."""
    return float(np.interp(ga_weeks, GA_WEEKS, HC_MM_AT_GA))


def draw_synthetic_ultrasound(size, semi_a_px, semi_b_px, angle_deg, center, speckle_level=18):
    """Render a grayscale image with speckle noise and an elliptical head boundary."""
    img = np.random.normal(loc=60, scale=speckle_level, size=(size, size)).astype(np.float32)
    img = cv2.GaussianBlur(img, (0, 0), sigmaX=1.2)

    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(mask, center, (int(semi_a_px), int(semi_b_px)), angle_deg, 0, 360, 255, thickness=-1)

    # Brighter "skull" ring just inside the boundary, darker interior -- a very
    # rough visual stand-in for the hyperechoic calvarium seen in real scans.
    ring = np.zeros_like(mask)
    cv2.ellipse(ring, center, (int(semi_a_px), int(semi_b_px)), angle_deg, 0, 360, 255, thickness=6)
    img = img + (mask.astype(np.float32) / 255.0) * 15.0
    img = img + (ring.astype(np.float32) / 255.0) * 90.0

    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, mask


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for i in range(N_CASES):
        case_id = f"{i:03d}_HC"
        ga_weeks = random.uniform(19, 37)
        true_hc_mm = hc_for_ga(ga_weeks) * random.uniform(0.94, 1.06)  # biological variance

        # pixel spacing chosen so the ellipse fits comfortably in the frame
        target_px_diameter = random.uniform(0.55, 0.72) * IMG_SIZE
        pixel_spacing_mm = (true_hc_mm / math.pi) / (target_px_diameter / 2)  # rough circle approx

        # eccentricity: semi-axes from true HC via Ramanujan approx (solve for a,b with fixed ratio)
        ratio = random.uniform(1.05, 1.35)  # a/b, fetal heads are not perfect circles
        # Ramanujan perimeter: P ~= pi*(3(a+b) - sqrt((3a+b)(a+3b)))
        denom = math.pi * (3 * (1 + ratio) - math.sqrt((3 * ratio + 1) * (3 + ratio)))
        b_guess = (true_hc_mm / pixel_spacing_mm) / denom
        a_guess = b_guess * ratio
        angle_deg = random.uniform(0, 180)
        cx = random.uniform(0.42, 0.58) * IMG_SIZE
        cy = random.uniform(0.42, 0.58) * IMG_SIZE

        img, mask = draw_synthetic_ultrasound(IMG_SIZE, a_guess, b_guess, angle_deg, (int(cx), int(cy)))

        # sonographer-recorded HC = true HC + human-like measurement noise
        # (Sarris et al., 2012: intraobserver 95% limit of agreement ~ +/-3.0%)
        noise_pct = np.random.normal(loc=0.0, scale=0.014)
        if random.random() < 0.12:  # a minority of harder cases to exercise Review/Re-measure tiers
            noise_pct += random.choice([-1, 1]) * random.uniform(0.03, 0.07)
        sonographer_hc_mm = true_hc_mm * (1 + noise_pct)

        img_name = f"{case_id}.png"
        ann_name = f"{case_id}_Annotation.png"
        cv2.imwrite(os.path.join(OUT_DIR, img_name), img)
        cv2.imwrite(os.path.join(OUT_DIR, ann_name), mask)

        rows.append({
            "filename": img_name,
            "pixel size(mm)": round(pixel_spacing_mm, 6),
            "head circumference (mm)": round(sonographer_hc_mm, 2),
        })

    csv_path = os.path.join(OUT_DIR, "training_set_pixel_size_and_HC.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "pixel size(mm)", "head circumference (mm)"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {N_CASES} synthetic cases in {OUT_DIR}")


if __name__ == "__main__":
    main()
