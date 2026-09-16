"""
fetal_hc_qa
===========

AI-Assisted Verification of Fetal Biometric Measurements for Improved
Diagnostic Accuracy in Obstetric Ultrasound.

A diagnostic-imaging QA / second-reader research pipeline that independently
measures fetal head circumference (HC) from a 2D ultrasound image, compares
it against the sonographer's recorded value, and classifies any discrepancy
into an evidence-based tier (Sarris et al., 2012).

This package is the Milestone 3 implementation of the architecture approved
in Milestone 2 (Architectural Design). See README.md and the Milestone 3
Implementation Plan for the two environment-driven deviations from that
design (segmentation backend and dataset source) and the plan to reconcile
them.
"""

__version__ = "0.3.0"
