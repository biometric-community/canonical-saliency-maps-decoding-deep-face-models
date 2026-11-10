"""REPORT.md, the committed output of the full run, against the paper's claims."""

from __future__ import annotations

import re


def metric(report: str, name: str) -> float:
    match = re.search(rf"- {re.escape(name)}: \*\*([\d.]+)%?\*\*", report)
    assert match, f"{name} missing from REPORT.md"
    return float(match.group(1))


def test_canonical_maps_cause_a_larger_average_drop_than_gradcam(report):
    """Sec. 5.1: with negative explanations a higher Average Drop % is better, and CIS wins."""
    assert metric(report, "CIS Average Drop %") > metric(report, "Grad-CAM Average Drop %")


def test_canonical_maps_win_on_most_images(report, targets):
    """Eq. (13), Figs. 10-11: removing the canonical map causes the largest drop on most images."""
    assert metric(report, "Win % (CIS > Grad-CAM)") > targets["win_pct_floor"]


def test_gender_classifier_is_far_above_chance(report):
    """The maps explain a trained gender classifier, so it has to work in the first place."""
    assert metric(report, "Accuracy") >= 90.0
