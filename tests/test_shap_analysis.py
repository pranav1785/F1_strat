"""Unit tests for SHAP analysis helpers."""

from __future__ import annotations

import numpy as np

from models.compound_rec.shap_analysis import _normalize_xgb_contribs


def test_normalize_xgb_contribs_handles_multiclass_last_axis_layout() -> None:
    contribs = np.arange(2 * 3 * 5, dtype=float).reshape(2, 3, 5)
    out = _normalize_xgb_contribs(contribs, n_features=4)
    assert out.shape == (2, 4, 3)


def test_normalize_xgb_contribs_handles_binary_2d_layout() -> None:
    contribs = np.arange(2 * 5, dtype=float).reshape(2, 5)
    out = _normalize_xgb_contribs(contribs, n_features=4)
    assert out.shape == (2, 4, 1)
