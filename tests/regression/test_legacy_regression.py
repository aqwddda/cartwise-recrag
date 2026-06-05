from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from tests.regression.legacy_harness import (
    run_legacy_evidence_cases,
    run_legacy_fusion_cases,
)


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "regression"


def load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def assert_float_close(actual: float, expected: float) -> None:
    assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6)


def test_legacy_fusion_matches_deterministic_fixture() -> None:
    actual = run_legacy_fusion_cases()
    expected = load_fixture("legacy_fusion_expected.json")

    assert set(actual) == set(expected)
    for case_name, expected_case in expected.items():
        actual_case = actual[case_name]
        assert actual_case["known_user"] == expected_case["known_user"]
        assert actual_case["final_parent_asins"] == expected_case["final_parent_asins"]
        assert actual_case["fusion_intent"]["search_query"] == expected_case["fusion_intent"]["search_query"]
        assert actual_case["fusion_intent"]["product_terms"] == expected_case["fusion_intent"]["product_terms"]
        for key, expected_value in expected_case["filter_constraints"].items():
            assert actual_case["filter_constraints"][key] == expected_value
        if "filtered_results" in expected_case:
            actual_filtered = [
                {
                    "parent_asin": result["parent_asin"],
                    "filter_reason": result["filter_reason"],
                }
                for result in actual_case["filtered_results"]
            ]
            assert actual_filtered == expected_case["filtered_results"]
        if "final_results" in expected_case:
            for actual_result, expected_result in zip(
                actual_case["final_results"],
                expected_case["final_results"],
                strict=True,
            ):
                assert actual_result["parent_asin"] == expected_result["parent_asin"]
                assert actual_result["rank"] == expected_result["rank"]
                assert actual_result["sources"] == expected_result["sources"]
                assert_float_close(
                    actual_result["fusion_score"],
                    expected_result["fusion_score"],
                )


def test_legacy_evidence_matches_deterministic_fixture() -> None:
    assert run_legacy_evidence_cases() == load_fixture("legacy_evidence_expected.json")
