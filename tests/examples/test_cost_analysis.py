"""Tests for the cost-accounting study.

Timing magnitudes are machine- and load-dependent, so nothing here asserts an
absolute duration or an ordering between the two ~millisecond contenders
(transfer matrix vs surrogate query). What is asserted is structure: the
measurements exist and are positive, the crossover arithmetic is consistent
with its inputs, and the closed form — three to four orders of magnitude below
everything else — really is the fastest.
"""

import json

import pytest

from examples import cost_analysis as ca


@pytest.fixture(scope="module")
def measured():
    return ca.measure(repeats=2)


class TestMeasure:
    def test_all_timings_positive_and_finite(self, measured):
        for key in (
            "t_closed_form_s",
            "t_tmm_s",
            "t_surrogate_k_query_s",
            "t_surrogate_forward_512pts_s",
        ):
            assert measured[key] > 0.0

    def test_training_time_comes_from_the_recorded_metrics(self, measured):
        recorded = json.load(open(ca.SURROGATE_METRICS))["summary"]["train_time_s"]
        assert measured["surrogate_train_time_s"] == recorded

    def test_closed_form_is_fastest_by_a_wide_margin(self, measured):
        # ~0.3 us of scalar cmath against milliseconds of matrix work; robust
        # to any realistic load. The TMM-vs-surrogate ordering is NOT asserted.
        assert measured["t_closed_form_s"] * 100 < measured["t_tmm_s"]
        assert measured["t_closed_form_s"] * 100 < measured["t_surrogate_k_query_s"]


class TestCrossovers:
    def test_arithmetic_consistent_with_inputs(self, measured):
        x = ca.crossovers(measured)
        q = measured["t_surrogate_k_query_s"]
        train = measured["surrogate_train_time_s"]
        for name, t_ref in (
            ("vs_tmm", measured["t_tmm_s"]),
            ("vs_hypothetical_60s_solver", 60.0),
            ("vs_hypothetical_600s_solver", 600.0),
        ):
            got = x[f"crossover_queries_{name}"]
            if t_ref > q:
                assert got == pytest.approx(train / (t_ref - q))
            else:
                assert got == -1.0

    def test_never_pays_off_against_the_closed_form(self, measured):
        """The honest headline: 0.3 us per closed-form query cannot be beaten
        by a network whose single query costs milliseconds."""
        x = ca.crossovers(measured)
        assert x["crossover_queries_vs_closed_form"] == -1.0

    def test_expensive_reference_crossovers_are_small(self, measured):
        x = ca.crossovers(measured)
        assert 0 < x["crossover_queries_vs_hypothetical_600s_solver"] < 1000
        assert (
            x["crossover_queries_vs_hypothetical_600s_solver"]
            < x["crossover_queries_vs_hypothetical_60s_solver"]
        )


def test_main_writes_schema(tmp_path):
    out = ca.main(["--repeats", "1", "--out-dir", str(tmp_path)])
    on_disk = json.load(open(tmp_path / "cost_summary.json"))
    assert set(on_disk) == set(
        {k: (v if not isinstance(v, list) else v) for k, v in out.items()}
    )
    assert on_disk["crossover_queries_vs_closed_form"] == -1.0
