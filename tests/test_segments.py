"""Segment-level observed uplift (no model involved)."""

from __future__ import annotations

import numpy as np
import pytest

from abtest import segments


def test_segments_partition_the_population(small_frame):
    frame = segments.add_segments(small_frame)
    table = segments.segment_uplift(frame)
    assert len(table) == 4
    assert table["n"].sum() == len(frame)
    assert set(table["segment"]) == set(segments.SEGMENT_ORDER)
    assert ((table["ci_low"] <= table["uplift"]) & (table["uplift"] <= table["ci_high"])).all()


def test_high_frequency_segments_have_positive_observed_uplift(small_frame):
    frame = segments.add_segments(small_frame)
    table = segments.segment_uplift(frame).set_index("segment")
    # The simulator gives a larger effect to 2+ order customers; the observed uplift
    # should be positive for those segments at this sample size.
    assert table.loc["2-3 orders", "uplift"] > 0
    assert table.loc["4+ orders", "uplift"] > 0


def test_segment_counts_match_manual_groupby(small_frame):
    frame = segments.add_segments(small_frame)
    table = segments.segment_uplift(frame).set_index("segment")
    counts = frame.groupby("segment")["user_id"].count()
    for segment in segments.SEGMENT_ORDER:
        assert table.loc[segment, "n"] == int(counts[segment])
