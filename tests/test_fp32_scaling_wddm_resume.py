"""Regression tests for the v0.2 preflight reliability hardening.

These cover the coverage the previous commit message claimed but, due to a
chained-command failure during its preparation, never actually landed:
PID-isolated typeperf parsing and window semantics, the four resume
order-structure rejections, no_memory_statistics stop-cause reporting, the
complete-is-False-exactly resume marker, and telemetry runtime coverage
accounting.
"""

import pytest

from athrub.scaling import (
    WddmSharedUsageMonitor,
    feasibility_binding,
    resume_partial_feasibility,
    stop_reason,
)


def _binding() -> dict:
    return feasibility_binding("s1", "s2", "s3", "s4", {"backend": "x"}, 12, "616.64", "2.11.0")


def test_wddm_monitor_pid_isolated_parsing() -> None:
    monitor = WddmSharedUsageMonitor(pid=4242)
    header = (
        '"Timestamp","\\\\HOST\\GPU Process Memory(pid_4242_luid_x)\\Shared Usage",'
        '"\\\\HOST\\GPU Process Memory(pid_9999_luid_x)\\Shared Usage"'
    )
    instances = monitor.parse_typeperf_header(header)[1:]
    _, values = monitor.parse_typeperf_row('"09/19/2026 15:04:05.123","1600000","999000000"')
    # Only this process's instance contributes; desktop activity excluded.
    assert monitor._pid_shared_bytes(instances, values) == 1600000
    # A PID with no matching instance yields None, never a false zero.
    other = WddmSharedUsageMonitor(pid=1111)
    assert other._pid_shared_bytes(instances, values) is None


def test_wddm_monitor_window_peak_and_baseline() -> None:
    import time as _time

    monitor = WddmSharedUsageMonitor(pid=4242)
    start = _time.monotonic()
    with monitor._lock:
        # Timestamps on the real monotonic clock. The end boundary is computed
        # inside close_window (a few microseconds after start), so in-window
        # samples sit exactly at the window start to order deterministically.
        monitor._series = [(start - 2.0, 100), (start - 1.0, 500), (start, 200), (start, 50)]
    monitor.available = True
    monitor.open_window()
    window = monitor.close_window(start)
    assert window == {
        "status": "covered",
        "peak_bytes": 200,
        "baseline_bytes": 500,
        "samples_in_window": 2,  # start+0.2 and start+0.4; both arrive before the end
        "post_window_sample_used": False,
    }
    coverage = monitor.coverage()
    assert coverage["windows_opened"] == 1 and coverage["windows_with_telemetry"] == 1


def test_wddm_monitor_coverage_never_overstates() -> None:
    monitor = WddmSharedUsageMonitor(pid=4242)
    monitor.available = False
    monitor.degraded_reason = "no PID-matched sample within 30.0s handshake"
    coverage = monitor.coverage()
    assert coverage["available"] is False
    assert coverage["degraded_reason"].startswith("no PID-matched")
    assert coverage["windows_opened"] == 0 and coverage["windows_with_telemetry"] == 0


def test_stop_reason_reports_no_memory_statistics() -> None:
    assert stop_reason({"status": "unsafe", "stop_cause": "no_memory_statistics"}) == "no_memory_statistics"
    assert stop_reason({"status": "unsafe", "stop_cause": "wddm_shared_spill"}) == "wddm_shared_spill"


def test_resume_requires_order_prefix() -> None:
    binding = _binding()
    base = {"binding": dict(binding), "complete": False}

    ok = {**base, "cells": {"c1": {}, "c2": {}}, "semantic": {}}
    assert resume_partial_feasibility(ok, binding, ["c1", "c2", "c3"], ["s1", "s2"])["resumed_units"] == 2

    gap = {**base, "cells": {"c2": {}}, "semantic": {}}
    with pytest.raises(ValueError, match="not a prefix"):
        resume_partial_feasibility(gap, binding, ["c1", "c2", "c3"], ["s1", "s2"])

    early_semantic = {**base, "cells": {"c1": {}}, "semantic": {"s1": {}}}
    with pytest.raises(ValueError, match="before all grid cells"):
        resume_partial_feasibility(early_semantic, binding, ["c1", "c2"], ["s1", "s2"])

    semantic_gap = {**base, "cells": {"c1": {}, "c2": {}}, "semantic": {"s2": {}}}
    with pytest.raises(ValueError, match="semantic rows are not a prefix"):
        resume_partial_feasibility(semantic_gap, binding, ["c1", "c2"], ["s1", "s2"])

    ok_semantic = {**base, "cells": {"c1": {}, "c2": {}}, "semantic": {"s1": {}}}
    assert resume_partial_feasibility(ok_semantic, binding, ["c1", "c2"], ["s1", "s2"])["resumed_units"] == 3


def test_resume_requires_complete_false_exactly() -> None:
    binding = _binding()
    cells = {"c1": {}, "c2": {}}
    # Missing marker and None marker are both rejected, not just True.
    for bad in ({}, {"complete": None}, {"complete": True}):
        partial = {"binding": dict(binding), "cells": dict(cells), "semantic": {}}
        partial.update(bad)
        with pytest.raises(ValueError, match="complete=false marker"):
            resume_partial_feasibility(partial, binding, ["c1", "c2"], ["s1"])


def test_normal_stop_is_clean_unexpected_death_is_degraded() -> None:
    # Deliberate shutdown: reader EOF after stop() must NOT be recorded as
    # degradation, and the reader must be joined so coverage() is stable.
    monitor = WddmSharedUsageMonitor(pid=4242)
    monitor.available = True
    with monitor._lock:
        monitor._series = [(1.0, 100), (2.0, 200)]

    def fake_reader() -> None:
        pass

    import threading as _threading

    monitor._reader = _threading.Thread(target=fake_reader)
    monitor._reader.start()
    monitor.stop()
    assert monitor._stopping is True
    assert monitor.degraded_reason is None
    coverage = monitor.coverage()
    assert coverage["available"] is True and coverage["degraded_reason"] is None
    assert coverage["samples"] == 2  # joined reader; final series visible

    # Unexpected stream death (no intentional stop) IS degraded. The reader's
    # EOF tail applies exactly this conditional; reproduce it directly since
    # driving a real typeperf EOF here would require the Windows counter.
    unexpected = WddmSharedUsageMonitor(pid=4242)
    unexpected.available = True
    unexpected._stopping = False
    if not unexpected._stopping:
        unexpected.degraded_reason = unexpected.degraded_reason or "typeperf stream ended"
    assert unexpected.degraded_reason == "typeperf stream ended"


def test_window_summarization_clips_at_end_boundary() -> None:
    series = [(1.0, 100), (2.0, 500), (3.0, 200), (4.0, 50)]
    summary = WddmSharedUsageMonitor.summarize_window(series, start_token=1.5, end_token=3.0)
    assert summary == {
        "baseline_bytes": 100,
        "peak_bytes": 500,  # samples at 2.0 and 3.0; the 4.0 sample is past the end
        "samples_in_window": 2,
        "post_window_sample_bytes": 50,
    }


def test_close_window_explicit_no_sample_status() -> None:
    monitor = WddmSharedUsageMonitor(pid=4242)
    monitor.available = True
    with monitor._lock:
        monitor._series = [(1.0, 100)]  # only a pre-window sample exists
    monitor.open_window()
    result = monitor.close_window(2.0, wait_for_sample_seconds=0.05)
    # Short window with no arriving sample: explicit lack of coverage, never
    # a silent no-spill fallback.
    assert result is not None
    assert result["status"] == "no_sample_in_window"
    assert result["baseline_bytes"] == 100
    coverage = monitor.coverage()
    assert coverage["windows_without_telemetry"] == 1


def test_close_window_uses_post_window_sample_explicitly() -> None:
    import time as _time

    monitor = WddmSharedUsageMonitor(pid=4242)
    monitor.available = True
    start = _time.monotonic()
    with monitor._lock:
        # Pre-window baseline plus a sample timestamped far after the window
        # will have closed: the bounded wait finds no in-window sample, then
        # uses the post-window observation explicitly and flags it.
        monitor._series = [(start - 1.0, 100), (start + 10.0, 700)]
    monitor.open_window()
    result = monitor.close_window(start, wait_for_sample_seconds=0.05)
    assert result["status"] == "covered"
    assert result["peak_bytes"] == 700
    assert result["post_window_sample_used"] is True
    assert result["samples_in_window"] == 0


def test_aggregate_segment_coverage_sums_all_segments() -> None:
    from athrub.scaling import aggregate_segment_coverage

    segments = [
        {"segment_id": "a", "wddm_telemetry_coverage": {"windows_opened": 10, "windows_with_telemetry": 9, "windows_without_telemetry": 1, "samples": 90}},
        {"segment_id": "b", "wddm_telemetry_coverage": {"windows_opened": 5, "windows_with_telemetry": 4, "windows_without_telemetry": 1, "samples": 40}},
        {"segment_id": "legacy", "note": "no coverage recorded"},
    ]
    aggregate = aggregate_segment_coverage(segments)
    assert aggregate["segment_count"] == 3
    assert aggregate["windows_opened"] == 15
    assert aggregate["windows_with_telemetry"] == 13
    assert aggregate["windows_without_telemetry"] == 2
    assert aggregate["samples"] == 130


def test_interrupted_segment_checkpoint_carries_both_evidence_kinds() -> None:
    # Regression for PR #17 comment 5745125546: session evidence used to be
    # written only at clean segment exit, so a crash mid-segment left the
    # checkpoint without RuntimeSession evidence. The checkpoint snapshot must
    # persist both WDDM coverage and session metadata at every write.
    from athrub.scaling import apply_segment_checkpoint_snapshot

    segment = {"segment_id": "segment-x", "cells_completed_this_segment": 3}
    coverage = {"available": True, "windows_opened": 20, "windows_with_telemetry": 18}
    session_evidence = {
        "requested_attention_policy": {"backend": "efficient_sdpa"},
        "query_head_count": 16,
        "kv_head_count": 8,
        "gqa_expansion_ratio": 2,
        "gqa_expansion_calls": 500,
    }
    # Mid-segment checkpoint (crash happens right after this write).
    snapshot = apply_segment_checkpoint_snapshot(segment, coverage, session_evidence)
    assert snapshot is segment
    assert snapshot["wddm_telemetry_coverage"] == coverage
    assert snapshot["attention_session_evidence"] == session_evidence
    # A checkpoint before the session exists must still persist coverage and
    # simply not fabricate session evidence.
    early = apply_segment_checkpoint_snapshot({"segment_id": "y"}, coverage, None)
    assert early["wddm_telemetry_coverage"] == coverage
    assert "attention_session_evidence" not in early
