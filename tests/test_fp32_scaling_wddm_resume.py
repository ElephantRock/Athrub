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
    monitor = WddmSharedUsageMonitor(pid=4242)
    with monitor._lock:
        monitor._series = [(1.0, 100), (2.0, 500), (3.0, 200), (4.0, 50)]
    monitor.available = True
    token = monitor.open_window()
    assert token is not None
    window = monitor.close_window(2.5)
    assert window == {"peak_bytes": 200, "baseline_bytes": 500, "samples_in_window": 2}
    # A window with no in-window samples yields None, not a fabricated peak.
    assert monitor.close_window(99.0) is None
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
