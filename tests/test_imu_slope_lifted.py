"""Xe bị nhấc / lật (pitch quá lớn) thì imu_slope phải BỎ mẫu, không tính vào phần XUỐNG."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import imu_slope as D  # noqa: E402


def _ctx(pitch_deg, dist=0.0, t=1.0):
    return SimpleNamespace(elapsed=lambda: t, distance=lambda: dist,
                           car=SimpleNamespace(scan=None, odom=SimpleNamespace(v=0.0)),
                           imu=SimpleNamespace(pitch=math.radians(pitch_deg), roll=0.0, yaw=0.0, gz=0.0))


def test_lifted_car_samples_dropped_in_manual_mode(monkeypatch):
    monkeypatch.setattr(D.Program, 'MANUAL', True)
    prog = D.Program()
    for p in (0.0, 3.0, -2.0):
        prog.step(_ctx(p), 0.05)
    for p in (-18.7, -43.0, -88.0, 25.0):                      # nhấc / lật
        assert prog.step(_ctx(p), 0.05) == (0.0, 0.0)
    assert len(prog.h['pitch']) == 3 and prog.drop == 4 and min(prog.h['pitch']) >= -2.0


def test_lifted_car_stops_in_auto_mode(monkeypatch):
    monkeypatch.setattr(D.Program, 'MANUAL', False)
    prog = D.Program()
    result = prog.step(_ctx(30.0), 0.05)
    assert hasattr(result, 'reason') and 'nhấc' in result.reason
