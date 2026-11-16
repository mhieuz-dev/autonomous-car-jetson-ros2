"""Xe bị nhấc / lật (pitch quá lớn) thì imu_doc phải BỎ mẫu, không tính vào phần XUỐNG."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cong_cu import imu_doc as D  # noqa: E402


def _ctx(pitch_do, qd=0.0, t=1.0):
    return SimpleNamespace(thoi_gian=lambda: t, quang_duong=lambda: qd,
                           xe=SimpleNamespace(scan=None, odom=SimpleNamespace(v=0.0)),
                           imu=SimpleNamespace(pitch=math.radians(pitch_do), roll=0.0, yaw=0.0, gz=0.0))


def test_nhac_xe_thi_bo_mau_o_che_do_tay(monkeypatch):
    monkeypatch.setattr(D.ChuongTrinh, 'TAY', True)
    gd = D.ChuongTrinh()
    for p in (0.0, 3.0, -2.0):
        gd.buoc(_ctx(p), 0.05)
    for p in (-18.7, -43.0, -88.0, 25.0):                      # nhấc / lật
        assert gd.buoc(_ctx(p), 0.05) == (0.0, 0.0)
    assert len(gd.h['pitch']) == 3 and gd.bo == 4 and min(gd.h['pitch']) >= -2.0


def test_nhac_xe_khi_tu_chay_thi_dung(monkeypatch):
    monkeypatch.setattr(D.ChuongTrinh, 'TAY', False)
    gd = D.ChuongTrinh()
    kq = gd.buoc(_ctx(30.0), 0.05)
    assert hasattr(kq, 'ly_do') and 'nhấc' in kq.ly_do
