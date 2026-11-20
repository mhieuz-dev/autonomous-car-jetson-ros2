"""Bộ dò vạch trên ảnh sinh từ sa bàn giả (đúng calib, đúng phép chiếu)."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.camera import giai_nen_xam_nua  # noqa: E402
from core.simulator import XeGiaLap  # noqa: E402
from core.lane_follow import tao_bo_do  # noqa: E402

XE = XeGiaLap(can_camera=True, nhanh=True)


def _do(x, y, yaw_do):
    XE.x, XE.y, XE.yaw = x, y, math.radians(yaw_do)
    XE.t += 1.0
    jpeg, _ = XE.anh_jpeg()
    dv = tao_bo_do(gia_lap=True)
    return dv.xu_ly(giai_nen_xam_nua(jpeg), XE.t)


def test_giua_lan_phai():
    kq = _do(5.0, 1.8, 90)                     # đường dọc trục y, vạch phải ở x = 5.2
    assert kq.thay and abs(kq.khoang_cach_truoc - 0.20) < 0.015


def test_lech_trai_thi_khoang_cach_tang():
    kq = _do(4.95, 1.8, 90)
    assert abs(kq.khoang_cach_truoc - 0.25) < 0.015


def _do_nhieu_khung(x, y, yaw_do, so_khung=6):
    """Bộ dò giữ độ cong lọc Kalman qua nhiều khung; một khung đơn lẻ còn nhiễu
    (góc psi ngoại suy từ độ cong), nên đo góc sau vài khung liên tiếp như khi chạy thật."""
    dv = tao_bo_do(gia_lap=True)
    kq = None
    for _ in range(so_khung):
        XE.x, XE.y, XE.yaw = x, y, math.radians(yaw_do)
        XE.t += 0.1
        jpeg, _ = XE.anh_jpeg()
        kq = dv.xu_ly(giai_nen_xam_nua(jpeg), XE.t)
    return kq


def test_xe_quay_trai_thi_vach_lech_phai():
    kq = _do_nhieu_khung(5.0, 1.8, 95)
    assert kq.psi < math.radians(-3)


def test_xe_quay_phai_thi_vach_lech_trai():
    kq = _do_nhieu_khung(5.0, 1.8, 85)
    assert kq.psi > math.radians(3)


def test_bo_vet_sang_gia_6cm_va_vach_ngang():
    # Đoạn s 6.9-7.4 có vệt sáng rộng 6 cm ở -0.27, s 7.8 có vạch dừng ngang.
    i = int((abs(XE.s - 7.2)).argmin())
    p, h = XE.tim[i] - 0.20 * XE.phap[i], math.degrees(XE.huong[i])
    kq = _do(p[0], p[1], h)
    assert kq.thay and abs(kq.khoang_cach_truoc - 0.20) < 0.02
    i = int((abs(XE.s - 7.5)).argmin())
    p, h = XE.tim[i] - 0.20 * XE.phap[i], math.degrees(XE.huong[i])
    kq = _do(p[0], p[1], h)
    assert kq.thay and abs(kq.khoang_cach_truoc - 0.20) < 0.02


def test_nhanh_hon_15hz():
    kq = _do(5.0, 1.8, 90)
    assert kq.ms < 30
