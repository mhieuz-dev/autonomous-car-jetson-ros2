"""Biển STOP ở vị trí ngẫu nhiên: S7 (sau hầm) cũng đứng yên STOP_DUNG_GIAY rồi chạy tiếp; biển nhỏ (xa) thì không."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import giai_doan.s1_xuat_phat as S1  # noqa: E402
from chung import khung  # noqa: E402
from chung.gia_lap import XeGiaLap  # noqa: E402


def chay_s7(den_lich):
    xe = XeGiaLap(can_camera=True, nhanh=True)
    xe.x, xe.y, xe.yaw = 5.0, 1.6, math.pi / 2           # sau hầm, giữa làn phải
    xe.t_xanh = 0.0
    xe.den_lich = den_lich
    ghi = []
    goc = xe.gui
    xe.gui = lambda v, w: (goc(v, w), ghi.append((xe.t, v)))
    import giai_doan.s7_bam_vach as S7
    cu, S7.THOI_GIAN_TOI_DA = S7.THOI_GIAN_TOI_DA, 9.0
    try:
        khung.chay(['s7_bam_vach'], gia_lap=True, nhanh=True, xem_cong=None, xe=xe)
    finally:
        S7.THOI_GIAN_TOI_DA = cu
    return ghi


def doan_dung(ghi, tu):
    dung = []
    for t, v in ghi:
        if t > tu and abs(v) < 1e-9:
            dung.append(t)
        elif dung:
            break
    return dung


def test_s7_dung_o_stop_roi_chay_tiep():
    ghi = chay_s7([(5.0, 6.0, 'stop', S1.STOP_CAO_MIN_PX + 20)])
    dung = doan_dung(ghi, 4.9)
    assert dung, 'S7 phải đứng yên khi thấy STOP đủ gần'
    assert S1.STOP_DUNG_GIAY - 0.3 <= dung[-1] - dung[0] <= S1.STOP_DUNG_GIAY + 0.4
    assert any(v > 0.05 for t, v in ghi if t > dung[-1] + 0.5), 'dừng xong phải chạy tiếp'


def test_s7_stop_con_xa_thi_khong_dung():
    ghi = chay_s7([(5.0, 6.0, 'stop', S1.STOP_CAO_MIN_PX - 20)])
    assert all(v > 0.0 for t, v in ghi if 5.0 <= t <= 8.5), 'STOP còn nhỏ (xa) thì không được dừng'
