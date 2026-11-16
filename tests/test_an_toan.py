"""Chặn sát mũi (mọi giai đoạn) và gỡ kẹt trong hầm, trên xe giả lập."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chung import cau_hinh, khung  # noqa: E402
from chung.gia_lap import XeGiaLap  # noqa: E402


def _chay(ds, xe, du_lieu=None):
    lenh = []
    goc = xe.gui

    def gui(v, w):
        goc(v, w)
        lenh.append((round(xe.t, 2), v, w))
    xe.gui = gui
    ma, ctx = khung.chay(ds, gia_lap=True, nhanh=True, xem_cong=None, xe=xe)
    return ctx, lenh


def test_chan_mui_dung_roi_chay_tiep(monkeypatch):
    monkeypatch.setattr(cau_hinh, 'CHAN_MUI_BAT', True)
    xe = XeGiaLap(can_camera=True, nhanh=True)
    xe.x, xe.y, xe.yaw = 5.0, 1.6, math.pi / 2          # sau hầm, giữa làn phải
    xe.vat_them = [(5.0, 2.35, 0.05, 0.0, 9.0)]          # hộp chắn ngay trên làn, mất ở giây 9
    xe.t_xanh = 0.0
    lenh_s7 = []
    goc = xe.gui
    xe.gui = lambda v, w: (goc(v, w), lenh_s7.append((xe.t, v, xe.y)))

    import giai_doan.s7_bam_vach as S7
    cu = S7.THOI_GIAN_TOI_DA
    S7.THOI_GIAN_TOI_DA = 16.0
    monkeypatch.setattr(S7, 'V_BAM', 0.15)              # mốc thời gian của bài này tính theo 0.15 m/s
    try:
        ma, ctx = khung.chay(['s7_bam_vach'], gia_lap=True, nhanh=True, xem_cong=None, xe=xe)
    finally:
        S7.THOI_GIAN_TOI_DA = cu
    assert xe.va_cham is None, f'va chạm {xe.va_cham}'
    dung = [y for t, v, y in lenh_s7 if 6.5 < t < 8.5]
    assert dung and all(abs(v) < 1e-9 for t, v, y in lenh_s7 if 6.5 < t < 8.5)   # đứng trước hộp
    khe = 2.35 - 0.05 - max(dung)                                               # mũi xe tới hộp
    assert khe > cau_hinh.LIDAR_TOI_CAN_M                                       # chưa chạm
    assert max(y for t, v, y in lenh_s7 if t > 12) > 2.45                       # hộp đi -> chạy tiếp


def test_ham_ket_thi_lui_roi_di_tiep(monkeypatch):
    from giai_doan import s5_ham
    monkeypatch.setattr(s5_ham, 'GO_KET_BAT', True)
    xe = XeGiaLap(can_camera=False, nhanh=True)
    xe.x, xe.y, xe.yaw = 4.8 + 0.0, 0.9, math.pi / 2     # trong đoạn hầm thẳng sau khúc cua
    xe.ket_den = 4.5                                      # 2 s đầu (sau hiệu chuẩn) bánh không lăn
    ctx, lenh = _chay(['s5_ham'], xe)
    assert any(v < 0 for t, v, w in lenh), 'không thấy lệnh lùi'
    assert ctx.du_lieu['lich_su'][-1][1].startswith('RA HẦM')
    assert xe.va_cham is None
