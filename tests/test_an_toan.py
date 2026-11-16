"""Chặn sát mũi (mọi lúc) và gỡ kẹt trong hầm, trên xe tự hành chạy sa bàn giả."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chung import cau_hinh  # noqa: E402
from chung.gia_lap import XeGiaLap  # noqa: E402
from test_tu_hanh import chay  # noqa: E402
from tu_hanh import tham_so as TS  # noqa: E402


def test_chan_mui_dung_roi_chay_tiep(monkeypatch):
    monkeypatch.setattr(cau_hinh, 'CHAN_MUI_BAT', True)
    monkeypatch.setattr(TS, 'V_BAM_VACH', 0.15)          # mốc thời gian của bài này tính theo 0.15 m/s
    xe = XeGiaLap(can_camera=True, nhanh=True, he_so_toc_do=TS.NGOC_MUI_DO, co_nguoi=False)
    xe.x, xe.y, xe.yaw = 5.0, 1.6, math.pi / 2          # sau hầm, giữa làn phải
    xe.vat_them = [(5.0, 2.35, 0.05, 0.0, 9.0)]          # hộp chắn ngay trên làn, mất ở giây 9
    y = []
    gui = xe.gui
    xe.gui = lambda v, w: (gui(v, w), y.append((xe.t, v, xe.y)))
    chay(xe=xe, toi_da=14.0)
    assert xe.va_cham is None, f'va chạm {xe.va_cham}'
    dung = [yy for t, v, yy in y if 6.5 < t < 8.5]
    assert dung and all(abs(v) < 1e-9 for t, v, _ in y if 6.5 < t < 8.5)      # đứng trước hộp
    assert 2.35 - 0.05 - max(dung) > cau_hinh.LIDAR_TOI_CAN_M                 # mũi xe chưa chạm hộp
    assert max(yy for t, v, yy in y if t > 12) > 2.45                          # hộp đi -> chạy tiếp


def test_ham_ket_thi_lui_roi_di_tiep(monkeypatch):
    monkeypatch.setattr(TS, 'GO_KET_BAT', True)
    xe = XeGiaLap(can_camera=True, nhanh=True, he_so_toc_do=TS.NGOC_MUI_DO, co_nguoi=False)
    xe.x, xe.y, xe.yaw = 4.8, 0.9, math.pi / 2          # trong đoạn hầm thẳng sau khúc cua
    xe.ket_den = 4.5                                     # 2 s đầu (sau hiệu chuẩn) bánh không lăn
    lenh = []
    gui = xe.gui
    xe.gui = lambda v, w: (gui(v, w), lenh.append((xe.t, v)))
    ctx, _, _ = chay(xe=xe, toi_da=30.0)
    hv = [ten for _, ten in ctx.du_lieu['hanh_vi']]
    assert any(v < 0 for t, v in lenh), 'không thấy lệnh lùi'
    assert 'qua_ham' in hv and 'ra_ham' in hv, hv
    assert xe.va_cham is None
