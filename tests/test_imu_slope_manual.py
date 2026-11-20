"""imu_slope --tay: khung KHÔNG được gửi lệnh nào trong lúc ghi (để tay đẩy được bánh)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import runtime  # noqa: E402
from core.simulator import XeGiaLap  # noqa: E402
from tools import imu_slope as D  # noqa: E402


def test_che_do_tay_khong_gui_lenh(monkeypatch):
    monkeypatch.setattr(D, 'THOI_GIAN_TOI_DA', 3.0)
    monkeypatch.setattr(D.ChuongTrinh, 'TAY', True)
    xe = XeGiaLap(can_camera=False, nhanh=True)
    n = {'goi': 0, 'dau': None, 'cuoi': None}
    goc = xe.gui
    xe.gui = lambda v, w: (n.__setitem__('goi', n['goi'] + 1), goc(v, w))[1]
    bd, kt = D.ChuongTrinh.bat_dau, D.ChuongTrinh.ket_thuc
    monkeypatch.setattr(D.ChuongTrinh, 'bat_dau', lambda self, ctx: (n.__setitem__('dau', n['goi']), bd(self, ctx))[1])
    monkeypatch.setattr(D.ChuongTrinh, 'ket_thuc', lambda self, ctx, ly: (n.__setitem__('cuoi', n['goi']), kt(self, ctx, ly))[1])
    runtime.chay('tools.imu_slope', gia_lap=True, ghi_bag=False, nhanh=True, xem_cong=None, xe=xe)
    # giữa bắt đầu và kết thúc chỉ được có lệnh DỪNG (v = 0) lúc hết giờ, không có lệnh chạy nào
    assert n['dau'] is not None and 0 <= n['cuoi'] - n['dau'] <= 1
    assert abs(xe.x) < 0.01                                          # xe không tự đi
