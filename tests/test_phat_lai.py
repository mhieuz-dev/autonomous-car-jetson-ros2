"""Dự phòng S7: ghi đường (đẩy tay) rồi phát lại bằng hướng IMU theo quãng đường, bản lai bám vạch + đường ghi."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chung import cau_hinh, duong_ghi, khung  # noqa: E402
from chung.gia_lap import XeGiaLap  # noqa: E402


def _ghi_bang_s7(tmp_path):
    """"Đẩy tay" trên sa bàn giả = cho S7 tự chạy một lượt từ sau hầm và ghi odom + yaw như s7_ghi_duong."""
    from test_gia_lap import chay_va_do
    import giai_doan.s7_bam_vach as S7
    xe = XeGiaLap(can_camera=True, nhanh=True)
    xe.x, xe.y, xe.yaw = 5.0, 1.6, math.pi / 2
    xe.t_xanh = 0.0
    g = duong_ghi.GhiDuong()
    goc = xe.gui
    bat_dau = []

    def gui(v, w):
        goc(v, w)
        if bat_dau:                                 # ghi theo odom + IMU giả (cùng nguồn xe thật dùng)
            g.cap_nhat(xe.odom.quang_duong, xe.imu.yaw, xe.odom.x, xe.odom.y, v, xe.t)
    xe.gui = gui
    cu, S7.THOI_GIAN_TOI_DA = S7.THOI_GIAN_TOI_DA, 40.0
    try:
        bat_dau.append(1)
        ma, ctx, xe, ghi = chay_va_do(['s7_bam_vach'], xe=xe)
    finally:
        S7.THOI_GIAN_TOI_DA = cu
    f = tmp_path / 'duong_s7.csv'
    g.luu(f)
    return f, (xe.x, xe.y, xe.yaw), g


def test_ghi_duong_moi_1cm_va_doc_lai(tmp_path):
    f, cuoi, g = _ghi_bang_s7(tmp_path)
    assert g.dai > 2.5 and len(g.mau) > 200
    d = duong_ghi.DuongDaGhi(f)
    assert abs(d.dai - g.dai) < 1e-6
    assert abs(d.huong(0.0)) < 0.05                                 # đầu: hướng 0 so với lúc bắt đầu
    assert d.huong(d.dai) > math.radians(60)                        # cuối: đã qua cua trái sau hầm


def test_phat_lai_thuan_theo_duong_ghi_ve_dung_cuoi(tmp_path, monkeypatch):
    import giai_doan.s7_phat_lai as P
    f, cuoi, g = _ghi_bang_s7(tmp_path)
    monkeypatch.setattr(cau_hinh, 'CALIB', tmp_path)
    monkeypatch.setattr(P, 'CHI_THEO_DUONG', True)
    xe = XeGiaLap(can_camera=True, nhanh=True)        # giai đoạn vẫn mở camera (kế thừa S7), chỉ không dùng
    xe.x, xe.y, xe.yaw = 5.0, 1.6, math.pi / 2
    xe.t_xanh = 0.0
    ma, ctx = khung.chay(['s7_phat_lai'], gia_lap=True, nhanh=True, xem_cong=None, xe=xe)
    assert ctx.du_lieu['lich_su'][0][1].startswith('hết đường đã ghi'), ctx.du_lieu['lich_su'][0]
    assert xe.va_cham is None
    lech = math.hypot(xe.x - cuoi[0], xe.y - cuoi[1])
    print(f'phát lại thuần: lệch cuối {lech * 100:.1f} cm sau {g.dai:.2f} m')
    assert lech < 0.15, f'lệch cuối {lech:.2f} m'


def test_phat_lai_lai_bam_vach_khi_co_vach(tmp_path, monkeypatch):
    from test_gia_lap import chay_va_do, lech
    f, cuoi, g = _ghi_bang_s7(tmp_path)
    monkeypatch.setattr(cau_hinh, 'CALIB', tmp_path)
    xe = XeGiaLap(can_camera=True, nhanh=True)
    xe.x, xe.y, xe.yaw = 5.0, 1.6, math.pi / 2
    xe.t_xanh = 0.0
    ma, ctx, xe, ghi = chay_va_do(['s7_phat_lai'], xe=xe)
    assert ctx.du_lieu['lich_su'][0][1].startswith('hết đường đã ghi')
    assert xe.va_cham is None
    s0 = ghi[0, 0]
    rms, _ = lech(ghi, s0 + 0.3, s0 + 3.0, -0.20)
    assert rms < 0.05, f'có vạch thì phải bám như S7, RMS {rms * 100:.1f} cm'
