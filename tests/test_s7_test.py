"""S7 chính thức (11/10): dự đoán giữ độ cong + cổng vạch phải hợp lý + chạy chậm khi dự đoán."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import giai_doan.s7_bam_vach as S7  # noqa: E402
from chung import lane  # noqa: E402


def _cung_tron(R, x):
    """Vạch là cung tròn bán kính R quẹo TRÁI, xe nằm trên tiếp tuyến tại x = 0: y(x) = R - sqrt(R^2 - x^2)."""
    return R - math.sqrt(R * R - x * x)


def test_doi_duong_cong_giu_dung_cung_tron_con_keo_thang_thi_lech():
    R, X, ds, n = 1.0, 0.35, 0.01, 30            # chạy 30 cm trên cung tròn bán kính 1 m
    xs = np.linspace(0.0, 0.9, 40)              # khớp tới X_FIT_MAX như bộ dò
    c2, b2, a2 = np.polyfit(xs, [_cung_tron(R, x) for x in xs], 2)
    he = (a2, b2, c2)
    y_th, psi_th = S7.y_psi(he, X)
    for _ in range(n):                           # xe chạy đúng trên cung: dyaw = ds / R, nên vạch trong hệ xe KHÔNG đổi
        dyaw = ds / R
        he = S7.doi_duong_cong(he, ds, dyaw, 1.0)
        y_th, psi_th = y_th + ds * math.tan(psi_th) - dyaw * X, psi_th - dyaw     # công thức S7 gốc
    y_dung = _cung_tron(R, X)
    sai_cong = abs(S7.y_psi(he, X)[0] - y_dung)
    sai_thang = abs(y_th - y_dung)
    assert sai_cong < 0.005, f'giữ cong lệch {sai_cong * 100:.2f} cm'
    assert sai_thang > 0.02 and sai_thang > 3 * sai_cong, f'kéo thẳng {sai_thang * 100:.1f} cm phải tệ hơn giữ cong {sai_cong * 100:.1f} cm'


def _bo_do(monkeypatch, chuoi_he):
    """Bộ dò test với bước 1-6 giả: mỗi khung trả về hệ số trong chuoi_he (None = không fit)."""
    do = S7.tao_bo_do(gia_lap=True)
    no = iter(chuoi_he)
    monkeypatch.setattr(lane.DoVach, 'tach', lambda self, xam, debug=False: (next(no), 300, {}))
    return do


def test_cong_hop_ly_bo_vach_xa_va_du_doan_tiep(monkeypatch):
    phai = (-0.20, 0.0, 0.0)          # vạch phải cách 20 cm
    xa = (-0.60, 0.0, 0.0)            # vạch làn bên cách 60 cm: ngoài dải hợp lý
    do = _bo_do(monkeypatch, [phai, phai, xa, xa, None])
    xam = np.zeros((240, 320), np.uint8)
    k1 = do.xu_ly(xam, 0.0)
    k2 = do.xu_ly(xam, 0.1)
    assert k2.thay and k2.do_moi and abs(k2.khoang_cach_truoc - 0.20) < 0.01
    k3 = do.xu_ly(xam, 0.2, ds=0.01)
    assert k3.thay and not k3.do_moi, 'vạch 60 cm bị cổng bỏ, phải giữ dự đoán'
    assert 'ngoài dải hợp lý' in k3.ly_do and abs(k3.khoang_cach_truoc - 0.20) < 0.02
    k4 = do.xu_ly(xam, 0.3, ds=0.01)
    assert k4.thay and not k4.do_moi
    k5 = do.xu_ly(xam, 0.2 + S7.GIU_HOP_LY_S + 0.1)     # quá GIU_HOP_LY_S không có vạch hợp lý -> mất vạch
    assert not k5.thay


def test_chay_cham_khi_dang_du_doan():
    dk = S7.DieuKhienLan(gia_lap=True)
    kq = lane.KetQuaVach()
    kq.thay, kq.do_moi, kq.khoang_cach_truoc, kq.kappa = True, False, 0.20, 0.0
    dk.bam.cap_nhat = lambda ctx, debug=False: (kq, True)
    dk.bam.kq = kq
    ctx = SimpleNamespace(xe=SimpleNamespace(gio=lambda: 1.0), imu=SimpleNamespace(yaw=0.0))
    v, w, het = dk.buoc(ctx, 0.05)
    assert abs(v - S7.V_DU_DOAN) < 1e-9 and not het
    kq.do_moi = True                              # đo được thật: tốc độ bình thường
    v2, _, _ = dk.buoc(ctx, 0.05)
    assert v2 > S7.V_DU_DOAN


def test_s7_test_chay_gia_lap_sau_ham():
    """S7 test chạy trên sa bàn giả từ sau hầm, không va chạm, bám tâm làn phải như S7 gốc."""
    from test_gia_lap import chay_va_do, lech
    from chung.gia_lap import XeGiaLap
    rms = {}
    for ten in ('s7_bam_vach',):
        xe = XeGiaLap(can_camera=True, nhanh=True)
        xe.x, xe.y, xe.yaw = 5.0, 1.6, math.pi / 2
        xe.t_xanh = 0.0
        import giai_doan.s7_bam_vach as S7m
        cu, S7m.THOI_GIAN_TOI_DA = S7m.THOI_GIAN_TOI_DA, 60.0
        try:
            ma, ctx, xe, ghi = chay_va_do([ten], xe=xe)
        finally:
            S7m.THOI_GIAN_TOI_DA = cu
        assert xe.va_cham is None, f'{ten}: va chạm {xe.va_cham}'
        s0 = ghi[0, 0]
        rms[ten] = lech(ghi, s0 + 0.3, s0 + 4.5, -0.20)[0]
    print(f"RMS sau hầm: S7 {rms['s7_bam_vach'] * 100:.2f} cm")
    assert rms['s7_bam_vach'] < 0.05
