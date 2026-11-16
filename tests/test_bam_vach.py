"""Bám vạch: dự đoán giữ độ cong, cổng vạch phải hợp lý, chạy chậm khi dự đoán, nhảy vạch thì xoay tìm lại."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chung import bam_vach as BV  # noqa: E402
from chung import lane  # noqa: E402
from tu_hanh import hanh_vi  # noqa: E402


def _cung_tron(R, x):
    """Vạch là cung tròn bán kính R quẹo TRÁI, xe nằm trên tiếp tuyến tại x = 0: y(x) = R - sqrt(R^2 - x^2)."""
    return R - math.sqrt(R * R - x * x)


def test_doi_duong_cong_giu_dung_cung_tron_con_keo_thang_thi_lech():
    R, X, ds, n = 1.0, 0.35, 0.01, 30            # chạy 30 cm trên cung tròn bán kính 1 m
    xs = np.linspace(0.0, 0.9, 40)              # khớp tới X_FIT_MAX như bộ dò
    c2, b2, a2 = np.polyfit(xs, [_cung_tron(R, x) for x in xs], 2)
    he = (a2, b2, c2)
    y_th, psi_th = BV.y_psi(he, X)
    for _ in range(n):                           # xe chạy đúng trên cung: dyaw = ds / R, nên vạch trong hệ xe KHÔNG đổi
        dyaw = ds / R
        he = BV.doi_duong_cong(he, ds, dyaw, 1.0)
        y_th, psi_th = y_th + ds * math.tan(psi_th) - dyaw * X, psi_th - dyaw     # dời kiểu kéo thẳng (bản cũ)
    y_dung = _cung_tron(R, X)
    sai_cong = abs(BV.y_psi(he, X)[0] - y_dung)
    sai_thang = abs(y_th - y_dung)
    assert sai_cong < 0.005, f'giữ cong lệch {sai_cong * 100:.2f} cm'
    assert sai_thang > 0.02 and sai_thang > 3 * sai_cong, f'kéo thẳng {sai_thang * 100:.1f} cm phải tệ hơn giữ cong {sai_cong * 100:.1f} cm'


def _bo_do(monkeypatch, chuoi_he):
    """Bộ dò test với bước 1-6 giả: mỗi khung trả về hệ số trong chuoi_he (None = không fit)."""
    do = BV.tao_bo_do(gia_lap=True)
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
    k5 = do.xu_ly(xam, 0.2 + BV.GIU_HOP_LY_S + 0.1)     # quá GIU_HOP_LY_S không có vạch hợp lý -> mất vạch
    assert not k5.thay


def test_chay_cham_khi_dang_du_doan():
    dk = BV.DieuKhienLan(gia_lap=True)
    kq = lane.KetQuaVach()
    kq.thay, kq.do_moi, kq.khoang_cach_truoc, kq.kappa = True, False, 0.20, 0.0
    imu = SimpleNamespace(yaw=0.0, gz=0.0)
    v, w, het = dk.lenh(kq, imu, 0.05)
    assert abs(v - BV.V_DU_DOAN) < 1e-9 and not het
    kq.do_moi = True                              # đo được thật: tốc độ bình thường
    v2, _, _ = dk.lenh(kq, imu, 0.05)
    assert v2 > BV.V_DU_DOAN


# ---------------------------------------------- nhảy sang vạch khác (hành vi) ---
def kq(kc, nhay=0.0):
    k = lane.KetQuaVach()
    k.thay = k.do_moi = True
    k.khoang_cach = k.khoang_cach_truoc = kc
    k.nhay_m = nhay
    return k


class DkGia:
    """Bộ điều khiển giả: luôn chạy thẳng 0,10 m/s, để chỉ thử luật nhảy vạch / xoay tìm của hành vi."""

    def __init__(self):
        self.bam = SimpleNamespace(do=SimpleNamespace(dat_lai=lambda: None))

    def lenh(self, kq, imu, dt, v_bam, t=None):
        return 0.10, 0.0, False

    def dat_lai(self):
        pass

    def mo_ta(self):
        return ''


def chay(chuoi):
    hv = hanh_vi.BamVach(bam=None)
    hv.dk = DkGia()
    th = SimpleNamespace(t=0.0, imu=SimpleNamespace(yaw=0.0, gz=0.0), vach_moi=True)
    ra = []
    for k in chuoi:
        th.vach = k
        ra.append(hv.lenh(th, 0.05))
        th.imu.yaw += 0.05 * ra[-1][1]           # xe xoay theo lệnh
    return ra


def test_nhay_bat_hop_ly_thi_dung_xoay_phai_roi_bam_tiep():
    ra = chay([kq(0.20)] * 3 + [kq(0.60, nhay=0.40)] + [kq(0.55)] * 20 + [kq(0.21)] * 3 + [kq(0.20)] * 2)
    assert ra[2] == (0.10, 0.0)
    assert ra[3] == (0.0, 0.0), 'nhảy vạch: phải đứng yên ngay'
    xoay = ra[4:24]
    assert all(v == 0.0 for v, w in xoay) and any(w < 0 for v, w in xoay), 'phải đứng yên và xoay PHẢI'
    assert ra[-1] == (0.10, 0.0), 'thấy vạch phải hợp lý 2 khung -> bám tiếp'


def test_nhay_nho_hoac_van_hop_ly_thi_khong_dung():
    ra = chay([kq(0.20)] * 3 + [kq(0.25, nhay=0.05)] + [kq(0.24, nhay=0.15)] + [kq(0.22)] * 3)
    assert all(v > 0 for v, w in ra)
