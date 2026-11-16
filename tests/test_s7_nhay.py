"""S7: nhảy sang vạch khác bất hợp lý -> đứng yên, xoay phải tới khi thấy vạch phải hợp lý rồi bám tiếp."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import giai_doan.s7_bam_vach as S7  # noqa: E402
from chung.lane import KetQuaVach  # noqa: E402


def kq(kc, nhay=0.0):
    k = KetQuaVach()
    k.thay = k.do_moi = True
    k.khoang_cach = k.khoang_cach_truoc = kc
    k.nhay_m = nhay
    return k


class DkGia:
    def __init__(self):
        self.so_khung = 0
        self.bam = SimpleNamespace(kq=kq(0.20), do=SimpleNamespace(dat_lai=lambda: None))

    def buoc(self, ctx, dt, v, debug=False):
        self.so_khung += 1
        return 0.10, 0.0, False

    def dat_lai(self):
        pass


def chay(chuoi):
    gd = S7.GiaiDoan()
    gd.dk = DkGia()
    ctx = SimpleNamespace(imu=SimpleNamespace(yaw=0.0), xem=None)
    ra = []
    for k in chuoi:
        gd.dk.bam.kq = k
        ra.append(gd.buoc(ctx, 0.05))
        ctx.imu.yaw += 0.05 * ra[-1][1]          # xe xoay theo lệnh
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
