"""Phần nhận biết dốc và thứ tự ưu tiên hành vi, không cần sa bàn giả."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tu_hanh import tham_so as TS  # noqa: E402
from tu_hanh.hanh_vi import ChoNguoi, DungBienStop, DungDen, QuaDoc, QuaHam, RaHam  # noqa: E402
from tu_hanh.nhan_biet import TheoDoiDoc, TinhHuong  # noqa: E402


def chay_doc(chuoi, dt=0.05):
    """chuoi = [(số giây, pitch độ)] -> các lần đổi trạng thái."""
    d, qd, doi = TheoDoiDoc(), 1.0, []
    for giay, p in chuoi:
        for _ in range(int(round(giay / dt))):
            qd += 0.3 * dt
            kq = d.cap_nhat(p, 0.0, qd, dt)
            if kq:
                doi.append(kq[1])
    return doi


def test_doc_len_dinh_xuong_het():
    assert chay_doc([(1, 0), (2, 5), (1, 0), (1.5, -4), (1, 0)]) == ['len', 'dinh', 'xuong', 'phang']


def test_rung_ngan_khong_doi_trang_thai():
    # pitch vọt +5 độ 0,5 s (ngắn hơn DOC_LEN_GIAY), rồi -3 độ 0,3 s: chỉ là rung
    assert chay_doc([(1, 0), (0.5, 5), (1, 0), (0.3, -3), (1, 0)]) == []


def test_doc_xuong_khong_can_len_truoc():
    assert chay_doc([(1, 0), (1.5, -4), (1, 0)]) == ['xuong', 'phang']


def test_dinh_dai_bat_thuong_thi_ve_phang():
    # lên rồi phẳng mãi (sàn trên cao): quá DINH_DAI_TOI_DA_M thì coi như phẳng, xe bám vạch lại được
    giay = TS.DINH_DAI_TOI_DA_M / 0.3 + 1.0
    assert chay_doc([(1, 0), (2, 5), (giay, 0)]) == ['len', 'dinh', 'phang']


def test_giu_huong_luc_vao_doc():
    d, qd = TheoDoiDoc(), 1.0
    for k in range(40):
        yaw = 0.10 if k < 2 else 0.30          # chân chạm dốc lúc hướng 0,10; giữa lúc chờ xác nhận thì xe lắc
        qd += 0.01
        d.cap_nhat(5.0, yaw, qd, 0.05)
    assert d.trang_thai == 'len' and abs(d.yaw_doc - 0.10) < 1e-9


def _th(**kv):
    th = TinhHuong()
    th.t = 10.0
    th.imu = SimpleNamespace(yaw=0.0, gz=0.0)
    for k, v in kv.items():
        setattr(th, k, v)
    return th


def test_tren_doc_khong_dung_vi_den_hay_bien():
    th = _th(doc='len', den='do', den_cao=90.0,
             bien=[{'ten': 'stop', 'tin': 0.9, 'hop': [0, 0, 50, 100]}])
    assert not DungDen().muon(th)
    assert not DungBienStop().muon(th)
    assert QuaDoc().muon(th)


def test_cho_nguoi_them_1_giay_sau_khi_trong():
    hv = ChoNguoi()
    assert hv.muon(_th(nguoi=True, nguoi_cach=0.4, t=10.0))
    assert hv.muon(_th(t=10.5))
    assert not hv.muon(_th(t=10.0 + TS.THOI_GIAN_TRONG + 0.05))


def test_trong_ham_va_ra_ham():
    assert QuaHam().muon(_th(trong_ham=True))
    ra = RaHam()
    assert ra.muon(_th(t_ra_ham=10.0, vach_lien=99))          # chưa đủ RA_HAM_TROI_MIN thì vẫn giữ hướng
    assert not ra.muon(_th(t=10.0 + TS.RA_HAM_TROI_MIN + 0.1, t_ra_ham=10.0, vach_lien=TS.RA_HAM_SO_KHUNG_VACH))
    assert not ra.muon(_th(t=10.0 + TS.RA_HAM_TROI_MAX + 0.1, t_ra_ham=10.0, vach_lien=0))
