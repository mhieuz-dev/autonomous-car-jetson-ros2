"""Dốc thật LÊN +3,6, XUỐNG -3,1 độ, CỘNG xe ngóc mũi theo tốc độ (30 x v^2, đo từ log chạy thật 10/10): sàn phẳng\nkhông được báo là dốc, dốc thật vẫn phải chuyển đúng."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chung import khung  # noqa: E402
from chung.gia_lap import XeGiaLap  # noqa: E402


def test_doc_that_3_6_do_qua_duoc_toi_ham():
    xe = XeGiaLap(can_camera=True, nhanh=True, doc_len=3.6, doc_xuong=-3.1, he_so_toc_do=30.0)
    ma, ctx = khung.chay(['s0_den_xanh', 's1_xuat_phat', 's2_len_doc', 's3_dinh_doc', 's4_xuong_doc', 's5_ham'],
                         gia_lap=True, nhanh=True, xem_cong=None, xe=xe)
    ten = [g for g, _, _ in ctx.du_lieu['lich_su']]
    ly_do = {g: r for g, r, _ in ctx.du_lieu['lich_su']}
    assert ten[:3] == ['s0_den_xanh', 's1_xuat_phat', 's2_len_doc'], ly_do
    assert ly_do['s1_xuat_phat'].startswith('tới chân dốc')
    assert ly_do['s2_len_doc'].startswith('lên tới đỉnh')
    assert 'bắt đầu xuống dốc' in ly_do.get('s3_dinh_doc', '') or 'hầm' in ly_do.get('s3_dinh_doc', ''), ly_do
    assert xe.va_cham is None


def test_san_phang_chay_nhanh_khong_bi_bao_la_doc(monkeypatch):
    """Lỗi 10/10: sàn PHẲNG, xe 0,30 m/s ngóc mũi +2,7 độ, S1 báo 'tới chân dốc' sau 0,44 m."""
    import giai_doan.s1_xuat_phat as S1
    monkeypatch.setattr(S1, 'THOI_GIAN_TOI_DA', 14.0)
    monkeypatch.setattr(S1, 'V_CHAY', 0.30)            # hệ số ngóc mũi 30 x v^2 được đo ở 0,30 m/s
    xe = XeGiaLap(can_camera=True, nhanh=True, co_nguoi=False, doc_len=0.0, doc_xuong=0.0, he_so_toc_do=40.0)
    ma, ctx = khung.chay(['s0_den_xanh', 's1_xuat_phat'], gia_lap=True, nhanh=True, xem_cong=None, xe=xe)
    ly_do = ctx.du_lieu['lich_su'][-1][1]
    assert 'chân dốc' not in ly_do, ly_do              # phải hết giờ chứ không được báo dốc
