"""Xe tự hành trên sa bàn giả: qua hết sa bàn, đổi chỗ dốc và hầm, người đi bộ, đèn, biển STOP."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chung import cau_hinh, khung  # noqa: E402
from chung.gia_lap import XeGiaLap  # noqa: E402
from tu_hanh import tham_so as TS  # noqa: E402


def chay(xe=None, toi_da=120.0, **sa_ban):
    """Chạy tới cuối đường giả (hoặc hết toi_da giây). Trả (ctx, xe, vết [t, s, lệch ngang, v])."""
    xe = xe or XeGiaLap(can_camera=True, nhanh=True, **sa_ban)
    vet = []
    gui = xe.gui

    def ghi(v, w):
        gui(v, w)
        i = int(np.argmin(np.hypot(xe.tim[:, 0] - xe.x, xe.tim[:, 1] - xe.y)))
        n = xe.phap[i]
        vet.append((xe.t, xe.s[i], (xe.x - xe.tim[i, 0]) * n[0] + (xe.y - xe.tim[i, 1]) * n[1], v))
        if xe.s[i] >= xe.s[-1] - 0.3:      # hết đường giả: dừng như người bấm Ctrl+C
            khung._DUNG['co'] = True

    xe.gui = ghi
    cu, TS.THOI_GIAN_TOI_DA = TS.THOI_GIAN_TOI_DA, toi_da
    try:
        _, ctx = khung.chay('tu_hanh.lai_xe', gia_lap=True, ghi_bag=False, nhanh=True, xem_cong=None, xe=xe)
    finally:
        TS.THOI_GIAN_TOI_DA = cu
    return ctx, xe, np.array(vet)


def hanh_vi(ctx):
    return [ten for _, ten in ctx.du_lieu['hanh_vi']]


def doc(ctx):
    return [tt for _, tt in ctx.du_lieu['doc']]


def lech(vet, s0, s1, dich):
    m = (vet[:, 1] >= s0) & (vet[:, 1] < s1)
    e = vet[m, 2] - dich
    return float(np.sqrt(np.mean(e ** 2))), float(np.abs(e).max())


@pytest.fixture(scope='module')
def ca_sa_ban():
    return chay()


def test_qua_het_sa_ban_mac_dinh(ca_sa_ban):
    ctx, xe, vet = ca_sa_ban
    hv = hanh_vi(ctx)
    assert xe.va_cham is None, f'va chạm {xe.va_cham}'
    assert vet[-1, 1] > xe.s[-1] - 0.4, 'phải chạy tới cuối đường giả'
    assert hv.index('qua_doc') < hv.index('qua_ham') < hv.index('ra_ham'), hv
    assert doc(ctx) == ['phang', 'len', 'dinh', 'xuong', 'phang']
    rms_truoc, _ = lech(vet, 0.6, 1.9, -0.20)      # bám vạch trước dốc: tâm làn phải
    _, max_ham = lech(vet, 4.1, 6.4, 0.0)          # trong hầm: giữa 2 tường
    rms_sau, _ = lech(vet, 7.5, 12.5, -0.20)       # bám vạch sau hầm
    print(f'RMS trước dốc {rms_truoc:.3f} | hầm max {max_ham:.3f} | sau hầm {rms_sau:.3f}')
    assert rms_truoc < 0.02
    assert max_ham < 0.25            # hầm rộng 0.77, xe rộng ~0.16: lệch 0.25 vẫn chưa chạm
    assert rms_sau < 0.03


def test_dung_cho_nguoi_di_bo(ca_sa_ban):
    ctx, xe, vet = ca_sa_ban
    assert 'cho_nguoi' in hanh_vi(ctx)
    dung = vet[(vet[:, 0] > 4.0) & (vet[:, 0] < 7.0)]          # người đứng trên làn từ giây 3 tới 7
    assert np.all(np.abs(dung[:, 3]) < 1e-9), 'có người trước mũi thì phải đứng yên'
    x_mui = dung[:, 1].max() - 0.5 + cau_hinh.LIDAR_TOI_CAN_M  # s = x + 0.5 trên đoạn thẳng đầu
    assert 0.9 - 0.06 - x_mui > 0.10, 'mũi xe phải cách người hơn 10 cm'
    assert np.any(vet[(vet[:, 0] > 8.5) & (vet[:, 0] < 10.0), 3] > 0.05), 'người đi rồi phải chạy tiếp'


def test_gap_ham_truoc_doc_van_qua_duoc():
    """Bản thi chạy theo thứ tự cố định nên không qua được sa bàn này. Dốc dời ra đoạn thẳng cuối, sau hầm."""
    ctx, xe, vet = chay(doan_doc=[(10.8, 11.4, 6.0), (12.0, 12.5, -5.0)], co_nguoi=False)
    hv = hanh_vi(ctx)
    assert xe.va_cham is None, f'va chạm {xe.va_cham}'
    assert vet[-1, 1] > xe.s[-1] - 0.4
    assert hv.index('qua_ham') < hv.index('qua_doc'), hv
    assert doc(ctx) == ['phang', 'len', 'dinh', 'xuong', 'phang']
    tren_doc = vet[(vet[:, 1] >= 10.8) & (vet[:, 1] < 12.5)]
    assert np.all(tren_doc[:, 3] > 0.0), 'không được đứng yên trên dốc'


def test_doc_that_3_6_do_va_xe_ngoc_mui():
    """Dốc thật lên +3,6 xuống -3,1 độ, xe ngóc mũi 30 x v^2 độ: vẫn nhận đủ lên, đỉnh, xuống."""
    ctx, xe, _ = chay(toi_da=30.0, doc_len=3.6, doc_xuong=-3.1, co_nguoi=False)
    assert doc(ctx)[:4] == ['phang', 'len', 'dinh', 'xuong'], ctx.du_lieu['doc']
    assert xe.va_cham is None


def test_san_phang_chay_nhanh_khong_bi_bao_la_doc(monkeypatch):
    """Bản thi (S1, ngưỡng 5 độ chưa trừ ngóc mũi) trên sàn PHẲNG này, 0,5 m/s: pitch +6,8 độ, báo tới chân dốc
    sau 0,64 m. Pitch đã trừ phần ngóc mũi thì không còn báo nhầm."""
    monkeypatch.setattr(TS, 'V_BAM_VACH', 0.5)
    ctx, xe, vet = chay(toi_da=15.0, doan_doc=[], ham=None, co_nguoi=False)
    assert vet[:, 3].max() > 0.45, 'phải chạy được tới 0,5 m/s'
    assert doc(ctx) == ['phang'], ctx.du_lieu['doc']
    assert 'qua_doc' not in hanh_vi(ctx)


def _xe_sau_ham(den_lich):
    xe = XeGiaLap(can_camera=True, nhanh=True, co_nguoi=False)
    xe.x, xe.y, xe.yaw = 5.0, 1.6, math.pi / 2             # đặt xe giữa đường, sau hầm, giữa làn phải
    xe.den_lich = den_lich
    return xe


def doan_dung(vet, tu):
    """Các mốc giờ của lần đứng yên đầu tiên sau giờ `tu`."""
    dung = []
    for t, _, _, v in vet:
        if t > tu and abs(v) < 1e-9:
            dung.append(t)
        elif dung:
            break
    return dung


def test_dat_xe_giua_duong_thi_bam_vach_ngay():
    ctx, xe, vet = chay(xe=_xe_sau_ham([]), toi_da=10.0)
    assert hanh_vi(ctx) == ['bam_vach']
    assert xe.va_cham is None


def test_dung_o_bien_stop_roi_chay_tiep():
    ctx, xe, vet = chay(xe=_xe_sau_ham([(5.0, 6.0, 'stop', TS.STOP_CAO_MIN_PX + 20)]), toi_da=10.0)
    dung = doan_dung(vet, 4.9)
    assert 'dung_stop' in hanh_vi(ctx)
    assert dung, 'thấy STOP đủ gần thì phải đứng yên'
    assert TS.STOP_DUNG_GIAY - 0.3 <= dung[-1] - dung[0] <= TS.STOP_DUNG_GIAY + 0.4
    assert np.any(vet[vet[:, 0] > dung[-1] + 0.5, 3] > 0.05), 'dừng xong phải chạy tiếp'


def test_bien_stop_con_xa_thi_khong_dung():
    ctx, _, vet = chay(xe=_xe_sau_ham([(5.0, 6.0, 'stop', TS.STOP_CAO_MIN_PX - 20)]), toi_da=9.0)
    assert 'dung_stop' not in hanh_vi(ctx)
    assert np.all(vet[(vet[:, 0] >= 5.0) & (vet[:, 0] <= 8.5), 3] > 0.0)


def test_den_do_thi_dung_den_xanh_thi_di():
    lich = [(4.0, 8.0, 'den do', TS.DEN_CAO_MIN_PX + 20), (8.0, 14.0, 'den xanh', TS.DEN_CAO_MIN_PX + 20)]
    ctx, _, vet = chay(xe=_xe_sau_ham(lich), toi_da=14.0)
    dung = doan_dung(vet, 3.9)
    assert 'dung_den' in hanh_vi(ctx)
    assert dung and dung[0] < 4.2 and dung[-1] > 7.9, 'đèn đỏ: đứng yên tới khi đèn xanh'
    assert np.any(vet[(vet[:, 0] > 9.5) & (vet[:, 0] < 11.0), 3] > 0.05), 'đèn xanh đủ lâu thì chạy'


def test_den_do_mai_thi_cho_toi_da_roi_di_tiep():
    lich = [(4.0, 60.0, 'den do', TS.DEN_CAO_MIN_PX + 20)]
    _, _, vet = chay(xe=_xe_sau_ham(lich), toi_da=4.0 + TS.DEN_CHO_TOI_DA + 3.0)
    dung = doan_dung(vet, 3.9)
    assert TS.DEN_CHO_TOI_DA - 0.2 <= dung[-1] - dung[0] <= TS.DEN_CHO_TOI_DA + 0.3
    sau = vet[(vet[:, 0] > dung[-1] + 1.0) & (vet[:, 0] < vet[-1, 0] - 0.5)]      # bỏ lệnh dừng lúc hết giờ chạy
    assert len(sau) and np.all(sau[:, 3] > 0.0), 'hết giờ chờ thì đi tiếp, không dừng lại ngay'
