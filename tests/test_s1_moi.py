"""S1 mới: bám vạch bird's eye, nhận đèn khi đủ lớn (đỏ/vàng dừng, xanh đi), nhớ hướng xe cho S2."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chung import khung  # noqa: E402
from chung.den_onnx import mau_den_lon  # noqa: E402
from chung.gia_lap import XeGiaLap  # noqa: E402


@pytest.fixture(autouse=True)
def _toc_do_cua_kich_ban(monkeypatch):
    """Mốc thời gian đèn/biển trong các kịch bản dưới đây được viết cho V_CHAY = 0,30 m/s.
    Thông số chạy thật đã chỉnh lên 0,5 m/s, nên ghim lại tốc độ của kịch bản."""
    import giai_doan.s1_xuat_phat as S1
    monkeypatch.setattr(S1, 'V_CHAY', 0.30)


def _chay(den_lich, ds=('s0_den_xanh', 's1_xuat_phat'), t_s1=None):
    xe = XeGiaLap(can_camera=True, nhanh=True, co_nguoi=False, doc_len=3.6, doc_xuong=-3.1, he_so_toc_do=30.0)
    xe.den_lich = den_lich
    vi_tri = []
    goc = xe.gui
    xe.gui = lambda v, w: (vi_tri.append((xe.t, xe.x, v)), goc(v, w))[1]
    ma, ctx = khung.chay(list(ds), gia_lap=True, nhanh=True, xem_cong=None, xe=xe)
    return ctx, xe, vi_tri


def test_mau_den_lon_uu_tien_do_va_bo_den_nho():
    d = lambda ten, cao, tin=0.9: {'ten': ten, 'tin': tin, 'hop': [10, 10, 40, 10 + cao]}  # noqa: E731
    assert mau_den_lon([d('den xanh', 60)], 40)[0] == 'xanh'
    assert mau_den_lon([d('den xanh', 60), d('den do', 50)], 40)[0] == 'do'      # đỏ thắng xanh
    assert mau_den_lon([d('den do', 20)], 40)[0] is None                         # đèn còn nhỏ = còn xa
    assert mau_den_lon([d('den do', 20)], 40)[1] == 20                           # nhưng vẫn báo cao để chỉnh ngưỡng
    assert mau_den_lon([d('den do', 80, 0.2)], 40)[0] is None                    # độ tin thấp
    assert mau_den_lon([d('stop', 80)], 40)[0] is None                           # không phải đèn


def test_den_do_lon_thi_dung_roi_xanh_thi_di():
    ctx, xe, vt = _chay([(5.0, 10.0, 'den do', 60), (10.0, 99.0, 'den xanh', 60)])
    dung = [x for t, x, v in vt if 5.6 <= t <= 9.5]
    assert dung and max(dung) - min(dung) < 0.05, 'xe phải đứng yên lúc đèn đỏ lớn'
    ly = ctx.du_lieu['lich_su'][-1]
    assert ly[0] == 's1_xuat_phat' and ly[1].startswith('tới chân dốc')          # xanh rồi vẫn tới được dốc


def test_den_do_con_nho_thi_khong_dung():
    ctx, xe, vt = _chay([(5.0, 12.0, 'den do', 20)])                             # cao 20 px < 40
    dung = [x for t, x, v in vt if 5.6 <= t <= 9.5]
    assert max(dung) - min(dung) > 0.4, 'đèn còn nhỏ (xa) thì xe không được dừng'


def test_s1_nho_huong_xe_cho_s2():
    ctx, xe, vt = _chay([])
    assert 'yaw_doc' in ctx.du_lieu and abs(ctx.du_lieu['yaw_doc']) < 0.2       # rad, xe đi gần thẳng


def test_s2_s3_s4_s5_giu_huong_da_nho():
    from giai_doan import s2_len_doc, s3_dinh_doc, s4_xuong_doc, s5_ham
    ctx = SimpleNamespace(du_lieu={'yaw_doc': 0.21})
    for m in (s2_len_doc, s3_dinh_doc, s4_xuong_doc, s5_ham):
        gd = m.GiaiDoan()
        gd.bat_dau(ctx)
        assert abs(gd.yaw - 0.21) < 1e-9, m.__name__


def test_roi_chi_nhan_nua_duoi_va_khung_ve_toa_do_goc(tmp_path, monkeypatch):
    import numpy as np
    from chung import den_onnx as D
    monkeypatch.setattr(D, 'ROI_FILE', tmp_path / 'roi.yaml')
    assert D.doc_roi() == (0, 240, 640, 480)                    # mặc định nửa dưới
    D.ghi_roi(200, 460)
    assert D.doc_roi() == (0, 200, 640, 460)
    try:
        D.ghi_roi(300, 100)
        assert False
    except ValueError:
        pass

    class Gia:
        def nhan(self, bgr):
            assert bgr.shape[:2] == (260, 640)                  # model chỉ thấy vùng ROI
            return [{'ten': 'stop', 'tin': 0.9, 'hop': [10.0, 5.0, 50.0, 50.0]}], 1.0
    kq, _ = D.nhan_trong_roi(Gia(), np.zeros((480, 640, 3), np.uint8), D.doc_roi())
    assert kq[0]['hop'] == [10.0, 205.0, 50.0, 250.0]           # cộng lại y1 = 200


def test_bien_stop_lon_va_dung_roi_di_tiep():
    from chung.den_onnx import bien_lon
    d = lambda ten, cao, tin=0.9: {'ten': ten, 'tin': tin, 'hop': [10, 10, 40, 10 + cao]}  # noqa: E731
    assert bien_lon([d('stop', 60)], 'stop', 40) == (True, 60)
    assert bien_lon([d('stop', 20)], 'stop', 40) == (False, 20)       # còn xa
    assert bien_lon([d('stop', 60, 0.3)], 'stop', 40, 0.6)[0] is False  # độ tin thấp
    assert bien_lon([d('re trai', 60)], 'stop', 40)[0] is False       # biển khác không dừng


def test_s1_dung_o_stop_roi_di_tiep():
    ctx, xe, vt = _chay([(5.0, 7.0, 'stop', 90)], ds=('s1_xuat_phat',))
    dung = []
    for t, x, v in vt:                       # chỉ lấy đoạn đứng yên LIỀN MẠCH đầu tiên sau t = 4
        if t > 4.0 and abs(v) < 1e-6:
            dung.append(t)
        elif dung:
            break
    import giai_doan.s1_xuat_phat as S1
    assert dung and S1.STOP_DUNG_GIAY - 0.5 <= dung[-1] - dung[0] <= S1.STOP_DUNG_GIAY + 0.6, \
        'xe phải đứng khoảng STOP_DUNG_GIAY rồi đi tiếp'
    assert any(t > dung[-1] and v > 0.2 for t, x, v in vt), 'sau khi dừng xong xe phải chạy tiếp'


def test_ve_ket_qua_chi_vung_bo_phan_tren():
    import numpy as np
    from chung.den_onnx import ve_ket_qua
    img = np.zeros((480, 640, 3), np.uint8)
    assert ve_ket_qua(img, [], vung=(0, 240, 640, 480), chi_vung=True).shape[:2] == (240, 640)
    assert ve_ket_qua(img, [], vung=(0, 240, 640, 480)).shape[:2] == (480, 640)


def test_s1_chi_nhan_stop_re_trai_re_phai():
    ctx, xe, vt = _chay([(3.0, 4.0, 're trai', 60), (4.5, 5.5, 're phai', 20), (6.0, 7.0, 'bien co doc', 80),
                         (7.5, 8.5, 're phai', 60)], ds=('s1_xuat_phat',))
    assert [h for _, h in ctx.du_lieu.get('bien_re', [])] == ['trai', 'phai'], 'rẽ phải nhỏ (20 px) và biển khác phải bị bỏ'
    assert min(v for t, x, v in vt[:-10] if t > 3.0) > 0.2, 'biển rẽ chưa có lệnh nên xe không được dừng'
