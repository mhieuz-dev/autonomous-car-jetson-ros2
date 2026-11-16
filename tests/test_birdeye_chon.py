"""Bước 4b: tự chọn lần dùng, vùng nhìn, độ phân giải bird's eye."""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chung import birdeye_chon as B  # noqa: E402
from chung import cau_hinh  # noqa: E402
from chung.camera import Calib, doc_bev  # noqa: E402
from chung.camera_pose import giai_pose, pose_theo_xe  # noqa: E402
from test_calib import CAM, K, D, _anh_ban_co_tren_san, _nap_cong_cu  # noqa: E402


@pytest.fixture
def moi(tmp_path, monkeypatch):
    """calib/ và calib_moi/ tạm; 2 ứng viên (bàn cờ ở 2 chỗ) như birdeye_banco lưu."""
    (tmp_path / 'calib').mkdir()
    (tmp_path / 'calib/camera.yaml').write_text(yaml.safe_dump(CAM))
    monkeypatch.setattr(cau_hinh, 'CALIB', tmp_path / 'calib')
    monkeypatch.setattr(cau_hinh, 'CALIB_MOI', tmp_path / 'calib_moi')
    bb = _nap_cong_cu()
    for x0 in (0.28, 0.36):
        g = _anh_ban_co_tren_san(x0=x0)
        g = np.clip(g + np.random.default_rng(0).normal(0, 2, g.shape), 0, 255).astype(np.uint8)
        cor, sz = bb.do_ban_co(g, 10, 7)
        res = giai_pose(cor, 10, 7, 0.019, K, D)
        rv, tv = pose_theo_xe(res, 0.06)
        B.luu_ung_vien(res, rv, tv, g, [('RMS', '0.2 px', True), ('roll', '0.8', True)], 0.06, 0.0, sz, 19.0, 1.0, 1.0)
    return tmp_path


def test_liet_ke_va_trung_vi(moi):
    ds = B.liet_ke()
    assert [u['n'] for u in ds] == [1, 2] and all(u['dat'] == 2 for u in ds)
    assert ds[0]['cao_m'] == pytest.approx(0.11, abs=0.003)
    rv, tv = B.tu_the_chon([1, 2])
    c = Calib.tu_pose(K, D, rv, tv)
    assert (c.cao_m, c.truoc_tam) == pytest.approx((0.11, 0.06), abs=0.003)


def test_de_xuat_hop_ly(moi):
    d = B.de_xuat([1])
    assert 0.1 < d['x_thay_gan_nhat'] < 0.4 and d['x_gan'] > d['x_thay_gan_nhat']
    assert 0.5 <= d['x_xa'] <= 1.5
    mm = [b['mm_hang'] for b in d['bang']]
    assert mm == sorted(mm)                       # càng xa, 1 hàng ảnh càng ứng với nhiều mm sàn


def test_xem_thu_va_tu_choi_so_sai(moi):
    p = {'x_gan': 0.25, 'x_xa': 0.9, 'y_trai': 0.4, 'y_phai': -0.4, 'mm_px': 5}
    img = B.ve_thu([1], p)
    assert img.ndim == 3 and img.shape[1] >= 400
    for sai in ({**p, 'x_xa': 0.2}, {**p, 'y_trai': -0.6}, {**p, 'mm_px': 0.1}, {**p, 'mm_px': 50}):
        with pytest.raises(ValueError):
            B.ve_thu([1], sai)
    with pytest.raises(ValueError, match='chưa chọn'):
        B.ve_thu([], p)


def test_luu_lua_chon_roi_code_lai_doc_duoc(moi):
    p = {'x_gan': 0.25, 'x_xa': 0.9, 'y_trai': 0.4, 'y_phai': -0.4, 'mm_px': 4}
    B.ghi_lua_chon([1, 2], p)
    for f in ('ngoai.yaml', 'bev.yaml'):
        shutil.copy(moi / 'calib_moi' / f, moi / 'calib' / f)
    assert Calib().cao_m == pytest.approx(0.11, abs=0.003)          # calib/ đầy đủ -> đọc được
    assert doc_bev() == {'x_gan': 0.25, 'x_xa': 0.9, 'y_trai': 0.4, 'y_phai': -0.4, 'm_px': 0.004}
    from giai_doan.s7_bam_vach import tao_bo_do
    do = tao_bo_do()
    assert (do.p.X_GAN, do.p.X_XA, do.be.m) == (0.25, 0.9, 0.004)   # S7 dùng đúng vùng nhìn đã chọn
    assert do.be.y_trai == 0.4 and do.be.y_phai == -0.4
