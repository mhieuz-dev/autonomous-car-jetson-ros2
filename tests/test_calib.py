"""Calib camera mới: không có số mặc định, tư thế lấy từ bàn cờ nằm trên sàn."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chung.camera import BirdEye, Calib, ChuaCalib, tu_pitch  # noqa: E402
from chung.camera_pose import giai_pose, pose_theo_xe  # noqa: E402

CAM = {'image_width': 640, 'image_height': 480,
       'camera_matrix': {'data': [552.83, 0, 311.43, 0, 551.9, 239.42, 0, 0, 1]},
       'distortion_coefficients': {'data': [0.2036, -1.1217, -0.0017, 0.0014, 1.745]}}
K = np.array(CAM['camera_matrix']['data']).reshape(3, 3)
D = np.array(CAM['distortion_coefficients']['data'])


def _dir(tmp_path, ngoai=None, cam=True):
    if cam:
        (tmp_path / 'camera.yaml').write_text(yaml.safe_dump(CAM))
    if ngoai is not None:
        (tmp_path / 'ngoai.yaml').write_text(yaml.safe_dump(ngoai))
    return tmp_path


def test_chua_calib_thi_bao_loi(tmp_path):
    with pytest.raises(ChuaCalib, match='CHƯA CALIB'):
        Calib(_dir(tmp_path, cam=False))


def test_chua_do_birdeye_thi_bao_loi_khong_dung_so_cu(tmp_path):
    with pytest.raises(ChuaCalib, match="BIRD'S EYE"):
        Calib(_dir(tmp_path, {'cao_m': 0.11}))        # có số tay nhưng không có tư thế
    with pytest.raises(ChuaCalib, match="BIRD'S EYE"):
        Calib(_dir(tmp_path))                          # không có ngoai.yaml


def _board(pitch_do, roll_do, x0, cao=0.11, truoc=0.05, nhieu=0.3, cols=10, rows=7, o=0.019):
    rv, tv = tu_pitch(np.radians(pitch_do), cao, truoc)
    R = cv2.Rodrigues(np.array([0, 0, np.radians(roll_do)]))[0] @ cv2.Rodrigues(rv)[0]
    tv = cv2.Rodrigues(np.array([0, 0, np.radians(roll_do)]))[0] @ tv
    rv = cv2.Rodrigues(R)[0].ravel()
    ii, jj = np.mgrid[0:rows, 0:cols]
    Pw = np.column_stack([x0 + ii.ravel() * o, (jj.ravel() - 4.5) * o, np.zeros(rows * cols)])
    px = cv2.projectPoints(Pw, rv, tv, K, D)[0].reshape(-1, 2)
    assert ((px[:, 0] > 0) & (px[:, 0] < 640) & (px[:, 1] > 0) & (px[:, 1] < 480)).all()
    return rv, tv, px + np.random.default_rng(1).normal(0, nhieu, px.shape), Pw


@pytest.mark.parametrize('pitch,roll,x0', [(10, 1.2, 0.25), (18, -0.8, 0.22), (5, 0.0, 0.30)])
def test_pose_tu_ban_co_tren_san(pitch, roll, x0):
    rv, tv, px, Pw = _board(pitch, roll, x0)
    r = giai_pose(px, 10, 7, 0.019, K, D)
    assert r['cao'] == pytest.approx(0.11, abs=0.002)            # chiều cao camera
    assert r['pitch'] == pytest.approx(pitch, abs=0.3)           # độ chúi
    assert r['roll'] == pytest.approx(roll, abs=0.3)
    assert abs(r['yaw']) < 0.3 and r['rms'] < 1.0
    # điểm sàn (hệ xe) qua calib dựng từ bàn cờ phải rơi đúng pixel thật
    rvx, tvx = pose_theo_xe(r, 0.05)
    c = Calib.tu_pose(K, D, rvx, tvx)
    assert (c.cao_m, c.truoc_tam) == pytest.approx((r['cao'], 0.05), abs=1e-6)
    thuc = cv2.projectPoints(np.array([[x0 + 3 * 0.019, 0.0, 0.0]]), rv, tv, K, D)[0].ravel()
    toi = np.array(c.mat_dat_sang_goc(np.array([x0 + 3 * 0.019]), np.array([0.0]))).ravel()
    assert np.abs(thuc - toi).max() < 1.0                         # sai < 1 pixel
    # và chiều ngược lại: pixel -> mặt sàn
    x, y = c.raw_sang_mat_dat(thuc[0], thuc[1])
    assert (x[0], y[0]) == pytest.approx((x0 + 3 * 0.019, 0.0), abs=0.002)


def test_birdeye_ban_co_thanh_o_vuong():
    """Trên bird's eye 1 mm/px các ô cờ 19 mm phải vuông và cạnh ~19 px."""
    x0_ban = 0.25
    rv, tv, px, Pw = _board(12, 0.5, x0_ban, truoc=0.0)   # gốc ngay dưới camera, như lúc calib
    r = giai_pose(px, 10, 7, 0.019, K, D)
    c = Calib.tu_pose(K, D, *pose_theo_xe(r, 0.0))
    from chung.camera_pose import goc_ban_co_theo_xe
    x, y = goc_ban_co_theo_xe(r)
    be = BirdEye(c, x.min() - 0.03, x.max() + 0.03, y.max() + 0.03, y.min() - 0.03, 0.001, ti_le_anh=1.0)
    # ảnh giả: dán lại bàn cờ thật bằng cách chiếu ô đen/trắng theo toạ độ sàn
    gx, gy = np.meshgrid(np.arange(640), np.arange(480))
    xs, ys = c.raw_sang_mat_dat(gx.ravel(), gy.ravel())
    o = 0.019
    ci = np.floor((np.nan_to_num(ys, nan=99) + 4.5 * o) / o).astype(int)      # cột ô: biên đúng tại các góc trong
    ri = np.floor((np.nan_to_num(xs, nan=99) - x0_ban) / o).astype(int)
    trong = (ci >= -1) & (ci <= 9) & (ri >= -1) & (ri <= 6)                   # 11 x 8 ô, ngoài là viền trắng
    img = np.where(trong, ((ci + ri) % 2) * 255, 255).astype(np.uint8).reshape(480, 640)
    bev = be.chieu(img)
    bev = cv2.copyMakeBorder(bev, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    ok, cor = cv2.findChessboardCorners(bev, (10, 7), cv2.CALIB_CB_ADAPTIVE_THRESH)
    assert ok
    cor = cor.reshape(7, 10, 2)
    ngang = np.linalg.norm(np.diff(cor, axis=1), axis=2).mean()
    doc = np.linalg.norm(np.diff(cor, axis=0), axis=2).mean()
    assert ngang == pytest.approx(19.0, rel=0.04) and doc == pytest.approx(19.0, rel=0.04)


def _anh_ban_co_tren_san(pitch=12.0, roll=0.8, x0=0.30, cao=0.11, truoc=0.05):
    """Ảnh camera thật-giả: bàn cờ 10x7 góc trong, ô 19 mm, nằm trên sàn, chụp qua méo của camera mới."""
    rv, tv = tu_pitch(np.radians(pitch), cao, truoc)
    Rr = cv2.Rodrigues(np.array([0, 0, np.radians(roll)]))[0]
    R = Rr @ cv2.Rodrigues(rv)[0]
    tv = Rr @ tv
    c = Calib.tu_pose(K, D, cv2.Rodrigues(R)[0].ravel(), tv)
    gx, gy = np.meshgrid(np.arange(640), np.arange(480))
    xs, ys = c.raw_sang_mat_dat(gx.ravel(), gy.ravel())
    o = 0.019
    ci = np.floor((np.nan_to_num(ys, nan=99) + 4.5 * o) / o).astype(int)
    ri = np.floor((np.nan_to_num(xs, nan=99) - x0) / o).astype(int)
    trong = (ci >= -1) & (ci <= 9) & (ri >= -1) & (ri <= 6)                  # 11 x 8 ô
    vien = (ci >= -3) & (ci <= 11) & (ri >= -3) & (ri <= 8)
    san = 120 + 25 * np.sin(np.nan_to_num(xs) * 40) * np.cos(np.nan_to_num(ys) * 37)
    img = np.where(trong, ((ci + ri) % 2) * 215 + 20, np.where(vien, 245, san)).astype(np.uint8).reshape(480, 640)
    return cv2.GaussianBlur(img, (3, 3), 0.8)


def _nap_cong_cu():
    import importlib.util
    spec = importlib.util.spec_from_file_location('birdeye_banco', Path(__file__).resolve().parents[1] / 'calib/birdeye_banco.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_cong_cu_birdeye_banco_tren_anh_gia():
    bb = _nap_cong_cu()
    gray = _anh_ban_co_tren_san()
    cor, sz = bb.do_ban_co(gray, 10, 7)
    assert cor is not None and sz == (10, 7)
    res = giai_pose(cor, 10, 7, 0.019, K, D)
    assert res['cao'] == pytest.approx(0.11, abs=0.003)
    assert res['pitch'] == pytest.approx(12.0, abs=0.4)
    assert res['roll'] == pytest.approx(0.8, abs=0.4)
    bev, asp, co, bc, m = bb.kiem_o_vuong(gray, K, D, res, sz, 19.0, 0.05, 0.0)
    assert asp == pytest.approx(1.0, abs=0.03) and co == pytest.approx(1.0, abs=0.03)   # ô vuông, đúng 19 mm


def test_khong_co_ban_co_thi_khong_dung_so():
    bb = _nap_cong_cu()
    assert bb.do_ban_co(np.full((480, 640), 120, np.uint8), 10, 7)[0] is None


@pytest.mark.parametrize('x0', [0.26, 0.36, 0.46])
def test_ban_co_gan_hay_xa_deu_do_duoc_va_dung(x0):
    """Camera thấp nên ô cờ bị ép mỏng: kéo giãn dọc phải dò được tới ~45 cm và vẫn chính xác."""
    bb = _nap_cong_cu()
    g = _anh_ban_co_tren_san(x0=x0)
    g = np.clip(g + np.random.default_rng(0).normal(0, 2, g.shape), 0, 255).astype(np.uint8)
    cor, sz = bb.do_ban_co(g, 10, 7)
    assert cor is not None
    res = giai_pose(cor, 10, 7, 0.019, K, D)
    assert res['cao'] == pytest.approx(0.11, abs=0.004) and res['pitch'] == pytest.approx(12.0, abs=0.4)
