"""Calib camera mới: không có số mặc định, tư thế lấy từ bàn cờ nằm trên sàn."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.camera import BirdEye, Calib, NotCalibrated, from_pitch  # noqa: E402
from core.camera_pose import solve_pose, pose_in_car               # noqa: E402

CAM = {'image_width': 640, 'image_height': 480,
       'camera_matrix': {'data': [552.83, 0, 311.43, 0, 551.9, 239.42, 0, 0, 1]},
       'distortion_coefficients': {'data': [0.2036, -1.1217, -0.0017, 0.0014, 1.745]}}
K = np.array(CAM['camera_matrix']['data']).reshape(3, 3)
D = np.array(CAM['distortion_coefficients']['data'])


def _dir(tmp_path, extr=None, cam=True):
    if cam:
        (tmp_path / 'camera.yaml').write_text(yaml.safe_dump(CAM))
    if extr is not None:
        (tmp_path / 'extrinsics.yaml').write_text(yaml.safe_dump(extr))
    return tmp_path


def test_missing_calib_raises(tmp_path):
    with pytest.raises(NotCalibrated, match='CHƯA CALIB'):
        Calib(_dir(tmp_path, cam=False))


def test_missing_birdeye_raises_instead_of_old_numbers(tmp_path):
    with pytest.raises(NotCalibrated, match="BIRD'S EYE"):
        Calib(_dir(tmp_path, {'height_m': 0.11}))        # có số tay nhưng không có tư thế
    with pytest.raises(NotCalibrated, match="BIRD'S EYE"):
        Calib(_dir(tmp_path))                          # không có extrinsics.yaml


def _board(pitch_deg, roll_deg, x0, height=0.11, front=0.05, noise=0.3, cols=10, rows=7, o=0.019):
    rv, tv = from_pitch(np.radians(pitch_deg), height, front)
    R = cv2.Rodrigues(np.array([0, 0, np.radians(roll_deg)]))[0] @ cv2.Rodrigues(rv)[0]
    tv = cv2.Rodrigues(np.array([0, 0, np.radians(roll_deg)]))[0] @ tv
    rv = cv2.Rodrigues(R)[0].ravel()
    ii, jj = np.mgrid[0:rows, 0:cols]
    Pw = np.column_stack([x0 + ii.ravel() * o, (jj.ravel() - 4.5) * o, np.zeros(rows * cols)])
    px = cv2.projectPoints(Pw, rv, tv, K, D)[0].reshape(-1, 2)
    assert ((px[:, 0] > 0) & (px[:, 0] < 640) & (px[:, 1] > 0) & (px[:, 1] < 480)).all()
    return rv, tv, px + np.random.default_rng(1).normal(0, noise, px.shape), Pw


@pytest.mark.parametrize('pitch,roll,x0', [(10, 1.2, 0.25), (18, -0.8, 0.22), (5, 0.0, 0.30)])
def test_pose_from_floor_chessboard(pitch, roll, x0):
    rv, tv, px, Pw = _board(pitch, roll, x0)
    r = solve_pose(px, 10, 7, 0.019, K, D)
    assert r['height'] == pytest.approx(0.11, abs=0.002)  # chiều cao camera
    assert r['pitch'] == pytest.approx(pitch, abs=0.3)    # độ chúi
    assert r['roll'] == pytest.approx(roll, abs=0.3)
    assert abs(r['yaw']) < 0.3 and r['rms'] < 1.0
    # điểm sàn (hệ xe) qua calib dựng từ bàn cờ phải rơi đúng pixel thật
    rvx, tvx = pose_in_car(r, 0.05)
    c = Calib.from_pose(K, D, rvx, tvx)
    assert (c.height_m, c.front_of_center) == pytest.approx((r['height'], 0.05), abs=1e-6)
    actual = cv2.projectPoints(np.array([[x0 + 3 * 0.019, 0.0, 0.0]]), rv, tv, K, D)[0].ravel()
    proj = np.array(c.ground_to_pixel(np.array([x0 + 3 * 0.019]), np.array([0.0]))).ravel()
    assert np.abs(actual - proj).max() < 1.0                         # sai < 1 pixel
    # và chiều ngược lại: pixel -> mặt sàn
    x, y = c.pixel_to_ground(actual[0], actual[1])
    assert (x[0], y[0]) == pytest.approx((x0 + 3 * 0.019, 0.0), abs=0.002)


def test_birdeye_turns_chessboard_into_squares():
    """Trên bird's eye 1 mm/px các ô cờ 19 mm phải vuông và cạnh ~19 px."""
    board_x0 = 0.25
    rv, tv, px, Pw = _board(12, 0.5, board_x0, front=0.0)   # gốc ngay dưới camera, như lúc calib
    r = solve_pose(px, 10, 7, 0.019, K, D)
    c = Calib.from_pose(K, D, *pose_in_car(r, 0.0))
    from core.camera_pose import board_corners_in_car
    x, y = board_corners_in_car(r)
    birdeye = BirdEye(c, x.min() - 0.03, x.max() + 0.03, y.max() + 0.03, y.min() - 0.03, 0.001, img_scale=1.0)
    # ảnh giả: dán lại bàn cờ thật bằng cách chiếu ô đen/trắng theo toạ độ sàn
    gx, gy = np.meshgrid(np.arange(640), np.arange(480))
    xs, ys = c.pixel_to_ground(gx.ravel(), gy.ravel())
    o = 0.019
    ci = np.floor((np.nan_to_num(ys, nan=99) + 4.5 * o) / o).astype(int)      # cột ô: biên đúng tại các góc trong
    ri = np.floor((np.nan_to_num(xs, nan=99) - board_x0) / o).astype(int)
    inside = (ci >= -1) & (ci <= 9) & (ri >= -1) & (ri <= 6)                   # 11 x 8 ô, ngoài là viền trắng
    img = np.where(inside, ((ci + ri) % 2) * 255, 255).astype(np.uint8).reshape(480, 640)
    bev = birdeye.warp(img)
    bev = cv2.copyMakeBorder(bev, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    ok, cor = cv2.findChessboardCorners(bev, (10, 7), cv2.CALIB_CB_ADAPTIVE_THRESH)
    assert ok
    cor = cor.reshape(7, 10, 2)
    horiz = np.linalg.norm(np.diff(cor, axis=1), axis=2).mean()
    slope = np.linalg.norm(np.diff(cor, axis=0), axis=2).mean()
    assert horiz == pytest.approx(19.0, rel=0.04) and slope == pytest.approx(19.0, rel=0.04)


def _board_on_floor_image(pitch=12.0, roll=0.8, x0=0.30, height=0.11, front=0.05):
    """Ảnh camera thật-giả: bàn cờ 10x7 góc trong, ô 19 mm, nằm trên sàn, chụp qua méo của camera mới."""
    rv, tv = from_pitch(np.radians(pitch), height, front)
    Rr = cv2.Rodrigues(np.array([0, 0, np.radians(roll)]))[0]
    R = Rr @ cv2.Rodrigues(rv)[0]
    tv = Rr @ tv
    c = Calib.from_pose(K, D, cv2.Rodrigues(R)[0].ravel(), tv)
    gx, gy = np.meshgrid(np.arange(640), np.arange(480))
    xs, ys = c.pixel_to_ground(gx.ravel(), gy.ravel())
    o = 0.019
    ci = np.floor((np.nan_to_num(ys, nan=99) + 4.5 * o) / o).astype(int)
    ri = np.floor((np.nan_to_num(xs, nan=99) - x0) / o).astype(int)
    inside = (ci >= -1) & (ci <= 9) & (ri >= -1) & (ri <= 6)                  # 11 x 8 ô
    border = (ci >= -3) & (ci <= 11) & (ri >= -3) & (ri <= 8)
    floor = 120 + 25 * np.sin(np.nan_to_num(xs) * 40) * np.cos(np.nan_to_num(ys) * 37)
    img = np.where(inside, ((ci + ri) % 2) * 215 + 20, np.where(border, 245, floor)).astype(np.uint8).reshape(480, 640)
    return cv2.GaussianBlur(img, (3, 3), 0.8)


def _load_tool():
    import importlib.util
    spec = importlib.util.spec_from_file_location('birdeye_chessboard', Path(__file__).resolve().parents[1] / 'tools/birdeye_chessboard.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_chessboard_tool_on_synthetic_image():
    bb = _load_tool()
    gray = _board_on_floor_image()
    cor, sz = bb.detect_board(gray, 10, 7)
    assert cor is not None and sz == (10, 7)
    res = solve_pose(cor, 10, 7, 0.019, K, D)
    assert res['height'] == pytest.approx(0.11, abs=0.003)
    assert res['pitch'] == pytest.approx(12.0, abs=0.4)
    assert res['roll'] == pytest.approx(0.8, abs=0.4)
    bev, asp, size, bc, m = bb.check_squares(gray, K, D, res, sz, 19.0, 0.05, 0.0)
    assert asp == pytest.approx(1.0, abs=0.03) and size == pytest.approx(1.0, abs=0.03)   # ô vuông, đúng 19 mm


def test_no_chessboard_no_numbers():
    bb = _load_tool()
    assert bb.detect_board(np.full((480, 640), 120, np.uint8), 10, 7)[0] is None


@pytest.mark.parametrize('x0', [0.26, 0.36, 0.46])
def test_near_and_far_chessboard_both_measured(x0):
    """Camera thấp nên ô cờ bị ép mỏng: kéo giãn dọc phải dò được tới ~45 cm và vẫn chính xác."""
    bb = _load_tool()
    g = _board_on_floor_image(x0=x0)
    g = np.clip(g + np.random.default_rng(0).normal(0, 2, g.shape), 0, 255).astype(np.uint8)
    cor, sz = bb.detect_board(g, 10, 7)
    assert cor is not None
    res = solve_pose(cor, 10, 7, 0.019, K, D)
    assert res['height'] == pytest.approx(0.11, abs=0.004) and res['pitch'] == pytest.approx(12.0, abs=0.4)
