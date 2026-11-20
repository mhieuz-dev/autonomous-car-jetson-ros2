"""Bước 4b: tự chọn lần dùng, vùng nhìn, độ phân giải bird's eye."""
import shutil

import cv2
import numpy as np
import pytest
import yaml

from core import birdeye_select as B
from core import settings
from core.camera import Calib, read_bev
from core.camera_pose import solve_pose, pose_in_car
from test_calib import CAM, K, D, _board_on_floor_image, _load_tool


@pytest.fixture
def new(tmp_path, monkeypatch):
    """config/ và calib_new/ tạm; 2 ứng viên (bàn cờ ở 2 chỗ) như birdeye_chessboard lưu."""
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config/camera.yaml').write_text(yaml.safe_dump(CAM))
    monkeypatch.setattr(settings, 'CONFIG_DIR', tmp_path / 'config')
    monkeypatch.setattr(settings, 'CALIB_NEW_DIR', tmp_path / 'calib_new')
    bb = _load_tool()
    for x0 in (0.28, 0.36):
        g = _board_on_floor_image(x0=x0)
        g = np.clip(g + np.random.default_rng(0).normal(0, 2, g.shape), 0, 255).astype(np.uint8)
        cor, sz = bb.detect_board(g, 10, 7)
        res = solve_pose(cor, 10, 7, 0.019, K, D)
        rv, tv = pose_in_car(res, 0.06)
        B.save_candidate(res, rv, tv, g, [('RMS', '0.2 px', True), ('roll', '0.8', True)], 0.06, 0.0, sz, 19.0, 1.0, 1.0)
    return tmp_path


def test_list_candidates_and_median(new):
    cands = B.list_candidates()
    assert [u['n'] for u in cands] == [1, 2] and all(u['passed'] == 2 for u in cands)
    assert cands[0]['height_m'] == pytest.approx(0.11, abs=0.003)
    rv, tv = B.chosen_pose([1, 2])
    c = Calib.from_pose(K, D, rv, tv)
    assert (c.height_m, c.front_of_center) == pytest.approx((0.11, 0.06), abs=0.003)


def test_suggestion_is_sane(new):
    d = B.suggest([1])
    assert 0.1 < d['x_nearest_seen'] < 0.4 and d['x_near'] > d['x_nearest_seen']
    assert 0.5 <= d['x_far'] <= 1.5
    mm = [b['mm_per_row'] for b in d['table']]
    assert mm == sorted(mm)                       # càng xa, 1 hàng ảnh càng ứng với nhiều mm sàn


def test_preview_and_reject_bad_params(new):
    p = {'x_near': 0.25, 'x_far': 0.9, 'y_left': 0.4, 'y_right': -0.4, 'mm_px': 5}
    img = B.preview([1], p)
    assert img.ndim == 3 and img.shape[1] >= 400
    for err in ({**p, 'x_far': 0.2}, {**p, 'y_left': -0.6}, {**p, 'mm_px': 0.1}, {**p, 'mm_px': 50}):
        with pytest.raises(ValueError):
            B.preview([1], err)
    with pytest.raises(ValueError, match='chưa chọn'):
        B.preview([], p)


def test_saved_selection_is_read_back(new):
    p = {'x_near': 0.25, 'x_far': 0.9, 'y_left': 0.4, 'y_right': -0.4, 'mm_px': 4}
    B.save_selection([1, 2], p)
    for f in ('extrinsics.yaml', 'bev.yaml'):
        shutil.copy(new / 'calib_new' / f, new / 'config' / f)
    assert Calib().height_m == pytest.approx(0.11, abs=0.003)          # config/ đầy đủ -> đọc được
    assert read_bev() == {'x_near': 0.25, 'x_far': 0.9, 'y_left': 0.4, 'y_right': -0.4, 'm_px': 0.004}
    from core.lane_follow import make_detector
    det = make_detector()
    assert (det.p.X_NEAR, det.p.X_FAR, det.birdeye.m) == (0.25, 0.9, 0.004)   # bộ dò dùng đúng vùng nhìn đã chọn
    assert det.birdeye.y_left == 0.4 and det.birdeye.y_right == -0.4
