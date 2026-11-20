"""Chọn tham số bird's eye SAU KHI đã tính xong (bước 4b trên trang calib, chạy trên laptop).

birdeye_chessboard (trên xe) mỗi lần chốt chỉ lưu một ỨNG VIÊN vào calib_new/candidates/:
    lan_NN.yaml  tư thế camera + số đo của lần đó,  lan_NN.png  ảnh xám lúc chốt.
Ở đây bạn tự chọn: những lần nào dùng (nhiều lần -> trung vị), VÙNG NHÌN (x gần/xa, y trái/phải)
và ĐỘ PHÂN GIẢI (mm/pixel), xem thử bird's eye với đúng các số đó, rồi mới ghi
calib_new/extrinsics.yaml (tư thế camera) + calib_new/bev.yaml (vùng nhìn + độ phân giải).
"""
import glob
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from . import settings
from .camera import BirdEye, Calib

MAX_MM_PER_ROW = 15.0        # gợi ý x xa: nơi 1 hàng ảnh còn ứng với <= 15 mm mặt sàn
MAX_PIXEL = 4_000_000


def candidates_dir(root=None):
    return Path(root or settings.CALIB_NEW_DIR) / 'candidates'


def save_candidate(res, rvec, tvec, gray, checks, front, lateral, sz, square_mm, aspect, size, root=None, extra=None):
    """Chạy trên XE: lưu một lần chốt. Trả đường dẫn file yaml."""
    d = candidates_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    n = len(list(d.glob('lan_*.yaml'))) + 1
    f = d / f'lan_{n:02d}.yaml'
    f.write_text(yaml.safe_dump({
        'n': n, 'time': time.strftime('%d/%m %H:%M:%S'),
        'height_m': round(float(res['height']), 5), 'pitch_deg': round(float(res['pitch']), 3),
        'roll_deg': round(float(res['roll']), 3), 'yaw_deg': round(float(res['yaw']), 3),
        'rms_px': round(float(res['rms']), 3),
        'square': None if aspect is None else round(float(aspect), 4),
        'has_squares': None if size is None else round(float(size), 4),
        'front_of_center_m': front, 'lateral_m': lateral, 'board': f'{sz[0]}x{sz[1]}', 'square_mm': square_mm,
        'checks': [[t, g, bool(ok)] for t, g, ok in checks],
        'rvec': [float(v) for v in rvec], 'tvec': [float(v) for v in tvec], **(extra or {})},
        sort_keys=False, allow_unicode=True))
    cv2.imwrite(str(d / f'lan_{n:02d}.png'), gray)
    return f


def list_candidates(root=None):
    cands = []
    for f in sorted(glob.glob(str(candidates_dir(root) / 'lan_*.yaml'))):
        y = yaml.safe_load(Path(f).read_text())
        y['image'] = f[:-5] + '.png'
        y['passed'] = sum(1 for k in y['checks'] if k[2])
        y['total'] = len(y['checks'])
        cands.append(y)
    return cands


def _selected(run_ids, root=None):
    cands = [u for u in list_candidates(root) if u['n'] in set(run_ids)]
    if not cands:
        raise ValueError('chưa chọn lần nào (hoặc chưa kéo kết quả từ xe: bấm "Kéo kết quả từ xe")')
    return cands


def chosen_pose(run_ids, root=None):
    """Trung vị tư thế của các lần được chọn (1 lần = chính nó). -> (rvec, tvec)."""
    cands = _selected(run_ids, root)
    return (np.median([u['rvec'] for u in cands], axis=0), np.median([u['tvec'] for u in cands], axis=0))


def _camera_intrinsics():
    c = yaml.safe_load((settings.CONFIG_DIR / 'camera.yaml').read_text())
    return (np.array(c['camera_matrix']['data'], float).reshape(3, 3),
            np.array(c['distortion_coefficients']['data'], float).ravel())


def suggest(run_ids, root=None):
    """Gợi ý vùng nhìn theo phần sàn camera thật sự thấy rõ + bảng mm/hàng ảnh để bạn tự cân nhắc."""
    K, D = _camera_intrinsics()
    rvec, tvec = chosen_pose(run_ids, root)
    c = Calib.from_pose(K, D, rvec, tvec)
    xs = np.arange(0.02, 3.0, 0.005)
    u, v = c.ground_to_pixel(xs, np.zeros_like(xs))
    found = (v >= 0) & (v <= c.img_h - 1) & (u >= 0) & (u <= c.width - 1)
    x_min = float(xs[found][0]) if found.any() else 0.15
    dv = np.abs(np.gradient(v, xs))                                   # pixel hàng ảnh trên mỗi mét
    mm_per_row = np.where(dv > 1e-6, 1000.0 / np.maximum(dv, 1e-6), np.inf)
    ok = found & (mm_per_row <= MAX_MM_PER_ROW)
    x_far = float(xs[ok][-1]) if ok.any() else 1.0
    table = []
    for x in np.arange(0.2, 1.61, 0.1):
        i = int(np.argmin(np.abs(xs - x)))
        ys = np.linspace(-1.2, 1.2, 481)
        uu, vv = c.ground_to_pixel(np.full_like(ys, x), ys)
        vis = (uu >= 0) & (uu <= c.width - 1) & (vv >= 0) & (vv <= c.img_h - 1)
        width = float(ys[vis][-1] - ys[vis][0]) if vis.any() else 0.0
        table.append({'x': round(float(x), 2), 'mm_per_row': round(float(mm_per_row[i]), 1), 'width': round(width, 2)})
    return {'x_nearest_seen': round(x_min, 3), 'x_near': round(float(np.ceil((x_min + 0.01) * 100) / 100), 2),
            'x_far': round(float(np.floor(min(x_far, 1.5) * 20) / 20), 2),
            'y_left': 0.5, 'y_right': -0.5, 'mm_px': 5.0, 'table': table}


def _check_params(p):
    xg, xx, yt, yp, mm = (float(p[k]) for k in ('x_near', 'x_far', 'y_left', 'y_right', 'mm_px'))
    if not (xx > xg):
        raise ValueError('x xa phải lớn hơn x gần')
    if not (yt > yp):
        raise ValueError('y trái phải lớn hơn y phải')
    if not (0.5 <= mm <= 20):
        raise ValueError('độ phân giải phải trong khoảng 0,5 tới 20 mm/pixel')
    if ((xx - xg) / (mm / 1000) + 1) * ((yt - yp) / (mm / 1000) + 1) > MAX_PIXEL:
        raise ValueError('ảnh bird\'s eye quá lớn (> 4 triệu pixel): tăng mm/pixel hoặc thu hẹp vùng nhìn')
    return xg, xx, yt, yp, mm


def preview(run_ids, p, root=None):
    """Bird's eye của ảnh lần ĐẦU trong run_ids với tham số p; có lưới 10 cm. -> ảnh BGR."""
    xg, xx, yt, yp, mm = _check_params(p)
    K, D = _camera_intrinsics()
    rvec, tvec = chosen_pose(run_ids, root)
    cand = _selected(run_ids, root)[0]
    gray = cv2.imread(cand['image'], cv2.IMREAD_GRAYSCALE)
    c = Calib.from_pose(K, D, rvec, tvec, (gray.shape[1], gray.shape[0]))
    birdeye = BirdEye(c, xg, xx, yt, yp, mm / 1000.0, img_scale=1.0)
    bev = birdeye.warp(gray)
    k = max(1, int(round(560.0 / bev.shape[1])))                     # ảnh nhỏ thì phóng to cho dễ nhìn
    img = cv2.cvtColor(cv2.resize(bev, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    for x in np.arange(np.ceil(xg * 10) / 10, xx + 1e-9, 0.1):
        r = int(round(birdeye.m_to_px(x, 0)[0] * k))
        cv2.line(img, (0, r), (img.shape[1], r), (0, 200, 255), 1)
        cv2.putText(img, f'x {x:.1f}', (4, max(12, r - 3)), 0, 0.42, (0, 255, 255), 1)
    for y in np.arange(np.ceil(yp * 10) / 10, yt + 1e-9, 0.1):
        c_ = int(round(birdeye.m_to_px(0, y)[1] * k))
        cv2.line(img, (c_, 0), (c_, img.shape[0]), (255, 200, 0) if abs(y) > 1e-9 else (0, 0, 255), 1)
        cv2.putText(img, f'{y:+.1f}', (c_ + 2, img.shape[0] - 24), 0, 0.4, (255, 255, 0), 1)
    cv2.putText(img, f"bird's eye {bev.shape[1]}x{bev.shape[0]} px, {mm:g} mm/px (x tien, y trai; do do = y 0)",
                (6, img.shape[0] - 8), 0, 0.45, (255, 255, 255), 1)
    if img.shape[1] > 900:
        f = 900.0 / img.shape[1]
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return img


def save_selection(run_ids, p, root=None, dest=None):
    """Ghi calib_new/extrinsics.yaml (tư thế = trung vị các lần chọn) + bev.yaml (vùng nhìn, độ phân giải)."""
    xg, xx, yt, yp, mm = _check_params(p)
    cands = _selected(run_ids, root)
    rvec, tvec = chosen_pose(run_ids, root)
    K, D = _camera_intrinsics()
    c = Calib.from_pose(K, D, rvec, tvec)
    dest = Path(dest or settings.CALIB_NEW_DIR)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'extrinsics.yaml').write_text(
        f'# chon tay {time.strftime("%d/%m %H:%M")} tu cac lan {sorted(u["n"] for u in cands)} (trung vi)\n' + yaml.safe_dump({
            'rvec': [round(float(v), 6) for v in rvec], 'tvec': [round(float(v), 6) for v in tvec],
            'height_m': round(c.height_m, 4), 'front_of_center_m': round(c.front_of_center, 4), 'lateral_m': round(c.lateral, 4),
            'chosen_runs': sorted(u['n'] for u in cands)}, sort_keys=False))
    (dest / 'bev.yaml').write_text(
        f'# vung nhin + do phan giai bird\'s eye, chon tay {time.strftime("%d/%m %H:%M")}\n' + yaml.safe_dump({
            'x_near': xg, 'x_far': xx, 'y_left': yt, 'y_right': yp, 'm_px': round(mm / 1000.0, 6)}, sort_keys=False))
    return {'height_m': c.height_m, 'front_of_center_m': c.front_of_center, 'times': sorted(u['n'] for u in cands)}
