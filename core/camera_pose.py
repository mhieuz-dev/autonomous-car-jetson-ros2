"""Tư thế camera so với mặt sàn từ MỘT bàn cờ đặt nằm phẳng trên sàn.

solvePnP cho tư thế bàn cờ -> camera. Mặt bàn cờ chính là mặt sàn, nên suy ra:
chiều cao camera, góc chúi (pitch), nghiêng ngang (roll), lệch hướng (yaw) so với cạnh bàn cờ.
Hệ xe: x phía trước, y bên trái, z lên. Trục "phía trước" của hệ xe = trục của bàn cờ
chạy ra xa camera nhất (đặt bàn cờ cho CẠNH của nó vuông góc / song song với hướng xe).
"""
import math

import cv2
import numpy as np


def board_points(cols, rows, square_m):
    """Toạ độ 3D các góc trong của bàn cờ (cols x rows), thứ tự như findChessboardCorners."""
    obj = np.zeros((cols * rows, 3), np.float64)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    return obj


def solve_pose(corners, cols, rows, square_m, K, D):
    """-> dict: rvec/tvec (hệ xe, gốc ngay DƯỚI camera), cao, pitch, roll, yaw (độ), rms (px), W (mat 3x3), C_b."""
    obj = board_points(cols, rows, square_m)
    pts = np.asarray(corners, np.float64).reshape(-1, 1, 2)
    best = None
    for flag in (cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_IPPE):       # lấy cách khớp ảnh tốt hơn
        try:
            ok, r_, t_ = cv2.solvePnP(obj, pts, K, D, flags=flag)
        except cv2.error:
            continue
        if not ok:
            continue
        r_, t_ = cv2.solvePnPRefineLM(obj, pts, K, D, r_, t_)
        pr_ = cv2.projectPoints(obj, r_, t_, K, D)[0].reshape(-1, 2)
        e_ = float(np.sqrt(np.mean(np.sum((pr_ - pts.reshape(-1, 2)) ** 2, axis=1))))
        if t_[2] > 0 and (best is None or e_ < best[0]):
            best = (e_, r_, t_)
    if best is None:
        return None
    rms, rv, tv = best
    R = cv2.Rodrigues(rv)[0]                                        # bàn cờ -> camera
    C_b = -R.T @ tv.ravel()                                         # camera trong hệ bàn cờ
    up = np.array([0.0, 0.0, 1.0]) * (1.0 if C_b[2] > 0 else -1.0)  # pháp tuyến hướng về phía camera
    a = max(((abs(R[2, i]), i) for i in (0, 1)))[1]
    sg = 1.0 if R[2, a] > 0 else -1.0                      # trục chạy RA XA camera
    fwd = np.zeros(3)
    fwd[a] = sg
    left = np.cross(up, fwd)
    W = np.column_stack([fwd, left, up])  # trục hệ xe viết trong hệ bàn cờ
    R_cw = R @ W                          # hệ xe -> camera
    height = float(C_b @ up)
    tvec = -R_cw @ np.array([0.0, 0.0, height])  # gốc dưới camera (truoc = 0)
    o_w = R_cw.T @ np.array([0.0, 0.0, 1.0])     # trục quang học trong hệ xe
    x_w = R_cw.T @ np.array([1.0, 0.0, 0.0])     # "bên phải" của camera trong hệ xe
    return {'rvec': cv2.Rodrigues(R_cw)[0].ravel(), 'tvec': tvec, 'height': height,
            'pitch': math.degrees(-math.asin(float(np.clip(o_w[2], -1, 1)))),
            'roll': math.degrees(math.asin(float(np.clip(x_w[2], -1, 1)))),
            'yaw': math.degrees(math.atan2(o_w[1], o_w[0])),
            'rms': rms, 'W': W, 'C_b': C_b, 'up': up, 'obj': obj}


def pose_in_car(res, front_m, lateral_m=0.0):
    """(rvec, tvec) trong hệ xe: camera nằm cách tâm xe front_m phía trước, lateral_m sang trái."""
    R = cv2.Rodrigues(res['rvec'])[0]
    return res['rvec'], -R @ np.array([front_m, lateral_m, res['height']])


def board_corners_in_car(res):
    """Toạ độ (x, y) trên mặt sàn, gốc dưới camera, của 4 góc ngoài lưới góc trong + toàn bộ điểm."""
    P = (res['obj'] - res['C_b'] + res['height'] * res['up']) @ res['W']
    return P[:, 0], P[:, 1]
