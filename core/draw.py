"""Vẽ hình để xem: LiDAR nhìn từ trên, ghép nhiều ảnh nhỏ thành lưới."""
import math

import cv2
import numpy as np

from .lidar import points_xy


def lidar_top_view(scan, radius=1.5, size=320, cones=(), text=''):
    """Ảnh LiDAR: xe ở giữa, mũi xe hướng LÊN. non = [(goc_giua_do, nua_goc_do, r, mau)]."""
    img = np.full((size, size, 3), 25, np.uint8)
    k = size / (2 * radius)
    c = size // 2

    def px(x, y):
        return int(c - y * k), int(c - x * k)

    for r in np.arange(0.5, radius + 1e-6, 0.5):
        cv2.circle(img, (c, c), int(r * k), (60, 60, 60), 1)
    for g0, half, r, color in cones:
        a0, a1 = math.radians(g0 - half), math.radians(g0 + half)
        pts = [px(0, 0)] + [px(r * math.cos(a), r * math.sin(a)) for a in np.linspace(a0, a1, 12)]
        cv2.polylines(img, [np.array(pts, np.int32)], True, color, 1)
    for x, y in points_xy(scan, radius):
        cv2.circle(img, px(x, y), 2, (0, 255, 255), -1)
    cv2.rectangle(img, px(0.10, 0.08), px(-0.13, -0.08), (255, 180, 0), 2)   # thân xe
    cv2.arrowedLine(img, (c, c), px(0.25, 0), (255, 180, 0), 2)
    if text:
        cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def grid(image, n_cols=4, size=(240, 180)):
    """Ghép các (tên, ảnh) thành một lưới, mỗi ô co=(rộng, cao), có chữ tên."""
    o = []
    for name, a in image:
        if a is None:
            a = np.zeros((size[1], size[0]), np.uint8)
        if a.ndim == 2:
            a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        a = cv2.resize(a, size, interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(a, (0, 0), (size[0], 18), (0, 0, 0), -1)
        cv2.putText(a, name, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        o.append(a)
    while len(o) % n_cols:
        o.append(np.zeros_like(o[0]))
    row = [np.hstack(o[i:i + n_cols]) for i in range(0, len(o), n_cols)]
    return np.vstack(row)
