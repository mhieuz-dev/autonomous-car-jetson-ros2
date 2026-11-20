"""Đọc LiDAR: vật phía trước, tường hầm.

Toàn bộ phần hầm (range_in_cone, cổng 2 tường, tunnel_pose, bend) CHÉP NGUYÊN
logic từ bản 04aba63e (vong1_xe_that/laptop/crc_solution: solution_node.py,
corridor_geometry.py), chỉ đổi thành hàm thuần nhận `scan` + tham số.
Quy ước góc: 0 độ = phía trước, + = bên TRÁI. Khoảng cách tính từ tâm LiDAR.
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Scan:
    angle_min: float
    angle_increment: float
    ranges: np.ndarray          # float, có thể chứa inf / nan
    range_min: float = 0.05
    t: float = 0.0


def goc_tia(scan):
    """Góc (rad, [-pi, pi]) của từng tia."""
    n = len(scan.ranges)
    inc = scan.angle_increment if scan.angle_increment else 2 * math.pi / n
    a = scan.angle_min + np.arange(n) * inc
    return (a + math.pi) % (2 * math.pi) - math.pi


def range_in_cone(scan, centre_deg=0.0, half_angle_deg=25.0, min_run=3, tol=0.06):
    """Vật gần nhất CÓ THẬT (>= min_run tia liền nhau cùng khoảng cách) trong nón.

    Bỏ được các điểm lẻ khi tia quét sượt mặt dốc. Trả None nếu nón trống.
    """
    if scan is None or len(scan.ranges) == 0:
        return None
    s = scan
    n = len(s.ranges)
    inc = s.angle_increment if s.angle_increment else (2 * math.pi / n)
    half = math.radians(half_angle_deg)
    cone = []
    for i, r in enumerate(s.ranges):
        ang = s.angle_min + i * inc - math.radians(centre_deg)
        ang = (ang + math.pi) % (2 * math.pi) - math.pi
        if abs(ang) > half or math.isnan(r):
            continue
        if not math.isinf(r) and r <= s.range_min:
            r = s.range_min
        cone.append((i, float(r)))
    if not cone:
        return None
    cone.sort(key=lambda t: t[0])
    best = float('inf')
    run_start = 0
    for k in range(len(cone) + 1):
        extend = (k < len(cone)
                  and not math.isinf(cone[k][1])
                  and k > run_start
                  and abs(cone[k][1] - cone[k - 1][1]) <= tol
                  and cone[k][0] - cone[k - 1][0] <= 2)
        if extend:
            continue
        if k - run_start >= min_run:
            seg = [r for _, r in cone[run_start:k] if not math.isinf(r)]
            if seg:
                best = min(best, min(seg))
        run_start = k
    return None if math.isinf(best) else best


def sector_far(scan, lo_deg, hi_deg, cap=3.0):
    """Phân vị 80 của khoảng cách giữa hai góc; không có tia về thì tính là cap."""
    if scan is None or len(scan.ranges) == 0:
        return 0.0
    ang = np.degrees(goc_tia(scan))
    r = np.asarray(scan.ranges, float)
    m = (ang >= lo_deg) & (ang <= hi_deg) & ~np.isnan(r)
    if not m.any():
        return 0.0
    v = r[m]
    v = np.where(np.isinf(v) | (v > cap), cap, np.maximum(v, scan.range_min))
    return float(np.percentile(v, 80))


def wall_points(scan, lo_deg, hi_deg, r_min, r_max):
    if scan is None:
        return np.empty((0, 2))
    ang = goc_tia(scan)
    deg = np.degrees(ang)
    r = np.asarray(scan.ranges, float)
    m = (r > r_min) & (r < r_max) & (deg >= lo_deg) & (deg <= hi_deg)
    return np.column_stack([r[m] * np.cos(ang[m]), r[m] * np.sin(ang[m])])


def wall_circle(points, min_pts):
    """Đường tròn khớp các điểm tường (Kasa). Trả (cx, cy, R) hoặc None."""
    if len(points) < min_pts:
        return None
    x, y = points.T
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    b = -(x * x + y * y)
    try:
        sol = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    cx, cy = -0.5 * sol[0], -0.5 * sol[1]
    disc = cx * cx + cy * cy - sol[2]
    if disc <= 0.0:
        return None
    return float(cx), float(cy), float(math.sqrt(disc))


def straight_wall(points):
    points = np.asarray(points, dtype=float)
    if len(points) < 10:
        return None
    centred = points - points.mean(axis=0)
    values, vectors = np.linalg.eigh(centred.T @ centred / len(points))
    tangent = vectors[:, -1]
    if tangent[0] < 0:
        tangent = -tangent
    span = float(np.ptp(centred @ tangent))
    residual = math.sqrt(max(0., float(values[0])))
    if span < .18 or residual > .018:
        return None
    return math.atan2(tangent[1], tangent[0]), span


def straight_corridor_heading(left, right):
    a, b = straight_wall(left), straight_wall(right)
    if a is None or b is None or max(a[1], b[1]) < .5:
        return None
    difference = math.atan2(math.sin(a[0] - b[0]), math.cos(a[0] - b[0]))
    if abs(difference) > .10:
        return None
    return math.atan2(a[1] * math.sin(a[0]) + b[1] * math.sin(b[0]),
                      a[1] * math.cos(a[0]) + b[1] * math.cos(b[0]))


def cong_ham(scan, p):
    """(trai, phai) nếu đang ở giữa 2 tường hầm, không thì None.

    p cần: SIDE_DEG, WALL_MAX, WIDTH, WIDTH_TOL.
    """
    left = range_in_cone(scan, 90.0, p.SIDE_DEG)
    right = range_in_cone(scan, -90.0, p.SIDE_DEG)
    if left is None or right is None:
        return None
    if left > p.WALL_MAX or right > p.WALL_MAX:
        return None
    if abs(left + right - p.WIDTH) > p.WIDTH_TOL:
        return None
    return left, right


def tunnel_pose(scan, p):
    """(offset, psi, kappa) của hành lang hầm, đo hoàn toàn từ tường, hoặc None.

    offset: tâm hành lang lệch sang TRÁI bao nhiêu (m); psi: hướng hành lang
    (+ = cong trái); kappa: độ cong (1/m). Giống hệt tunnel_pose bản 04a.
    p cần thêm: WALL_FIT_MIN, WALL_FIT_MAX, WALL_FIT_MIN_PTS, PSI_MAX, KAPPA_MAX.
    """
    gate = cong_ham(scan, p)
    if gate is None:
        return None
    near_l, near_r = gate
    offset = 0.5 * (near_l - near_r)
    half = 0.5 * (near_l + near_r)
    walls = [wall_points(scan, 20., 160., p.WALL_FIT_MIN, p.WALL_FIT_MAX),
             wall_points(scan, -160., -20., p.WALL_FIT_MIN, p.WALL_FIT_MAX)]
    straight_heading = straight_corridor_heading(*walls)
    if straight_heading is not None:
        return offset, max(-p.PSI_MAX, min(p.PSI_MAX, straight_heading)), 0.0
    best = None
    for pts in walls:
        fit = wall_circle(pts, p.WALL_FIT_MIN_PTS)
        if fit is None:
            continue
        cx, cy, radius = fit
        d = math.hypot(cx, cy)
        if d < 1e-3 or not (0.05 < radius < 60.0):
            continue
        r_c = radius - half if d < radius else radius + half
        if r_c < 0.05:
            continue
        off = cy * (1.0 - r_c / d)
        sgn = 1.0 if cy >= 0.0 else -1.0
        kappa = sgn / r_c
        psi = math.atan2(-sgn * cx / d, sgn * cy / d)
        if best is None or abs(off - offset) < abs(best[0] - offset):
            best = (off, psi, kappa)
    if best is None:
        return offset, 0.0, 0.0
    _, psi, kappa = best
    return (offset, max(-p.PSI_MAX, min(p.PSI_MAX, psi)),
            max(-p.KAPPA_MAX, min(p.KAPPA_MAX, kappa)))


class CuaHam:
    """Phản ứng khúc cua trong hầm (bản 04a `_tunnel_bend`): tường trước < BEND_FRONT
    thì chậm lại và quẹo về phía nhìn được xa hơn. Chọn phía một lần mỗi khúc cua."""

    def __init__(self):
        self.phia = None

    def tinh(self, scan, p, v_ham):
        front = range_in_cone(scan, 0.0, 25.0)
        if front is None or front >= p.BEND_FRONT:
            if front is None or front >= p.BEND_FRONT + 0.10:
                self.phia = None
            return None
        if self.phia is None:
            left, right = sector_far(scan, 15.0, 75.0), sector_far(scan, -75.0, -15.0)
            self.phia = 1.0 if left >= right else -1.0
        frac = max(0.0, min(1.0, (front - 0.25) / max(p.BEND_FRONT - 0.25, 1e-3)))
        return 0.04 + (v_ham - 0.04) * frac, self.phia * p.BEND_OMEGA


def diem_xy(scan, r_max=3.0):
    """Mọi điểm LiDAR (x trước, y trái) trong bán kính r_max, để vẽ."""
    if scan is None:
        return np.empty((0, 2))
    a = goc_tia(scan)
    r = np.asarray(scan.ranges, float)
    m = np.isfinite(r) & (r > scan.range_min) & (r < r_max)
    return np.column_stack([r[m] * np.cos(a[m]), r[m] * np.sin(a[m])])
