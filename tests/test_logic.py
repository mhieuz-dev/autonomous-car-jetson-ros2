"""Test phần logic thuần (không cần xe, không cần ROS)."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.imu import IMU                                              # noqa: E402
from core.lidar import Scan, tunnel_gate, range_in_cone, tunnel_pose  # noqa: E402
from core import tunnel                                               # noqa: E402
from core.detector import largest_light                               # noqa: E402
from core.pid import RateLimiter, HeadingHold, GyroLoop, wrap_angle   # noqa: E402
from autonomy.perception import has_pedestrian, pedestrian_threshold  # noqa: E402

G = 9.81


def _scan_from_walls(wall, noise=0.0, rng=None):
    """360 tia 1 độ từ gốc (0,0) tới các đoạn [(x1,y1,x2,y2)]."""
    a = np.radians(np.arange(-180, 180, 1.0))
    r = np.full(a.shape, np.inf)
    for x1, y1, x2, y2 in wall:
        for k, ang in enumerate(a):
            dx, dy = math.cos(ang), math.sin(ang)
            ex, ey = x2 - x1, y2 - y1
            light = dx * ey - dy * ex
            if abs(light) < 1e-9:
                continue
            t = (x1 * ey - y1 * ex) / light
            u = (x1 * dy - y1 * dx) / light
            if t > 0 and 0 <= u <= 1:
                r[k] = min(r[k], t)
    if noise and rng is not None:
        r = np.where(np.isfinite(r), r + rng.normal(0, noise, r.shape), r)
    return Scan(-math.pi, math.radians(1.0), r, 0.05)


# ------------------------------------------------------------------- IMU ---
def test_imu_calibration_and_pitch_sign():
    imu = IMU()
    imu.start_sampling()
    for i in range(100):                       # đứng yên, gyro lệch 0.01 rad/s
        imu.update(i * 0.01, 0.0, 0.0, G, 0.0, 0.0, 0.01)
    count = imu.finish_sampling()
    assert abs(count['gravity'] - G) < 1e-6 and abs(count['bias_gz'] - 0.01) < 1e-9
    t = 1.0
    for i in range(300):                       # mũi ngóc lên 6 độ
        q = math.radians(6)
        t += 0.01
        imu.update(t, -G * math.sin(q), 0.0, G * math.cos(q), 0.0, 0.0, 0.01)
    assert imu.pitch > math.radians(5)  # DƯƠNG khi leo dốc
    assert abs(imu.yaw) < 1e-6          # bias đã trừ: không trôi


def test_imu_yaw_left_is_positive():
    imu = IMU()
    imu.start_sampling()
    for i in range(20):
        imu.update(i * 0.01, 0, 0, G, 0, 0, 0)
    imu.finish_sampling()
    for i in range(100):                       # quay trái 0.5 rad/s trong 1 s
        imu.update(0.2 + i * 0.01, 0, 0, G, 0, 0, 0.5)
    assert abs(imu.yaw - 0.5) < 0.02


# ------------------------------------------------------------------- PID ---
def test_heading_hold_returns_to_zero():
    imu = SimpleNamespace(yaw=math.radians(10), gz=0.0)
    w, err = HeadingHold(1.5).compute(imu, 0.0, 0.05)
    assert w < 0 and abs(err + math.radians(10)) < 1e-9     # lệch trái -> quẹo phải


def test_angle_wrap():
    assert abs(wrap_angle(math.radians(350)) - math.radians(-10)) < 1e-9


def test_rate_limiter():
    gh = RateLimiter(accel=0.6)
    v, _ = gh.parser(0.3, 0.0, 0.1)
    assert abs(v - 0.06) < 1e-9


def test_gyro_loop_compensates_drift():
    gyro = GyroLoop(kp=0.5, ki=1.0)
    w_real, drift = 0.0, 0.05                   # bánh lệch làm xe tự quay 0.05 rad/s
    for _ in range(400):
        w_sent = gyro.parser(0.15, 0.0, w_real, 0.02)
        w_real = w_sent + drift
    assert abs(w_real) < 0.005


# ------------------------------------------------------------ người đi bộ ---
def test_pedestrian_in_zone():
    thresh = pedestrian_threshold()
    near = thresh - 0.05
    scan = _scan_from_walls([(near, -0.05, near, 0.05)])
    size, d = has_pedestrian(scan)
    assert size and abs(d - near) < 0.01


def test_object_on_side_is_not_pedestrian():
    scan = _scan_from_walls([(0.30, 0.30, 0.35, 0.30)])  # cột biển bên trái, ngoài vùng
    size, _ = has_pedestrian(scan)
    assert not size


# -------------------------------------------------------- hầm (logic 04a) ---
def test_tunnel_gate_and_pose_straight():
    # Xe lệch phải 0.15 m trong hành lang thẳng rộng 0.77
    wall = [(-1.0, 0.385 + 0.15, 1.5, 0.385 + 0.15), (-1.0, -0.385 + 0.15, 1.5, -0.385 + 0.15)]
    scan = _scan_from_walls(wall)
    assert tunnel_gate(scan, tunnel.P) is not None
    off, psi, kappa = tunnel_pose(scan, tunnel.P)
    assert abs(off - 0.15) < 0.01 and abs(psi) < 0.02 and kappa == 0.0
    w = tunnel.steer_cmd(off, psi, kappa, 0.15)
    assert w > 0                              # tâm hầm bên trái -> quẹo trái


def test_not_a_tunnel_when_too_wide():
    wall = [(-1.0, 0.6, 1.5, 0.6), (-1.0, -0.6, 1.5, -0.6)]   # tổng 1.2 m
    assert tunnel_gate(_scan_from_walls(wall), tunnel.P) is None


def test_range_in_cone_ignores_outliers():
    r = np.full(360, np.inf)
    r[180] = 0.15                             # 1 tia lẻ (sượt mặt dốc)
    assert range_in_cone(Scan(-math.pi, math.radians(1), r, 0.05), 0.0, 25.0) is None
    r[178:183] = 0.40                         # vật thật: 5 tia liền
    assert abs(range_in_cone(Scan(-math.pi, math.radians(1), r, 0.05), 0.0, 25.0) - 0.40) < 1e-9


def test_tunnel_bend_turns_to_open_side():
    p = tunnel.P
    wall = [(0.45, -0.6, 0.45, 0.3),          # tường chắn phía trước
             (-0.5, -0.385, 0.45, -0.385),     # tường phải chạy tới góc
             (-0.5, 0.385, 0.0, 0.385)]        # bên trái mở ra phía trước
    from core.lidar import TunnelBend
    result = TunnelBend().compute(_scan_from_walls(wall), p, 0.15)
    assert result is not None and result[1] > 0 and result[0] < 0.15


# ------------------------------------------------------------------ đèn ---
def test_red_wins_and_must_be_large():
    d = lambda name, height=60, conf=0.9: {'name': name, 'conf': conf, 'box': [10, 10, 40, 10 + height]}  # noqa: E731
    assert largest_light([d('green_light')], 40)[0] == 'green'
    assert largest_light([d('green_light', 90), d('red_light', 60)], 40)[0] == 'red'          # thấy cả đỏ lẫn xanh: dừng
    assert largest_light([d('red_light', 30)], 40) == (None, 30, 0.0)                      # còn xa: chưa tính, vẫn báo cao
    assert largest_light([d('red_light', 80, 0.6)], 40, min_conf=0.5, min_conf_red=0.75)[0] is None   # đỏ cần tin cao hơn
    assert largest_light([d('stop'), d('pedestrian')], 40)[0] is None
