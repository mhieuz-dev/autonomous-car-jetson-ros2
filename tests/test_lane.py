"""Bộ dò vạch trên ảnh sinh từ sa bàn giả (đúng calib, đúng phép chiếu)."""
import math

from core.camera import decode_gray_half
from core.simulator import SimCar
from core.lane_follow import make_detector

CAR = SimCar(need_camera=True, fast=True)


def _measure(x, y, yaw_deg):
    CAR.x, CAR.y, CAR.yaw = x, y, math.radians(yaw_deg)
    CAR.t += 1.0
    jpeg, _ = CAR.camera_jpeg()
    dv = make_detector(sim=True)
    return dv.process(decode_gray_half(jpeg), CAR.t)


def test_centre_of_right_lane():
    result = _measure(5.0, 1.8, 90)                     # đường dọc trục y, vạch phải ở x = 5.2
    assert result.found and abs(result.dist_ahead - 0.20) < 0.015


def test_left_offset_increases_distance():
    result = _measure(4.95, 1.8, 90)
    assert abs(result.dist_ahead - 0.25) < 0.015


def _measure_frames(x, y, yaw_deg, n_frames=6):
    """Bộ dò giữ độ cong lọc Kalman qua nhiều khung; một khung đơn lẻ còn nhiễu
    (góc psi ngoại suy từ độ cong), nên đo góc sau vài khung liên tiếp như khi chạy thật."""
    dv = make_detector(sim=True)
    result = None
    for _ in range(n_frames):
        CAR.x, CAR.y, CAR.yaw = x, y, math.radians(yaw_deg)
        CAR.t += 0.1
        jpeg, _ = CAR.camera_jpeg()
        result = dv.process(decode_gray_half(jpeg), CAR.t)
    return result


def test_car_turns_left_line_shifts_right():
    result = _measure_frames(5.0, 1.8, 95)
    assert result.psi < math.radians(-3)


def test_car_turns_right_line_shifts_left():
    result = _measure_frames(5.0, 1.8, 85)
    assert result.psi > math.radians(3)


def test_rejects_6cm_glare_and_cross_line():
    # Đoạn s 6.9-7.4 có vệt sáng rộng 6 cm ở -0.27, s 7.8 có vạch dừng ngang.
    i = int((abs(CAR.s - 7.2)).argmin())
    p, h = CAR.centerline[i] - 0.20 * CAR.normal[i], math.degrees(CAR.heading[i])
    result = _measure(p[0], p[1], h)
    assert result.found and abs(result.dist_ahead - 0.20) < 0.02
    i = int((abs(CAR.s - 7.5)).argmin())
    p, h = CAR.centerline[i] - 0.20 * CAR.normal[i], math.degrees(CAR.heading[i])
    result = _measure(p[0], p[1], h)
    assert result.found and abs(result.dist_ahead - 0.20) < 0.02


def test_faster_than_15hz():
    result = _measure(5.0, 1.8, 90)
    assert result.ms < 30
