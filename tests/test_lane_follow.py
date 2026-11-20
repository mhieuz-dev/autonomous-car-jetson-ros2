"""Bám vạch: dự đoán giữ độ cong, cổng vạch phải hợp lý, chạy chậm khi dự đoán, nhảy vạch thì xoay tìm lại."""
import math
from types import SimpleNamespace

import numpy as np

from core import lane_follow as BV
from core import lane
from autonomy import behaviors


def _arc(R, x):
    """Vạch là cung tròn bán kính R quẹo TRÁI, xe nằm trên tiếp tuyến tại x = 0: y(x) = R - sqrt(R^2 - x^2)."""
    return R - math.sqrt(R * R - x * x)


def test_shift_curve_keeps_arc_straight_shift_drifts():
    R, X, ds, n = 1.0, 0.35, 0.01, 30  # chạy 30 cm trên cung tròn bán kính 1 m
    xs = np.linspace(0.0, 0.9, 40)     # khớp tới X_FIT_MAX như bộ dò
    c2, b2, a2 = np.polyfit(xs, [_arc(R, x) for x in xs], 2)
    coef = (a2, b2, c2)
    y_straight, psi_straight = BV.y_psi(coef, X)
    for _ in range(n):                           # xe chạy đúng trên cung: dyaw = ds / R, nên vạch trong hệ xe KHÔNG đổi
        dyaw = ds / R
        coef = BV.shift_curve(coef, ds, dyaw, 1.0)
        y_straight, psi_straight = y_straight + ds * math.tan(psi_straight) - dyaw * X, psi_straight - dyaw     # dời kiểu kéo thẳng (bản cũ)
    y_true = _arc(R, X)
    err_curve = abs(BV.y_psi(coef, X)[0] - y_true)
    err_straight = abs(y_straight - y_true)
    assert err_curve < 0.005, f'giữ cong lệch {err_curve * 100:.2f} cm'
    assert err_straight > 0.02 and err_straight > 3 * err_curve, f'kéo thẳng {err_straight * 100:.1f} cm phải tệ hơn giữ cong {err_curve * 100:.1f} cm'


def _detector(monkeypatch, coef_seq):
    """Bộ dò test với bước 1-6 giả: mỗi khung trả về hệ số trong coef_seq (None = không fit)."""
    det = BV.make_detector(sim=True)
    feed = iter(coef_seq)
    monkeypatch.setattr(lane.LaneDetector, 'extract', lambda self, gray_img, debug=False: (next(feed), 300, {}))
    return det


def test_plausibility_gate_drops_far_line_and_predicts(monkeypatch):
    right = (-0.20, 0.0, 0.0)  # vạch phải cách 20 cm
    far = (-0.60, 0.0, 0.0)    # vạch làn bên cách 60 cm: ngoài dải hợp lý
    det = _detector(monkeypatch, [right, right, far, far, None])
    gray_img = np.zeros((240, 320), np.uint8)
    k1 = det.process(gray_img, 0.0)
    k2 = det.process(gray_img, 0.1)
    assert k2.found and k2.fresh and abs(k2.dist_ahead - 0.20) < 0.01
    k3 = det.process(gray_img, 0.2, ds=0.01)
    assert k3.found and not k3.fresh, 'vạch 60 cm bị cổng bỏ, phải giữ dự đoán'
    assert 'ngoài dải hợp lý' in k3.reason and abs(k3.dist_ahead - 0.20) < 0.02
    k4 = det.process(gray_img, 0.3, ds=0.01)
    assert k4.found and not k4.fresh
    k5 = det.process(gray_img, 0.2 + BV.HOLD_PLAUSIBLE_S + 0.1)     # quá HOLD_PLAUSIBLE_S không có vạch hợp lý -> mất vạch
    assert not k5.found


def test_slows_down_while_predicting():
    ctrl = BV.LaneController(sim=True)
    result = lane.LaneResult()
    result.found, result.fresh, result.dist_ahead, result.kappa = True, False, 0.20, 0.0
    imu = SimpleNamespace(yaw=0.0, gz=0.0)
    v, w, expired = ctrl.command(result, imu, 0.05)
    assert abs(v - BV.V_PREDICT) < 1e-9 and not expired
    result.fresh = True                              # đo được thật: tốc độ bình thường
    v2, _, _ = ctrl.command(result, imu, 0.05)
    assert v2 > BV.V_PREDICT


# ---------------------------------------------- nhảy sang vạch khác (hành vi) ---
def result(gap, jump=0.0):
    k = lane.LaneResult()
    k.found = k.fresh = True
    k.distance = k.dist_ahead = gap
    k.jump_m = jump
    return k


class FakeController:
    """Bộ điều khiển giả: luôn chạy thẳng 0,10 m/s, để chỉ thử luật nhảy vạch / xoay tìm của hành vi."""

    def __init__(self):
        self.tracker = SimpleNamespace(det=SimpleNamespace(reset=lambda: None))

    def command(self, result, imu, dt, v_follow, t=None):
        return 0.10, 0.0, False

    def reset(self):
        pass

    def describe(self):
        return ''


def run(seq):
    beh = behaviors.FollowLane(tracker=None)
    beh.ctrl = FakeController()
    sit = SimpleNamespace(t=0.0, imu=SimpleNamespace(yaw=0.0, gz=0.0), new_line=True)
    out = []
    for k in seq:
        sit.line = k
        out.append(beh.command(sit, 0.05))
        sit.imu.yaw += 0.05 * out[-1][1]           # xe xoay theo lệnh
    return out


def test_implausible_jump_stops_turns_right_then_follows():
    out = run([result(0.20)] * 3 + [result(0.60, jump=0.40)] + [result(0.55)] * 20 + [result(0.21)] * 3 + [result(0.20)] * 2)
    assert out[2] == (0.10, 0.0)
    assert out[3] == (0.0, 0.0), 'nhảy vạch: phải đứng yên ngay'
    rot = out[4:24]
    assert all(v == 0.0 for v, w in rot) and any(w < 0 for v, w in rot), 'phải đứng yên và xoay PHẢI'
    assert out[-1] == (0.10, 0.0), 'thấy vạch phải hợp lý 2 khung -> bám tiếp'


def test_small_or_plausible_jump_does_not_stop():
    out = run([result(0.20)] * 3 + [result(0.25, jump=0.05)] + [result(0.24, jump=0.15)] + [result(0.22)] * 3)
    assert all(v > 0 for v, w in out)
