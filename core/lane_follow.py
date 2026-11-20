"""Bám vạch liền bên phải: bộ dò 8 bước (core/lane.py) + giữ độ cong khi mất vạch + PID theo khoảng cách.

  d     = khoảng cách tới vạch phải, đo ở điểm X_LOOK (0.35 m) phía trước mũi xe.
          Đo ở đây vì camera NHÌN RÕ điểm này; kéo dài đường cong về sát xe (camera
          không thấy) thì sai góc tới 5 độ.
  e     = d - D_TARGET          (e > 0: xe đang xa vạch / lệch trái -> phải quẹo PHẢI)
  kappa_cmd = -(KP*e + KI*integral(e) + KD*de/dt)
  w     = v * kappa_cmd
  KP = 2 / X_LOOK^2 là luật "pure pursuit": điểm tâm làn cách X_LOOK phía trước
  nằm đúng trên cung tròn xe sẽ chạy, nên trên khúc cua tròn xe bám đúng tâm làn
  mà không cần đoán góc / độ cong.
  v     = V_FOLLOW giảm theo |e| và độ cong vạch
Không đo được vạch (hoặc vạch đo ngoài dải hợp lý HOP_LY_GAN..XA): dời cả đường cong theo odom + IMU (giữ độ cong)
tối đa HOLD_PLAUSIBLE_S s, chạy chậm V_PREDICT; lâu hơn thì MẤT VẠCH: chạy chậm giữ hướng.
"""
import math

import cv2
import numpy as np

from . import lane
from .camera import BirdEye, active_calib, read_bev, decode_gray_half
from .pid import PID, HeadingHold, clamp

# ===== THAM SỐ PID KHOẢNG CÁCH (sửa, Ctrl+S, chạy lại) =======================
V_FOLLOW = 0.25               # m/s trên đường thẳng
V_MIN = 0.1
SLOW_PER_ERR = 1.5    # v *= 1 - SLOW_PER_ERR*|e| (e tính bằng m)
SLOW_PER_CURV = 0.35  # v /= 1 + SLOW_PER_CURV*|kappa|
D_TARGET = 0.20       # m: muốn tâm xe cách vạch phải bấy nhiêu (nửa làn 0.40)
KP = 16.0             # 1/m^2; pure pursuit = 2 / X_LOOK^2 = 16.3 khi X_LOOK 0.35
KI = 0.0              # bù lệch cố định (vòng gyro trong khung đã bù bánh lệch)
KD = 1.0              # giảm dao động: phản ứng theo tốc độ thay đổi của e
W_MAX = 1.2
V_LINE_LOST = 0.1  # chạy chậm giữ hướng khi mất vạch
LINE_LOST_S = 3.0  # mất vạch lâu hơn (s) thì báo mất hẳn (bên gọi quyết định dừng / xoay tìm)

# ===== THAM SỐ BỘ DÒ VẠCH (xem giải thích từng bước trong core/lane.py) ====
DETECTOR_PARAMS = dict(
    X_NEAR=0.15, X_FAR=1.00, M_PX=0.005,
    Y_ROI_LEFT=0.05, Y_ROI_RIGHT=-0.60,
    MAX_WIDTH_M=0.045, MIN_WIDTH_M=0.010,
    MIN_THRESH=18, K_STD=3.0,
    GAP_JOIN_M=0.06, MIN_LEN_M=0.12, MAX_ANGLE_DEG=50.0,
    Y_EXPECTED=-0.20, WINDOW_M=0.08,
    RANSAC_ITERS=40, RANSAC_TOL_M=0.012, MIN_POINTS=25, MIN_CENTER_X_M=0.08, C_MAX=1.5,
    X_LOOK=0.35, WEIGHT_TAU_M=0.40, X_FIT_MAX=0.90, JUMP_GATE_M=0.05, MAX_HOLD=0.5,
)
# ===== DỰ ĐOÁN GIỮ ĐỘ CONG + CỔNG VẠCH PHẢI HỢP LÝ =================
HOLD_PLAUSIBLE_S = 3.0   # không có vạch hợp lý thì dự đoán tiếp bấy nhiêu giây rồi mới coi là MẤT VẠCH (bản cũ: 0.5)
PLAUSIBLE_NEAR_M = 0.10  # vạch phải hợp lý: cách tâm xe tại X_LOOK từ ...
PLAUSIBLE_FAR_M = 0.30   # ... tới chừng này (m). Cua gắt vạch đúng cũng có thể ra ngoài: nới XA nếu cần
V_PREDICT = V_LINE_LOST  # m/s lúc chạy bằng dự đoán (không có vạch hợp lý)
X_BEHIND_M = 0.30        # dời đường cong: lấy điểm từ bấy nhiêu mét SAU xe tới X_FAR để khớp lại (giữ phần đã qua)
# ===========================================================================


def shift_curve(coef, ds, dyaw, x_far, x_behind=X_BEHIND_M):
    """Đường y = a + b x + c x^2 trong hệ xe CŨ -> hệ xe MỚI sau khi xe tiến ds (m) và quay dyaw (rad, + trái).
    Điểm cố định trên sàn: p' = R(-dyaw) (p - [ds, 0]); dời 40 điểm rồi khớp lại bậc 2 (phép dời không đổi hình
    dạng nên độ cong được giữ, chỉ vị trí và hướng đổi theo xe)."""
    a, b, c = coef
    xs = np.linspace(-x_behind, x_far, 40)
    ys = a + b * xs + c * xs * xs
    cos_a, sin_a = math.cos(dyaw), math.sin(dyaw)
    x = xs - ds
    c2, b2, a2 = np.polyfit(cos_a * x + sin_a * ys, -sin_a * x + cos_a * ys, 2)
    return float(a2), float(b2), float(c2)


def y_psi(coef, x):
    a, b, c = coef
    return a + b * x + c * x * x, math.atan(b + 2 * c * x)


class CurveHoldDetector(lane.LaneDetector):
    """Bộ dò 8 bước của Champion + (1) dự đoán giữ độ cong, (2) cổng vạch phải hợp lý."""

    def __init__(self, birdeye, p, near=PLAUSIBLE_NEAR_M, far=PLAUSIBLE_FAR_M):
        super().__init__(birdeye, p)
        self.near, self.far = near, far
        self.out_of_band = None          # khoảng cách của vạch vừa bị cổng bỏ (m), None = không bỏ gì

    # (1) ---------------------------------------------------------------
    def predict(self, ds, dyaw):
        if self.y is None:
            return
        if self.coef is None:
            return super().predict(ds, dyaw)
        p = self.p
        new_coef = shift_curve(self.coef, ds, dyaw, p.X_FAR)
        y_old, psi_old = y_psi(self.coef, p.X_LOOK)
        y_new, psi_new = y_psi(new_coef, p.X_LOOK)
        self.y += y_new - y_old             # giữ phần Kalman đã lọc, chỉ cộng mức dời do xe chuyển động
        self.psi += psi_new - psi_old
        self.coef = new_coef
        a, b, c = new_coef
        self.kappa = 2 * c / (1 + b * b) ** 1.5
        self.Py += p.Q_Y
        self.Ppsi += p.Q_PSI

    # (2) ---------------------------------------------------------------
    def extract(self, gray_half, debug=False):
        coef, count, image = super().extract(gray_half, debug)
        self.out_of_band = None
        if coef is not None:
            gap = -y_psi(coef, self.p.X_LOOK)[0]          # y âm = bên phải
            if not (self.near <= gap <= self.far):
                self.out_of_band = gap
                if debug and '1 birdeye' in image:
                    image['6 fit'] = self._draw_fit(image['1 birdeye'], None)
                    self._draw_curve(image['6 fit'], coef, (140, 140, 140))
                return None, count, image
        return coef, count, image

    def process(self, gray_half, t, ds=0.0, dyaw=0.0, debug=False):
        result = super().process(gray_half, t, ds, dyaw, debug)
        result.out_of_band = self.out_of_band
        if self.out_of_band is not None:
            result.reason = f'vạch cách {self.out_of_band:.2f} m: ngoài dải hợp lý {self.near:.2f}..{self.far:.2f}'
        if debug and result.found and not result.fresh and self.coef is not None and '6 fit' in result.image:
            self._draw_curve(result.image['6 fit'], self.coef, (0, 220, 255))    # vàng: đường cong đang dự đoán
        return result

    def _draw_curve(self, img, coef, color):
        a, b, c = coef
        xs = np.linspace(self.p.X_NEAR, self.p.X_FAR, 40)
        r, k = self.birdeye.m_to_px(xs, a + b * xs + c * xs * xs)
        cv2.polylines(img, [np.column_stack([k, r]).astype(np.int32)], False, color, 2)


def make_detector(sim=False):
    """Vùng nhìn + độ phân giải bird's eye: lấy từ config/bev.yaml (bạn chọn ở bước 4b); không có thì dùng
    X_NEAR, X_FAR, M_PX trong DETECTOR_PARAMS và bề rộng ±0,5 m."""
    p = lane.params(**DETECTOR_PARAMS)
    p.MAX_HOLD = HOLD_PLAUSIBLE_S
    bev = None if sim else read_bev()
    y_left, y_right = 0.5, -0.5
    if bev:
        p.X_NEAR, p.X_FAR, p.M_PX = bev['x_near'], bev['x_far'], bev['m_px']
        y_left, y_right = bev['y_left'], bev['y_right']
    birdeye = BirdEye(active_calib(sim), x_near=p.X_NEAR, x_far=p.X_FAR, y_left=y_left, y_right=y_right, m_px=p.M_PX)
    return CurveHoldDetector(birdeye, p)


class LaneTracker:
    """Gom: lấy khung mới -> dò vạch (dự đoán bằng odom + IMU) -> kết quả."""

    def __init__(self, sim=False):
        self.det = make_detector(sim)
        self.old_count = 0
        self.result = lane.LaneResult()
        self.old_dist = None
        self.old_yaw = None
        self.image = {}

    def update(self, ctx, debug=False):
        jpeg, count = ctx.car.camera_jpeg()
        if jpeg is None or count == self.old_count:
            return self.result, False
        self.old_count = count
        dist, yaw = ctx.car.odom.distance, ctx.imu.yaw
        ds = 0.0 if self.old_dist is None else dist - self.old_dist
        dyaw = 0.0 if self.old_yaw is None else yaw - self.old_yaw
        self.old_dist, self.old_yaw = dist, yaw
        self.result = self.det.process(decode_gray_half(jpeg), ctx.car.now(), ds, dyaw, debug)
        self.image = self.result.image
        return self.result, True


class LaneController:
    """PID khoảng cách tới vạch phải (xem đầu file), tính lệnh từ kết quả dò vạch của LaneTracker.

    command() trả (v, w, expired). Mất vạch thì chạy chậm giữ hướng; expired = True khi mất vạch lâu hơn
    LINE_LOST_S giây, bên gọi tự quyết dừng hay chạy tiếp."""

    def __init__(self, sim=False, tracker=None):
        self.tracker = tracker or LaneTracker(sim)
        self.pid = PID(KP, KI, KD, i_max=0.2)
        self.heading = HeadingHold(1.5, 0.0, 0.1, 0.6)
        self.t_lost = 0.0
        self.yaw_hold = None
        self.e = 0.0
        self._t = 0.0

    def reset(self):
        self.pid.reset()
        self.heading.reset()

    def command(self, result, imu, dt, v_follow=V_FOLLOW, t=None):
        if t is not None:
            self._t = t
        if not result.found:
            self.t_lost += dt
            if self.yaw_hold is None:
                self.yaw_hold = imu.yaw
            w, _ = self.heading.compute(imu, self.yaw_hold, dt)
            return V_LINE_LOST, w, self.t_lost > LINE_LOST_S
        self.t_lost, self.yaw_hold = 0.0, None
        self.e = result.dist_ahead - D_TARGET
        v = v_follow * max(0.0, 1.0 - SLOW_PER_ERR * abs(self.e)) / (1.0 + SLOW_PER_CURV * abs(result.kappa))
        v = max(V_MIN, v)
        kappa_cmd = -self.pid.compute(self.e, dt)
        w = clamp(v * kappa_cmd, -W_MAX, W_MAX)
        if not result.fresh and v > V_PREDICT:          # chạy bằng dự đoán: chậm lại, giữ nguyên độ cong lệnh
            w, v = w * V_PREDICT / v, V_PREDICT
        return v, w, False

    def describe(self):
        result = self.tracker.result
        if not result.found:
            return f'MẤT VẠCH {self.t_lost:.1f} s ({result.reason})'
        predict = '' if result.fresh else f'DỰ ĐOÁN {self._t - self.tracker.det.t_measure:.1f} s ({result.reason or "không đo được"}) | '
        return predict + (f'd={result.dist_ahead:.3f} m e={self.e:+.3f} psi={math.degrees(result.psi):+.1f} '
                f'k={result.kappa:+.2f} diem={result.n_points} {result.ms:.0f} ms{" (giữ)" if not result.fresh else ""}')
