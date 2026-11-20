"""PID, giữ hướng bằng IMU, giới hạn gia tốc của lệnh."""
import math


def wrap_angle(a):
    """Đưa góc về [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class PID:
    def __init__(self, kp, ki=0.0, kd=0.0, i_max=1.0, out_max=None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i_max = i_max
        self.out_max = out_max
        self.integral = 0.0
        self.e_prev = None

    def compute(self, e, dt, de=None):
        """e = sai số; de = đạo hàm sai số nếu đã có (ví dụ -gyro), không thì tự tính."""
        if dt > 0:
            self.integral = clamp(self.integral + e * dt, -self.i_max, self.i_max)
        if de is None:
            de = 0.0 if (self.e_prev is None or dt <= 0) else (e - self.e_prev) / dt
        self.e_prev = e
        out = self.kp * e + self.ki * self.integral + self.kd * de
        return out if self.out_max is None else clamp(out, -self.out_max, self.out_max)

    def reset(self):
        self.integral = 0.0
        self.e_prev = None


class HeadingHold:
    """w (rad/s) để giữ yaw đúng yaw_target. Đạo hàm lấy thẳng từ gyro (không nhiễu vi phân)."""

    def __init__(self, kp, ki=0.0, kd=0.0, w_max=1.0, i_max=0.3):
        self.pid = PID(kp, ki, kd, i_max=i_max, out_max=w_max)

    def compute(self, imu, yaw_target, dt):
        e = wrap_angle(yaw_target - imu.yaw)
        return self.pid.compute(e, dt, de=-imu.gz), e

    def reset(self):
        self.pid.reset()


class GyroLoop:
    """Vòng trong: sửa w gửi xuống driver để tốc độ quay THẬT (gyro) bằng w muốn.

    Giống imu_yaw_loop.py của bản cũ: driver chỉ nhìn encoder nên bánh lệch / trượt
    làm xe tự quay; vòng này đo bằng gyro rồi bù. Tích phân = phần lệch cố định.
    """

    def __init__(self, kp=0.5, ki=1.0, i_max=0.3, corr_max=0.5):
        self.kp, self.ki, self.i_max, self.corr_max = kp, ki, i_max, corr_max
        self.integral = 0.0

    def parser(self, v, w, gz, dt):
        if abs(v) < 1e-3 and abs(w) < 1e-3:
            self.integral = 0.0
            return w
        err = w - gz
        self.integral = clamp(self.integral + err * dt, -self.i_max, self.i_max)
        return w + clamp(self.kp * err + self.ki * self.integral, -self.corr_max, self.corr_max)


class RateLimiter:
    """Giới hạn độ lớn và tốc độ thay đổi của (v, w), giống apply_limits của bản 04a."""

    def __init__(self, accel=0.6, w_max=1.5, w_accel=6.0):
        self.accel, self.w_max, self.w_accel = accel, w_max, w_accel
        self.v = 0.0
        self.w = 0.0

    def parser(self, v, w, dt):
        dv = self.accel * dt
        self.v = clamp(v, self.v - dv, self.v + dv)
        w = clamp(w, -self.w_max, self.w_max)
        dw = self.w_accel * dt
        self.w = clamp(w, self.w - dw, self.w + dw)
        return self.v, self.w

    def reset(self, v=0.0, w=0.0):
        self.v, self.w = v, w
