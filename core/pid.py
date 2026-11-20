"""PID, giữ hướng bằng IMU, giới hạn gia tốc của lệnh."""
import math


def goc(a):
    """Đưa góc về [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def kep(x, lo, hi):
    return max(lo, min(hi, x))


class PID:
    def __init__(self, kp, ki=0.0, kd=0.0, i_max=1.0, ra_max=None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i_max = i_max
        self.ra_max = ra_max
        self.tich = 0.0
        self.e_truoc = None

    def tinh(self, e, dt, de=None):
        """e = sai số; de = đạo hàm sai số nếu đã có (ví dụ -gyro), không thì tự tính."""
        if dt > 0:
            self.tich = kep(self.tich + e * dt, -self.i_max, self.i_max)
        if de is None:
            de = 0.0 if (self.e_truoc is None or dt <= 0) else (e - self.e_truoc) / dt
        self.e_truoc = e
        ra = self.kp * e + self.ki * self.tich + self.kd * de
        return ra if self.ra_max is None else kep(ra, -self.ra_max, self.ra_max)

    def reset(self):
        self.tich = 0.0
        self.e_truoc = None


class GiuHuong:
    """w (rad/s) để giữ yaw đúng yaw_dich. Đạo hàm lấy thẳng từ gyro (không nhiễu vi phân)."""

    def __init__(self, kp, ki=0.0, kd=0.0, w_max=1.0, i_max=0.3):
        self.pid = PID(kp, ki, kd, i_max=i_max, ra_max=w_max)

    def tinh(self, imu, yaw_dich, dt):
        e = goc(yaw_dich - imu.yaw)
        return self.pid.tinh(e, dt, de=-imu.gz), e

    def reset(self):
        self.pid.reset()


class VongGyro:
    """Vòng trong: sửa w gửi xuống driver để tốc độ quay THẬT (gyro) bằng w muốn.

    Giống imu_yaw_loop.py của bản cũ: driver chỉ nhìn encoder nên bánh lệch / trượt
    làm xe tự quay; vòng này đo bằng gyro rồi bù. Tích phân = phần lệch cố định.
    """

    def __init__(self, kp=0.5, ki=1.0, i_max=0.3, sua_max=0.5):
        self.kp, self.ki, self.i_max, self.sua_max = kp, ki, i_max, sua_max
        self.tich = 0.0

    def ap(self, v, w, gz, dt):
        if abs(v) < 1e-3 and abs(w) < 1e-3:
            self.tich = 0.0
            return w
        sai = w - gz
        self.tich = kep(self.tich + sai * dt, -self.i_max, self.i_max)
        return w + kep(self.kp * sai + self.ki * self.tich, -self.sua_max, self.sua_max)


class GioiHan:
    """Giới hạn độ lớn và tốc độ thay đổi của (v, w), giống apply_limits của bản 04a."""

    def __init__(self, gia_toc=0.6, w_max=1.5, w_doi=6.0):
        self.gia_toc, self.w_max, self.w_doi = gia_toc, w_max, w_doi
        self.v = 0.0
        self.w = 0.0

    def ap(self, v, w, dt):
        dv = self.gia_toc * dt
        self.v = kep(v, self.v - dv, self.v + dv)
        w = kep(w, -self.w_max, self.w_max)
        dw = self.w_doi * dt
        self.w = kep(w, self.w - dw, self.w + dw)
        return self.v, self.w

    def dat_lai(self, v=0.0, w=0.0):
        self.v, self.w = v, w
