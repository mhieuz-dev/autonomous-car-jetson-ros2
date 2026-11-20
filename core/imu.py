"""Góc xe từ IMU: yaw (hướng), pitch (dốc), roll.

IMU của xe KHÔNG gửi sẵn góc (orientation rỗng), chỉ có gyro + gia tốc, nên:
  - yaw   = tích phân gyro z sau khi trừ độ lệch (bias) đo lúc đứng yên.
            + = quay TRÁI. Gốc 0 = hướng mũi xe lúc hiệu chuẩn.
  - pitch = lọc bù: gyro y ngắn hạn + hướng trọng lực từ gia tốc dài hạn.
            + = mũi xe NGÓC LÊN (đang leo dốc). Gốc 0 = lúc hiệu chuẩn.
  - roll  = tương tự pitch, chỉ để hiển thị.
Công thức pitch giống bản 04a (atan2(-ax, hypot(ay, az)) trộn với gyro y), trên
bag 08/10 nó cho số DƯƠNG khi leo dốc. imu_test.py kiểm lại: nâng mũi xe -> dương.
Nếu ra âm thì đổi PITCH_SIGN = -1.
"""
import math

PITCH_SIGN = 1.0     # đổi thành -1.0 nếu imu_test báo nâng mũi xe mà pitch âm
FILTER_ALPHA = 0.98  # phần tin gyro trong lọc bù (0.98 như bản 04a)


class IMU:
    def __init__(self, pitch_sign=PITCH_SIGN, alpha=FILTER_ALPHA):
        self.sign = pitch_sign
        self.k = alpha
        self.bias = [0.0, 0.0, 0.0]  # gyro x, y, z lúc đứng yên (rad/s)
        self.q = None                # pitch thô (quy ước công thức cũ)
        self.r = None                # roll thô
        self.q0 = 0.0
        self.r0 = 0.0
        self.yaw = 0.0
        self.t = None
        self.gz = 0.0                        # tốc độ quay hiện tại (đã trừ bias), rad/s
        self.n_samples = 0
        self._samples = None                     # đang gom mẫu để hiệu chuẩn

    # ------------------------------------------------------------ cập nhật ---
    def update(self, t, ax, ay, az, gx, gy, gz):
        q_acc = math.atan2(-ax, math.hypot(ay, az))
        r_acc = math.atan2(ay, az)
        if self._samples is not None:
            self._samples.append((ax, ay, az, gx, gy, gz, q_acc, r_acc))
        dt = None if self.t is None else t - self.t
        self.t = t
        self.n_samples += 1
        if self.q is None or dt is None or not 0.0 < dt < 0.2:
            self.q = q_acc if self.q is None else self.q
            self.r = r_acc if self.r is None else self.r
            return
        self.q = self.k * (self.q + (gy - self.bias[1]) * dt) + (1 - self.k) * q_acc
        self.r = self.k * (self.r + (gx - self.bias[0]) * dt) + (1 - self.k) * r_acc
        self.gz = gz - self.bias[2]
        self.yaw += self.gz * dt

    @property
    def pitch(self):
        return 0.0 if self.q is None else self.sign * (self.q - self.q0)

    @property
    def roll(self):
        return 0.0 if self.r is None else self.r - self.r0

    # ----------------------------------------------------------- hiệu chuẩn ---
    def start_sampling(self):
        self._samples = []

    def finish_sampling(self):
        """Xe đứng yên trong lúc gom: lấy bias gyro, đặt yaw = 0, pitch = roll = 0.

        Trả dict số đo để kiểm checklist (trọng lực, bias, độ rung gyro).
        """
        samples, self._samples = self._samples or [], None
        if len(samples) < 10:
            raise RuntimeError(f'IMU chỉ có {len(samples)} mẫu trong lúc hiệu chuẩn (cần >= 10)')
        n = len(samples)
        mean = [sum(m[i] for m in samples) / n for i in range(8)]
        jitter = math.sqrt(sum((m[5] - mean[5]) ** 2 for m in samples) / n)
        self.bias = [mean[3], mean[4], mean[5]]
        self.q = self.q0 = mean[6]
        self.r = self.r0 = mean[7]
        self.yaw = 0.0
        return {'n_samples': n, 'gravity': math.sqrt(mean[0] ** 2 + mean[1] ** 2 + mean[2] ** 2),
                'bias_gz': mean[5], 'bias_gy': mean[4], 'jitter_gz': jitter}

    def zero_yaw(self):
        self.yaw = 0.0
