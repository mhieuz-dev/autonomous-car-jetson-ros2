"""Góc xe từ IMU: yaw (hướng), pitch (dốc), roll.

IMU của xe KHÔNG gửi sẵn góc (orientation rỗng), chỉ có gyro + gia tốc, nên:
  - yaw   = tích phân gyro z sau khi trừ độ lệch (bias) đo lúc đứng yên.
            + = quay TRÁI. Gốc 0 = hướng mũi xe lúc hiệu chuẩn.
  - pitch = lọc bù: gyro y ngắn hạn + hướng trọng lực từ gia tốc dài hạn.
            + = mũi xe NGÓC LÊN (đang leo dốc). Gốc 0 = lúc hiệu chuẩn.
  - roll  = tương tự pitch, chỉ để hiển thị.
Công thức pitch giống bản 04a (atan2(-ax, hypot(ay, az)) trộn với gyro y), trên
bag 08/10 nó cho số DƯƠNG khi leo dốc. imu_test.py kiểm lại: nâng mũi xe -> dương.
Nếu ra âm thì đổi DAU_PITCH = -1.
"""
import math

DAU_PITCH = 1.0      # đổi thành -1.0 nếu imu_test báo nâng mũi xe mà pitch âm
HE_SO_LOC = 0.98     # phần tin gyro trong lọc bù (0.98 như bản 04a)


class IMU:
    def __init__(self, dau_pitch=DAU_PITCH, he_so=HE_SO_LOC):
        self.dau = dau_pitch
        self.k = he_so
        self.bias = [0.0, 0.0, 0.0]          # gyro x, y, z lúc đứng yên (rad/s)
        self.q = None                        # pitch thô (quy ước công thức cũ)
        self.r = None                        # roll thô
        self.q0 = 0.0
        self.r0 = 0.0
        self.yaw = 0.0
        self.t = None
        self.gz = 0.0                        # tốc độ quay hiện tại (đã trừ bias), rad/s
        self.so_mau = 0
        self._gom = None                     # đang gom mẫu để hiệu chuẩn

    # ------------------------------------------------------------ cập nhật ---
    def cap_nhat(self, t, ax, ay, az, gx, gy, gz):
        q_acc = math.atan2(-ax, math.hypot(ay, az))
        r_acc = math.atan2(ay, az)
        if self._gom is not None:
            self._gom.append((ax, ay, az, gx, gy, gz, q_acc, r_acc))
        dt = None if self.t is None else t - self.t
        self.t = t
        self.so_mau += 1
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
        return 0.0 if self.q is None else self.dau * (self.q - self.q0)

    @property
    def roll(self):
        return 0.0 if self.r is None else self.r - self.r0

    # ----------------------------------------------------------- hiệu chuẩn ---
    def bat_dau_gom(self):
        self._gom = []

    def ket_thuc_gom(self):
        """Xe đứng yên trong lúc gom: lấy bias gyro, đặt yaw = 0, pitch = roll = 0.

        Trả dict số đo để kiểm checklist (trọng lực, bias, độ rung gyro).
        """
        mau, self._gom = self._gom or [], None
        if len(mau) < 10:
            raise RuntimeError(f'IMU chỉ có {len(mau)} mẫu trong lúc hiệu chuẩn (cần >= 10)')
        n = len(mau)
        tb = [sum(m[i] for m in mau) / n for i in range(8)]
        do_rung = math.sqrt(sum((m[5] - tb[5]) ** 2 for m in mau) / n)
        self.bias = [tb[3], tb[4], tb[5]]
        self.q = self.q0 = tb[6]
        self.r = self.r0 = tb[7]
        self.yaw = 0.0
        return {'so_mau': n, 'trong_luc': math.sqrt(tb[0] ** 2 + tb[1] ** 2 + tb[2] ** 2),
                'bias_gz': tb[5], 'bias_gy': tb[4], 'rung_gz': do_rung}

    def dat_yaw_0(self):
        self.yaw = 0.0
