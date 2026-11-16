"""ĐƯỜNG ĐÃ GHI (dự phòng cho S7): ghi lúc ĐẨY TAY từ đầu S7 tới đích, chạy lại bằng hướng IMU theo quãng đường.

Ghi mỗi BUOC_M mét (quãng đường odom từ đầu S7): s, hướng (rad, so với hướng lúc đầu S7), x, y, v. x, y là vị trí
tính bằng quãng đường encoder + hướng IMU (hệ lúc bắt đầu ghi, x = phía trước), cùng cách xe tự tính lúc phát lại. Chạy lại: ở quãng đường s, giữ hướng = huong(s) + yaw lúc đầu S7. Encoder chỉ để đo quãng đường, hướng lấy từ IMU
nên bánh trượt không làm lệch hướng; điểm bắt đầu phải giống lúc ghi (cùng là đầu S7 trong một lượt full map).
File: calib/duong_s7.csv (ghi tạm vào calib_moi/duong_s7.csv trên xe, `chay.sh keo` rồi `chay.sh nhan_calib`).
"""
import csv
import math

import numpy as np

from . import cau_hinh

BUOC_M = 0.01
TEN_FILE = 'duong_s7.csv'
TEN_FILE_STOP = 'duong_stop.csv'   # đường đẩy tay sau biển STOP (s7_stop_ghi ghi, s7_bam_vach chạy lại)


def goc(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class GhiDuong:
    """Gom mẫu (s, huong, x, y, v) mỗi BUOC_M; gọi cap_nhat mỗi chu kỳ với odom + yaw hiện tại."""

    def __init__(self):
        self.mau = []            # [(s, huong, x, y, v)]
        self.s0 = self.yaw0 = None
        self.x0 = self.y0 = 0.0
        self.s_cuoi = -1e9
        self.t_cuoi = None
        self.s_tich = 0.0         # quãng đường đã tích phân vào x, y
        self.x = self.y = 0.0

    def cap_nhat(self, quang_duong, yaw, x, y, v, t):
        if self.s0 is None:
            self.s0, self.yaw0, self.x0, self.y0 = quang_duong, yaw, x, y
        s = quang_duong - self.s0
        h = goc(yaw - self.yaw0)
        ds = s - self.s_tich               # tích phân vị trí mỗi chu kỳ (x, y odom không dùng: hệ của odom lệch IMU)
        self.x += ds * math.cos(h)
        self.y += ds * math.sin(h)
        self.s_tich = s
        if s - self.s_cuoi < BUOC_M:
            return False
        self.s_cuoi = s
        self.mau.append((round(s, 4), round(h, 5), round(self.x, 4), round(self.y, 4), round(v, 3)))
        return True

    @property
    def dai(self):
        return self.mau[-1][0] if self.mau else 0.0

    def luu(self, duong_dan):
        duong_dan.parent.mkdir(parents=True, exist_ok=True)
        with open(duong_dan, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['s_m', 'huong_rad', 'x_m', 'y_m', 'v_mps'])
            w.writerows(self.mau)
        return len(self.mau)


class DuongDaGhi:
    """Đọc calib/duong_s7.csv; huong(s) nội suy theo quãng đường (góc, nội suy qua sin/cos)."""

    def __init__(self, duong_dan=None):
        duong_dan = duong_dan or (cau_hinh.CALIB / TEN_FILE)
        with open(duong_dan, newline='') as f:
            r = list(csv.DictReader(f))
        if len(r) < 2:
            raise ValueError(f'{duong_dan}: chưa có đường ghi (ít hơn 2 mẫu)')
        self.s = np.array([float(d['s_m']) for d in r])
        h = np.array([float(d['huong_rad']) for d in r])
        self.cos, self.sin = np.cos(h), np.sin(h)
        self.v = np.array([float(d['v_mps']) for d in r])
        self.xy = np.array([[float(d['x_m']), float(d['y_m'])] for d in r])
        self.dai = float(self.s[-1])

    def huong(self, s):
        """Hướng (rad, so với hướng đầu S7) tại quãng đường s; ngoài khoảng thì lấy đầu / cuối."""
        return math.atan2(float(np.interp(s, self.s, self.sin)), float(np.interp(s, self.s, self.cos)))

    def diem(self, s):
        """Vị trí (x, y) đã ghi tại quãng đường s."""
        return float(np.interp(s, self.s, self.xy[:, 0])), float(np.interp(s, self.s, self.xy[:, 1]))

    def toc_do(self, s):
        return float(np.interp(s, self.s, self.v))
