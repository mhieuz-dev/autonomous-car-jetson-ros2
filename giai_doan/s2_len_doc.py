#!/usr/bin/env python3
"""S2: LÊN DỐC. Lệch hướng ít -> giữ yaw 0 bằng IMU, tăng tốc theo pitch để leo được.
Lệch nhiều -> chậm lại, chỉnh hướng về yaw 0 trước, xong mới tăng tốc.
Không bao giờ đứng yên / lùi trên dốc (luật). Hết dốc (pitch về gần 0) -> S3.

Chạy riêng: đặt xe ở chân dốc, thẳng hướng hầm:  bash chay.sh s2_len_doc
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chung import khung  # noqa: E402
from chung.pid import GiuHuong, kep  # noqa: E402

# ===== THAM SỐ (sửa, Ctrl+S, chạy lại) =====================================
V_CO_BAN = 0.3            # m/s trên dốc (hằng số)
K_PITCH = 0.0              # m/s cộng thêm cho mỗi độ pitch DƯƠNG (dốc thật chỉ ~1-1,5 độ nên cần hệ số lớn: 1,2 độ -> +0,14)
V_MIN = 0.30               # không thấp hơn (luật: không đứng yên trên dốc)
V_MAX = 0.3
LECH_NHIEU_DO = 8.0        # |lệch yaw| > số này -> chế độ chỉnh hướng
V_CHINH_HUONG = 0.12       # tốc độ trong lúc chỉnh hướng (vẫn > 0)
KP_HUONG = 1.8
KI_HUONG = 0.3
KD_HUONG = 0.10
W_MAX = 0.8

NGUONG_DINH_DO = 5.0       # pitch < số này (TẠM: xe chạy nhanh đã ngóc +3, dốc +3,6 thêm vào ~ +6,5) ...
THOI_GIAN_DINH = 0.6       # ... liên tục bao lâu thì coi là lên tới đỉnh -> S3
QUANG_DUONG_MIN = 0.20     # leo ít nhất bấy nhiêu mét rồi mới xét đỉnh
THOI_GIAN_TOI_DA = 15.0
# ===========================================================================


class GiaiDoan(khung.GiaiDoanCoSo):
    TEN = 's2_len_doc'
    HZ = 20

    def __init__(self):
        self.THOI_GIAN_TOI_DA = THOI_GIAN_TOI_DA
        self.huong = GiuHuong(KP_HUONG, KI_HUONG, KD_HUONG, W_MAX)
        self.t_dinh = 0.0
        self.sai = 0.0
        self.che_do = ''
        self.v = 0.0
        self.yaw = 0.0

    def bat_dau(self, ctx):
        self.yaw = ctx.du_lieu.get('yaw_doc', 0.0)     # hướng xe lúc tới chân dốc (S1); chạy riêng thì 0 = hướng lúc bắt đầu

    def buoc(self, ctx, dt):
        imu = ctx.imu
        pitch_do = math.degrees(imu.pitch)
        if pitch_do < NGUONG_DINH_DO and ctx.quang_duong() > QUANG_DUONG_MIN:
            self.t_dinh += dt
            if self.t_dinh >= THOI_GIAN_DINH:
                return khung.KetThuc(f'lên tới đỉnh (pitch {pitch_do:+.1f} độ)')
        else:
            self.t_dinh = 0.0
        w, self.sai = self.huong.tinh(imu, self.yaw, dt)
        if abs(math.degrees(self.sai)) > LECH_NHIEU_DO:
            self.che_do, self.v = 'CHỈNH HƯỚNG', V_CHINH_HUONG
        else:
            self.che_do = 'leo thẳng'
            self.v = kep(V_CO_BAN + K_PITCH * max(0.0, pitch_do), V_MIN, V_MAX)
        return self.v, w

    def trang_thai(self):
        return f'{self.che_do} | v đặt {self.v:.2f} | lệch yaw {math.degrees(self.sai):+.1f} độ'


if __name__ == '__main__':
    khung.main(__file__)
