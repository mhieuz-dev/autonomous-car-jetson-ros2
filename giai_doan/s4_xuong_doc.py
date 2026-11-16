#!/usr/bin/env python3
"""S4: XUỐNG DỐC thẳng vào miệng hầm. Khoá yaw 0 bằng IMU, không lùi, không dừng.
LiDAR thấy cổng hầm (2 tường, tham số trong s5_ham.py) đủ số khung -> S5.

Chạy riêng: đặt xe đầu dốc xuống, thẳng hướng hầm:  bash chay.sh s4_xuong_doc
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chung import khung  # noqa: E402
from chung.pid import GiuHuong  # noqa: E402
from giai_doan import s5_ham  # noqa: E402

# ===== THAM SỐ (sửa, Ctrl+S, chạy lại) =====================================
V_XUONG = 0.15             # m/s (tăng tốc vào hầm; trước là 0.15)
KP_HUONG = 2.0
KI_HUONG = 0.3
KD_HUONG = 0.10
W_MAX = 0.8
QUANG_DUONG_TOI_DA = 1.5   # quá bấy nhiêu mét chưa thấy hầm -> dừng
THOI_GIAN_TOI_DA = 15.0
# ===========================================================================


class GiaiDoan(khung.GiaiDoanCoSo):
    TEN = 's4_xuong_doc'
    HZ = 20

    def __init__(self):
        self.THOI_GIAN_TOI_DA = THOI_GIAN_TOI_DA
        self.huong = GiuHuong(KP_HUONG, KI_HUONG, KD_HUONG, W_MAX)
        self.ham = s5_ham.DemHam()
        self.sai = 0.0
        self.yaw = 0.0

    def bat_dau(self, ctx):
        self.yaw = ctx.du_lieu.get('yaw_doc', 0.0)

    def buoc(self, ctx, dt):
        if self.ham.cap_nhat(ctx.xe.scan) >= s5_ham.SO_KHUNG_VAO:
            ctx.du_lieu['da_thay_ham'] = True
            return khung.KetThuc('VÀO HẦM (LiDAR thấy 2 tường)')
        if ctx.quang_duong() > QUANG_DUONG_TOI_DA:
            return khung.KetThuc(f'đi {QUANG_DUONG_TOI_DA} m chưa thấy hầm', dung_han=True)
        w, self.sai = self.huong.tinh(ctx.imu, self.yaw, dt)
        return V_XUONG, w

    def trang_thai(self):
        return f'lệch yaw {math.degrees(self.sai):+.1f} độ | cổng hầm {self.ham.lien}/{s5_ham.SO_KHUNG_VAO}'


if __name__ == '__main__':
    khung.main(__file__)
