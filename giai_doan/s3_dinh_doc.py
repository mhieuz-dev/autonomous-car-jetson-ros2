#!/usr/bin/env python3
"""S3: ĐỈNH DỐC. Đỉnh dốc và miệng hầm thẳng hàng: giữ đúng yaw 0 (góc lúc xuất phát),
chạy chậm đều. Pitch âm (bắt đầu xuống) -> S4. Nếu LiDAR đã thấy hầm luôn -> nhảy S5.

Chạy riêng: đặt xe trên đỉnh dốc, thẳng hướng hầm:  bash chay.sh s3_dinh_doc
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chung import khung  # noqa: E402
from chung.pid import GiuHuong  # noqa: E402
from giai_doan import s5_ham  # noqa: E402

# ===== THAM SỐ (sửa, Ctrl+S, chạy lại) =====================================
V_DINH = 0.25              # m/s (tăng tốc vào hầm; trước là 0.15)
KP_HUONG = 1.8
KI_HUONG = 0.3
KD_HUONG = 0.10
W_MAX = 0.8
NGUONG_XUONG_DO = -2.2     # pitch < số này (TẠM: phẳng ở 0,15 m/s dao động -1,9..+1,5; XUỐNG thật -3,1) ...
THOI_GIAN_XUONG = 0.5      # ... liên tục bao lâu thì sang S4
QUANG_DUONG_TOI_DA = 1.5   # quá bấy nhiêu mét chưa thấy dốc xuống / hầm -> dừng
THOI_GIAN_TOI_DA = 15.0
# ===========================================================================


class GiaiDoan(khung.GiaiDoanCoSo):
    TEN = 's3_dinh_doc'
    HZ = 20

    def __init__(self):
        self.THOI_GIAN_TOI_DA = THOI_GIAN_TOI_DA
        self.huong = GiuHuong(KP_HUONG, KI_HUONG, KD_HUONG, W_MAX)
        self.ham = s5_ham.DemHam()
        self.t_xuong = 0.0
        self.sai = 0.0
        self.yaw = 0.0

    def bat_dau(self, ctx):
        self.yaw = ctx.du_lieu.get('yaw_doc', 0.0)

    def buoc(self, ctx, dt):
        if self.ham.cap_nhat(ctx.xe.scan) >= s5_ham.SO_KHUNG_VAO:
            ctx.du_lieu['da_thay_ham'] = True
            return khung.KetThuc('LiDAR đã thấy hầm', tiep='s5_ham')
        if math.degrees(ctx.imu.pitch) < NGUONG_XUONG_DO:
            self.t_xuong += dt
            if self.t_xuong >= THOI_GIAN_XUONG:
                return khung.KetThuc(f'bắt đầu xuống dốc (pitch {math.degrees(ctx.imu.pitch):+.1f} độ)')
        else:
            self.t_xuong = 0.0
        if ctx.quang_duong() > QUANG_DUONG_TOI_DA:
            return khung.KetThuc(f'đi {QUANG_DUONG_TOI_DA} m chưa thấy dốc xuống/hầm', dung_han=True)
        w, self.sai = self.huong.tinh(ctx.imu, self.yaw, dt)
        return V_DINH, w

    def trang_thai(self):
        return f'lệch yaw {math.degrees(self.sai):+.1f} độ | cổng hầm {self.ham.lien}'


if __name__ == '__main__':
    khung.main(__file__)
