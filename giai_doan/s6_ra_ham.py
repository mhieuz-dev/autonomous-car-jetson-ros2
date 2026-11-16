#!/usr/bin/env python3
"""S6: RA HẦM. Trôi chậm THOI_GIAN_TROI giây, giữ đúng hướng lúc ra hầm (IMU).
Camera vẫn dò vạch suốt lúc trôi (bộ dò dùng chung tham số với S7) để S7 bắt
vạch ngay. Nếu CHO_PHEP_SOM và đã thấy vạch đủ khung thì sang S7 sớm.

Chạy riêng: đặt xe ở cửa ra hầm:  bash chay.sh s6_ra_ham
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chung import khung  # noqa: E402
from chung.pid import GiuHuong  # noqa: E402
from chung.bam_vach import BamVach  # noqa: E402

# ===== THAM SỐ (sửa, Ctrl+S, chạy lại) =====================================
V_TROI = 0.15
THOI_GIAN_TROI = 2.0       # giây: ra hầm trôi ĐÚNG 2 s rồi mới sang S7 (bird's eye dò vạch)
CHO_PHEP_SOM = False       # thấy vạch chắc chắn thì sang S7 trước khi hết giờ trôi
SO_KHUNG_THAY = 5          # số khung liên tiếp đo được vạch mới tính là thấy
THOI_GIAN_TROI_MIN = 0.5   # trôi ít nhất bấy nhiêu giây rồi mới cho sang sớm
KP_HUONG = 1.5
KI_HUONG = 0.0
KD_HUONG = 0.10
W_MAX = 0.6
THOI_GIAN_TOI_DA = 10.0
# ===========================================================================


class GiaiDoan(khung.GiaiDoanCoSo):
    TEN = 's6_ra_ham'
    HZ = 20
    CAN_CAMERA = True

    def __init__(self):
        self.THOI_GIAN_TOI_DA = THOI_GIAN_TOI_DA
        self.huong = GiuHuong(KP_HUONG, KI_HUONG, KD_HUONG, W_MAX)
        self.bam = None
        self.yaw_giu = 0.0
        self.lien = 0

    def bat_dau(self, ctx):
        self.bam = BamVach(ctx.gia_lap)
        self.yaw_giu = ctx.du_lieu.get('yaw_ra_ham', ctx.imu.yaw)

    def buoc(self, ctx, dt):
        kq, moi = self.bam.cap_nhat(ctx)
        if moi:
            self.lien = self.lien + 1 if kq.do_moi else 0
        t = ctx.thoi_gian()
        if CHO_PHEP_SOM and t >= THOI_GIAN_TROI_MIN and self.lien >= SO_KHUNG_THAY:
            return khung.KetThuc(f'thấy vạch phải sau {t:.1f} s')
        if t >= THOI_GIAN_TROI:
            return khung.KetThuc(f'hết {THOI_GIAN_TROI:.1f} s trôi')
        w, _ = self.huong.tinh(ctx.imu, self.yaw_giu, dt)
        return V_TROI, w

    def trang_thai(self):
        if self.bam is None:
            return ''
        kq = self.bam.kq
        d = f'd={kq.khoang_cach:.2f}' if kq.thay else 'chưa thấy vạch'
        return f'giữ yaw {math.degrees(self.yaw_giu):+.0f} độ | {d} | liên tiếp {self.lien}'


if __name__ == '__main__':
    khung.main(__file__)
