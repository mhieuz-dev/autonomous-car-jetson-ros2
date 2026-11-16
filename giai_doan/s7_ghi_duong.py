#!/usr/bin/env python3
"""S7 GHI ĐƯỜNG (dự phòng): từ đầu S7 tới đích xe KHÔNG tự chạy, bạn ĐẨY TAY theo đường thi; xe chỉ ghi lại quãng
đường (encoder qua odom) + hướng IMU mỗi 1 cm. Tới đích bấm DỪNG KHẨN / Ctrl+C: file được lưu khi kết thúc.

Để điểm bắt đầu đúng như lúc thi: đặt S7_DUNG = 's7_ghi_duong' trong chung/cau_hinh.py rồi CHẠY TẤT CẢ (full map);
xe tự chạy S0 tới S6, vào S7 thì thả bánh tự do, bạn đẩy. Hoặc chạy riêng: bash chay.sh s7_ghi_duong (đặt xe đúng chỗ
S7 bắt đầu, sau khi ra hầm trôi 2 s).
File ghi: calib_moi/duong_s7.csv trên xe -> `bash chay.sh keo` -> `bash chay.sh nhan_calib` (chép vào calib/) -> commit.
Chạy lại bằng s7_phat_lai (S7_DUNG = 's7_phat_lai').
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chung import cau_hinh, khung  # noqa: E402
from chung.duong_ghi import TEN_FILE, GhiDuong  # noqa: E402
from chung.log import log  # noqa: E402

# ===== THAM SỐ =============================================================
THOI_GIAN_TOI_DA = 600.0   # s: đẩy tay tối đa bấy nhiêu (bấm dừng khẩn khi tới đích là xong sớm)
# ===========================================================================


class GiaiDoan(khung.GiaiDoanCoSo):
    TEN = 's7_ghi_duong'
    HZ = 20
    CAN_CAMERA = False
    KHONG_GUI = True             # KHÔNG gửi /cmd_vel: bánh quay tự do để đẩy tay

    def __init__(self):
        self.THOI_GIAN_TOI_DA = THOI_GIAN_TOI_DA
        self.ghi = GhiDuong()
        self.v = 0.0
        self.t_cu = None
        self.qd_cu = None

    def bat_dau(self, ctx):
        log(f'GHI ĐƯỜNG: xe thả bánh tự do, ĐẨY TAY từ đây tới đích rồi bấm DỪNG KHẨN. Ghi mỗi 1 cm '
            f'(hướng IMU so với hướng lúc này {math.degrees(ctx.imu.yaw):+.1f} độ)')

    def buoc(self, ctx, dt):
        o = ctx.xe.odom
        t = ctx.xe.gio()
        if self.t_cu is not None and t > self.t_cu:
            self.v = (o.quang_duong - self.qd_cu) / (t - self.t_cu)
        self.t_cu, self.qd_cu = t, o.quang_duong
        self.ghi.cap_nhat(o.quang_duong, ctx.imu.yaw, o.x, o.y, self.v, t)
        return 0.0, 0.0

    def ket_thuc(self, ctx, ly_do):
        ra = cau_hinh.CALIB_MOI / TEN_FILE
        n = self.ghi.luu(ra)
        log(f'GHI ĐƯỜNG: đã lưu {n} mẫu, dài {self.ghi.dai:.2f} m -> {ra}. '
            f'Kéo về: bash chay.sh keo, rồi bash chay.sh nhan_calib')

    def trang_thai(self):
        return f'đã ghi {self.ghi.dai:.2f} m, {len(self.ghi.mau)} mẫu, v đẩy {self.v:.2f} m/s'


if __name__ == '__main__':
    khung.main(__file__)
