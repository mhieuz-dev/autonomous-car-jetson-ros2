#!/usr/bin/env python3
"""S7 STOP GHI: bám vạch như s7_bam_vach; gặp biển STOP thì dừng STOP_DUNG_GIAY như thường, rồi THẢ BÁNH (không gửi
lệnh) để bạn ĐẨY TAY đoạn khó. Xe ghi quãng đường (encoder) + hướng IMU + vị trí tự tính mỗi 1 cm, so với chỗ dừng STOP.
Đẩy xong bấm "DỪNG GHI" trên bảng điều khiển (hoặc DỪNG / Ctrl+C): file được lưu, giai đoạn kết thúc.

File: calib_moi/duong_stop.csv trên xe -> kéo về rồi chép vào calib/ (bash chay.sh keo, hoặc scp ... rồi cp sang calib/).
S7 chính thức (s7_bam_vach) tự chạy lại đường này sau STOP đầu tiên. Chạy riêng: đặt xe ở đầu S7 (sau hầm) rồi bash chay.sh s7_stop_ghi.
Full map: đặt S7_DUNG = 's7_stop_ghi' trong chung/cau_hinh.py.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chung import cau_hinh, khung  # noqa: E402
from chung.duong_ghi import TEN_FILE_STOP as TEN_FILE, GhiDuong  # noqa: E402
from chung.log import log  # noqa: E402
from giai_doan import s7_bam_vach as S7  # noqa: E402

# ===== THAM SỐ =============================================================
THOI_GIAN_TOI_DA = 900.0   # s: bám vạch + đẩy tay tối đa bấy nhiêu
# ===========================================================================


class GiaiDoan(S7.GiaiDoan):
    TEN = 's7_stop_ghi'
    PHAT_STOP = False            # lượt này GHI đường mới, không chạy lại đường cũ

    def __init__(self):
        super().__init__()
        self.THOI_GIAN_TOI_DA = THOI_GIAN_TOI_DA
        self.KHONG_GUI = False
        self.ghi = None
        self.v = 0.0
        self.t_cu = self.qd_cu = None

    def bat_dau(self, ctx):
        super().bat_dau(ctx)
        log('S7 STOP GHI: bám vạch tới biển STOP; dừng xong thì THẢ BÁNH, bạn đẩy tay, xong bấm DỪNG GHI')

    def buoc(self, ctx, dt):
        if self.ghi is None:
            d = getattr(ctx, 'dung_stop', None)
            if S7.so_lan_stop(ctx) > self.stop0 and ctx.xe.gio() >= d.den:      # đã dừng đủ ở STOP (đã gửi v = 0)
                ctx.xe.dung()
                self.KHONG_GUI = True
                self.ghi = GhiDuong()
                log(f'S7 STOP GHI: BẮT ĐẦU GHI. Xe thả bánh tự do, ĐẨY TAY đoạn cần ghi rồi bấm DỪNG GHI '
                    f'(hướng gốc {math.degrees(ctx.imu.yaw):+.1f} độ)')
            else:
                return super().buoc(ctx, dt)
        o = ctx.xe.odom
        t = ctx.xe.gio()
        if self.t_cu is not None and t > self.t_cu:
            self.v = (o.quang_duong - self.qd_cu) / (t - self.t_cu)
        self.t_cu, self.qd_cu = t, o.quang_duong
        self.ghi.cap_nhat(o.quang_duong, ctx.imu.yaw, o.x, o.y, self.v, t)
        return 0.0, 0.0

    def ket_thuc(self, ctx, ly_do):
        if self.ghi is None:
            log('S7 STOP GHI: chưa gặp STOP nên KHÔNG có gì để lưu')
            return
        ra = cau_hinh.CALIB_MOI / TEN_FILE
        n = self.ghi.luu(ra)
        log(f'S7 STOP GHI: đã lưu {n} mẫu, dài {self.ghi.dai:.2f} m -> {ra}')

    def trang_thai(self):
        if self.ghi is None:
            return 'chờ STOP | ' + super().trang_thai()
        return f'ĐANG GHI ĐẨY TAY: {self.ghi.dai:.2f} m, {len(self.ghi.mau)} mẫu, v đẩy {self.v:.2f} m/s'


if __name__ == '__main__':
    khung.main(__file__)
