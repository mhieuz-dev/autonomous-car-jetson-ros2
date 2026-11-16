#!/usr/bin/env python3
"""S7 PHÁT LẠI (bản LAI, dự phòng cho s7_bam_vach): bám vạch phải như S7; khi KHÔNG thấy vạch thì chạy theo ĐƯỜNG ĐÃ GHI
(calib/duong_s7.csv, ghi bằng s7_ghi_duong lúc đẩy tay): ở quãng đường s (odom từ đầu S7) giữ hướng IMU = hướng đã
ghi tại s. Không còn "mất vạch 3 s thì xoay tìm". Hết quãng đường đã ghi thì dừng (tới đích).
Biển STOP vẫn dừng (khung, STOP_CA_MAP). Nhảy vạch bất hợp lý thì bỏ vạch, đi theo đường ghi thay vì đứng xoay.
Chỉ bám vạch khi khung đó ĐO ĐƯỢC vạch thật; đang DỰ ĐOÁN (giữ độ cong mò) hoặc hướng xe lệch hướng đã ghi quá
LECH_GHI_MAX_DO thì theo đường ghi.
CHI_THEO_DUONG = True: bỏ camera, chạy thuần theo đường ghi (để kiểm tra đường ghi).

Dùng cho full map: đặt S7_DUNG = 's7_phat_lai' trong chung/cau_hinh.py. Chạy riêng: bash chay.sh s7_phat_lai
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chung import khung  # noqa: E402
from chung.duong_ghi import DuongDaGhi  # noqa: E402
from chung.log import log  # noqa: E402
from chung.pid import GiuHuong  # noqa: E402
from giai_doan import s7_bam_vach as S7  # noqa: E402

# ===== THAM SỐ (sửa, Ctrl+S, chạy lại) =====================================
V_THEO_DUONG = 0.15        # m/s khi đi theo đường ghi (không thấy vạch)
CHI_THEO_DUONG = False     # True = không dùng camera, chạy thuần theo đường ghi (kiểm tra đường ghi)
KP_HUONG = 2.0
KI_HUONG = 0.2
KD_HUONG = 0.10
W_MAX = 0.8
LECH_GHI_MAX_DO = 30.0     # bám vạch mà hướng xe lệch hướng đã ghi quá số này (độ) -> bỏ vạch, theo đường ghi
THEM_CUOI_M = 0.05         # chạy quá cuối đường ghi bấy nhiêu mét rồi dừng
THOI_GIAN_TOI_DA = 180.0
# ===========================================================================


class GiaiDoan(S7.GiaiDoan):
    TEN = 's7_phat_lai'
    PHAT_STOP = False            # có cách theo đường ghi riêng (calib/duong_s7.csv)

    def __init__(self):
        super().__init__()
        self.THOI_GIAN_TOI_DA = THOI_GIAN_TOI_DA
        self.duong = None
        self.huong = GiuHuong(KP_HUONG, KI_HUONG, KD_HUONG, W_MAX)
        self.yaw0 = 0.0
        self.theo_duong = False
        self.qd_theo_duong = 0.0
        self.s_cuoi = 0.0

    def bat_dau(self, ctx):
        self.duong = DuongDaGhi()
        self.yaw0 = ctx.imu.yaw
        if not CHI_THEO_DUONG:
            super().bat_dau(ctx)
        log(f'S7 PHÁT LẠI: đường ghi dài {self.duong.dai:.2f} m ({len(self.duong.s)} mẫu), hướng gốc '
            f'{math.degrees(self.yaw0):+.1f} độ | {"CHỈ theo đường ghi" if CHI_THEO_DUONG else "bám vạch, mất vạch thì theo đường ghi"}')

    def _lenh_theo_duong(self, ctx, dt, s):
        yaw_dich = self.yaw0 + self.duong.huong(s)
        w, _ = self.huong.tinh(ctx.imu, yaw_dich, dt)
        return V_THEO_DUONG, w

    def buoc(self, ctx, dt):
        s = self.s_cuoi = ctx.quang_duong()
        if s >= self.duong.dai + THEM_CUOI_M:
            return khung.KetThuc(f'hết đường đã ghi ({self.duong.dai:.2f} m) -> tới đích')
        if CHI_THEO_DUONG:
            return self._lenh_theo_duong(ctx, dt, s)
        v, w, _ = self.dk.buoc(ctx, dt, S7.V_BAM, debug=ctx.xem is not None and ctx.xem.dang_xem('vach'))
        kq = self.dk.bam.kq
        moi = self.dk.so_khung != self._so_khung
        self._so_khung = self.dk.so_khung
        nhay = (S7.NHAY_BAT and moi and kq.nhay_m > S7.NHAY_MAX_M
                and abs(kq.khoang_cach_truoc - S7.D_DICH) > S7.SAI_SO_BAT_HOP_LY_M)
        if nhay:
            log(f'S7 PHÁT LẠI: NHẢY VẠCH {kq.nhay_m * 100:.0f} cm -> bỏ vạch, đi theo đường ghi')
            self.dk.bam.do.dat_lai()
            self.dk.dat_lai()
        lech = abs(math.degrees(math.atan2(math.sin(ctx.imu.yaw - self.yaw0 - self.duong.huong(s)),
                                           math.cos(ctx.imu.yaw - self.yaw0 - self.duong.huong(s)))))
        # chỉ tin vạch khi ĐO THẬT khung này (không tin DỰ ĐOÁN mò) và hướng không lệch xa đường đã ghi
        if kq.thay and kq.do_moi and not nhay and lech <= LECH_GHI_MAX_DO:
            if self.theo_duong:
                log(f'S7 PHÁT LẠI: thấy vạch lại sau {self.qd_theo_duong:.2f} m theo đường ghi -> bám vạch')
                self.theo_duong = False
            return v, w
        if not self.theo_duong:
            self.theo_duong, self.qd_theo_duong = True, 0.0
            self.huong.reset()
            vi_sao = (f'lệch đường ghi {lech:.0f} độ' if kq.thay and kq.do_moi and lech > LECH_GHI_MAX_DO
                      else 'đang dự đoán, không đo được vạch' if kq.thay else 'không thấy vạch')
            log(f'S7 PHÁT LẠI: {vi_sao} ở s = {s:.2f} m -> đi theo đường ghi, hướng {math.degrees(self.duong.huong(s)):+.0f} độ')
        self.qd_theo_duong += V_THEO_DUONG * dt
        self.dk.t_mat = 0.0                   # không để S7 đếm mất vạch (không xoay tìm)
        return self._lenh_theo_duong(ctx, dt, s)

    def trang_thai(self):
        if self.duong is None:
            return ''
        s = self.s_cuoi
        them = f'THEO ĐƯỜNG GHI (hướng ghi {math.degrees(self.duong.huong(s)):+.0f} độ)' if self.theo_duong or CHI_THEO_DUONG else 'bám vạch'
        return f'{them} | ' + ('' if self.dk is None else self.dk.mo_ta())


if __name__ == '__main__':
    khung.main(__file__)
