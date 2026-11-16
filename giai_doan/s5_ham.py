#!/usr/bin/env python3
"""S5: QUA HẦM bằng LiDAR, logic GIỮ NGUYÊN của bản 04aba63e.

  - Nhận hầm: tường ở 2 bên (nón ±SIDE_DEG quanh ±90 độ), mỗi bên < WALL_MAX,
    tổng 2 bên = WIDTH ± WIDTH_TOL. Vào: SO_KHUNG_VAO khung liền; ra: mất SO_KHUNG_RA khung.
  - Trong hầm: bám tâm hành lang (offset, psi, kappa đo từ tường, khớp đường tròn
    khi tường cong) bằng luật lái của 04a: kappa_lenh = K_FF*kappa + boost*K_E*offset + K_PSI*psi.
  - Khúc cua: tường phía trước < BEND_FRONT -> chậm lại, quẹo BEND_OMEGA về phía nhìn xa hơn.
  - Gỡ kẹt (MỚI, không có ở 04a): bị chặn mũi hoặc bánh không lăn quá GO_KET_SAU s thì
    lùi thẳng LUI_QUANG_DUONG m (nhìn LiDAR phía sau), rồi quét lại và đi tiếp.
Nhận cổng hầm và luật lái 04a nằm ở chung/ham.py (S3, S4 cũng dùng để biết đã tới hầm).
Chạy riêng: đặt xe trước miệng hầm  bash chay.sh s5_ham
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chung import cau_hinh, khung  # noqa: E402
from chung.ham import (BEND_FRONT, P, SIDE_DEG, SO_KHUNG_RA, SO_KHUNG_VAO, V_HAM, WALL_MAX,  # noqa: E402,F401
                       DemHam, lenh_lai)
from chung.lidar import CuaHam, range_in_cone, tunnel_pose  # noqa: E402
from chung.log import log  # noqa: E402
from chung.pid import GiuHuong  # noqa: E402
from chung.ve import lidar_tren_xuong  # noqa: E402

# ===== THAM SỐ (luật lái và nhận cổng hầm nằm ở chung/ham.py) ===========
GO_KET_BAT = False         # TẮT mặc định (user chốt 10/10: hầm giữ đúng 04a). True = bật gỡ kẹt
GO_KET_SAU = 1.5           # bị chặn mũi / lệnh chạy mà bánh không lăn quá bấy nhiêu giây ...
LUI_V = -0.08              # ... thì lùi THẲNG với tốc độ này
LUI_QUANG_DUONG = 0.12     # lùi bấy nhiêu mét rồi quét lại, đi tiếp (không giới hạn số lần)
SAU_TRONG_M = 0.08         # dừng lùi sớm nếu sau cản sau còn ít hơn bấy nhiêu mét
QUANG_DUONG_CHO_VAO = 1.0  # chạy riêng S5: chạy thẳng tìm hầm tối đa bấy nhiêu mét
THOI_GIAN_TOI_DA = 45.0
# ===========================================================================


class GiaiDoan(khung.GiaiDoanCoSo):
    TEN = 's5_ham'
    HZ = 20

    def __init__(self):
        self.THOI_GIAN_TOI_DA = THOI_GIAN_TOI_DA
        self.cua = CuaHam()
        self.pose = None
        self.thay = 0
        self.mat = 0
        self.trong_ham = False
        self.t_scan = None
        self.bend = None
        self.huong = GiuHuong(1.5, 0.0, 0.1, 0.8)
        self.t_ket = 0.0
        self.lui_tu = None         # quãng đường lúc bắt đầu lùi (None = không lùi)
        self.so_lan_lui = 0
        self.v_truoc = 0.0
        self.yaw = 0.0

    def bat_dau(self, ctx):
        # Đi từ S4 sang thì S4 đã thấy cổng hầm đủ số khung.
        self.trong_ham = ctx.du_lieu.get('da_thay_ham', False)
        self.yaw = ctx.du_lieu.get('yaw_doc', 0.0)

    def buoc(self, ctx, dt):
        scan = ctx.xe.scan
        if scan is not None and scan.t != self.t_scan:
            self.t_scan = scan.t
            pose = tunnel_pose(scan, P)
            self.bend = self.cua.tinh(scan, P, V_HAM) if self.trong_ham else None
            if pose is not None:
                self.pose, self.thay, self.mat = pose, self.thay + 1, 0
                if not self.trong_ham and self.thay >= SO_KHUNG_VAO:
                    self.trong_ham = True
            elif self.trong_ham and self.bend is not None:
                self.mat = 0                       # tường phía trước: khúc cua, chưa phải cửa ra
            else:
                self.thay, self.mat = 0, self.mat + 1
            if self.trong_ham and self.mat >= SO_KHUNG_RA:
                ctx.du_lieu['yaw_ra_ham'] = ctx.imu.yaw
                return khung.KetThuc(f'RA HẦM (mất tường {SO_KHUNG_RA} khung)')
        if not self.trong_ham:
            if ctx.quang_duong() > QUANG_DUONG_CHO_VAO:
                return khung.KetThuc(f'không thấy hầm sau {QUANG_DUONG_CHO_VAO} m', dung_han=True)
            w, _ = self.huong.tinh(ctx.imu, self.yaw, dt)
            return V_HAM, w
        if GO_KET_BAT:
            lui = self._go_ket(ctx, dt, scan)
            if lui is not None:
                return lui
        if self.bend is not None:
            v, w = self.bend
        elif self.pose is not None:
            v, w = V_HAM, lenh_lai(*self.pose, V_HAM)
        else:
            v, w = V_HAM, 0.0
        self.v_truoc = v
        return v, w

    def _go_ket(self, ctx, dt, scan):
        """Kẹt (chặn mũi, hoặc ra lệnh chạy mà bánh không lăn) quá GO_KET_SAU s -> lùi thẳng."""
        if self.lui_tu is None:
            ket = ctx.chan_mui or (self.v_truoc > 0.03 and abs(ctx.xe.odom.v) < 0.015)
            self.t_ket = self.t_ket + dt if ket else 0.0
            if self.t_ket < GO_KET_SAU:
                return None
            self.lui_tu = ctx.quang_duong()
            self.so_lan_lui += 1
            log(f'S5: KẸT {self.t_ket:.1f} s -> lùi {LUI_QUANG_DUONG} m rồi đi tiếp (lần {self.so_lan_lui})')
        sau = range_in_cone(scan, 180.0, 25.0)
        het_cho = sau is not None and sau < cau_hinh.LIDAR_TOI_CAN_SAU_M + SAU_TRONG_M
        if ctx.quang_duong() - self.lui_tu >= LUI_QUANG_DUONG or het_cho:
            log('S5: lùi xong' + (' (sau sát vật)' if het_cho else '') + ', quét lại và đi tiếp')
            self.lui_tu, self.t_ket, self.v_truoc = None, 0.0, 0.0
            self.cua.phia = None              # chọn lại phía quẹo khúc cua
            return None
        self.v_truoc = LUI_V
        return LUI_V, 0.0

    def trang_thai(self):
        if not self.trong_ham:
            return f'tìm hầm (thấy {self.thay}/{SO_KHUNG_VAO})'
        o, psi, k = self.pose or (0, 0, 0)
        cua = f' | CUA {self.bend[1]:+.1f}' if self.bend else ''
        cua += ' | ĐANG LÙI' if self.lui_tu is not None else ''
        return f'TRONG HẦM offset {o:+.3f} psi {math.degrees(psi):+.0f} k {k:+.1f} mất {self.mat}{cua}'

    def ve(self, ctx):
        s = ctx.xe.scan
        truoc = range_in_cone(s, 0.0, 25.0)
        ctx.xem.dat_anh('lidar', lidar_tren_xuong(
            s, non=[(90, SIDE_DEG, WALL_MAX, (0, 255, 0)), (-90, SIDE_DEG, WALL_MAX, (0, 255, 0)),
                    (0, 25, BEND_FRONT, (0, 0, 255) if self.bend else (120, 120, 120))],
            chu=f'truoc {truoc if truoc else 0:.2f} m | {self.trang_thai()[:30]}'))


if __name__ == '__main__':
    khung.main(__file__)
