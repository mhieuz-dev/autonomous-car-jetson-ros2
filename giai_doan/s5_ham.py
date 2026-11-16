#!/usr/bin/env python3
"""S5: QUA HẦM bằng LiDAR, logic GIỮ NGUYÊN của bản 04aba63e.

  - Nhận hầm: tường ở 2 bên (nón ±SIDE_DEG quanh ±90 độ), mỗi bên < WALL_MAX,
    tổng 2 bên = WIDTH ± WIDTH_TOL. Vào: SO_KHUNG_VAO khung liền; ra: mất SO_KHUNG_RA khung.
  - Trong hầm: bám tâm hành lang (offset, psi, kappa đo từ tường, khớp đường tròn
    khi tường cong) bằng luật lái của 04a: kappa_lenh = K_FF*kappa + boost*K_E*offset + K_PSI*psi.
  - Khúc cua: tường phía trước < BEND_FRONT -> chậm lại, quẹo BEND_OMEGA về phía nhìn xa hơn.
  - Gỡ kẹt (MỚI, không có ở 04a): bị chặn mũi hoặc bánh không lăn quá GO_KET_SAU s thì
    lùi thẳng LUI_QUANG_DUONG m (nhìn LiDAR phía sau), rồi quét lại và đi tiếp.
Các tham số tường ở đây cũng được S3, S4 dùng để biết đã tới hầm.
Chạy riêng: đặt xe trước miệng hầm  bash chay.sh s5_ham
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chung import cau_hinh, khung  # noqa: E402
from chung.log import log  # noqa: E402
from chung.lidar import CuaHam, cong_ham, range_in_cone, tunnel_pose  # noqa: E402
from chung.pid import GiuHuong  # noqa: E402
from chung.ve import lidar_tren_xuong  # noqa: E402

# ===== THAM SỐ (giá trị đúng như lúc 04a chạy trên xe, car_onboard.sh) ======
WIDTH = 0.77               # tổng khoảng cách trái + phải tới tường (m)
WIDTH_TOL = 0.08
WALL_MAX = 0.70            # mỗi bên không xa hơn
SIDE_DEG = 12.0            # nửa góc nón nhìn ngang
WALL_FIT_MIN = 0.08        # điểm tường dùng để khớp: khoảng cách trong [MIN, MAX]
WALL_FIT_MAX = 0.95
WALL_FIT_MIN_PTS = 10
PSI_MAX = 0.45             # rad
KAPPA_MAX = 3.2            # 1/m
BEND_FRONT = 0.60          # tường trước < số này = khúc cua
BEND_OMEGA = 0.6           # rad/s khi quẹo ở khúc cua

V_HAM = 0.13               # m/s (04a là 0.13, user chốt 0.15)
K_E = 5.0                  # luật lái 04a
K_PSI = 3.5
K_FF = 1.0
EDGE_MARGIN = 0.12
EDGE_BOOST = 2.0
EDGE_BOOST_MAX = 4.0
SO_KHUNG_VAO = 3
SO_KHUNG_RA = 6
GO_KET_BAT = False         # TẮT mặc định (user chốt 10/10: hầm giữ đúng 04a). True = bật gỡ kẹt
GO_KET_SAU = 1.5           # bị chặn mũi / lệnh chạy mà bánh không lăn quá bấy nhiêu giây ...
LUI_V = -0.08              # ... thì lùi THẲNG với tốc độ này
LUI_QUANG_DUONG = 0.12     # lùi bấy nhiêu mét rồi quét lại, đi tiếp (không giới hạn số lần)
SAU_TRONG_M = 0.08         # dừng lùi sớm nếu sau cản sau còn ít hơn bấy nhiêu mét
QUANG_DUONG_CHO_VAO = 1.0  # chạy riêng S5: chạy thẳng tìm hầm tối đa bấy nhiêu mét
THOI_GIAN_TOI_DA = 45.0
# ===========================================================================

P = SimpleNamespace(WIDTH=WIDTH, WIDTH_TOL=WIDTH_TOL, WALL_MAX=WALL_MAX, SIDE_DEG=SIDE_DEG,
                    WALL_FIT_MIN=WALL_FIT_MIN, WALL_FIT_MAX=WALL_FIT_MAX,
                    WALL_FIT_MIN_PTS=WALL_FIT_MIN_PTS, PSI_MAX=PSI_MAX, KAPPA_MAX=KAPPA_MAX,
                    BEND_FRONT=BEND_FRONT, BEND_OMEGA=BEND_OMEGA)


class DemHam:
    """Đếm số khung LiDAR liên tiếp thấy cổng hầm (chỉ đếm khi có scan MỚI)."""

    def __init__(self):
        self.t_scan = None
        self.lien = 0

    def cap_nhat(self, scan):
        if scan is None or scan.t == self.t_scan:
            return self.lien
        self.t_scan = scan.t
        self.lien = self.lien + 1 if cong_ham(scan, P) is not None else 0
        return self.lien


def lenh_lai(offset, psi, kappa, v):
    """Luật lái bản 04a (Controller.steer + apply ở trạng thái TUNNEL)."""
    excess = abs(offset) - EDGE_MARGIN
    boost = 1.0
    if excess > 0.0:
        boost = min(1.0 + EDGE_BOOST * (excess / max(EDGE_MARGIN, 1e-3)), EDGE_BOOST_MAX)
    kappa_lenh = K_FF * kappa + (boost * K_E * offset + K_PSI * psi)
    return v * kappa_lenh


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
