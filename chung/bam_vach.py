"""Bám vạch liền bên phải: bộ dò 8 bước (chung/lane.py) + giữ độ cong khi mất vạch + PID theo khoảng cách.

  d     = khoảng cách tới vạch phải, đo ở điểm X_NHIN (0.35 m) phía trước mũi xe.
          Đo ở đây vì camera NHÌN RÕ điểm này; kéo dài đường cong về sát xe (camera
          không thấy) thì sai góc tới 5 độ.
  e     = d - D_DICH          (e > 0: xe đang xa vạch / lệch trái -> phải quẹo PHẢI)
  kappa_lenh = -(KP*e + KI*tich_phan(e) + KD*de/dt)
  w     = v * kappa_lenh
  KP = 2 / X_NHIN^2 là luật "pure pursuit": điểm tâm làn cách X_NHIN phía trước
  nằm đúng trên cung tròn xe sẽ chạy, nên trên khúc cua tròn xe bám đúng tâm làn
  mà không cần đoán góc / độ cong.
  v     = V_BAM giảm theo |e| và độ cong vạch
Không đo được vạch (hoặc vạch đo ngoài dải hợp lý HOP_LY_GAN..XA): dời cả đường cong theo odom + IMU (giữ độ cong)
tối đa GIU_HOP_LY_S s, chạy chậm V_DU_DOAN; lâu hơn thì MẤT VẠCH: chạy chậm giữ hướng.
"""
import math

import cv2
import numpy as np

from . import lane
from .camera import BirdEye, calib_dung, doc_bev, giai_nen_xam_nua
from .pid import PID, GiuHuong, kep

# ===== THAM SỐ PID KHOẢNG CÁCH (sửa, Ctrl+S, chạy lại) =======================
V_BAM = 0.25               # m/s trên đường thẳng
V_MIN = 0.1
GIAM_THEO_E = 1.5          # v *= 1 - GIAM_THEO_E*|e| (e tính bằng m)
GIAM_THEO_CONG = 0.35      # v /= 1 + GIAM_THEO_CONG*|kappa|
D_DICH = 0.20              # m: muốn tâm xe cách vạch phải bấy nhiêu (nửa làn 0.40)
KP = 16.0                  # 1/m^2; pure pursuit = 2 / X_NHIN^2 = 16.3 khi X_NHIN 0.35
KI = 0.0                   # bù lệch cố định (vòng gyro trong khung đã bù bánh lệch)
KD = 1.0                   # giảm dao động: phản ứng theo tốc độ thay đổi của e
W_MAX = 1.2
V_MAT_VACH = 0.1          # chạy chậm giữ hướng khi mất vạch
MAT_VACH_DUNG = 3.0        # mất vạch lâu hơn (s) thì báo mất hẳn (bên gọi quyết định dừng / xoay tìm)

# ===== THAM SỐ BỘ DÒ VẠCH (xem giải thích từng bước trong chung/lane.py) ====
DO_VACH = dict(
    X_GAN=0.15, X_XA=1.00, M_PX=0.005,
    Y_ROI_TRAI=0.05, Y_ROI_PHAI=-0.60,
    DO_RONG_MAX_M=0.045, DO_RONG_MIN_M=0.010,
    NGUONG_MIN=18, K_STD=3.0,
    NOI_DOC_M=0.06, DAI_MIN_M=0.12, GOC_MAX_DO=50.0,
    Y_KY_VONG=-0.20, CUA_SO_M=0.08,
    RANSAC_LAN=40, RANSAC_SAI_M=0.012, DIEM_MIN=25, TAM_X_MIN_M=0.08, C_MAX=1.5,
    X_NHIN=0.35, TRONG_SO_TAU_M=0.40, X_FIT_MAX=0.90, CONG_NHAY_M=0.05, GIU_TOI_DA=0.5,
)
# ===== DỰ ĐOÁN GIỮ ĐỘ CONG + CỔNG VẠCH PHẢI HỢP LÝ =================
GIU_HOP_LY_S = 3.0         # không có vạch hợp lý thì dự đoán tiếp bấy nhiêu giây rồi mới coi là MẤT VẠCH (bản cũ: 0.5)
HOP_LY_GAN_M = 0.10        # vạch phải hợp lý: cách tâm xe tại X_NHIN từ ...
HOP_LY_XA_M = 0.30         # ... tới chừng này (m). Cua gắt vạch đúng cũng có thể ra ngoài: nới XA nếu cần
V_DU_DOAN = V_MAT_VACH           # m/s lúc chạy bằng dự đoán (không có vạch hợp lý)
X_SAU_M = 0.30             # dời đường cong: lấy điểm từ bấy nhiêu mét SAU xe tới X_XA để khớp lại (giữ phần đã qua)
# ===========================================================================


def doi_duong_cong(he, ds, dyaw, x_xa, x_sau=X_SAU_M):
    """Đường y = a + b x + c x^2 trong hệ xe CŨ -> hệ xe MỚI sau khi xe tiến ds (m) và quay dyaw (rad, + trái).
    Điểm cố định trên sàn: p' = R(-dyaw) (p - [ds, 0]); dời 40 điểm rồi khớp lại bậc 2 (phép dời không đổi hình
    dạng nên độ cong được giữ, chỉ vị trí và hướng đổi theo xe)."""
    a, b, c = he
    xs = np.linspace(-x_sau, x_xa, 40)
    ys = a + b * xs + c * xs * xs
    co, si = math.cos(dyaw), math.sin(dyaw)
    x = xs - ds
    c2, b2, a2 = np.polyfit(co * x + si * ys, -si * x + co * ys, 2)
    return float(a2), float(b2), float(c2)


def y_psi(he, x):
    a, b, c = he
    return a + b * x + c * x * x, math.atan(b + 2 * c * x)


class DoVachGiuCong(lane.DoVach):
    """Bộ dò 8 bước của Champion + (1) dự đoán giữ độ cong, (2) cổng vạch phải hợp lý."""

    def __init__(self, birdeye, p, gan=HOP_LY_GAN_M, xa=HOP_LY_XA_M):
        super().__init__(birdeye, p)
        self.gan, self.xa = gan, xa
        self.ngoai_dai = None          # khoảng cách của vạch vừa bị cổng bỏ (m), None = không bỏ gì

    # (1) ---------------------------------------------------------------
    def du_doan(self, ds, dyaw):
        if self.y is None:
            return
        if self.he is None:
            return super().du_doan(ds, dyaw)
        p = self.p
        he_moi = doi_duong_cong(self.he, ds, dyaw, p.X_XA)
        y_cu, psi_cu = y_psi(self.he, p.X_NHIN)
        y_moi, psi_moi = y_psi(he_moi, p.X_NHIN)
        self.y += y_moi - y_cu             # giữ phần Kalman đã lọc, chỉ cộng mức dời do xe chuyển động
        self.psi += psi_moi - psi_cu
        self.he = he_moi
        a, b, c = he_moi
        self.kappa = 2 * c / (1 + b * b) ** 1.5
        self.Py += p.Q_Y
        self.Ppsi += p.Q_PSI

    # (2) ---------------------------------------------------------------
    def tach(self, xam_nua, debug=False):
        he, so, anh = super().tach(xam_nua, debug)
        self.ngoai_dai = None
        if he is not None:
            kc = -y_psi(he, self.p.X_NHIN)[0]          # y âm = bên phải
            if not (self.gan <= kc <= self.xa):
                self.ngoai_dai = kc
                if debug and '1 birdeye' in anh:
                    anh['6 fit'] = self._ve_fit(anh['1 birdeye'], None)
                    self._ve_cong(anh['6 fit'], he, (140, 140, 140))
                return None, so, anh
        return he, so, anh

    def xu_ly(self, xam_nua, t, ds=0.0, dyaw=0.0, debug=False):
        kq = super().xu_ly(xam_nua, t, ds, dyaw, debug)
        kq.ngoai_dai = self.ngoai_dai
        if self.ngoai_dai is not None:
            kq.ly_do = f'vạch cách {self.ngoai_dai:.2f} m: ngoài dải hợp lý {self.gan:.2f}..{self.xa:.2f}'
        if debug and kq.thay and not kq.do_moi and self.he is not None and '6 fit' in kq.anh:
            self._ve_cong(kq.anh['6 fit'], self.he, (0, 220, 255))    # vàng: đường cong đang dự đoán
        return kq

    def _ve_cong(self, img, he, mau):
        a, b, c = he
        xs = np.linspace(self.p.X_GAN, self.p.X_XA, 40)
        r, k = self.be.m_sang_px(xs, a + b * xs + c * xs * xs)
        cv2.polylines(img, [np.column_stack([k, r]).astype(np.int32)], False, mau, 2)


def tao_bo_do(gia_lap=False):
    """Vùng nhìn + độ phân giải bird's eye: lấy từ calib/bev.yaml (bạn chọn ở bước 4b); không có thì dùng
    X_GAN, X_XA, M_PX trong DO_VACH và bề rộng ±0,5 m."""
    p = lane.tham_so(**DO_VACH)
    p.GIU_TOI_DA = GIU_HOP_LY_S
    bev = None if gia_lap else doc_bev()
    y_trai, y_phai = 0.5, -0.5
    if bev:
        p.X_GAN, p.X_XA, p.M_PX = bev['x_gan'], bev['x_xa'], bev['m_px']
        y_trai, y_phai = bev['y_trai'], bev['y_phai']
    be = BirdEye(calib_dung(gia_lap), x_gan=p.X_GAN, x_xa=p.X_XA, y_trai=y_trai, y_phai=y_phai, m_px=p.M_PX)
    return DoVachGiuCong(be, p)


class BamVach:
    """Gom: lấy khung mới -> dò vạch (dự đoán bằng odom + IMU) -> kết quả."""

    def __init__(self, gia_lap=False):
        self.do = tao_bo_do(gia_lap)
        self.so_cu = 0
        self.kq = lane.KetQuaVach()
        self.qd_cu = None
        self.yaw_cu = None
        self.anh = {}

    def cap_nhat(self, ctx, debug=False):
        jpeg, so = ctx.xe.anh_jpeg()
        if jpeg is None or so == self.so_cu:
            return self.kq, False
        self.so_cu = so
        qd, yaw = ctx.xe.odom.quang_duong, ctx.imu.yaw
        ds = 0.0 if self.qd_cu is None else qd - self.qd_cu
        dyaw = 0.0 if self.yaw_cu is None else yaw - self.yaw_cu
        self.qd_cu, self.yaw_cu = qd, yaw
        self.kq = self.do.xu_ly(giai_nen_xam_nua(jpeg), ctx.xe.gio(), ds, dyaw, debug)
        self.anh = self.kq.anh
        return self.kq, True


class DieuKhienLan:
    """PID khoảng cách tới vạch phải (xem đầu file). lenh() chỉ tính lệnh từ kết quả dò vạch đã có;
    buoc() tự lấy khung mới qua BamVach rồi gọi lenh().

    Trả (v, w, mat_qua_lau). Mất vạch thì chạy chậm giữ hướng; mat_qua_lau = True khi mất vạch lâu hơn
    MAT_VACH_DUNG giây, bên gọi tự quyết dừng hay chạy tiếp."""

    def __init__(self, gia_lap=False, bam=None):
        self.bam = bam or BamVach(gia_lap)
        self.pid = PID(KP, KI, KD, i_max=0.2)
        self.huong = GiuHuong(1.5, 0.0, 0.1, 0.6)
        self.t_mat = 0.0
        self.yaw_giu = None
        self.e = 0.0
        self.so_khung = 0
        self._t = 0.0

    def dat_lai(self):
        self.pid.reset()
        self.huong.reset()

    def buoc(self, ctx, dt, v_bam=V_BAM, debug=False):
        self._t = ctx.xe.gio()
        kq, moi = self.bam.cap_nhat(ctx, debug=debug)
        if moi:
            self.so_khung += 1
        return self.lenh(kq, ctx.imu, dt, v_bam)

    def lenh(self, kq, imu, dt, v_bam=V_BAM, t=None):
        if t is not None:
            self._t = t
        if not kq.thay:
            self.t_mat += dt
            if self.yaw_giu is None:
                self.yaw_giu = imu.yaw
            w, _ = self.huong.tinh(imu, self.yaw_giu, dt)
            return V_MAT_VACH, w, self.t_mat > MAT_VACH_DUNG
        self.t_mat, self.yaw_giu = 0.0, None
        self.e = kq.khoang_cach_truoc - D_DICH
        v = v_bam * max(0.0, 1.0 - GIAM_THEO_E * abs(self.e)) / (1.0 + GIAM_THEO_CONG * abs(kq.kappa))
        v = max(V_MIN, v)
        kappa_lenh = -self.pid.tinh(self.e, dt)
        w = kep(v * kappa_lenh, -W_MAX, W_MAX)
        if not kq.do_moi and v > V_DU_DOAN:          # chạy bằng dự đoán: chậm lại, giữ nguyên độ cong lệnh
            w, v = w * V_DU_DOAN / v, V_DU_DOAN
        return v, w, False

    def mo_ta(self):
        kq = self.bam.kq
        if not kq.thay:
            return f'MẤT VẠCH {self.t_mat:.1f} s ({kq.ly_do})'
        du_doan = '' if kq.do_moi else f'DỰ ĐOÁN {self._t - self.bam.do.t_do:.1f} s ({kq.ly_do or "không đo được"}) | '
        return du_doan + (f'd={kq.khoang_cach_truoc:.3f} m e={self.e:+.3f} psi={math.degrees(kq.psi):+.1f} '
                f'k={kq.kappa:+.2f} diem={kq.so_diem} {kq.ms:.0f} ms{" (giữ)" if not kq.do_moi else ""}')
