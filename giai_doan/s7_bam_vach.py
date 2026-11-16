#!/usr/bin/env python3
"""S7: BÁM VẠCH LIỀN BÊN PHẢI bằng PID theo KHOẢNG CÁCH từ xe tới vạch.

  d     = khoảng cách tới vạch phải, đo ở điểm X_NHIN (0.35 m) phía trước mũi xe,
          bằng bộ dò 8 bước (chung/lane.py). Đo ở đây vì camera NHÌN RÕ điểm này;
          kéo dài đường cong về sát xe (camera không thấy) thì sai góc tới 5 độ.
  e     = d - D_DICH          (e > 0: xe đang xa vạch / lệch trái -> phải quẹo PHẢI)
  kappa_lenh = -(KP*e + KI*tich_phan(e) + KD*de/dt)
  w     = v * kappa_lenh
  KP = 2 / X_NHIN^2 là luật "pure pursuit": điểm tâm làn cách X_NHIN phía trước
  nằm đúng trên cung tròn xe sẽ chạy, nên trên khúc cua tròn xe bám đúng tâm làn
  mà không cần đoán góc / độ cong.
  v     = V_BAM giảm theo |e| và độ cong vạch
Không đo được vạch (hoặc vạch đo ngoài dải hợp lý HOP_LY_GAN..XA): dời cả đường cong theo odom + IMU (giữ độ cong)
tối đa GIU_HOP_LY_S s, chạy chậm V_DU_DOAN; lâu hơn thì MẤT VẠCH: chạy chậm giữ hướng, quá MAT_VACH_DUNG s thì
đứng xoay phải tìm vạch (xem dưới).
Biển STOP: dừng STOP_DUNG_GIAY (khung, STOP_CA_MAP). STOP ĐẦU TIÊN trong S7: dừng xong thì CHẠY LẠI ĐƯỜNG ĐẨY TAY
(calib/duong_stop.csv, ghi bằng s7_stop_ghi) bằng pure pursuit trên vị trí tự tính (encoder + IMU, gốc là chỗ dừng
STOP), không nhìn vạch; hết quãng đường đã đẩy thì bám vạch lại với V_BAM_SAU. Không có file thì chỉ dừng rồi bám tiếp.
Chạy riêng: đặt xe giữa làn phải:  bash chay.sh s7_bam_vach
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chung import cau_hinh, khung, lane  # noqa: E402
from chung.camera import BirdEye, calib_dung, doc_bev, giai_nen_xam_nua  # noqa: E402
from chung.duong_ghi import TEN_FILE_STOP, DuongDaGhi, goc as chuan_goc  # noqa: E402
from chung.log import log  # noqa: E402
from chung.pid import PID, GiuHuong, kep  # noqa: E402
from chung.ve import luoi  # noqa: E402

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
MAT_VACH_DUNG = 3.0        # mất vạch lâu hơn (s) thì DỪNG rồi XOAY PHẢI TÌM VẠCH (S7)
TIM_W = 0.6                # rad/s: tốc độ xoay phải mỗi nhịp tìm vạch
TIM_XOAY_GIAY = 0.25       # xoay bấy nhiêu giây (~8,6 độ ở 0.6 rad/s) ...
TIM_DUNG_GIAY = 0.5        # ... rồi đứng yên bấy nhiêu giây cho camera nhìn rõ, lặp lại tới khi thấy vạch
TIM_GOC_MAX_DO = 360.0     # xoay tổng quá số độ này vẫn chưa thấy vạch thì dừng hẳn
# Nhảy sang vạch khác: bộ dò nhận vạch mới lệch > NHAY_MAX_M so với vạch đang bám VÀ sai số mới > SAI_SO_BAT_HOP_LY_M
# -> đứng yên ngay, xoay phải từng nhịp (như trên) tới khi thấy vạch phải HỢP LÝ: khoảng cách tâm xe -> vạch
# (quy về ngay cạnh xe, không đổi khi xe xoay tại chỗ) trong [VACH_PHAI_GAN_M, VACH_PHAI_XA_M], N_KHUNG_OK khung liên tiếp.
NHAY_BAT = True
NHAY_MAX_M = 0.12          # vạch đổi chỗ đột ngột hơn chừng này (m) ... (làn 0.40, vạch kế bên cách ~0.40)
SAI_SO_BAT_HOP_LY_M = 0.12 # ... và |khoảng cách - D_DICH| sau khi nhảy lớn hơn chừng này (m) thì coi là bám nhầm vạch
VACH_PHAI_GAN_M = 0.10     # vạch phải hợp lý: cách tâm xe từ ...
VACH_PHAI_XA_M = 0.30      # ... tới chừng này (m)
N_KHUNG_OK = 2
THOI_GIAN_TOI_DA = 180.0
# ===== CHẠY LẠI ĐƯỜNG ĐẨY TAY SAU BIỂN STOP (ghi bằng s7_stop_ghi -> calib/duong_stop.csv) =====
PHAT_SAU_STOP = True       # True: STOP ĐẦU TIÊN trong S7 dừng xong thì chạy lại đường đẩy tay rồi bám vạch tiếp
PHAT_V = 0.15              # m/s khi chạy lại đường đẩy tay
PHAT_NHIN_TRUOC_M = 0.20   # pure pursuit: nhắm điểm trên đường ghi phía trước bấy nhiêu mét
PHAT_W_MAX = 1.0           # rad/s
V_BAM_SAU = 0.45           # m/s bám vạch SAU KHI chạy hết đường đẩy tay (trước STOP dùng V_BAM)

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
# ===== DỰ ĐOÁN GIỮ ĐỘ CONG + CỔNG VẠCH PHẢI HỢP LÝ (bản thử 11/10, nay là chính thức) =======
GIU_HOP_LY_S = 3.0         # không có vạch hợp lý thì dự đoán tiếp bấy nhiêu giây rồi mới coi là MẤT VẠCH (bản cũ: 0.5)
HOP_LY_GAN_M = VACH_PHAI_GAN_M   # vạch phải hợp lý: cách tâm xe tại X_NHIN từ ... 
HOP_LY_XA_M = VACH_PHAI_XA_M     # ... tới (0.30). Cua gắt vạch đúng cũng có thể ra ngoài: nới XA nếu cần
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
    """Gom: lấy khung mới -> dò vạch (dự đoán bằng odom + IMU) -> kết quả. Dùng chung với S6."""

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
    """Bám vạch phải bằng bird's eye + PID khoảng cách (xem đầu file). DÙNG CHUNG: S7 và S1 (tham số ở file này).

    buoc(ctx, dt, v_bam) -> (v, w, mat_qua_lau). Mất vạch thì chạy chậm giữ hướng; mat_qua_lau = True khi mất
    vạch lâu hơn MAT_VACH_DUNG giây (S7 dừng hẳn, S1 thì cứ giữ hướng chạy tiếp)."""

    def __init__(self, gia_lap=False):
        self.bam = BamVach(gia_lap)
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
        if not kq.thay:
            self.t_mat += dt
            if self.yaw_giu is None:
                self.yaw_giu = ctx.imu.yaw
            w, _ = self.huong.tinh(ctx.imu, self.yaw_giu, dt)
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


def so_lan_stop(ctx):
    """Số lần đã dừng ở STOP từ đầu map (bộ đếm chung chung/bien_stop.DungStop)."""
    d = getattr(ctx, 'dung_stop', None)
    return 0 if d is None else d.so_lan


class GiaiDoan(khung.GiaiDoanCoSo):
    TEN = 's7_bam_vach'
    HZ = 20
    CAN_CAMERA = True
    PHAT_STOP = True                     # lớp con tắt: s7_stop_ghi (đang ghi đường), s7_phat_lai (đường ghi riêng)

    def __init__(self):
        self.THOI_GIAN_TOI_DA = THOI_GIAN_TOI_DA
        self.dk = None
        self.tim = None                  # đang xoay tìm vạch: {'t', 'goc', 'yaw', 'vi': 'mat' | 'nhay', 'ok'}
        self._so_khung = 0
        self.v_bam = V_BAM
        self.duong_stop = None           # đường đẩy tay sau STOP (None = gặp STOP chỉ dừng rồi bám tiếp)
        self.stop0 = 0                   # số lần STOP lúc vào S7: STOP mới trong S7 là khi bộ đếm vượt số này
        self.phat = None                 # đang chạy lại: {'s0', 'yaw0', 's', 'x', 'y', 'lech'}
        self.phat_xong = False

    @property
    def bam(self):
        return None if self.dk is None else self.dk.bam

    def bat_dau(self, ctx):
        self.dk = DieuKhienLan(ctx.gia_lap)
        log(f'S7: bộ dò vạch bird\'s eye {self.bam.do.be.so_cot}x{self.bam.do.be.so_hang} px, '
            f'đích cách vạch phải {D_DICH:.2f} m')
        self.stop0 = so_lan_stop(ctx)
        if PHAT_SAU_STOP and self.PHAT_STOP:
            f = cau_hinh.CALIB / TEN_FILE_STOP
            try:
                self.duong_stop = DuongDaGhi(f)
                log(f'S7: STOP đầu tiên dừng xong thì chạy lại đường đẩy tay {self.duong_stop.dai:.2f} m '
                    f'({len(self.duong_stop.s)} mẫu), rồi bám vạch {V_BAM_SAU:.2f} m/s')
            except (OSError, ValueError) as e:
                log(f'S7: không đọc được {f} ({e}) -> gặp STOP chỉ dừng rồi bám vạch tiếp')

    def _phat_sau_stop(self, ctx):
        """STOP đầu tiên trong S7 vừa dừng xong -> chạy lại đường đẩy tay. Trả (v, w) khi đang chạy lại, None = bám vạch."""
        if self.duong_stop is None or self.phat_xong:
            return None
        if self.phat is None:
            d = getattr(ctx, 'dung_stop', None)
            if d is None or d.so_lan <= self.stop0 or ctx.xe.gio() < d.den:
                return None
            self.phat = {'s0': ctx.quang_duong(), 'yaw0': ctx.imu.yaw, 's': 0.0, 'x': 0.0, 'y': 0.0, 'lech': 0.0}
            log(f'S7: hết dừng STOP -> CHẠY LẠI ĐƯỜNG ĐẨY TAY {self.duong_stop.dai:.2f} m '
                f'(hướng gốc {math.degrees(ctx.imu.yaw):+.1f} độ)')
        lenh = self._lenh_phat(ctx)
        if lenh is not None:
            return lenh
        log(f'S7: hết đường đẩy tay ({self.duong_stop.dai:.2f} m) -> bám vạch tiếp, {V_BAM_SAU:.2f} m/s')
        self.phat, self.phat_xong, self.v_bam, self.tim = None, True, V_BAM_SAU, None
        self.dk.bam.do.dat_lai()
        self.dk.dat_lai()
        return None

    def _lenh_phat(self, ctx):
        """Pure pursuit trên đường đẩy tay, vị trí tự tính = quãng đường encoder + hướng IMU (giống lúc ghi).
        Lệch ngang được kéo về chứ không cộng dồn như chỉ giữ hướng. None = đã hết đường."""
        p, duong = self.phat, self.duong_stop
        s = ctx.quang_duong() - p['s0']
        h = chuan_goc(ctx.imu.yaw - p['yaw0'])
        ds = s - p['s']
        p['x'] += ds * math.cos(h)
        p['y'] += ds * math.sin(h)
        p['s'] = s
        if s >= duong.dai:
            return None
        sx, sy = duong.diem(s)
        p['lech'] = math.hypot(sx - p['x'], sy - p['y'])
        nhin = s + PHAT_NHIN_TRUOC_M
        tx, ty = duong.diem(min(nhin, duong.dai))
        if nhin > duong.dai:                       # cuối đường: kéo dài theo hướng cuối để vẫn có điểm nhắm
            hc = duong.huong(duong.dai)
            tx, ty = tx + (nhin - duong.dai) * math.cos(hc), ty + (nhin - duong.dai) * math.sin(hc)
        dx, dy = tx - p['x'], ty - p['y']
        alpha = chuan_goc(math.atan2(dy, dx) - h)
        return PHAT_V, kep(2.0 * PHAT_V * math.sin(alpha) / max(0.05, math.hypot(dx, dy)), -PHAT_W_MAX, PHAT_W_MAX)

    def _xoay_tim(self, ctx, dt):
        """Một nhịp tìm: xoay phải TIM_XOAY_GIAY rồi đứng TIM_DUNG_GIAY. Trả (v, w) hoặc KetThuc nếu xoay quá nhiều."""
        t = self.tim
        t['goc'] += abs(math.atan2(math.sin(ctx.imu.yaw - t['yaw']), math.cos(ctx.imu.yaw - t['yaw'])))
        t['yaw'] = ctx.imu.yaw
        if math.degrees(t['goc']) > TIM_GOC_MAX_DO:
            return khung.KetThuc(f'xoay phải {TIM_GOC_MAX_DO:.0f} độ vẫn không thấy vạch', dung_han=True)
        t['t'] = (t['t'] + dt) % (TIM_XOAY_GIAY + TIM_DUNG_GIAY)
        return 0.0, (-TIM_W if t['t'] < TIM_XOAY_GIAY else 0.0)

    def buoc(self, ctx, dt):
        lenh = self._phat_sau_stop(ctx)
        if lenh is not None:
            return lenh
        v, w, het = self.dk.buoc(ctx, dt, self.v_bam, debug=ctx.xem is not None and ctx.xem.dang_xem('vach'))
        kq = self.dk.bam.kq
        moi = self.dk.so_khung != self._so_khung
        self._so_khung = self.dk.so_khung

        # ---- nhảy sang vạch khác bất hợp lý: đứng yên ngay, xoay phải tìm lại vạch phải ----
        if (NHAY_BAT and moi and self.tim is None and kq.nhay_m > NHAY_MAX_M
                and abs(kq.khoang_cach_truoc - D_DICH) > SAI_SO_BAT_HOP_LY_M):
            log(f'S7: NHẢY VẠCH {kq.nhay_m * 100:.0f} cm, cách vạch mới {kq.khoang_cach_truoc:.2f} m '
                f'(đích {D_DICH:.2f}) -> ĐỨNG YÊN, xoay phải tìm lại vạch phải')
            self.tim = {'t': 0.0, 'goc': 0.0, 'yaw': ctx.imu.yaw, 'vi': 'nhay', 'ok': 0}
            self.dk.bam.do.dat_lai()           # quên vạch nhầm, dò lại từ đầu
            self.dk.dat_lai()
            return 0.0, 0.0
        if self.tim is not None and self.tim['vi'] == 'nhay':
            if moi:
                hop_ly = kq.thay and kq.do_moi and VACH_PHAI_GAN_M <= kq.khoang_cach <= VACH_PHAI_XA_M
                self.tim['ok'] = self.tim['ok'] + 1 if hop_ly else 0
            if self.tim['ok'] >= N_KHUNG_OK:
                log(f'S7: THẤY LẠI VẠCH PHẢI cách {kq.khoang_cach:.2f} m sau khi xoay phải '
                    f'{math.degrees(self.tim["goc"]):.0f} độ -> bám vạch tiếp')
                self.tim = None
                self.dk.dat_lai()
                return v, w
            return self._xoay_tim(ctx, dt)

        # ---- mất vạch quá MAT_VACH_DUNG: đứng yên, xoay phải từng nhịp tới khi thấy vạch ----
        if not het:
            if self.tim is not None:
                log(f'S7: THẤY LẠI VẠCH sau khi xoay phải {math.degrees(self.tim["goc"]):.0f} độ -> bám vạch tiếp')
                self.tim = None
            return v, w
        if self.tim is None:
            self.tim = {'t': 0.0, 'goc': 0.0, 'yaw': ctx.imu.yaw, 'vi': 'mat', 'ok': 0}
            log(f'S7: MẤT VẠCH {MAT_VACH_DUNG:.0f} s -> dừng, xoay phải từng chút tìm vạch')
        return self._xoay_tim(ctx, dt)

    def trang_thai(self):
        if self.dk is None:
            return ''
        if self.phat is not None:
            p = self.phat
            return (f'CHẠY LẠI ĐƯỜNG ĐẨY TAY {p["s"]:.2f}/{self.duong_stop.dai:.2f} m | '
                    f'lệch đường ghi {p["lech"] * 100:.0f} cm')
        return (f'TÌM VẠCH (đã xoay phải {math.degrees(self.tim["goc"]):.0f} độ) | ' if self.tim else '') + self.dk.mo_ta()

    def ve(self, ctx):
        if not ctx.xem.dang_xem('vach'):
            ctx.xem.anh.setdefault('vach', None)     # chỉ báo là có hình, chưa ai xem thì khỏi vẽ
            return
        if self.bam and self.bam.anh:
            ctx.xem.dat_anh('vach', luoi(list(self.bam.anh.items())))
        else:
            ctx.xem.dat_anh('vach', luoi([('chưa có khung debug', None)]))


if __name__ == '__main__':
    khung.main(__file__)
