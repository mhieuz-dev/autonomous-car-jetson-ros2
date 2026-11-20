"""Dò VẠCH LIỀN BÊN PHẢI: 8 bước, mỗi bước có ảnh debug riêng.

  0 Giải nén     JPEG -> xám, nửa độ phân giải (camera.giai_nen_xam_nua)
  1 Bird's eye   1 lần cv2.remap từ ảnh gốc còn méo (camera.BirdEye)
  2 ROI          chỉ giữ dải [Y_ROI_PHAI, Y_ROI_TRAI] (nửa phải), trước xe X_GAN..X_XA
  3 Tách vạch    LỌC THEO BỀ RỘNG (top-hat ngang): vạch sơn ~2 cm sáng hơn nền 2 bên
                 thì giữ, vệt sáng rộng hơn DO_RONG_MAX_M (vệt phản chiếu 6 cm trên
                 bạt) bị triệt. Ngưỡng = max(NGUONG_MIN, trung bình + K_STD * độ lệch).
  4 Morphology   mở (bỏ chấm lẻ < DO_RONG_MIN_M) rồi đóng theo chiều dọc (nối vạch đứt)
  5 Thành phần   giữ vùng DÀI >= DAI_MIN_M, chạy DỌC (lệch < GOC_MAX_DO so với hướng xe);
                 vùng nằm ngang (vạch dừng, vạch người đi bộ) bị loại
  6 Fit          chọn vùng gần vị trí dự đoán (khung trước) hoặc gần Y_KY_VONG; khớp
                 y = a + b x + c x^2 (mét) bằng RANSAC, rồi khớp lại có trọng số
                 ưu tiên điểm gần xe (TRONG_SO_TAU_M)
  7 Lọc thời gian Kalman trên (y tại X_NHIN, góc psi); dự đoán theo odom + yaw;
                 bỏ khung nhảy > CONG_NHAY_M; mất vạch thì giữ dự đoán tối đa GIU_TOI_DA s
Đầu ra: khoảng cách tới vạch phải đo tại X_NHIN phía trước (khoang_cach_truoc, dùng
để lái) và ngoại suy về ngay cạnh xe (khoang_cach, để xem), góc psi, độ cong.
Bước 8 (PID theo khoảng cách) nằm trong core/lane_follow.py.
"""
import math
import time
import warnings
from types import SimpleNamespace

import cv2
import numpy as np

warnings.simplefilter('ignore', getattr(np, 'exceptions', np).RankWarning)

MAC_DINH = dict(
    X_GAN=0.15, X_XA=1.00, M_PX=0.005,      # vùng bird's eye (m) và độ phân giải (m/pixel)
    Y_ROI_TRAI=0.05, Y_ROI_PHAI=-0.45,      # dải ngang giữ lại (m, + = trái)
    DO_RONG_MAX_M=0.045,                    # vệt sáng rộng hơn này bị triệt (vạch thật 0.02)
    DO_RONG_MIN_M=0.010,                    # vệt hẹp hơn này coi là nhiễu
    NGUONG_MIN=18, K_STD=3.0,               # ngưỡng top-hat
    NOI_DOC_M=0.06,                         # nối vạch đứt theo chiều dọc
    DAI_MIN_M=0.12, GOC_MAX_DO=50.0,        # vùng giữ lại: đủ dài và chạy dọc
    Y_KY_VONG=-0.20,                        # vạch phải nằm ở đâu khi xe giữa làn
    CUA_SO_M=0.08,                          # khi đang bám: chỉ nhận điểm cách dự đoán < này
    RANSAC_LAN=40, RANSAC_SAI_M=0.012, DIEM_MIN=25, TAM_X_MIN_M=0.15, C_MAX=1.5,
    X_NHIN=0.35,                            # đo khoảng cách tới vạch tại điểm này phía trước
    TRONG_SO_TAU_M=0.40,                    # fit: điểm xa hơn X_GAN + tau nặng nhẹ đi ~1/e lần
    X_FIT_MAX=0.90,                         # chỉ khớp điểm gần hơn (xa: cung tròn khác bậc 2)
    CONG_NHAY_M=0.05, GIU_TOI_DA=0.5,       # Kalman: bỏ khung nhảy, giữ khi mất
    Q_Y=0.0004, Q_PSI=0.004, R_Y=0.0001, R_PSI=0.01,
)


def tham_so(**doi):
    p = dict(MAC_DINH)
    p.update({k: v for k, v in doi.items() if k in MAC_DINH})
    return SimpleNamespace(**p)


class KetQuaVach:
    def __init__(self):
        self.thay = False          # có vạch (đo được hoặc đang giữ dự đoán)
        self.do_moi = False        # khung này đo được thật
        self.y = math.nan          # y của vạch tại X_NHIN (m, + = trái)
        self.psi = 0.0             # hướng vạch NGAY CẠNH XE so với hướng xe (rad, + = trái)
        self.kappa = 0.0
        self.khoang_cach = math.nan  # từ TÂM XE tới vạch phải (m, dương), ngoại suy về x = 0
        self.khoang_cach_truoc = math.nan  # tại X_NHIN phía trước (đo trực tiếp, dùng để lái)
        self.so_diem = 0
        self.ms = 0.0
        self.ly_do = ''
        self.nhay_m = 0.0          # > 0: khung này NHẬN vạch mới lệch chừng này (m) so với vạch đang bám (sau 3 khung bỏ nhảy)


class DoVach:
    def __init__(self, birdeye, p=None):
        self.be = birdeye
        self.p = p or tham_so()
        p = self.p
        m = birdeye.m
        hang, cot = np.mgrid[0:birdeye.so_hang, 0:birdeye.so_cot]
        x, y = birdeye.px_sang_m(hang, cot)
        self.roi = ((y <= p.Y_ROI_TRAI) & (y >= p.Y_ROI_PHAI) &
                    (x >= p.X_GAN) & (x <= p.X_XA)).astype(np.uint8)
        k = max(3, int(round(p.DO_RONG_MAX_M / m)) | 1)
        self.k_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
        self.k_mo = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, int(round(p.DO_RONG_MIN_M / m))), 1))
        self.k_dong = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, int(round(p.NOI_DOC_M / m)))))
        self.rng = np.random.default_rng(0)
        self.dat_lai()

    def dat_lai(self):
        self.y = None              # trạng thái Kalman
        self.psi = 0.0
        self.Py = self.Ppsi = 1.0
        self.he = None             # hệ số fit gần nhất (a, b, c)
        self.t_do = -1e9
        self.lien_mat = 0
        self.kappa = 0.0

    # --------------------------------------------------------------- 7a ---
    def du_doan(self, ds, dyaw):
        """Xe chạy thêm ds (m) và quay dyaw (rad, + trái) từ khung trước."""
        if self.y is None:
            return
        p = self.p
        self.y += ds * math.tan(self.psi) - dyaw * p.X_NHIN
        self.psi -= dyaw
        self.Py += p.Q_Y
        self.Ppsi += p.Q_PSI
        if self.he is not None:
            a, b, c = self.he
            self.he = (a + ds * b, b - dyaw, c)

    # --------------------------------------------------------------- 1-6 ---
    def tach(self, xam_nua, debug=False):
        """Bước 1-6 trên một khung. Trả (hệ số (a,b,c) hoặc None, số điểm, dict ảnh debug)."""
        p, be = self.p, self.be
        anh = {}
        bev = be.chieu(xam_nua)                                          # 1
        roi = bev * self.roi                                             # 2
        tophat = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, self.k_tophat)  # 3
        vals = tophat[self.roi > 0]
        nguong = max(p.NGUONG_MIN, float(vals.mean() + p.K_STD * vals.std())) if vals.size else 255
        mask = ((tophat > nguong) * 255).astype(np.uint8)
        mo = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.k_mo)          # 4
        dong = cv2.morphologyEx(mo, cv2.MORPH_CLOSE, self.k_dong)
        n, nhan, stats, _ = cv2.connectedComponentsWithStats(dong, connectivity=8)  # 5
        giu = np.zeros(n, bool)
        dai_min = p.DAI_MIN_M / be.m
        for i in range(1, n):
            x0, y0, w, h, dt = stats[i]
            if max(w, h) < dai_min or dt < 6:
                continue
            ys, xs = np.nonzero(nhan[y0:y0 + h, x0:x0 + w] == i)
            if len(xs) < 6:
                continue
            cov = np.cov(np.vstack([xs, ys]).astype(float))
            vals_e, vecs = np.linalg.eigh(cov)
            truc = vecs[:, -1]                                   # (cột, hàng) hướng chính
            do_dai = 4.0 * math.sqrt(max(vals_e[-1], 0.0))
            goc_doc = math.degrees(math.atan2(abs(truc[0]), abs(truc[1])))   # 0 = chạy dọc
            if do_dai >= dai_min and goc_doc <= p.GOC_MAX_DO:
                giu[i] = True
        ok = giu[nhan]
        hang, cot = np.nonzero(ok)
        x_m, y_m = be.px_sang_m(hang, cot)
        he, so = self._chon_va_fit(x_m, y_m, nhan[hang, cot])          # 6
        if debug:
            anh['0 goc (nua)'] = xam_nua
            anh['1 birdeye'] = bev
            anh['2 roi'] = roi
            anh['3 tophat'] = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)
            anh['3b nguong'] = mask
            anh['4 morph'] = dong
            mau = np.zeros((*nhan.shape, 3), np.uint8)
            for i in range(1, n):
                mau[nhan == i] = (0, 200, 0) if giu[i] else (0, 0, 160)
            anh['5 thanh phan'] = mau
            anh['6 fit'] = self._ve_fit(bev, he)
        return he, so, anh

    def _chon_va_fit(self, x, y, nhan):
        p = self.p
        if len(x) < p.DIEM_MIN:
            return None, 0
        if self.he is not None:
            a, b, c = self.he
            gan = np.abs(y - (a + b * x + c * x * x)) < p.CUA_SO_M
        else:
            gan = np.zeros(len(x), bool)
            nhan_ds = np.unique(nhan)
            tot, diem_tot = None, 1e9
            for i in nhan_ds:                        # vùng có y trung bình gần Y_KY_VONG nhất
                m = nhan == i
                d = abs(float(np.median(y[m])) - p.Y_KY_VONG) - 0.002 * m.sum() ** 0.5
                if d < diem_tot:
                    tot, diem_tot = i, d
            if tot is not None:
                gan = nhan == tot
        gan &= x <= p.X_FIT_MAX
        x, y = x[gan], y[gan]
        if len(x) < p.DIEM_MIN or np.ptp(x) < p.TAM_X_MIN_M:
            return None, int(len(x))
        if len(x) > 600:
            k = self.rng.choice(len(x), 600, replace=False)
            x, y = x[k], y[k]
        best, best_in = None, None
        for _ in range(p.RANSAC_LAN):
            k = self.rng.choice(len(x), 3, replace=False)
            if np.ptp(x[k]) < 0.05:
                continue
            he = np.polyfit(x[k], y[k], 2)
            if abs(he[0]) > p.C_MAX:
                he = np.r_[0.0, np.polyfit(x[k], y[k], 1)]
            trong = np.abs(np.polyval(he, x) - y) < p.RANSAC_SAI_M
            if best_in is None or trong.sum() > best_in.sum():
                best, best_in = he, trong
        if best is None or best_in.sum() < p.DIEM_MIN:
            return None, 0 if best_in is None else int(best_in.sum())
        xs, ys = x[best_in], y[best_in]
        # Trọng số giảm theo khoảng cách: khoảng cách / góc cần ĐÚNG NGAY CẠNH XE.
        # Không trọng số thì vạch thẳng gần + cong xa bị khớp thành "cong đều",
        # d tại xe sai 5 cm (giả lập, đầu khúc cua).
        tr = np.exp(-(xs - p.X_GAN) / p.TRONG_SO_TAU_M)
        he = (np.polyfit(xs, ys, 2, w=tr) if np.ptp(xs) > 0.3
              else np.r_[0.0, np.polyfit(xs, ys, 1, w=tr)])
        if abs(he[0]) > p.C_MAX:
            he = np.r_[0.0, np.polyfit(xs, ys, 1, w=tr)]
        c, b, a = he
        return (float(a), float(b), float(c)), int(best_in.sum())

    def _ve_fit(self, bev, he):
        img = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
        be = self.be
        if he is not None:
            a, b, c = he
            xs = np.linspace(self.p.X_GAN, self.p.X_XA, 40)
            ys = a + b * xs + c * xs * xs
            r, k = be.m_sang_px(xs, ys)
            cv2.polylines(img, [np.column_stack([k, r]).astype(np.int32)], False, (0, 0, 255), 2)
            r, k = be.m_sang_px(xs, ys + 0.20)
            cv2.polylines(img, [np.column_stack([k, r]).astype(np.int32)], False, (0, 255, 0), 1)
        r0, k0 = be.m_sang_px(self.p.X_NHIN, 0.0)
        cv2.circle(img, (int(k0), int(r0)), 3, (255, 0, 0), -1)
        return img

    # ----------------------------------------------------------------- 7 ---
    def xu_ly(self, xam_nua, t, ds=0.0, dyaw=0.0, debug=False):
        p = self.p
        t0 = time.monotonic()
        self.du_doan(ds, dyaw)
        he, so, anh = self.tach(xam_nua, debug)
        kq = KetQuaVach()
        kq.so_diem = so
        if he is not None:
            a, b, c = he
            y_do = a + b * p.X_NHIN + c * p.X_NHIN ** 2
            psi_do = math.atan(b + 2 * c * p.X_NHIN)
            if self.y is not None and abs(y_do - self.y) > p.CONG_NHAY_M and self.lien_mat < 3:
                kq.ly_do = f'bỏ khung nhảy {abs(y_do - self.y) * 100:.0f} cm'
                self.lien_mat += 1
            else:
                if self.y is None or self.lien_mat >= 3:
                    if self.y is not None:
                        kq.nhay_m = abs(y_do - self.y)
                    self.y, self.psi, self.Py, self.Ppsi = y_do, psi_do, p.R_Y, p.R_PSI
                else:
                    gy = self.Py / (self.Py + p.R_Y)
                    gp = self.Ppsi / (self.Ppsi + p.R_PSI)
                    self.y += gy * (y_do - self.y)
                    self.psi += gp * (psi_do - self.psi)
                    self.Py *= (1 - gy)
                    self.Ppsi *= (1 - gp)
                self.he = he
                self.kappa = 2 * c / (1 + b * b) ** 1.5
                self.t_do = t
                self.lien_mat = 0
                kq.do_moi = True
        else:
            self.lien_mat += 1
            kq.ly_do = f'không fit được ({so} điểm)'
        if self.y is not None and t - self.t_do <= p.GIU_TOI_DA:
            kq.thay = True
            # Quy về ngay vị trí xe (x = 0): góc và khoảng cách đo ở X_NHIN đã chứa
            # sẵn độ cong; để nguyên mà cộng thêm bù cong K_FF*kappa thì bù 2 lần
            # (giả lập: ôm vào trong khúc cua 8 cm).
            psi0 = self.psi - self.kappa * p.X_NHIN
            y0 = self.y - p.X_NHIN * math.tan(psi0) - 0.5 * self.kappa * p.X_NHIN ** 2
            kq.y, kq.psi, kq.kappa = self.y, psi0, self.kappa
            kq.khoang_cach = -y0
            kq.khoang_cach_truoc = -self.y
        elif self.y is not None and t - self.t_do > p.GIU_TOI_DA:
            self.dat_lai()
        kq.ms = (time.monotonic() - t0) * 1000
        kq.anh = anh
        return kq
