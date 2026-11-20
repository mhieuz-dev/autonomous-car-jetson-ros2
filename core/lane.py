"""Dò VẠCH LIỀN BÊN PHẢI: 8 bước, mỗi bước có ảnh debug riêng.

  0 Giải nén     JPEG -> xám, nửa độ phân giải (camera.decode_gray_half)
  1 Bird's eye   1 lần cv2.remap từ ảnh gốc còn méo (camera.BirdEye)
  2 ROI          chỉ giữ dải [Y_ROI_RIGHT, Y_ROI_LEFT] (nửa phải), trước xe X_NEAR..X_FAR
  3 Tách vạch    LỌC THEO BỀ RỘNG (top-hat ngang): vạch sơn ~2 cm sáng hơn nền 2 bên
                 thì giữ, vệt sáng rộng hơn MAX_WIDTH_M (vệt phản chiếu 6 cm trên
                 bạt) bị triệt. Ngưỡng = max(MIN_THRESH, trung bình + K_STD * độ lệch).
  4 Morphology   mở (bỏ chấm lẻ < MIN_WIDTH_M) rồi đóng theo chiều dọc (nối vạch đứt)
  5 Thành phần   giữ vùng DÀI >= MIN_LEN_M, chạy DỌC (lệch < MAX_ANGLE_DEG so với hướng xe);
                 vùng nằm ngang (vạch dừng, vạch người đi bộ) bị loại
  6 Fit          chọn vùng gần vị trí dự đoán (khung trước) hoặc gần Y_EXPECTED; khớp
                 y = a + b x + c x^2 (mét) bằng RANSAC, rồi khớp lại có trọng số
                 ưu tiên điểm gần xe (WEIGHT_TAU_M)
  7 Lọc thời gian Kalman trên (y tại X_LOOK, góc psi); dự đoán theo odom + yaw;
                 bỏ khung nhảy > JUMP_GATE_M; mất vạch thì giữ dự đoán tối đa MAX_HOLD s
Đầu ra: khoảng cách tới vạch phải đo tại X_LOOK phía trước (dist_ahead, dùng
để lái) và ngoại suy về ngay cạnh xe (distance, để xem), góc psi, độ cong.
Bước 8 (PID theo khoảng cách) nằm trong core/lane_follow.py.
"""
import math
import time
import warnings
from types import SimpleNamespace

import cv2
import numpy as np

warnings.simplefilter('ignore', getattr(np, 'exceptions', np).RankWarning)

DEFAULTS = dict(
    X_NEAR=0.15, X_FAR=1.00, M_PX=0.005,  # vùng bird's eye (m) và độ phân giải (m/pixel)
    Y_ROI_LEFT=0.05, Y_ROI_RIGHT=-0.45,   # dải ngang giữ lại (m, + = trái)
    MAX_WIDTH_M=0.045,                    # vệt sáng rộng hơn này bị triệt (vạch thật 0.02)
    MIN_WIDTH_M=0.010,                    # vệt hẹp hơn này coi là nhiễu
    MIN_THRESH=18, K_STD=3.0,             # ngưỡng top-hat
    GAP_JOIN_M=0.06,                      # nối vạch đứt theo chiều dọc
    MIN_LEN_M=0.12, MAX_ANGLE_DEG=50.0,   # vùng giữ lại: đủ dài và chạy dọc
    Y_EXPECTED=-0.20,                     # vạch phải nằm ở đâu khi xe giữa làn
    WINDOW_M=0.08,                        # khi đang bám: chỉ nhận điểm cách dự đoán < này
    RANSAC_ITERS=40, RANSAC_TOL_M=0.012, MIN_POINTS=25, MIN_CENTER_X_M=0.15, C_MAX=1.5,
    X_LOOK=0.35,                     # đo khoảng cách tới vạch tại điểm này phía trước
    WEIGHT_TAU_M=0.40,               # fit: điểm xa hơn X_NEAR + tau nặng nhẹ đi ~1/e lần
    X_FIT_MAX=0.90,                  # chỉ khớp điểm gần hơn (xa: cung tròn khác bậc 2)
    JUMP_GATE_M=0.05, MAX_HOLD=0.5,  # Kalman: bỏ khung nhảy, giữ khi mất
    Q_Y=0.0004, Q_PSI=0.004, R_Y=0.0001, R_PSI=0.01,
)


def params(**overrides):
    p = dict(DEFAULTS)
    p.update({k: v for k, v in overrides.items() if k in DEFAULTS})
    return SimpleNamespace(**p)


class LaneResult:
    def __init__(self):
        self.found = False  # có vạch (đo được hoặc đang giữ dự đoán)
        self.fresh = False  # khung này đo được thật
        self.y = math.nan   # y của vạch tại X_LOOK (m, + = trái)
        self.psi = 0.0      # hướng vạch NGAY CẠNH XE so với hướng xe (rad, + = trái)
        self.kappa = 0.0
        self.distance = math.nan    # từ TÂM XE tới vạch phải (m, dương), ngoại suy về x = 0
        self.dist_ahead = math.nan  # tại X_LOOK phía trước (đo trực tiếp, dùng để lái)
        self.n_points = 0
        self.ms = 0.0
        self.reason = ''
        self.jump_m = 0.0          # > 0: khung này NHẬN vạch mới lệch chừng này (m) so với vạch đang bám (sau 3 khung bỏ nhảy)


class LaneDetector:
    def __init__(self, birdeye, p=None):
        self.birdeye = birdeye
        self.p = p or params()
        p = self.p
        m = birdeye.m
        row, col = np.mgrid[0:birdeye.n_rows, 0:birdeye.n_cols]
        x, y = birdeye.px_to_m(row, col)
        self.roi = ((y <= p.Y_ROI_LEFT) & (y >= p.Y_ROI_RIGHT) &
                    (x >= p.X_NEAR) & (x <= p.X_FAR)).astype(np.uint8)
        k = max(3, int(round(p.MAX_WIDTH_M / m)) | 1)
        self.k_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
        self.k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, int(round(p.MIN_WIDTH_M / m))), 1))
        self.k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, int(round(p.GAP_JOIN_M / m)))))
        self.rng = np.random.default_rng(0)
        self.reset()

    def reset(self):
        self.y = None              # trạng thái Kalman
        self.psi = 0.0
        self.Py = self.Ppsi = 1.0
        self.coef = None             # hệ số fit gần nhất (a, b, c)
        self.t_measure = -1e9
        self.lost_streak = 0
        self.kappa = 0.0

    # --------------------------------------------------------------- 7a ---
    def predict(self, ds, dyaw):
        """Xe chạy thêm ds (m) và quay dyaw (rad, + trái) từ khung trước."""
        if self.y is None:
            return
        p = self.p
        self.y += ds * math.tan(self.psi) - dyaw * p.X_LOOK
        self.psi -= dyaw
        self.Py += p.Q_Y
        self.Ppsi += p.Q_PSI
        if self.coef is not None:
            a, b, c = self.coef
            self.coef = (a + ds * b, b - dyaw, c)

    # --------------------------------------------------------------- 1-6 ---
    def extract(self, gray_half, debug=False):
        """Bước 1-6 trên một khung. Trả (hệ số (a,b,c) hoặc None, số điểm, dict ảnh debug)."""
        p, birdeye = self.p, self.birdeye
        image = {}
        bev = birdeye.warp(gray_half)                                    # 1
        roi = bev * self.roi                                             # 2
        tophat = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, self.k_tophat)  # 3
        vals = tophat[self.roi > 0]
        thresh = max(p.MIN_THRESH, float(vals.mean() + p.K_STD * vals.std())) if vals.size else 255
        mask = ((tophat > thresh) * 255).astype(np.uint8)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.k_open)          # 4
        line = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.k_close)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(line, connectivity=8)  # 5
        keep = np.zeros(n, bool)
        min_len = p.MIN_LEN_M / birdeye.m
        for i in range(1, n):
            x0, y0, w, h, dt = stats[i]
            if max(w, h) < min_len or dt < 6:
                continue
            ys, xs = np.nonzero(labels[y0:y0 + h, x0:x0 + w] == i)
            if len(xs) < 6:
                continue
            cov = np.cov(np.vstack([xs, ys]).astype(float))
            vals_e, vecs = np.linalg.eigh(cov)
            axis_vec = vecs[:, -1]                                   # (cột, hàng) hướng chính
            length = 4.0 * math.sqrt(max(vals_e[-1], 0.0))
            tilt_deg = math.degrees(math.atan2(abs(axis_vec[0]), abs(axis_vec[1])))   # 0 = chạy dọc
            if length >= min_len and tilt_deg <= p.MAX_ANGLE_DEG:
                keep[i] = True
        ok = keep[labels]
        row, col = np.nonzero(ok)
        x_m, y_m = birdeye.px_to_m(row, col)
        coef, count = self._select_and_fit(x_m, y_m, labels[row, col])          # 6
        if debug:
            image['0 goc (nua)'] = gray_half
            image['1 birdeye'] = bev
            image['2 roi'] = roi
            image['3 tophat'] = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)
            image['3b nguong'] = mask
            image['4 morph'] = line
            color = np.zeros((*labels.shape, 3), np.uint8)
            for i in range(1, n):
                color[labels == i] = (0, 200, 0) if keep[i] else (0, 0, 160)
            image['5 thanh phan'] = color
            image['6 fit'] = self._draw_fit(bev, coef)
        return coef, count, image

    def _select_and_fit(self, x, y, labels):
        p = self.p
        if len(x) < p.MIN_POINTS:
            return None, 0
        if self.coef is not None:
            a, b, c = self.coef
            near = np.abs(y - (a + b * x + c * x * x)) < p.WINDOW_M
        else:
            near = np.zeros(len(x), bool)
            label_ids = np.unique(labels)
            good, good_pts = None, 1e9
            for i in label_ids:                        # vùng có y trung bình gần Y_EXPECTED nhất
                m = labels == i
                d = abs(float(np.median(y[m])) - p.Y_EXPECTED) - 0.002 * m.sum() ** 0.5
                if d < good_pts:
                    good, good_pts = i, d
            if good is not None:
                near = labels == good
        near &= x <= p.X_FIT_MAX
        x, y = x[near], y[near]
        if len(x) < p.MIN_POINTS or np.ptp(x) < p.MIN_CENTER_X_M:
            return None, int(len(x))
        if len(x) > 600:
            k = self.rng.choice(len(x), 600, replace=False)
            x, y = x[k], y[k]
        best, best_in = None, None
        for _ in range(p.RANSAC_ITERS):
            k = self.rng.choice(len(x), 3, replace=False)
            if np.ptp(x[k]) < 0.05:
                continue
            coef = np.polyfit(x[k], y[k], 2)
            if abs(coef[0]) > p.C_MAX:
                coef = np.r_[0.0, np.polyfit(x[k], y[k], 1)]
            inside = np.abs(np.polyval(coef, x) - y) < p.RANSAC_TOL_M
            if best_in is None or inside.sum() > best_in.sum():
                best, best_in = coef, inside
        if best is None or best_in.sum() < p.MIN_POINTS:
            return None, 0 if best_in is None else int(best_in.sum())
        xs, ys = x[best_in], y[best_in]
        # Trọng số giảm theo khoảng cách: khoảng cách / góc cần ĐÚNG NGAY CẠNH XE.
        # Không trọng số thì vạch thẳng gần + cong xa bị khớp thành "cong đều",
        # d tại xe sai 5 cm (giả lập, đầu khúc cua).
        tr = np.exp(-(xs - p.X_NEAR) / p.WEIGHT_TAU_M)
        coef = (np.polyfit(xs, ys, 2, w=tr) if np.ptp(xs) > 0.3
              else np.r_[0.0, np.polyfit(xs, ys, 1, w=tr)])
        if abs(coef[0]) > p.C_MAX:
            coef = np.r_[0.0, np.polyfit(xs, ys, 1, w=tr)]
        c, b, a = coef
        return (float(a), float(b), float(c)), int(best_in.sum())

    def _draw_fit(self, bev, coef):
        img = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
        birdeye = self.birdeye
        if coef is not None:
            a, b, c = coef
            xs = np.linspace(self.p.X_NEAR, self.p.X_FAR, 40)
            ys = a + b * xs + c * xs * xs
            r, k = birdeye.m_to_px(xs, ys)
            cv2.polylines(img, [np.column_stack([k, r]).astype(np.int32)], False, (0, 0, 255), 2)
            r, k = birdeye.m_to_px(xs, ys + 0.20)
            cv2.polylines(img, [np.column_stack([k, r]).astype(np.int32)], False, (0, 255, 0), 1)
        r0, k0 = birdeye.m_to_px(self.p.X_LOOK, 0.0)
        cv2.circle(img, (int(k0), int(r0)), 3, (255, 0, 0), -1)
        return img

    # ----------------------------------------------------------------- 7 ---
    def process(self, gray_half, t, ds=0.0, dyaw=0.0, debug=False):
        p = self.p
        t0 = time.monotonic()
        self.predict(ds, dyaw)
        coef, count, image = self.extract(gray_half, debug)
        result = LaneResult()
        result.n_points = count
        if coef is not None:
            a, b, c = coef
            y_meas = a + b * p.X_LOOK + c * p.X_LOOK ** 2
            psi_deg = math.atan(b + 2 * c * p.X_LOOK)
            if self.y is not None and abs(y_meas - self.y) > p.JUMP_GATE_M and self.lost_streak < 3:
                result.reason = f'bỏ khung nhảy {abs(y_meas - self.y) * 100:.0f} cm'
                self.lost_streak += 1
            else:
                if self.y is None or self.lost_streak >= 3:
                    if self.y is not None:
                        result.jump_m = abs(y_meas - self.y)
                    self.y, self.psi, self.Py, self.Ppsi = y_meas, psi_deg, p.R_Y, p.R_PSI
                else:
                    gy = self.Py / (self.Py + p.R_Y)
                    gp = self.Ppsi / (self.Ppsi + p.R_PSI)
                    self.y += gy * (y_meas - self.y)
                    self.psi += gp * (psi_deg - self.psi)
                    self.Py *= (1 - gy)
                    self.Ppsi *= (1 - gp)
                self.coef = coef
                self.kappa = 2 * c / (1 + b * b) ** 1.5
                self.t_measure = t
                self.lost_streak = 0
                result.fresh = True
        else:
            self.lost_streak += 1
            result.reason = f'không fit được ({count} điểm)'
        if self.y is not None and t - self.t_measure <= p.MAX_HOLD:
            result.found = True
            # Quy về ngay vị trí xe (x = 0): góc và khoảng cách đo ở X_LOOK đã chứa
            # sẵn độ cong; để nguyên mà cộng thêm bù cong K_FF*kappa thì bù 2 lần
            # (giả lập: ôm vào trong khúc cua 8 cm).
            psi0 = self.psi - self.kappa * p.X_LOOK
            y0 = self.y - p.X_LOOK * math.tan(psi0) - 0.5 * self.kappa * p.X_LOOK ** 2
            result.y, result.psi, result.kappa = self.y, psi0, self.kappa
            result.distance = -y0
            result.dist_ahead = -self.y
        elif self.y is not None and t - self.t_measure > p.MAX_HOLD:
            self.reset()
        result.ms = (time.monotonic() - t0) * 1000
        result.image = image
        return result
