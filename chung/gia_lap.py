"""Xe GIẢ LẬP để thử code không cần xe (`--gia-lap`).

Sa bàn giả: xuất phát -> người đi bộ -> dốc (+6 độ) -> đỉnh -> xuống (-5 độ) ->
hầm có khúc cua TRÁI 90 độ -> ra hầm -> đường có vạch liền phải + vạch đứt giữa,
một vệt sáng giả rộng 6 cm và một vạch dừng ngang để thử bộ dò vạch.
Sinh đủ IMU 100 Hz, /scan 10 Hz (360 tia), odom 25 Hz, camera 15 Hz (ảnh JPEG
méo đúng như calib), nên MỌI script giai đoạn chạy y như trên xe thật.
Đây là sa bàn tượng trưng để kiểm luồng code, KHÔNG thay được thử trên xe.
"""
import math
import time

import cv2
import numpy as np

from .camera import Calib
from .lidar import Scan
from .ros_io import OdomTin

G = 9.81
LAN = 0.40              # bề rộng 1 làn
HAM_RONG = 0.77         # tường cách nhau (đo trên xe: tổng 2 bên 0.77)


def _duong_tim():
    """Tim đường (vạch đứt giữa) lấy mẫu mỗi 1 cm: điểm, hướng, quãng đường s."""
    pts = []

    def thang(a, b):
        n = max(2, int(math.dist(a, b) / 0.01))
        for i in range(n):
            k = i / n
            pts.append((a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k))

    def cung(tam, r, a0, a1):
        n = max(2, int(abs(a1 - a0) * r / 0.01))
        for i in range(n):
            a = a0 + (a1 - a0) * i / n
            pts.append((tam[0] + r * math.cos(a), tam[1] + r * math.sin(a)))

    thang((-0.5, 0.0), (4.3, 0.0))
    cung((4.3, 0.5), 0.5, -math.pi / 2, 0.0)          # cua trái 90 độ trong hầm
    thang((4.8, 0.5), (4.8, 3.0))
    cung((3.3, 3.0), 1.5, 0.0, math.pi / 2)            # cua trái êm sau hầm
    thang((3.3, 4.5), (0.5, 4.5))
    p = np.array(pts)
    d = np.gradient(p, axis=0)
    huong = np.arctan2(d[:, 1], d[:, 0])
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    return p, huong, s


class XeGiaLap:
    gia_lap = True

    def __init__(self, can_camera=False, nhanh=False, t_xanh=3.0, co_nguoi=True, w_troi=0.03,
                 doc_len=6.0, doc_xuong=-5.0, he_so_toc_do=0.0):
        self.doc_len, self.doc_xuong = doc_len, doc_xuong     # độ dốc lên / xuống (độ)
        self.den_lich = []       # [(t_bat_dau, t_ket_thuc, 'den do'|'den vang'|'den xanh', cao_px)] đèn dọc đường (giả lập)
        self.he_so_toc_do = he_so_toc_do     # xe ngóc mũi thêm he_so * v^2 độ khi chạy (xe thật: ~30)
        self.nhanh = nhanh
        self.t = 0.0
        self.imu = None
        self.t_xanh = t_xanh
        self.co_nguoi = co_nguoi
        self.w_troi = w_troi                 # bánh lệch: xe tự quay khi chạy thẳng
        self.x, self.y, self.yaw = 0.0, -0.20, 0.0
        self.v = self.w = 0.0
        self.v_lenh = self.w_lenh = 0.0
        self.q = 0.0
        self.va_cham = None
        self.vat_them = []          # [(cx, cy, nửa cạnh, t_bắt_đầu, t_kết_thúc)] vật cản thả thêm để test
        self.ket_den = -1.0         # tới giờ này bánh KHÔNG lăn tới được (giả lập xe kẹt)
        self.odom = OdomTin()
        self.scan = None
        self._t_imu = self._t_scan = self._t_odom = 0.0
        self._luc = {}
        self._rng = np.random.default_rng(1)
        self.tim, self.huong, self.s = _duong_tim()
        self.phap = np.column_stack([-np.sin(self.huong), np.cos(self.huong)])  # pháp tuyến trái
        self._tuong = self._lam_tuong()
        self._anh_cache = (None, -1)
        self._so_anh = 0
        if can_camera:
            self._lam_ban_do()
            self._lam_tia_camera()

    # ------------------------------------------------------------ sa bàn ---
    def _doan(self, s0, s1, lech):
        m = (self.s >= s0) & (self.s <= s1)
        return self.tim[m] + lech * self.phap[m]

    def _lam_tuong(self):
        s_vao, s_ra = 4.0, 6.485
        segs = []
        for lech in (+HAM_RONG / 2, -HAM_RONG / 2):
            p = self._doan(s_vao, s_ra, lech)[::2]
            segs += [[p[i][0], p[i][1], p[i + 1][0], p[i + 1][1]] for i in range(len(p) - 1)]
        return np.array(segs)

    def _vat(self):
        ds = [self._hop(cx, cy, h) for cx, cy, h, t0, t1 in self.vat_them if t0 <= self.t <= t1]
        return np.vstack(ds) if ds else np.empty((0, 4))

    @staticmethod
    def _hop(cx, cy, h):
        c = [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]
        return np.array([[*c[i], *c[(i + 1) % 4]] for i in range(4)])

    def _nguoi(self):
        if not self.co_nguoi or not (self.t_xanh <= self.t <= self.t_xanh + 4.0):
            return np.empty((0, 4))
        cx, cy, h = 0.9, -0.2, 0.06
        c = [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]
        return np.array([[*c[i], *c[(i + 1) % 4]] for i in range(4)])

    def _lam_ban_do(self):
        self.bd_m = 0.004
        self.bd_x0, self.bd_y0 = -1.0, -1.0
        w, h = int(7.5 / self.bd_m), int(7.0 / self.bd_m)
        nen = self._rng.normal(110, 10, (h, w)).astype(np.float32)
        bd = cv2.GaussianBlur(nen, (5, 5), 0)

        def to(p):
            return np.round(np.column_stack([(p[:, 0] - self.bd_x0) / self.bd_m,
                                             (p[:, 1] - self.bd_y0) / self.bd_m])).astype(np.int32)

        def ve(p, rong, sang):
            cv2.polylines(bd, [to(p)], False, float(sang), max(1, int(round(rong / self.bd_m))))

        smax = self.s[-1]
        ve(self._doan(0, smax, -LAN), 0.02, 235)                       # vạch liền phải
        for s0 in np.arange(0, smax, 0.20):                             # vạch đứt giữa
            ve(self._doan(s0, s0 + 0.10, 0.0), 0.02, 235)
        ve(self._doan(0, smax, +LAN), 0.02, 235)                        # mép trái
        ve(self._doan(6.9, 7.4, -0.27), 0.06, 200)                      # vệt sáng giả 6 cm
        i = int(np.searchsorted(self.s, 7.8))                           # vạch dừng ngang
        ve(np.array([self.tim[i] - LAN * self.phap[i], self.tim[i]]), 0.05, 235)
        self.ban_do = np.clip(bd, 0, 255).astype(np.uint8)

    def _lam_tia_camera(self):
        c = Calib.gia_lap()
        self.calib = c
        jj, ii = np.mgrid[0:240, 0:320]
        x, y = c.raw_sang_mat_dat(2 * ii.ravel() + 0.5, 2 * jj.ravel() + 0.5)
        ok = np.isfinite(x) & (x < 3.5)
        self._cam_x = np.where(ok, x, 0.0).reshape(240, 320)
        self._cam_y = np.where(ok, y, 0.0).reshape(240, 320)
        self._cam_ok = ok.reshape(240, 320)

    # ------------------------------------------------------------ vật lý ---
    def _pitch_dich(self):
        toc_do = math.radians(self.he_so_toc_do * self.v * self.v)       # xe thật ngóc mũi khi chạy nhanh
        if abs(self.y) > 0.6 or self.x > 3.4:
            return toc_do
        if 1.5 <= self.x < 2.2:
            return math.radians(self.doc_len) + toc_do
        if 2.8 <= self.x < 3.3:
            return math.radians(self.doc_xuong) + toc_do
        return toc_do

    def _buoc(self, h):
        tau = 0.15
        v_lenh = 0.0 if (self.t < self.ket_den and self.v_lenh > 0) else self.v_lenh
        self.v += (v_lenh - self.v) * min(1.0, h / tau)
        w_dich = self.w_lenh + (self.w_troi if abs(self.v) > 0.02 else 0.0)
        self.w += (w_dich - self.w) * min(1.0, h / tau)
        self.v -= 0.4 * math.sin(self.q) * h          # lên dốc thì chậm lại
        q_truoc = self.q
        self.q += (self._pitch_dich() - self.q) * min(1.0, h / 0.2)
        self.yaw += self.w * h
        ds = self.v * h
        self.x += ds * math.cos(self.yaw)
        self.y += ds * math.sin(self.yaw)
        self.odom.quang_duong += abs(ds)
        self.t += h
        if self.t - self._t_imu >= 0.01:
            self._t_imu = self.t
            self._luc['imu'] = self.t
            if self.imu is not None:
                r = self._rng.normal
                self.imu.cap_nhat(self.t, -G * math.sin(self.q) + r(0, 0.05), r(0, 0.05),
                                  G * math.cos(self.q) + r(0, 0.05), r(0, 0.002) + 0.001,
                                  (self.q - q_truoc) / h + r(0, 0.002) + 0.002,
                                  self.w + r(0, 0.002) + 0.004)
        if self.t - self._t_odom >= 0.04:
            self._t_odom = self.t
            self._luc['odom'] = self.t
            o = self.odom
            o.t, o.x, o.y, o.v, o.w = self.t, self.x, self.y, self.v, self.w
        if self.t - self._t_scan >= 0.1:
            self._t_scan = self.t
            self._luc['scan'] = self.t
            self.scan = self._quet()

    def _quet(self):
        segs = np.vstack([self._tuong, self._nguoi(), self._vat()])
        a = np.radians(np.arange(-180, 180, 1.0))
        dx, dy = np.cos(a + self.yaw), np.sin(a + self.yaw)
        r = np.full(a.shape, np.inf)
        if len(segs):
            px, py = segs[:, 0] - self.x, segs[:, 1] - self.y
            ex, ey = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
            den = dx[:, None] * ey[None] - dy[:, None] * ex[None]
            with np.errstate(divide='ignore', invalid='ignore'):
                t = (px[None] * ey[None] - py[None] * ex[None]) / den
                u = (px[None] * dy[:, None] - py[None] * dx[:, None]) / den
            hit = (np.abs(den) > 1e-9) & (t > 0) & (u >= 0) & (u <= 1)
            t = np.where(hit, t, np.inf).min(axis=1)
            r = np.where(np.isfinite(t), t + self._rng.normal(0, 0.004, a.shape), np.inf)
            if np.min(r) < 0.08 and self.va_cham is None:
                self.va_cham = (round(self.t, 2), round(self.x, 2), round(self.y, 2))
        return Scan(-math.pi, math.radians(1.0), r, 0.05, self.t)

    def _ve_anh(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        wx = self.x + c * self._cam_x - s * self._cam_y
        wy = self.y + s * self._cam_x + c * self._cam_y
        mx = ((wx - self.bd_x0) / self.bd_m).astype(np.float32)
        my = ((wy - self.bd_y0) / self.bd_m).astype(np.float32)
        nua = cv2.remap(self.ban_do, mx, my, cv2.INTER_LINEAR, borderValue=90)
        nua[~self._cam_ok] = 60
        full = cv2.resize(nua, (640, 480), interpolation=cv2.INTER_LINEAR)
        return cv2.imencode('.jpg', full, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()

    # ---------------------------------------------------------------- API ---
    def gio(self):
        return self.t

    def ngu(self, dt):
        if not self.nhanh:
            time.sleep(max(0.0, dt))
        n = max(1, int(round(dt / 0.005)))
        for _ in range(n):
            self._buoc(dt / n)

    def gan_imu(self, imu):
        self.imu = imu

    def anh_jpeg(self):
        if not hasattr(self, 'ban_do'):
            return None, 0
        k = int(self.t * 15)
        if k != self._anh_cache[1]:
            self._anh_cache = (self._ve_anh(), k)
            self._so_anh += 1
        return self._anh_cache[0], self._so_anh

    def den_doc_duong(self):
        """Kết quả nhận đèn giả lập cho S1 (cùng dạng NhanDen.nhan): [{'ten','tin','hop'}]."""
        return [{'ten': ten, 'tin': 0.9, 'hop': [300, 100, 330, 100 + cao]}
                for t0, t1, ten, cao in self.den_lich if t0 <= self.t <= t1]

    def den_gia_lap(self):
        return 'den xanh' if self.t >= self.t_xanh else 'den do'

    def tuoi(self, ten):
        if ten == 'camera':
            return 0.0 if hasattr(self, 'ban_do') else math.inf
        t = self._luc.get(ten)
        return math.inf if t is None else self.t - t

    def gui(self, v, w):
        self.v_lenh, self.w_lenh = float(v), float(w)

    def dung(self):
        self.gui(0.0, 0.0)

    def dong(self):
        self.dung()
