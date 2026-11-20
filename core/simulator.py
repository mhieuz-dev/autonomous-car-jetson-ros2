"""Xe GIẢ LẬP để thử code không cần xe (`--sim`).

Sa bàn giả: xuất phát -> người đi bộ -> dốc (+6 độ) -> đỉnh -> xuống (-5 độ) ->
hầm có khúc cua TRÁI 90 độ -> ra hầm -> đường có vạch liền phải + vạch đứt giữa,
một vệt sáng giả rộng 6 cm và một vạch dừng ngang để thử bộ dò vạch.
Sinh đủ IMU 100 Hz, /scan 10 Hz (360 tia), odom 25 Hz, camera 15 Hz (ảnh JPEG
méo đúng như calib), nên mọi script lái chạy y như trên xe thật.
Vị trí dốc và hầm tính theo quãng đường s dọc tim đường (ramp_segments, ham), nên
đổi chỗ được để thử xe gặp tình huống theo thứ tự khác.
Đây là sa bàn tượng trưng để kiểm luồng code, KHÔNG thay được thử trên xe.
"""
import math
import time

import cv2
import numpy as np

from .camera import Calib
from .lidar import Scan
from .ros_io import Odom

G = 9.81
LANE_W = 0.40    # bề rộng 1 làn
TUNNEL_W = 0.77  # tường cách nhau (đo trên xe: tổng 2 bên 0.77)


def _centerline():
    """Tim đường (vạch đứt giữa) lấy mẫu mỗi 1 cm: điểm, hướng, quãng đường s."""
    pts = []

    def straight(a, b):
        n = max(2, int(math.dist(a, b) / 0.01))
        for i in range(n):
            k = i / n
            pts.append((a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k))

    def arc(center, r, a0, a1):
        n = max(2, int(abs(a1 - a0) * r / 0.01))
        for i in range(n):
            a = a0 + (a1 - a0) * i / n
            pts.append((center[0] + r * math.cos(a), center[1] + r * math.sin(a)))

    straight((-0.5, 0.0), (4.3, 0.0))
    arc((4.3, 0.5), 0.5, -math.pi / 2, 0.0)          # cua trái 90 độ trong hầm
    straight((4.8, 0.5), (4.8, 3.0))
    arc((3.3, 3.0), 1.5, 0.0, math.pi / 2)            # cua trái êm sau hầm
    straight((3.3, 4.5), (0.5, 4.5))
    p = np.array(pts)
    d = np.gradient(p, axis=0)
    heading = np.arctan2(d[:, 1], d[:, 0])
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    return p, heading, s


class SimCar:
    sim = True

    def __init__(self, need_camera=False, fast=False, t_pedestrian=3.0, has_pedestrian=True, w_drift=0.03,
                 slope_up=6.0, slope_down=-5.0, speed_factor=30.0, ramp_segments=None, tunnel=(4.0, 6.485)):
        # [(s đầu, s cuối, độ dốc)] dọc tim đường; mặc định dốc lên rồi xuống ngay trước hầm
        self.ramp_segments = ramp_segments if ramp_segments is not None else [(2.0, 2.7, slope_up), (3.3, 3.8, slope_down)]
        self.tunnel = tunnel              # (s vào, s ra) của hầm, None = sa bàn không có hầm
        self.light_schedule = []          # [(t_bat_dau, t_ket_thuc, 'red_light'|'yellow_light'|'green_light', cao_px)] đèn dọc đường (giả lập)
        self.speed_factor = speed_factor  # xe ngóc mũi thêm alpha * v^2 độ khi chạy (xe thật ~30, log 10/10)
        self.fast = fast
        self.t = 0.0
        self.imu = None
        self.t_pedestrian = t_pedestrian     # người đi bộ đứng trên làn từ giây này, trong 4 s
        self.has_pedestrian = has_pedestrian
        self.w_drift = w_drift                 # bánh lệch: xe tự quay khi chạy thẳng
        self.x, self.y, self.yaw = 0.0, -0.20, 0.0
        self.v = self.w = 0.0
        self.v_cmd = self.w_cmd = 0.0
        self.q = 0.0
        self.collided = None
        self.extra_obstacles = []  # [(cx, cy, nửa cạnh, t_bắt_đầu, t_kết_thúc)] vật cản thả thêm để test
        self.stuck_until = -1.0    # tới giờ này bánh KHÔNG lăn tới được (giả lập xe kẹt)
        self.odom = Odom()
        self.scan = None
        self._t_imu = self._t_scan = self._t_odom = 0.0
        self._stamps = {}
        self._rng = np.random.default_rng(1)
        self.centerline, self.heading, self.s = _centerline()
        self.normal = np.column_stack([-np.sin(self.heading), np.cos(self.heading)])  # pháp tuyến trái
        self._walls = self._build_walls()
        self._image_cache = (None, -1)
        self._frame_count = 0
        if need_camera:
            self._build_map()
            self._build_camera_rays()

    # ------------------------------------------------------------ sa bàn ---
    def _segment(self, s0, s1, lateral):
        m = (self.s >= s0) & (self.s <= s1)
        return self.centerline[m] + lateral * self.normal[m]

    def _build_walls(self):
        if self.tunnel is None:
            return np.empty((0, 4))
        s_in, s_out = self.tunnel
        segs = []
        for lateral in (+TUNNEL_W / 2, -TUNNEL_W / 2):
            p = self._segment(s_in, s_out, lateral)[::2]
            segs += [[p[i][0], p[i][1], p[i + 1][0], p[i + 1][1]] for i in range(len(p) - 1)]
        return np.array(segs)

    def _obstacles(self):
        ds = [self._box(cx, cy, h) for cx, cy, h, t0, t1 in self.extra_obstacles if t0 <= self.t <= t1]
        return np.vstack(ds) if ds else np.empty((0, 4))

    @staticmethod
    def _box(cx, cy, h):
        c = [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]
        return np.array([[*c[i], *c[(i + 1) % 4]] for i in range(4)])

    def _pedestrian(self):
        if not self.has_pedestrian or not (self.t_pedestrian <= self.t <= self.t_pedestrian + 4.0):
            return np.empty((0, 4))
        cx, cy, h = 0.9, -0.2, 0.06
        c = [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]
        return np.array([[*c[i], *c[(i + 1) % 4]] for i in range(4)])

    def _build_map(self):
        self.map_m_px = 0.004
        self.map_x0, self.map_y0 = -1.0, -1.0
        w, h = int(7.5 / self.map_m_px), int(7.0 / self.map_m_px)
        nen = self._rng.normal(110, 10, (h, w)).astype(np.float32)
        canvas = cv2.GaussianBlur(nen, (5, 5), 0)

        def to(p):
            return np.round(np.column_stack([(p[:, 0] - self.map_x0) / self.map_m_px,
                                             (p[:, 1] - self.map_y0) / self.map_m_px])).astype(np.int32)

        def draw(p, width, brightness):
            cv2.polylines(canvas, [to(p)], False, float(brightness), max(1, int(round(width / self.map_m_px))))

        smax = self.s[-1]
        draw(self._segment(0, smax, -LANE_W), 0.02, 235)  # vạch liền phải
        for s0 in np.arange(0, smax, 0.20):               # vạch đứt giữa
            draw(self._segment(s0, s0 + 0.10, 0.0), 0.02, 235)
        draw(self._segment(0, smax, +LANE_W), 0.02, 235)  # mép trái
        draw(self._segment(6.9, 7.4, -0.27), 0.06, 200)   # vệt sáng giả 6 cm
        i = int(np.searchsorted(self.s, 7.8))             # vạch dừng ngang
        draw(np.array([self.centerline[i] - LANE_W * self.normal[i], self.centerline[i]]), 0.05, 235)
        self.map_img = np.clip(canvas, 0, 255).astype(np.uint8)

    def _build_camera_rays(self):
        c = Calib.sim()
        self.calib = c
        jj, ii = np.mgrid[0:240, 0:320]
        x, y = c.pixel_to_ground(2 * ii.ravel() + 0.5, 2 * jj.ravel() + 0.5)
        ok = np.isfinite(x) & (x < 3.5)
        self._cam_x = np.where(ok, x, 0.0).reshape(240, 320)
        self._cam_y = np.where(ok, y, 0.0).reshape(240, 320)
        self._cam_ok = ok.reshape(240, 320)

    # ------------------------------------------------------------ vật lý ---
    def _target_pitch(self):
        speed = math.radians(self.speed_factor * self.v * self.v)       # xe thật ngóc mũi khi chạy nhanh
        d = np.hypot(self.centerline[:, 0] - self.x, self.centerline[:, 1] - self.y)
        i = int(np.argmin(d))
        if d[i] > 0.6:                     # ra khỏi đường thì coi như sàn phẳng
            return speed
        for s0, s1, det in self.ramp_segments:
            if s0 <= self.s[i] < s1:
                return math.radians(det) + speed
        return speed

    def _step(self, h):
        tau = 0.15
        v_cmd = 0.0 if (self.t < self.stuck_until and self.v_cmd > 0) else self.v_cmd
        self.v += (v_cmd - self.v) * min(1.0, h / tau)
        w_target = self.w_cmd + (self.w_drift if abs(self.v) > 0.02 else 0.0)
        self.w += (w_target - self.w) * min(1.0, h / tau)
        self.v -= 0.4 * math.sin(self.q) * h          # lên dốc thì chậm lại
        q_prev = self.q
        self.q += (self._target_pitch() - self.q) * min(1.0, h / 0.2)
        self.yaw += self.w * h
        ds = self.v * h
        self.x += ds * math.cos(self.yaw)
        self.y += ds * math.sin(self.yaw)
        self.odom.distance += abs(ds)
        self.t += h
        if self.t - self._t_imu >= 0.01:
            self._t_imu = self.t
            self._stamps['imu'] = self.t
            if self.imu is not None:
                r = self._rng.normal
                self.imu.update(self.t, -G * math.sin(self.q) + r(0, 0.05), r(0, 0.05),
                                  G * math.cos(self.q) + r(0, 0.05), r(0, 0.002) + 0.001,
                                  (self.q - q_prev) / h + r(0, 0.002) + 0.002,
                                  self.w + r(0, 0.002) + 0.004)
        if self.t - self._t_odom >= 0.04:
            self._t_odom = self.t
            self._stamps['odom'] = self.t
            o = self.odom
            o.t, o.x, o.y, o.v, o.w = self.t, self.x, self.y, self.v, self.w
        if self.t - self._t_scan >= 0.1:
            self._t_scan = self.t
            self._stamps['scan'] = self.t
            self.scan = self._sweep()

    def _sweep(self):
        segs = np.vstack([self._walls, self._pedestrian(), self._obstacles()])
        a = np.radians(np.arange(-180, 180, 1.0))
        dx, dy = np.cos(a + self.yaw), np.sin(a + self.yaw)
        r = np.full(a.shape, np.inf)
        if len(segs):
            px, py = segs[:, 0] - self.x, segs[:, 1] - self.y
            ex, ey = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
            light = dx[:, None] * ey[None] - dy[:, None] * ex[None]
            with np.errstate(divide='ignore', invalid='ignore'):
                t = (px[None] * ey[None] - py[None] * ex[None]) / light
                u = (px[None] * dy[:, None] - py[None] * dx[:, None]) / light
            hit = (np.abs(light) > 1e-9) & (t > 0) & (u >= 0) & (u <= 1)
            t = np.where(hit, t, np.inf).min(axis=1)
            r = np.where(np.isfinite(t), t + self._rng.normal(0, 0.004, a.shape), np.inf)
            if np.min(r) < 0.08 and self.collided is None:
                self.collided = (round(self.t, 2), round(self.x, 2), round(self.y, 2))
        return Scan(-math.pi, math.radians(1.0), r, 0.05, self.t)

    def _render(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        wx = self.x + c * self._cam_x - s * self._cam_y
        wy = self.y + s * self._cam_x + c * self._cam_y
        mx = ((wx - self.map_x0) / self.map_m_px).astype(np.float32)
        my = ((wy - self.map_y0) / self.map_m_px).astype(np.float32)
        half = cv2.remap(self.map_img, mx, my, cv2.INTER_LINEAR, borderValue=90)
        half[~self._cam_ok] = 60
        full = cv2.resize(half, (640, 480), interpolation=cv2.INTER_LINEAR)
        return cv2.imencode('.jpg', full, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()

    # ---------------------------------------------------------------- API ---
    def now(self):
        return self.t

    def sleep(self, dt):
        if not self.fast:
            time.sleep(max(0.0, dt))
        n = max(1, int(round(dt / 0.005)))
        for _ in range(n):
            self._step(dt / n)

    def attach_imu(self, imu):
        self.imu = imu

    def camera_jpeg(self):
        if not hasattr(self, 'map_img'):
            return None, 0
        k = int(self.t * 15)
        if k != self._image_cache[1]:
            self._image_cache = (self._render(), k)
            self._frame_count += 1
        return self._image_cache[0], self._frame_count

    def lights_on_road(self):
        """Đèn / biển giả theo light_schedule (cùng dạng Detector.detect): [{'name','conf','box'}]."""
        return [{'name': name, 'conf': 0.9, 'box': [300, 100, 330, 100 + height]}
                for t0, t1, name, height in self.light_schedule if t0 <= self.t <= t1]

    def age(self, name):
        if name == 'camera':
            return 0.0 if hasattr(self, 'map_img') else math.inf
        t = self._stamps.get(name)
        return math.inf if t is None else self.t - t

    def send(self, v, w):
        self.v_cmd, self.w_cmd = float(v), float(w)

    def stop(self):
        self.send(0.0, 0.0)

    def close(self):
        self.stop()
