"""Nhận biết tình huống: gom camera, LiDAR, IMU, odom thành một bức tranh (Situation) cho mỗi nhịp lái.

Không giả định thứ tự sa bàn. Dốc nhận bằng pitch (đã trừ phần xe ngóc mũi theo tốc độ), hầm bằng 2 tường
LiDAR, người đi bộ bằng LiDAR phía trước, đèn và biển bằng YOLO, vạch bằng bird's eye. Mỗi phần tự theo dõi
trạng thái của nó; hành vi (autonomy/behaviors.py) chỉ đọc kết quả.
"""
import math
import threading
import time

import numpy as np

from core import tunnel
from core.lane_follow import LaneTracker
from core.detector import largest_light
from core.lidar import TunnelBend, points_xy, tunnel_pose
from core.log import log

from . import params as TS


def pedestrian_threshold():
    """Khoảng cách (tính từ LiDAR) mà người đi bộ gần hơn thì phải dừng."""
    return TS.LIDAR_TO_BUMPER_M + TS.CLEARANCE_M + TS.V_LANE * TS.REACTION_S


def has_pedestrian(scan):
    """(có người trước mũi?, khoảng cách gần nhất trong dải trước mũi, tính từ LiDAR)."""
    p = points_xy(scan, 2.0) if scan is not None else []
    if len(p) == 0:
        return False, math.inf
    zone = p[(p[:, 0] > 0.0) & (np.abs(p[:, 1]) < TS.HALF_WIDTH_M)]
    near = float(zone[:, 0].min()) if len(zone) else math.inf
    return int((zone[:, 0] < pedestrian_threshold()).sum()) >= TS.PED_MIN_POINTS, near


class Situation:
    """Bức tranh ở một nhịp lái. Perception cập nhật, hành vi chỉ đọc."""

    def __init__(self):
        self.t = 0.0
        self.imu = None
        self.v = 0.0         # m/s theo odom
        self.distance = 0.0  # m cộng dồn từ lúc mở xe
        self.scan = None
        self.nose_blocked = False  # khung vừa cấm chạy tới vì vật sát mũi (core/runtime.py)
        self.pitch_deg = 0.0       # pitch đã trừ phần ngóc mũi theo tốc độ (độ)
        self.slope = 'flat'        # 'flat' | 'up' | 'crest' | 'down'
        self.slope_yaw = 0.0       # hướng xe lúc bắt đầu vào dốc
        self.tunnel_gate = 0       # số khung LiDAR liền thấy cổng hầm
        self.in_tunnel = False
        self.tunnel_geom = None    # (offset, psi, kappa) của hành lang, đo từ tường
        self.tunnel_bend = None    # (v, w) khi tường chắn trước = khúc cua
        self.t_exit_tunnel = None  # giờ vừa ra hầm lần gần nhất
        self.exit_yaw = 0.0
        self.pedestrian = False
        self.ped_dist = math.inf
        self.signs = []          # khung YOLO mới nhất [{'name', 'conf', 'box'}]
        self.light = None        # 'red' | 'yellow' | 'green' | None: đèn ĐỦ GẦN
        self.light_height = 0.0  # px, đèn lớn nhất thấy được
        self.line = None         # core.lane.LaneResult
        self.new_line = False    # nhịp này có khung camera mới
        self.line_streak = 0     # số khung liền đo được vạch thật


class SlopeTracker:
    """flat -> up -> crest -> down -> flat theo pitch đã bù tốc độ. Mỗi lần đổi, điều kiện phải đúng liên tục
    đủ lâu (SLOPE_*_S, CREST_S) để rung của xe không làm đổi nhầm. Gặp dốc xuống mà không lên trước cũng nhận."""

    def __init__(self):
        self.status = 'flat'
        self.slope_yaw = 0.0
        self._candidate = None
        self._t_hold = 0.0
        self._yaw_start = 0.0
        self._crest_dist = 0.0

    def _conditions(self, p):
        """[(trạng thái kế, điều kiện đúng chưa, phải giữ bao lâu)] của trạng thái hiện tại."""
        state = self.status
        if state == 'flat':
            return [('up', p > TS.SLOPE_UP_DEG, TS.SLOPE_UP_S), ('down', p < TS.SLOPE_DOWN_DEG, TS.SLOPE_DOWN_S)]
        if state == 'up':
            return [('crest', p < TS.CREST_DEG, TS.CREST_S)]
        if state == 'crest':
            return [('down', p < TS.SLOPE_DOWN_DEG, TS.SLOPE_DOWN_S)]
        return [('flat', p > TS.SLOPE_END_DEG, TS.SLOPE_END_S)]

    def update(self, pitch_deg, yaw, distance, dt):
        """Trả (cũ, mới) khi vừa đổi trạng thái, không thì None."""
        if distance < TS.SLOPE_MIN_DIST_M:
            return None
        old = self.status
        if old == 'crest' and distance - self._crest_dist > TS.CREST_MAX_LEN_M:
            log(f'DỐC: đỉnh dài quá {TS.CREST_MAX_LEN_M} m mà chưa xuống, coi như sàn phẳng')
            return self._switch('flat')
        stop = [(cand, keep) for cand, ok, keep in self._conditions(pitch_deg) if ok]
        if not stop:
            self._candidate = None
            return None
        cand, keep = stop[0]
        if cand != self._candidate:
            self._candidate, self._t_hold, self._yaw_start = cand, 0.0, yaw
        self._t_hold += dt
        if self._t_hold < keep:
            return None
        if old == 'flat':
            self.slope_yaw = self._yaw_start      # hướng lúc chân vừa chạm dốc, giữ suốt con dốc
        if cand == 'crest':
            self._crest_dist = distance
        return self._switch(cand)

    def _switch(self, new):
        old, self.status, self._candidate = self.status, new, None
        return old, new


class TunnelTracker:
    """Vào hầm khi thấy cổng hầm tunnel.ENTER_FRAMES khung liền; ra khi mất tường tunnel.EXIT_FRAMES khung
    (tường chắn phía trước ở khúc cua không tính là mất)."""

    def __init__(self):
        self.counts = tunnel.TunnelGate()
        self.bend = TunnelBend()
        self.inside = False
        self.lost = 0
        self.pose = None
        self.bend_cmd = None
        self._t_scan = None

    def update(self, scan):
        """Trả 'enter' / 'exit' khi vừa vào / ra hầm, không thì None."""
        port = self.counts.update(scan)
        if scan is None or scan.t == self._t_scan:
            return None
        self._t_scan = scan.t
        if not self.inside:
            if port < tunnel.ENTER_FRAMES:
                return None
            self.inside, self.lost, self.pose, self.bend = True, 0, None, TunnelBend()
            return 'enter'
        pose = tunnel_pose(scan, tunnel.P)
        self.bend_cmd = self.bend.compute(scan, tunnel.P, tunnel.V_TUNNEL)
        if pose is not None:
            self.pose, self.lost = pose, 0
        elif self.bend_cmd is not None:
            self.lost = 0
        else:
            self.lost += 1
        if self.lost < tunnel.EXIT_FRAMES:
            return None
        self.inside, self.pose, self.bend_cmd = False, None, None
        return 'exit'


class SignWatcher:
    """Đèn + biển bằng YOLO. Trên xe chạy ở luồng riêng (TensorRT ~100 ms/khung), mỗi nhịp lái chỉ lấy kết quả
    mới nhất, cũ quá 2 s coi như không thấy. Sa bàn giả lấy thẳng đèn / biển dựng sẵn (car.lights_on_road)."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.run = True
        self._result, self._t = [], 0.0
        if ctx.sim or not (TS.LIGHTS_ON or TS.SIGNS_ON):
            return
        if TS.LIGHT_SOURCE == 'laptop':
            log('Nhận đèn/biển bằng model trên LAPTOP qua DDS (bảng điều khiển phải bật "Nhận biển bằng laptop")')
            return
        from core.detector import Detector, read_roi
        self.roi = read_roi()
        self.backend = Detector()
        log(f'Nhận đèn/biển bằng {self.backend.name}, ROI hàng {self.roi[1]}..{self.roi[3]}'
            + (f' | lỗi đã gặp: {self.backend.error}' if self.backend.error else ''))
        if not self.backend.ready:
            log('CẢNH BÁO: không chạy được model đèn/biển, xe chạy KHÔNG nhận đèn và biển')
            return
        threading.Thread(target=self._loop, daemon=True).start()

    def result(self):
        ctx = self.ctx
        if ctx.sim:
            return ctx.car.lights_on_road()
        if TS.LIGHT_SOURCE == 'laptop':
            n = getattr(ctx.car, 'remote_detections', None)
            return n['result'] if n is not None and time.monotonic() - n['t'] < 2.0 else []
        return self._result if time.monotonic() - self._t < 2.0 else []

    def _loop(self):
        from core.camera import decode_color
        from core.detector import detect_in_roi, draw_detections
        old_count = 0
        while self.run:
            jpeg, count = self.ctx.car.camera_jpeg()
            if jpeg is None or count == old_count:
                time.sleep(0.03)
                continue
            old_count = count
            t_start = time.monotonic()
            bgr = decode_color(jpeg)
            result, _ = detect_in_roi(self.backend, bgr, self.roi)
            self._result, self._t = result, time.monotonic()
            if self.ctx.viewer is not None:
                self.ctx.viewer.set_image('light', draw_detections(bgr, result, zone=self.roi, zone_only=True))
            remaining = TS.LIGHT_PERIOD_S - (time.monotonic() - t_start)
            if remaining > 0:             # model chậm hơn chu kỳ thì chạy liên tục, không ngủ thêm
                time.sleep(remaining)


class Perception:
    def __init__(self, ctx):
        self.sit = Situation()
        self.sit.imu = ctx.imu
        self.tracker = LaneTracker(ctx.sim)
        self.slope = SlopeTracker()
        self.tunnel = TunnelTracker()
        self.signs = SignWatcher(ctx)
        self._v2 = 0.0

    def update(self, ctx, dt):
        sit, car, imu = self.sit, ctx.car, ctx.imu
        sit.t = car.now()
        sit.v, sit.distance, sit.scan = car.odom.v, car.odom.distance, car.scan
        sit.nose_blocked = ctx.nose_blocked

        self._v2 += (sit.v * sit.v - self._v2) * min(1.0, dt / TS.NOSE_UP_LAG_S)
        sit.pitch_deg = math.degrees(imu.pitch) - TS.NOSE_UP_DEG * self._v2
        overrides = self.slope.update(sit.pitch_deg, imu.yaw, sit.distance, dt)
        if overrides:
            log(f'DỐC: {overrides[0]} -> {overrides[1]} (pitch {sit.pitch_deg:+.1f} độ đã bù tốc độ, '
                f'giữ hướng {math.degrees(self.slope.slope_yaw):+.1f} độ)')
        sit.slope, sit.slope_yaw = self.slope.status, self.slope.slope_yaw

        event = self.tunnel.update(car.scan)
        if event == 'enter':
            log('HẦM: vào hầm (LiDAR thấy 2 tường đúng bề rộng)')
        elif event == 'exit':
            sit.t_exit_tunnel, sit.exit_yaw = sit.t, imu.yaw
            self.tracker.det.reset()           # trong hầm vạch đo bậy, dò lại từ đầu
            sit.line_streak = 0
            log(f'HẦM: ra hầm (mất tường {tunnel.EXIT_FRAMES} khung), hướng {math.degrees(imu.yaw):+.1f} độ')
        sit.tunnel_gate, sit.in_tunnel = self.tunnel.counts.streak, self.tunnel.inside
        sit.tunnel_geom, sit.tunnel_bend = self.tunnel.pose, self.tunnel.bend_cmd

        pedestrian, sit.ped_dist = has_pedestrian(car.scan)
        # tường / mặt dốc ở cổng hầm hay bị nhận là người; trên dốc thì luật cấm dừng
        sit.pedestrian = pedestrian and sit.slope == 'flat' and not sit.in_tunnel and sit.tunnel_gate < 2

        sit.signs = self.signs.result()
        sit.light, sit.light_height, _ = largest_light(sit.signs, TS.LIGHT_MIN_HEIGHT_PX, TS.LIGHT_MIN_CONF, TS.RED_MIN_CONF)

        debug = ctx.viewer is not None and ctx.viewer.is_watched('line')
        sit.line, sit.new_line = self.tracker.update(ctx, debug=debug)
        if sit.new_line:
            sit.line_streak = sit.line_streak + 1 if sit.line.fresh else 0
        return sit

    def stop(self):
        self.signs.run = False
