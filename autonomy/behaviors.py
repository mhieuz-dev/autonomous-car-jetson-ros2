"""Các hành vi lái. Mỗi nhịp, driver.py hỏi mọi hành vi `wants(sit)` (có cần nắm lái không), hành vi đầu tiên
theo thứ tự ưu tiên trả lời có thì ra lệnh `command(sit, dt)` -> (v, w), hoặc (v, w, 'gap') = đặt ngay không qua
giới hạn gia tốc (dùng khi dừng).

Ưu tiên (make_behaviors cuối file): an toàn trước, rồi luật giao thông, rồi địa hình, cuối cùng là bám vạch.
Hành vi không được chọn vẫn được hỏi `wants` mỗi nhịp, nên bộ đếm giờ của nó luôn đúng.
"""
import math

from core import settings, tunnel
from core.lane_follow import D_TARGET, LaneController
from core.stop_sign import StopSignLogic
from core.lidar import range_in_cone
from core.log import log
from core.pid import HeadingHold, wrap_angle

from . import params as TS

_LIGHT_NAMES = {'red': 'ĐỎ', 'yellow': 'VÀNG', 'green': 'XANH'}


class Behavior:
    NAME = '?'

    def __init__(self):
        self.reason = ''    # vì sao muốn nắm lái (in ra log khi đổi hành vi)
        self.finish = None  # đặt chuỗi lý do khi hành vi muốn dừng hẳn lượt chạy

    def wants(self, sit):
        return False

    def start(self, sit):
        """Gọi khi vừa được chọn."""

    def command(self, sit, dt):
        raise NotImplementedError

    def describe(self):
        return ''


class WaitPedestrian(Behavior):
    """Người đi bộ trước mũi (LiDAR): đứng yên tới khi vùng trước trống liên tục CLEAR_TIME_S giây."""
    NAME = 'pedestrian'

    def __init__(self):
        super().__init__()
        self.t_seen = None           # lần cuối còn thấy người, None = không chờ
        self.times = 0

    def wants(self, sit):
        if sit.pedestrian:
            if self.t_seen is None:
                self.times += 1
                self.reason = f'người cách LiDAR {sit.ped_dist:.2f} m (lần {self.times})'
            self.t_seen = sit.t
            return True
        if self.t_seen is not None and sit.t - self.t_seen < TS.CLEAR_TIME_S and sit.slope == 'flat':
            return True
        self.t_seen = None
        return False

    def command(self, sit, dt):
        return 0.0, 0.0, 'gap'

    def describe(self):
        return f'chờ người (lần {self.times})'


class StopAtLight(Behavior):
    """Đèn đỏ / vàng đủ gần: dừng, chờ đèn xanh GREEN_STREAK lần nhận liền. Chờ quá LIGHT_MAX_WAIT_S giây thì đi tiếp
    và bỏ qua đèn LIGHT_IGNORE_S giây (thà chạy còn hơn kẹt cả lượt vì nhận nhầm). Không dừng trên dốc."""
    NAME = 'traffic_light'

    def __init__(self):
        super().__init__()
        self.t_stop = None
        self.green = 0
        self.times = 0
        self.ignore_until = -1.0

    def wants(self, sit):
        if not TS.LIGHTS_ON or sit.slope != 'flat' or sit.t < self.ignore_until:
            self.t_stop = None
            return False
        if self.t_stop is None:
            if sit.light not in ('red', 'yellow'):
                return False
            self.t_stop, self.green = sit.t, 0
            self.times += 1
            self.reason = f'đèn {_LIGHT_NAMES[sit.light]} cao {sit.light_height:.0f} px (lần {self.times})'
            return True
        self.green = self.green + 1 if sit.light == 'green' else 0
        if self.green >= TS.GREEN_STREAK * max(1, int(TS.LIGHT_PERIOD_S * TS.HZ)):
            log('ĐÈN: xanh -> đi tiếp')
            self.t_stop = None
            return False
        if sit.t - self.t_stop > TS.LIGHT_MAX_WAIT_S:
            log(f'ĐÈN: CẢNH BÁO chờ quá {TS.LIGHT_MAX_WAIT_S:.0f} s chưa thấy xanh -> đi tiếp, '
                f'bỏ qua đèn {TS.LIGHT_IGNORE_S:.0f} s (kiểm tra nhận đèn)')
            self.t_stop, self.ignore_until = None, sit.t + TS.LIGHT_IGNORE_S
            return False
        return True

    def command(self, sit, dt):
        return 0.0, 0.0, 'gap'

    def describe(self):
        return f'chờ xanh ({self.green} lần xanh liền)'


class StopAtSign(Behavior):
    """Biển STOP đủ gần: đứng yên STOP_WAIT_S giây rồi đi tiếp, sau đó bỏ qua STOP STOP_IGNORE_S giây để chạy
    qua khỏi biển (luật ở core/stop_sign.py). Không dừng trên dốc."""
    NAME = 'stop_sign'

    def __init__(self):
        super().__init__()
        self.logic = StopSignLogic()

    def wants(self, sit):
        if not TS.SIGNS_ON or sit.slope != 'flat':
            return False
        stop = self.logic.step(sit.signs, sit.t, 'STOP', TS.STOP_MIN_HEIGHT_PX, TS.SIGN_MIN_CONF,
                              TS.STOP_WAIT_S, TS.STOP_IGNORE_S)
        self.reason = f'biển STOP (lần {self.logic.n_times})'
        return stop

    def command(self, sit, dt):
        return 0.0, 0.0, 'gap'


class DriveTunnel(Behavior):
    """Trong hầm chỉ tin LiDAR: bám tâm hành lang bằng luật lái 04a (core/tunnel.py), khúc cua thì chậm lại quẹo về
    phía mở. UNSTICK_ON: kẹt quá UNSTICK_AFTER_S giây thì lùi thẳng một đoạn rồi đi tiếp."""
    NAME = 'tunnel'

    def __init__(self):
        super().__init__()
        self.t_stuck = 0.0
        self.reverse_from = None           # quãng đường lúc bắt đầu lùi (None = không lùi)
        self.n_reverses = 0
        self.v_prev = 0.0

    def wants(self, sit):
        self.reason = 'LiDAR thấy 2 tường'
        return sit.in_tunnel

    def start(self, sit):
        self.t_stuck, self.reverse_from, self.v_prev = 0.0, None, 0.0

    def command(self, sit, dt):
        if TS.UNSTICK_ON:
            reverse = self._unstick(sit, dt)
            if reverse is not None:
                return reverse
        if sit.tunnel_bend is not None:
            v, w = sit.tunnel_bend
        elif sit.tunnel_geom is not None:
            v, w = tunnel.V_TUNNEL, tunnel.steer_cmd(*sit.tunnel_geom, tunnel.V_TUNNEL)
        else:
            v, w = tunnel.V_TUNNEL, 0.0
        self.v_prev = v
        return v, w

    def _unstick(self, sit, dt):
        """Kẹt (chặn mũi, hoặc ra lệnh chạy mà bánh không lăn) quá UNSTICK_AFTER_S giây -> lùi thẳng."""
        if self.reverse_from is None:
            stuck = sit.nose_blocked or (self.v_prev > 0.03 and abs(sit.v) < 0.015)
            self.t_stuck = self.t_stuck + dt if stuck else 0.0
            if self.t_stuck < TS.UNSTICK_AFTER_S:
                return None
            self.reverse_from = sit.distance
            self.n_reverses += 1
            log(f'HẦM: KẸT {self.t_stuck:.1f} s -> lùi {TS.REVERSE_DIST_M} m rồi đi tiếp (lần {self.n_reverses})')
        after = range_in_cone(sit.scan, 180.0, 25.0)
        wait_over = after is not None and after < settings.LIDAR_TO_REAR_BUMPER_M + TS.REAR_CLEAR_M
        if sit.distance - self.reverse_from >= TS.REVERSE_DIST_M or wait_over:
            log('HẦM: lùi xong' + (' (sau sát vật)' if wait_over else '') + ', đi tiếp')
            self.reverse_from, self.t_stuck, self.v_prev = None, 0.0, 0.0
            return None
        self.v_prev = TS.REVERSE_V
        return TS.REVERSE_V, 0.0

    def describe(self):
        return 'ĐANG LÙI' if self.reverse_from is not None else ''


class DriveRamp(Behavior):
    """Trên dốc camera chúc lên / xuống nên bird's eye đo vạch sai: giữ hướng lúc vào dốc bằng IMU, tốc độ theo
    phần dốc. Không bao giờ đứng yên hay lùi trên dốc (luật thi)."""
    NAME = 'ramp'

    def __init__(self):
        super().__init__()
        self.heading = HeadingHold(TS.KP_SLOPE_HEADING, TS.KI_SLOPE_HEADING, TS.KD_SLOPE_HEADING, TS.W_MAX_SLOPE)
        self.err = 0.0

    def wants(self, sit):
        self.reason = f'dốc {sit.slope}'
        return sit.slope != 'flat'

    def start(self, sit):
        self.heading.reset()

    def command(self, sit, dt):
        w, self.err = self.heading.compute(sit.imu, sit.slope_yaw, dt)
        if abs(math.degrees(self.err)) > TS.HEADING_DEV_DEG:
            return TS.V_REALIGN, w
        v = {'up': TS.V_UPHILL, 'crest': TS.V_CREST}.get(sit.slope, TS.V_DOWNHILL)
        return v, w

    def describe(self):
        return f'lệch hướng {math.degrees(self.err):+.1f} độ'


class ExitTunnel(Behavior):
    """Vừa ra hầm camera chưa bắt được vạch: giữ hướng lúc ra hầm, chạy chậm ít nhất EXIT_COAST_MIN_S giây; thấy vạch
    EXIT_LINE_FRAMES khung liền (hoặc hết EXIT_COAST_MAX_S giây) thì trả lái cho bám vạch."""
    NAME = 'tunnel_exit'

    def __init__(self):
        super().__init__()
        self.heading = HeadingHold(1.5, 0.0, 0.10, 0.6)

    def wants(self, sit):
        if sit.t_exit_tunnel is None:
            return False
        t = sit.t - sit.t_exit_tunnel
        self.reason = 'vừa ra hầm'
        return t < TS.EXIT_COAST_MAX_S and (t < TS.EXIT_COAST_MIN_S or sit.line_streak < TS.EXIT_LINE_FRAMES)

    def start(self, sit):
        self.heading.reset()

    def command(self, sit, dt):
        w, _ = self.heading.compute(sit.imu, sit.exit_yaw, dt)
        return TS.V_EXIT_TUNNEL, w

    def describe(self):
        return 'giữ hướng chờ thấy vạch'


class FollowLane(Behavior):
    """Mặc định: bám vạch phải (core/lane_follow.py). Vạch nhảy sang chỗ bất hợp lý (nhận nhầm vạch làn bên) thì đứng
    yên xoay phải tìm lại; mất vạch quá LINE_LOST_S giây cũng vậy. Xoay hết SEARCH_MAX_DEG vẫn không thấy = hết
    đường, dừng hẳn lượt chạy."""
    NAME = 'lane_follow'

    def __init__(self, tracker):
        super().__init__()
        self.ctrl = LaneController(tracker=tracker)
        self.search = None              # đang xoay tìm vạch: {'t', 'angle', 'yaw', 'why': 'lost' | 'jump', 'ok'}

    def wants(self, sit):
        self.reason = ''
        return True

    def start(self, sit):
        self.ctrl.reset()
        self.search = None

    def command(self, sit, dt):
        v, w, expired = self.ctrl.command(sit.line, sit.imu, dt, TS.V_LANE, t=sit.t)
        result, new = sit.line, sit.new_line

        # ---- nhảy sang vạch khác bất hợp lý: đứng yên ngay, xoay phải tìm lại vạch phải ----
        if (TS.JUMP_CHECK_ON and new and self.search is None and result.jump_m > TS.JUMP_MAX_M
                and abs(result.dist_ahead - D_TARGET) > TS.PLAUSIBLE_ERR_M):
            log(f'VẠCH: nhảy {result.jump_m * 100:.0f} cm, cách vạch mới {result.dist_ahead:.2f} m '
                f'(đích {D_TARGET:.2f}) -> đứng yên, xoay phải tìm lại vạch phải')
            self.search = {'t': 0.0, 'angle': 0.0, 'yaw': sit.imu.yaw, 'why': 'jump', 'ok': 0}
            self.ctrl.tracker.det.reset()           # quên vạch nhầm, dò lại từ đầu
            self.ctrl.reset()
            return 0.0, 0.0
        if self.search is not None and self.search['why'] == 'jump':
            if new:
                plausible = result.found and result.fresh and TS.RIGHT_LINE_NEAR_M <= result.distance <= TS.RIGHT_LINE_FAR_M
                self.search['ok'] = self.search['ok'] + 1 if plausible else 0
            if self.search['ok'] >= TS.N_FRAMES_OK:
                log(f'VẠCH: thấy lại vạch phải cách {result.distance:.2f} m sau khi xoay '
                    f'{math.degrees(self.search["angle"]):.0f} độ -> bám tiếp')
                self.search = None
                self.ctrl.reset()
                return v, w
            return self._search_turn(sit, dt)

        # ---- mất vạch quá LINE_LOST_S: đứng yên, xoay phải từng nhịp tới khi thấy vạch ----
        if not expired:
            if self.search is not None:
                log(f'VẠCH: thấy lại vạch sau khi xoay {math.degrees(self.search["angle"]):.0f} độ -> bám tiếp')
                self.search = None
            return v, w
        if self.search is None:
            self.search = {'t': 0.0, 'angle': 0.0, 'yaw': sit.imu.yaw, 'why': 'lost', 'ok': 0}
            log('VẠCH: mất vạch quá lâu -> dừng, xoay phải từng chút tìm vạch')
        return self._search_turn(sit, dt)

    def _search_turn(self, sit, dt):
        """Một nhịp tìm: xoay phải SEARCH_TURN_S rồi đứng SEARCH_PAUSE_S."""
        t = self.search
        t['angle'] += abs(wrap_angle(sit.imu.yaw - t['yaw']))
        t['yaw'] = sit.imu.yaw
        if math.degrees(t['angle']) > TS.SEARCH_MAX_DEG:
            self.finish = f'xoay phải {TS.SEARCH_MAX_DEG:.0f} độ vẫn không thấy vạch (hết đường)'
            return 0.0, 0.0, 'gap'
        t['t'] = (t['t'] + dt) % (TS.SEARCH_TURN_S + TS.SEARCH_PAUSE_S)
        return 0.0, (-TS.SEARCH_W if t['t'] < TS.SEARCH_TURN_S else 0.0)

    def describe(self):
        search = f'TÌM VẠCH (đã xoay {math.degrees(self.search["angle"]):.0f} độ) | ' if self.search else ''
        return search + self.ctrl.describe()


def make_behaviors(tracker):
    """Mọi hành vi, ưu tiên cao trước. bam = FollowLane của lớp nhận biết (dùng chung khung camera đã dò)."""
    return [WaitPedestrian(), StopAtLight(), StopAtSign(), DriveTunnel(), DriveRamp(), ExitTunnel(), FollowLane(tracker)]
