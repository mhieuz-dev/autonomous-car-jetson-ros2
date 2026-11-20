"""Khung chạy một chương trình lái: mở xe, hiệu chuẩn IMU, gọi chương trình mỗi nhịp, dừng xe an toàn.

Chương trình (autonomy/driver.py, tools/imu_slope.py) là module có lớp Program với:
    NAME, HZ, MAX_RUN_TIME, NEED_CAMERA
    start(ctx)            gọi 1 lần lúc bắt đầu
    step(ctx, dt)         trả (v, w), (v, w, 'gap') = đặt ngay không qua giới hạn
                          gia tốc (dừng gấp), hoặc Finish(...)
    status()              chuỗi ngắn đưa vào log
    draw(ctx)             (tuỳ chọn) đặt hình lên trang xem
Khung lo: kiểm cảm biến, dừng khẩn (Ctrl+C / file /tmp/champion_stop / mất
cảm biến), giới hạn gia tốc, vòng gyro, chặn sát mũi, log mỗi 0,5 s, ghi bag,
gửi trạng thái lên trang xem.
"""
import importlib
import math
import signal
import subprocess
import time

from . import settings
from .imu import IMU
from .lidar import range_in_cone
from .log import log, open_log
from .pid import RateLimiter, GyroLoop


class Finish:
    def __init__(self, reason, failed=False):
        self.reason = reason
        self.failed = failed    # True = dừng vì sự cố (mã thoát 1), False = xong việc


class ProgramBase:
    NAME = '?'
    HZ = 20
    MAX_RUN_TIME = 60.0
    NEED_CAMERA = False
    RATE_LIMITS = dict(accel=0.6, w_max=1.5, w_accel=6.0)

    def start(self, ctx):
        pass

    def step(self, ctx, dt):
        raise NotImplementedError

    def status(self):
        return ''

    def draw(self, ctx):
        pass

    def finish(self, ctx, reason):
        """Gọi khi chương trình kết thúc vì BẤT KỲ lý do nào (xong, dừng tay, hết giờ, mất cảm biến)."""
        pass


class Context:
    def __init__(self, car, imu, viewer, sim):
        self.car, self.imu, self.viewer, self.sim = car, imu, viewer, sim
        self.records = {}           # chương trình ghi lại để xem sau / test đọc
        self.nose_blocked = self._nose_blocked_prev = False
        self._dist0 = 0.0
        self.t0 = 0.0

    def distance(self):
        """Mét đã chạy từ lúc chương trình bắt đầu (odom)."""
        return self.car.odom.distance - self._dist0

    def elapsed(self):
        return self.car.now() - self.t0


_STOP = {'flag': False, 'reason': ''}


def _install_signals():
    def h(sig, _frame):
        _STOP['flag'] = True
        _STOP['reason'] = signal.Signals(sig).name
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, h)


def should_stop():
    if _STOP['flag']:
        return f'dừng tay ({_STOP["reason"]})'
    if settings.STOP_FILE.exists():
        return 'dừng khẩn (file /tmp/champion_stop)'
    return None


def wait_sensors(car, name, max_time=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < max_time:
        missing = [n for n in name if car.age(n) > 1.0]
        if not missing:
            return []
        if should_stop():
            return missing
        car.sleep(0.05)
    return [n for n in name if car.age(n) > 1.0]


def calibrate_imu(car, imu, seconds=2.0):
    """Xe đứng yên `seconds` giây: đo bias gyro, đặt yaw = 0 theo mũi xe, pitch = 0."""
    log(f'IMU: đứng yên {seconds:.0f} s để hiệu chuẩn (đừng chạm xe)...')
    imu.start_sampling()
    t0 = car.now()
    while car.now() - t0 < seconds:
        car.send(0.0, 0.0)
        car.sleep(0.05)
    count = imu.finish_sampling()
    log(f'IMU: {count["n_samples"]} mẫu | trọng lực {count["gravity"]:.2f} m/s2 | '
        f'bias gz {count["bias_gz"]:+.4f} rad/s | rung gz {count["jitter_gz"]:.4f} | yaw = 0, pitch = 0')
    return count


def _record_bag(name):
    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = settings.LOG_DIR / f'bag_{time.strftime("%Y%m%d_%H%M%S")}_{name}'
    topics = [settings.TOPIC_SCAN, settings.TOPIC_IMU, settings.TOPIC_ODOM,
              settings.TOPIC_CMD, settings.TOPIC_CAMERA]
    try:
        p = subprocess.Popen(['ros2', 'bag', 'record', '-o', str(out), *topics],
                             stdout=open(str(out) + '.log', 'w'), stderr=subprocess.STDOUT)
        log(f'Ghi bag: {out}')
        return p
    except OSError as e:
        log(f'CẢNH BÁO: không ghi được bag ({e})')
        return None


def _run_one(prog, ctx):
    car = ctx.car
    gh = RateLimiter(**prog.RATE_LIMITS)
    gyro = GyroLoop(settings.GYRO_KP, settings.GYRO_KI, settings.GYRO_I_MAX) if settings.GYRO_LOOP_ON else None
    ctx._dist0 = car.odom.distance
    ctx.t0 = car.now()
    prog.start(ctx)
    needed = ['imu', 'scan', 'odom'] + (['camera'] if prog.NEED_CAMERA else [])
    period = 1.0 / prog.HZ
    t_prev = car.now()
    t_log = t_draw = -1e9
    while True:
        reason = should_stop()
        if reason:
            return Finish(reason, failed=True)
        old = [n for n in needed if car.age(n) > (1.0 if n == 'camera' else 0.5)]
        if old:
            car.stop()
            return Finish(f'MẤT CẢM BIẾN {old}', failed=True)
        now_t = car.now()
        dt = max(1e-3, now_t - t_prev)
        t_prev = now_t
        if now_t - ctx.t0 > prog.MAX_RUN_TIME:
            car.stop()
            return Finish(f'quá thời gian tối đa {prog.MAX_RUN_TIME:.0f} s', failed=True)
        result = prog.step(ctx, dt)
        if isinstance(result, Finish):
            return result
        ctx.nose_blocked = False
        if settings.NOSE_GUARD_ON and result[0] > 0.0:
            d = range_in_cone(car.scan, 0.0, settings.NOSE_GUARD_HALF_DEG)
            if d is not None and d < settings.LIDAR_TO_BUMPER_M + settings.NOSE_GUARD_DIST_M:
                ctx.nose_blocked = True
                result = (0.0, result[1], 'gap')         # cấm chạy tới, vẫn cho quay
        if ctx.nose_blocked != ctx._nose_blocked_prev:
            ctx._nose_blocked_prev = ctx.nose_blocked
            log('CHẶN MŨI: vật sát cản trước -> không chạy tới (vẫn quay / lùi được)' if ctx.nose_blocked
                else 'CHẶN MŨI: hết vật cản -> chạy tiếp')
        if len(result) == 3 and result[2] == 'gap':      # dừng gấp: bỏ qua giới hạn gia tốc
            gh.reset(result[0], result[1])
            v, w = result[0], result[1]
        else:
            v, w = gh.parser(result[0], result[1], dt)
        w_sent = gyro.parser(v, w, ctx.imu.gz, dt) if gyro else w
        if not getattr(prog, 'NO_SEND', False):       # chế độ ĐẨY TAY: không gửi lệnh nào để bánh quay tự do
            car.send(v, w_sent)
        imu = ctx.imu
        status = prog.status()
        if ctx.viewer is not None:
            ctx.viewer.update(program=prog.NAME, v=round(v, 3), w=round(w, 3),
                             yaw=round(math.degrees(imu.yaw), 1), pitch=round(math.degrees(imu.pitch), 2),
                             roll=round(math.degrees(imu.roll), 2), distance=round(ctx.distance(), 2),
                             t=round(ctx.elapsed(), 1), detail=status)
            if now_t - t_draw >= 0.2:
                t_draw = now_t
                prog.draw(ctx)
        if now_t - t_log >= 0.5:
            t_log = now_t
            log(f'{prog.NAME:13s} t={ctx.elapsed():5.1f} v={v:+.2f} w={w:+.2f} '
                f'yaw={math.degrees(imu.yaw):+6.1f} pitch={math.degrees(imu.pitch):+5.1f} '
                f'qd={ctx.distance():4.2f}m | {status}')
        remaining = period - (car.now() - now_t) if not car.sim else period
        car.sleep(max(0.0, remaining))


def run_one(prog, ctx):
    result = _run_one(prog, ctx)
    try:
        prog.finish(ctx, result.reason)
    except Exception as e:      # noqa: BLE001 - báo cáo lỗi không được làm hỏng việc dừng xe
        log(f'CẢNH BÁO: lỗi trong finish của {prog.NAME}: {e}')
    return result


def run(name, sim=False, record_bag=True, fast=False, view_port=settings.VIEW_PORT, car=None):
    """Chạy chương trình trong module `name` (vd 'autonomy.driver') tới khi xong / dừng. Trả (mã thoát, ngữ cảnh)."""
    _install_signals()
    _STOP['flag'] = False
    try:
        settings.STOP_FILE.unlink()
    except OSError:
        pass
    log_name = name.split('.')[-1]
    if not fast:
        log(f'Log: {open_log(log_name + ("_gialap" if sim else ""))}')
    prog_cls = importlib.import_module(name).Program
    if car is None:
        if sim:
            from .simulator import SimCar
            car = SimCar(need_camera=prog_cls.NEED_CAMERA, fast=fast)
        else:
            from .ros_io import RealCar
            car = RealCar(need_camera=prog_cls.NEED_CAMERA)
    imu = IMU()
    car.attach_imu(imu)
    viewer = None
    if view_port:
        from .viewer import Viewer
        viewer = Viewer(view_port)
    ctx = Context(car, imu, viewer, sim)
    bag = _record_bag(log_name) if (record_bag and not sim) else None
    code = 0
    try:
        log(f'Chạy: {name} | {"GIẢ LẬP" if sim else "XE THẬT"}')
        missing = wait_sensors(car, ['imu', 'scan', 'odom'] + (['camera'] if prog_cls.NEED_CAMERA else []))
        if missing:
            log(f'LỖI: không nhận được cảm biến {missing} sau 5 s. Kiểm `docker ps`, crc_bringup.')
            return 2, ctx
        ctx.records['imu_calib'] = calibrate_imu(car, imu)
        prog = prog_cls()
        log(f'==> BẮT ĐẦU {prog.NAME}')
        result = run_one(prog, ctx)
        log(f'<== XONG {prog.NAME}: {result.reason} (sau {ctx.elapsed():.1f} s, {ctx.distance():.2f} m)')
        ctx.records['history'] = [(prog.NAME, result.reason, round(ctx.elapsed(), 2))]
        code = 1 if result.failed else 0
    finally:
        car.stop()
        if bag is not None:
            bag.send_signal(signal.SIGINT)
            try:
                bag.wait(timeout=5)
            except subprocess.TimeoutExpired:
                bag.kill()
        car.close()
        if viewer is not None:
            viewer.close()
        log('Đã gửi lệnh dừng xe (v = 0, w = 0).')
    return code, ctx


class Tool:
    """Mở xe cho một công cụ tự viết vòng lặp riêng. Dùng:  with Tool('imu_test') as c: ...

    c.xe, c.imu, c.xem; c.running() = chưa bị Ctrl+C / dừng khẩn. Khi thoát luôn gửi v = w = 0.
    """

    def __init__(self, name, need_camera=False, sim=False, calibrate=True, view_port=settings.VIEW_PORT):
        self.name, self.need_camera, self.sim = name, need_camera, sim
        self.calibrate, self.view_port = calibrate, view_port

    def __enter__(self):
        _install_signals()
        _STOP['flag'] = False
        try:
            settings.STOP_FILE.unlink()
        except OSError:
            pass
        log(f'Log: {open_log(self.name + ("_gialap" if self.sim else ""))}')
        if self.sim:
            from .simulator import SimCar
            self.car = SimCar(need_camera=self.need_camera)
        else:
            from .ros_io import RealCar
            self.car = RealCar(need_camera=self.need_camera, node_name=f'champion_{self.name}')
        self.imu = IMU()
        self.car.attach_imu(self.imu)
        self.viewer = None
        if self.view_port:
            from .viewer import Viewer
            self.viewer = Viewer(self.view_port)
        missing = wait_sensors(self.car, ['imu', 'scan', 'odom'] + (['camera'] if self.need_camera else []))
        if missing:
            self.__exit__(None, None, None)
            raise SystemExit(f'LỖI: không nhận được cảm biến {missing} sau 5 s. Kiểm `docker ps`, crc_bringup.')
        self.imu_stats = calibrate_imu(self.car, self.imu) if self.calibrate else None
        return self

    def running(self):
        return should_stop() is None

    def __exit__(self, *_):
        try:
            self.car.stop()
            self.car.close()
        finally:
            if self.viewer is not None:
                self.viewer.close()
            log('Đã gửi lệnh dừng xe (v = 0, w = 0).')
        return False
