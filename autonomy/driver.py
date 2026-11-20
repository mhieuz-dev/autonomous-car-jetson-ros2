"""Chương trình lái tự hành: mỗi nhịp nhận biết tình huống -> chọn hành vi theo ưu tiên -> ra lệnh (v, w).

Phần chung của mọi script nằm ở core/runtime.py: dừng khẩn (Ctrl+C, file /tmp/champion_stop, mất cảm biến),
hiệu chuẩn IMU, giới hạn gia tốc, vòng gyro, chặn sát mũi, log, ghi bag, trang xem hình.
"""
import math

from core import runtime
from core.log import log
from core.draw import lidar_top_view, grid

from . import params as TS
from .behaviors import make_behaviors
from .perception import Perception, pedestrian_threshold


class Program(runtime.ProgramBase):
    NAME = 'autonomy'
    HZ = TS.HZ
    NEED_CAMERA = True

    def __init__(self):
        self.MAX_RUN_TIME = TS.MAX_RUN_TIME
        self.perc = None
        self.behaviors = []
        self.current = None
        self.sit = None

    def start(self, ctx):
        self.perc = Perception(ctx)
        self.behaviors = make_behaviors(self.perc.tracker)
        ctx.records['behavior'] = []  # [(giờ, tên)] mỗi lần đổi hành vi, để xem lại / test
        ctx.records['slope'] = []     # [(giờ, trạng thái dốc)] mỗi lần đổi
        log('Tự hành, ưu tiên: ' + ' > '.join(beh.NAME for beh in self.behaviors))

    def step(self, ctx, dt):
        sit = self.sit = self.perc.update(ctx, dt)
        if not ctx.records['slope'] or ctx.records['slope'][-1][1] != sit.slope:
            ctx.records['slope'].append((round(sit.t, 2), sit.slope))
        wants = [beh for beh in self.behaviors if beh.wants(sit)]     # hỏi hết để bộ đếm giờ của từng hành vi luôn đúng
        beh = wants[0]
        if beh is not self.current:
            old = self.current.NAME if self.current else '-'
            log(f'HÀNH VI: {old} -> {beh.NAME}' + (f' ({beh.reason})' if beh.reason else ''))
            ctx.records['behavior'].append((round(sit.t, 2), beh.NAME))
            self.current = beh
            beh.start(sit)
        command = beh.command(sit, dt)
        if beh.finish:
            return runtime.Finish(beh.finish, failed=True)
        return command

    def finish(self, ctx, reason):
        if self.perc is not None:
            self.perc.stop()

    def status(self):
        sit, beh = self.sit, self.current
        if sit is None or beh is None:
            return ''
        tunnel = 'TRONG HẦM' if sit.in_tunnel else f'cổng hầm {sit.tunnel_gate}'
        return f'{beh.NAME} | dốc {sit.slope} {sit.pitch_deg:+.1f} độ | {tunnel} | {beh.describe()}'

    def draw(self, ctx):
        sit = self.sit
        if sit is None:
            return
        d = pedestrian_threshold()
        ctx.viewer.set_image('lidar', lidar_top_view(
            sit.scan, cones=[(0, math.degrees(math.atan2(TS.HALF_WIDTH_M, d)), d, (0, 0, 255) if sit.pedestrian else (0, 255, 0))],
            text=self.current.NAME if self.current else ''))
        if ctx.viewer.is_watched('line'):
            image = self.perc.tracker.image
            ctx.viewer.set_image('line', grid(list(image.items())) if image else grid([('chưa có khung debug', None)]))
        else:
            ctx.viewer.image.setdefault('line', None)     # chỉ báo là có hình, chưa ai xem thì khỏi vẽ
