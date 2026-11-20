"""imu_slope --manual: khung KHÔNG được gửi lệnh nào trong lúc ghi (để tay đẩy được bánh)."""

from core import runtime
from core.simulator import SimCar
from tools import imu_slope as D


def test_manual_mode_sends_no_commands(monkeypatch):
    monkeypatch.setattr(D, 'MAX_RUN_TIME', 3.0)
    monkeypatch.setattr(D.Program, 'MANUAL', True)
    car = SimCar(need_camera=False, fast=True)
    n = {'calls': 0, 'first': None, 'last': None}
    orig_send = car.send
    car.send = lambda v, w: (n.__setitem__('calls', n['calls'] + 1), orig_send(v, w))[1]
    orig_start, orig_finish = D.Program.start, D.Program.finish
    monkeypatch.setattr(D.Program, 'start', lambda self, ctx: (n.__setitem__('first', n['calls']), orig_start(self, ctx))[1])
    monkeypatch.setattr(D.Program, 'finish', lambda self, ctx, why: (n.__setitem__('last', n['calls']), orig_finish(self, ctx, why))[1])
    runtime.run('tools.imu_slope', sim=True, record_bag=False, fast=True, view_port=None, car=car)
    # giữa bắt đầu và kết thúc chỉ được có lệnh DỪNG (v = 0) lúc hết giờ, không có lệnh chạy nào
    assert n['first'] is not None and 0 <= n['last'] - n['first'] <= 1
    assert abs(car.x) < 0.01                                          # xe không tự đi
