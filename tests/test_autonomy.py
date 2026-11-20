"""Xe tự hành trên sa bàn giả: qua hết sa bàn, đổi chỗ dốc và hầm, người đi bộ, đèn, biển STOP."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import settings, runtime  # noqa: E402
from core.simulator import SimCar   # noqa: E402
from autonomy import params as TS   # noqa: E402


def run(car=None, max_time=120.0, **track):
    """Chạy tới cuối đường giả (hoặc hết max_time giây). Trả (ctx, xe, vết [t, s, lệch ngang, v])."""
    car = car or SimCar(need_camera=True, fast=True, **track)
    trace = []
    send = car.send

    def write(v, w):
        send(v, w)
        i = int(np.argmin(np.hypot(car.centerline[:, 0] - car.x, car.centerline[:, 1] - car.y)))
        n = car.normal[i]
        trace.append((car.t, car.s[i], (car.x - car.centerline[i, 0]) * n[0] + (car.y - car.centerline[i, 1]) * n[1], v))
        if car.s[i] >= car.s[-1] - 0.3:      # hết đường giả: dừng như người bấm Ctrl+C
            runtime._STOP['flag'] = True

    car.send = write
    old, TS.MAX_RUN_TIME = TS.MAX_RUN_TIME, max_time
    try:
        _, ctx = runtime.run('autonomy.driver', sim=True, record_bag=False, fast=True, view_port=None, car=car)
    finally:
        TS.MAX_RUN_TIME = old
    return ctx, car, np.array(trace)


def behavior(ctx):
    return [name for _, name in ctx.records['behavior']]


def slope(ctx):
    return [status for _, status in ctx.records['slope']]


def lateral(trace, s0, s1, dest):
    m = (trace[:, 1] >= s0) & (trace[:, 1] < s1)
    e = trace[m, 2] - dest
    return float(np.sqrt(np.mean(e ** 2))), float(np.abs(e).max())


@pytest.fixture(scope='module')
def full_track():
    return run()


def test_drives_default_track(full_track):
    ctx, car, trace = full_track
    beh = behavior(ctx)
    assert car.collided is None, f'va chạm {car.collided}'
    assert trace[-1, 1] > car.s[-1] - 0.4, 'phải chạy tới cuối đường giả'
    assert beh.index('ramp') < beh.index('tunnel') < beh.index('tunnel_exit'), beh
    assert slope(ctx) == ['flat', 'up', 'crest', 'down', 'flat']
    rms_before, _ = lateral(trace, 0.6, 1.9, -0.20)  # bám vạch trước dốc: tâm làn phải
    _, max_tunnel = lateral(trace, 4.1, 6.4, 0.0)    # trong hầm: giữa 2 tường
    rms_after, _ = lateral(trace, 7.5, 12.5, -0.20)  # bám vạch sau hầm
    print(f'RMS trước dốc {rms_before:.3f} | hầm max {max_tunnel:.3f} | sau hầm {rms_after:.3f}')
    assert rms_before < 0.02
    assert max_tunnel < 0.25            # hầm rộng 0.77, xe rộng ~0.16: lệch 0.25 vẫn chưa chạm
    assert rms_after < 0.03


def test_stops_for_pedestrian(full_track):
    ctx, car, trace = full_track
    assert 'pedestrian' in behavior(ctx)
    stop = trace[(trace[:, 0] > 4.0) & (trace[:, 0] < 7.0)]          # người đứng trên làn từ giây 3 tới 7
    assert np.all(np.abs(stop[:, 3]) < 1e-9), 'có người trước mũi thì phải đứng yên'
    x_nose = stop[:, 1].max() - 0.5 + settings.LIDAR_TO_BUMPER_M  # s = x + 0.5 trên đoạn thẳng đầu
    assert 0.9 - 0.06 - x_nose > 0.10, 'mũi xe phải cách người hơn 10 cm'
    assert np.any(trace[(trace[:, 0] > 8.5) & (trace[:, 0] < 10.0), 3] > 0.05), 'người đi rồi phải chạy tiếp'


def test_tunnel_before_ramp_still_works():
    """Bản thi chạy theo thứ tự cố định nên không qua được sa bàn này. Dốc dời ra đoạn thẳng cuối, sau hầm."""
    ctx, car, trace = run(ramp_segments=[(10.8, 11.4, 6.0), (12.0, 12.5, -5.0)], has_pedestrian=False)
    beh = behavior(ctx)
    assert car.collided is None, f'va chạm {car.collided}'
    assert trace[-1, 1] > car.s[-1] - 0.4
    assert beh.index('tunnel') < beh.index('ramp'), beh
    assert slope(ctx) == ['flat', 'up', 'crest', 'down', 'flat']
    on_slope = trace[(trace[:, 1] >= 10.8) & (trace[:, 1] < 12.5)]
    assert np.all(on_slope[:, 3] > 0.0), 'không được đứng yên trên dốc'


def test_real_3_6_deg_ramp_with_nose_up():
    """Dốc thật lên +3,6 xuống -3,1 độ, xe ngóc mũi 30 x v^2 độ: vẫn nhận đủ lên, đỉnh, xuống."""
    ctx, car, _ = run(max_time=30.0, slope_up=3.6, slope_down=-3.1, has_pedestrian=False)
    assert slope(ctx)[:4] == ['flat', 'up', 'crest', 'down'], ctx.records['slope']
    assert car.collided is None


def test_fast_on_flat_floor_is_not_a_ramp(monkeypatch):
    """Bản thi (S1, ngưỡng 5 độ chưa trừ ngóc mũi) trên sàn PHẲNG này, 0,5 m/s: pitch +6,8 độ, báo tới chân dốc
    sau 0,64 m. Pitch đã trừ phần ngóc mũi thì không còn báo nhầm."""
    monkeypatch.setattr(TS, 'V_LANE', 0.5)
    ctx, car, trace = run(max_time=15.0, ramp_segments=[], tunnel=None, has_pedestrian=False)
    assert trace[:, 3].max() > 0.45, 'phải chạy được tới 0,5 m/s'
    assert slope(ctx) == ['flat'], ctx.records['slope']
    assert 'ramp' not in behavior(ctx)


def _car_after_tunnel(light_schedule):
    car = SimCar(need_camera=True, fast=True, has_pedestrian=False)
    car.x, car.y, car.yaw = 5.0, 1.6, math.pi / 2             # đặt xe giữa đường, sau hầm, giữa làn phải
    car.light_schedule = light_schedule
    return car


def stop_spans(trace, t_from):
    """Các mốc giờ của lần đứng yên đầu tiên sau giờ `t_from`."""
    stop = []
    for t, _, _, v in trace:
        if t > t_from and abs(v) < 1e-9:
            stop.append(t)
        elif stop:
            break
    return stop


def test_start_mid_track_follows_lane_at_once():
    ctx, car, trace = run(car=_car_after_tunnel([]), max_time=10.0)
    assert behavior(ctx) == ['lane_follow']
    assert car.collided is None


def test_stops_at_stop_sign_then_continues():
    ctx, car, trace = run(car=_car_after_tunnel([(5.0, 6.0, 'stop', TS.STOP_MIN_HEIGHT_PX + 20)]), max_time=10.0)
    stop = stop_spans(trace, 4.9)
    assert 'stop_sign' in behavior(ctx)
    assert stop, 'thấy STOP đủ gần thì phải đứng yên'
    assert TS.STOP_WAIT_S - 0.3 <= stop[-1] - stop[0] <= TS.STOP_WAIT_S + 0.4
    assert np.any(trace[trace[:, 0] > stop[-1] + 0.5, 3] > 0.05), 'dừng xong phải chạy tiếp'


def test_far_stop_sign_is_ignored():
    ctx, _, trace = run(car=_car_after_tunnel([(5.0, 6.0, 'stop', TS.STOP_MIN_HEIGHT_PX - 20)]), max_time=9.0)
    assert 'stop_sign' not in behavior(ctx)
    assert np.all(trace[(trace[:, 0] >= 5.0) & (trace[:, 0] <= 8.5), 3] > 0.0)


def test_red_stops_green_goes():
    schedule = [(4.0, 8.0, 'red_light', TS.LIGHT_MIN_HEIGHT_PX + 20), (8.0, 14.0, 'green_light', TS.LIGHT_MIN_HEIGHT_PX + 20)]
    ctx, _, trace = run(car=_car_after_tunnel(schedule), max_time=14.0)
    stop = stop_spans(trace, 3.9)
    assert 'traffic_light' in behavior(ctx)
    assert stop and stop[0] < 4.2 and stop[-1] > 7.9, 'đèn đỏ: đứng yên tới khi đèn xanh'
    assert np.any(trace[(trace[:, 0] > 9.5) & (trace[:, 0] < 11.0), 3] > 0.05), 'đèn xanh đủ lâu thì chạy'


def test_endless_red_waits_max_then_goes():
    schedule = [(4.0, 60.0, 'red_light', TS.LIGHT_MIN_HEIGHT_PX + 20)]
    _, _, trace = run(car=_car_after_tunnel(schedule), max_time=4.0 + TS.LIGHT_MAX_WAIT_S + 3.0)
    stop = stop_spans(trace, 3.9)
    assert TS.LIGHT_MAX_WAIT_S - 0.2 <= stop[-1] - stop[0] <= TS.LIGHT_MAX_WAIT_S + 0.3
    after = trace[(trace[:, 0] > stop[-1] + 1.0) & (trace[:, 0] < trace[-1, 0] - 0.5)]      # bỏ lệnh dừng lúc hết giờ chạy
    assert len(after) and np.all(after[:, 3] > 0.0), 'hết giờ chờ thì đi tiếp, không dừng lại ngay'
