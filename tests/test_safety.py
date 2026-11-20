"""Chặn sát mũi (mọi lúc) và gỡ kẹt trong hầm, trên xe tự hành chạy sa bàn giả."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import settings          # noqa: E402
from core.simulator import SimCar  # noqa: E402
from test_autonomy import run      # noqa: E402
from autonomy import params as TS  # noqa: E402


def test_nose_guard_stops_then_resumes(monkeypatch):
    monkeypatch.setattr(settings, 'NOSE_GUARD_ON', True)
    monkeypatch.setattr(TS, 'V_LANE', 0.15)          # mốc thời gian của bài này tính theo 0.15 m/s
    car = SimCar(need_camera=True, fast=True, has_pedestrian=False)
    car.x, car.y, car.yaw = 5.0, 1.6, math.pi / 2        # sau hầm, giữa làn phải
    car.extra_obstacles = [(5.0, 2.35, 0.05, 0.0, 9.0)]  # hộp chắn ngay trên làn, mất ở giây 9
    y = []
    send = car.send
    car.send = lambda v, w: (send(v, w), y.append((car.t, v, car.y)))
    run(car=car, max_time=14.0)
    assert car.collided is None, f'va chạm {car.collided}'
    stop = [yy for t, v, yy in y if 6.5 < t < 8.5]
    assert stop and all(abs(v) < 1e-9 for t, v, _ in y if 6.5 < t < 8.5)  # đứng trước hộp
    assert 2.35 - 0.05 - max(stop) > settings.LIDAR_TO_BUMPER_M           # mũi xe chưa chạm hộp
    assert max(yy for t, v, yy in y if t > 12) > 2.45                     # hộp đi -> chạy tiếp


def test_stuck_in_tunnel_reverses_then_continues(monkeypatch):
    monkeypatch.setattr(TS, 'UNSTICK_ON', True)
    car = SimCar(need_camera=True, fast=True, has_pedestrian=False)
    car.x, car.y, car.yaw = 4.8, 0.9, math.pi / 2  # trong đoạn hầm thẳng sau khúc cua
    car.stuck_until = 4.5                          # 2 s đầu (sau hiệu chuẩn) bánh không lăn
    command = []
    send = car.send
    car.send = lambda v, w: (send(v, w), command.append((car.t, v)))
    ctx, _, _ = run(car=car, max_time=30.0)
    beh = [name for _, name in ctx.records['behavior']]
    assert any(v < 0 for t, v in command), 'không thấy lệnh lùi'
    assert 'tunnel' in beh and 'tunnel_exit' in beh, beh
    assert car.collided is None
