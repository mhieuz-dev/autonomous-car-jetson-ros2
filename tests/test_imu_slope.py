"""imu_slope: chia phần dốc + ngưỡng gợi ý + chạy thử trên xe giả lập (dốc nhẹ ~1,2 độ như dốc thật)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import runtime           # noqa: E402
from core.simulator import SimCar  # noqa: E402
from tools import imu_slope as D   # noqa: E402


def _profile():
    rng = np.random.default_rng(0)
    seg = [(0.0, 40), (1.2, 70), (0.05, 25), (-1.2, 70), (0.0, 40)]          # phẳng, lên, giữa, xuống, phẳng (20 Hz)
    p = np.concatenate([v + rng.normal(0, 0.15, n) for v, n in seg])
    t = np.arange(len(p)) / 20.0
    return t, t * 0.2, p


def test_split_phases_order_and_numbers():
    t, dist, p = _profile()
    labels = D.split_phases(p)
    order = [n for i, n in enumerate(labels) if i == 0 or n != labels[i - 1]]
    assert order == ['FLAT', 'UP', 'MIDDLE', 'DOWN', 'AFTER']
    stats = D.summarize(t, dist, p, labels)
    assert abs(stats['UP']['tv'] - 1.2) < 0.15 and abs(stats['DOWN']['tv'] + 1.2) < 0.15
    assert abs(stats['MIDDLE']['tv']) < 0.3 and abs(stats['FLAT']['tv']) < 0.15


def test_hints_fall_between_phases():
    t, dist, p = _profile()
    gy = {name: gt for name, _, gt, _ in D.hint(D.summarize(t, dist, p, D.split_phases(p)))}
    assert 0.3 < gy['SLOPE_UP_DEG'] < 1.0                # trên nhiễu phẳng, dưới dốc lên (~1,2)
    assert 0.1 < gy['CREST_DEG'] < 0.9
    assert -1.0 < gy['SLOPE_DOWN_DEG'] < -0.1


def test_sim_car_drives_over_ramp_to_tunnel():
    car = SimCar(need_camera=False, fast=True, has_pedestrian=False, slope_up=1.3, slope_down=-1.2)
    code, ctx = runtime.run('tools.imu_slope', sim=True, record_bag=False, fast=True, view_port=None, car=car)
    assert ctx.records['history'][-1][1].startswith('vào hầm'), ctx.records['history']
    assert car.collided is None
