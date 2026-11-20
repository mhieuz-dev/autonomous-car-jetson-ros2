"""Phần nhận biết dốc và thứ tự ưu tiên hành vi, không cần sa bàn giả."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autonomy import params as TS  # noqa: E402
from autonomy.behaviors import WaitPedestrian, StopAtSign, StopAtLight, DriveRamp, DriveTunnel, ExitTunnel  # noqa: E402
from autonomy.perception import SlopeTracker, Situation  # noqa: E402


def run_slope(seq, dt=0.05):
    """chuoi = [(số giây, pitch độ)] -> các lần đổi trạng thái."""
    d, dist, overrides = SlopeTracker(), 1.0, []
    for seconds, p in seq:
        for _ in range(int(round(seconds / dt))):
            dist += 0.3 * dt
            result = d.update(p, 0.0, dist, dt)
            if result:
                overrides.append(result[1])
    return overrides


def test_slope_up_crest_down_end():
    assert run_slope([(1, 0), (2, 5), (1, 0), (1.5, -4), (1, 0)]) == ['up', 'crest', 'down', 'flat']


def test_short_bump_keeps_state():
    # pitch vọt +5 độ 0,5 s (ngắn hơn SLOPE_UP_S), rồi -3 độ 0,3 s: chỉ là rung
    assert run_slope([(1, 0), (0.5, 5), (1, 0), (0.3, -3), (1, 0)]) == []


def test_downhill_without_uphill_first():
    assert run_slope([(1, 0), (1.5, -4), (1, 0)]) == ['down', 'flat']


def test_long_crest_falls_back_to_flat():
    # lên rồi phẳng mãi (sàn trên cao): quá CREST_MAX_LEN_M thì coi như phẳng, xe bám vạch lại được
    seconds = TS.CREST_MAX_LEN_M / 0.3 + 1.0
    assert run_slope([(1, 0), (2, 5), (seconds, 0)]) == ['up', 'crest', 'flat']


def test_keeps_heading_from_ramp_entry():
    d, dist = SlopeTracker(), 1.0
    for k in range(40):
        yaw = 0.10 if k < 2 else 0.30          # chân chạm dốc lúc hướng 0,10; giữa lúc chờ xác nhận thì xe lắc
        dist += 0.01
        d.update(5.0, yaw, dist, 0.05)
    assert d.status == 'up' and abs(d.slope_yaw - 0.10) < 1e-9


def _sit(**kv):
    sit = Situation()
    sit.t = 10.0
    sit.imu = SimpleNamespace(yaw=0.0, gz=0.0)
    for k, v in kv.items():
        setattr(sit, k, v)
    return sit


def test_no_stops_for_lights_or_signs_on_ramp():
    sit = _sit(slope='up', light='red', light_height=90.0,
             signs=[{'name': 'stop', 'conf': 0.9, 'box': [0, 0, 50, 100]}])
    assert not StopAtLight().wants(sit)
    assert not StopAtSign().wants(sit)
    assert DriveRamp().wants(sit)


def test_waits_1s_after_pedestrian_leaves():
    beh = WaitPedestrian()
    assert beh.wants(_sit(pedestrian=True, ped_dist=0.4, t=10.0))
    assert beh.wants(_sit(t=10.5))
    assert not beh.wants(_sit(t=10.0 + TS.CLEAR_TIME_S + 0.05))


def test_in_tunnel_and_exit():
    assert DriveTunnel().wants(_sit(in_tunnel=True))
    out = ExitTunnel()
    assert out.wants(_sit(t_exit_tunnel=10.0, line_streak=99))          # chưa đủ EXIT_COAST_MIN_S thì vẫn giữ hướng
    assert not out.wants(_sit(t=10.0 + TS.EXIT_COAST_MIN_S + 0.1, t_exit_tunnel=10.0, line_streak=TS.EXIT_LINE_FRAMES))
    assert not out.wants(_sit(t=10.0 + TS.EXIT_COAST_MAX_S + 0.1, t_exit_tunnel=10.0, line_streak=0))
