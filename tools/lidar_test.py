#!/usr/bin/env python3
"""Kiểm LiDAR (XE KHÔNG CHẠY). In mỗi 0,5 s: khoảng cách trước / trái / phải / sau,
có người đi bộ theo tham số autonomy/params.py không, cổng hầm theo core/tunnel.py có đạt không.
Hình LiDAR nhìn từ trên: http://<ip-xe>:8097 (hoặc bảng điều khiển).

Checklist:
  1. Đặt hộp ngay TRƯỚC mũi xe, đo bằng thước từ MÉP CẢN TRƯỚC tới hộp = 30 cm.
     Số "trước" phải = 0.30 + LIDAR_TO_BUMPER_M (core/settings.py). Lệch thì sửa LIDAR_TO_BUMPER_M ở đó.
  2. Đưa tay vào bên TRÁI xe -> số "trái" giảm. Ngược thì sửa LIDAR_YAW_OFFSET_DEG trong core/settings.py.
  3. Đặt xe giữa 2 tấm chắn cách nhau ~0.77 m -> dòng "HẦM: ĐẠT".
    bash scripts/car.sh lidar_test
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.tunnel import P, SIDE_DEG, WALL_MAX                         # noqa: E402
from core.runtime import Tool                                         # noqa: E402
from core.lidar import tunnel_gate, range_in_cone, tunnel_pose        # noqa: E402
from core.log import log                                              # noqa: E402
from core.draw import lidar_top_view                                  # noqa: E402
from autonomy import params as TS                                     # noqa: E402
from autonomy.perception import has_pedestrian, pedestrian_threshold  # noqa: E402


def f(x):
    return ' ---' if x is None else f'{x:4.2f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--sim', action='store_true')
    a = parser.parse_args()
    with Tool('lidar_test', sim=a.sim, calibrate=False) as c:
        log(f'Người đi bộ: dừng khi vật trong vùng rộng ±{TS.HALF_WIDTH_M} m, gần hơn '
            f'{pedestrian_threshold():.2f} m tính từ LiDAR (= {TS.LIDAR_TO_BUMPER_M} tới cản + {TS.CLEARANCE_M} hở '
            f'+ {TS.V_LANE * TS.REACTION_S:.3f} phản ứng)')
        t_old = None
        while c.running():
            s = c.car.scan
            if s is None or s.t == t_old:
                c.car.sleep(0.05)
                continue
            t_old = s.t
            front, left = range_in_cone(s, 0, 10), range_in_cone(s, 90, 10)
            right, after = range_in_cone(s, -90, 10), range_in_cone(s, 180, 10)
            pedestrian, near = has_pedestrian(s)
            gate = tunnel_gate(s, P)
            pose = tunnel_pose(s, P)
            tunnel = (f'HẦM: ĐẠT trái {gate[0]:.2f} + phải {gate[1]:.2f} = {sum(gate):.2f}'
                   + (f' | offset {pose[0]:+.3f} psi {math.degrees(pose[1]):+.0f} độ' if pose else '')
                   if gate else 'HẦM: không')
            log(f'trước {f(front)} | trái {f(left)} | phải {f(right)} | sau {f(after)} | '
                f'người: {"CÓ -> DỪNG" if pedestrian else "không"} ({near:.2f}) | {tunnel}')
            if c.viewer:
                c.viewer.set_image('lidar', lidar_top_view(
                    s, cones=[(0, math.degrees(math.atan2(TS.HALF_WIDTH_M, pedestrian_threshold())), pedestrian_threshold(),
                             (0, 0, 255) if pedestrian else (0, 255, 0)),
                            (90, SIDE_DEG, WALL_MAX, (255, 0, 255)),
                            (-90, SIDE_DEG, WALL_MAX, (255, 0, 255))],
                    text=f'truoc {f(front)} trai {f(left)} phai {f(right)}'), always=True)
                c.viewer.update(program='lidar_test', detail=tunnel)
            c.car.sleep(0.4)


if __name__ == '__main__':
    main()
