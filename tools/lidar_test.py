#!/usr/bin/env python3
"""Kiểm LiDAR (XE KHÔNG CHẠY). In mỗi 0,5 s: khoảng cách trước / trái / phải / sau,
có người đi bộ theo tham số autonomy/params.py không, cổng hầm theo core/tunnel.py có đạt không.
Hình LiDAR nhìn từ trên: http://<ip-xe>:8097 (hoặc bảng điều khiển).

Checklist:
  1. Đặt hộp ngay TRƯỚC mũi xe, đo bằng thước từ MÉP CẢN TRƯỚC tới hộp = 30 cm.
     Số "trước" phải = 0.30 + LIDAR_TOI_CAN_M (core/settings.py). Lệch thì sửa LIDAR_TOI_CAN_M ở đó.
  2. Đưa tay vào bên TRÁI xe -> số "trái" giảm. Ngược thì sửa LIDAR_LECH_GOC_DO trong core/settings.py.
  3. Đặt xe giữa 2 tấm chắn cách nhau ~0.77 m -> dòng "HẦM: ĐẠT".
    bash scripts/car.sh lidar_test
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.tunnel import P, SIDE_DEG, WALL_MAX  # noqa: E402
from core.runtime import CongCu  # noqa: E402
from core.lidar import cong_ham, range_in_cone, tunnel_pose  # noqa: E402
from core.log import log  # noqa: E402
from core.draw import lidar_tren_xuong  # noqa: E402
from autonomy import params as TS  # noqa: E402
from autonomy.perception import co_nguoi, nguong_nguoi  # noqa: E402


def f(x):
    return ' ---' if x is None else f'{x:4.2f}'


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--gia-lap', action='store_true')
    a = ap.parse_args()
    with CongCu('lidar_test', gia_lap=a.gia_lap, hieu_chuan=False) as c:
        log(f'Người đi bộ: dừng khi vật trong vùng rộng ±{TS.NUA_RONG_M} m, gần hơn '
            f'{nguong_nguoi():.2f} m tính từ LiDAR (= {TS.LIDAR_TOI_CAN_M} tới cản + {TS.KHOANG_HO_M} hở '
            f'+ {TS.V_BAM_VACH * TS.THOI_GIAN_PHAN_UNG:.3f} phản ứng)')
        t_cu = None
        while c.dang_chay():
            s = c.xe.scan
            if s is None or s.t == t_cu:
                c.xe.ngu(0.05)
                continue
            t_cu = s.t
            truoc, trai = range_in_cone(s, 0, 10), range_in_cone(s, 90, 10)
            phai, sau = range_in_cone(s, -90, 10), range_in_cone(s, 180, 10)
            nguoi, gan = co_nguoi(s)
            gate = cong_ham(s, P)
            pose = tunnel_pose(s, P)
            ham = (f'HẦM: ĐẠT trái {gate[0]:.2f} + phải {gate[1]:.2f} = {sum(gate):.2f}'
                   + (f' | offset {pose[0]:+.3f} psi {math.degrees(pose[1]):+.0f} độ' if pose else '')
                   if gate else 'HẦM: không')
            log(f'trước {f(truoc)} | trái {f(trai)} | phải {f(phai)} | sau {f(sau)} | '
                f'người: {"CÓ -> DỪNG" if nguoi else "không"} ({gan:.2f}) | {ham}')
            if c.xem:
                c.xem.dat_anh('lidar', lidar_tren_xuong(
                    s, non=[(0, math.degrees(math.atan2(TS.NUA_RONG_M, nguong_nguoi())), nguong_nguoi(),
                             (0, 0, 255) if nguoi else (0, 255, 0)),
                            (90, SIDE_DEG, WALL_MAX, (255, 0, 255)),
                            (-90, SIDE_DEG, WALL_MAX, (255, 0, 255))],
                    chu=f'truoc {f(truoc)} trai {f(trai)} phai {f(phai)}'), luon=True)
                c.xem.cap_nhat(chuong_trinh='lidar_test', chi_tiet=ham)
            c.xe.ngu(0.4)


if __name__ == '__main__':
    main()
