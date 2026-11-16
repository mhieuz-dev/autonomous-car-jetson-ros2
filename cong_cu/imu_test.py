#!/usr/bin/env python3
"""Kiểm IMU (XE KHÔNG CHẠY). Hiệu chuẩn 2 s rồi làm theo 3 bước có đếm giờ:
  B1 (8 s)  NÂNG MŨI XE lên ~10 độ rồi đặt xuống  -> pitch lớn nhất phải DƯƠNG > 5 độ
  B2 (10 s) XOAY XE sang TRÁI đúng 90 độ (theo mép bàn / thước) rồi để yên -> yaw ~ +90
  B3 (10 s) để YÊN                                   -> yaw trôi < 0.5 độ
Trang xem (http://<ip-xe>:8097) và bảng điều khiển có mô hình xe 3D xoay theo IMU.
    bash chay.sh imu_test            bash chay.sh imu_test --chi-xem   (chỉ in số, không làm bước)
"""
import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chung.khung import CongCu  # noqa: E402
from chung.log import log  # noqa: E402

TRONG_LUC = (9.70, 9.95)     # m/s2 chấp nhận
BIAS_MAX = 0.005             # rad/s
RUNG_MAX = 0.01              # rad/s (độ lệch chuẩn gyro z khi đứng yên)


def dat(ok):
    return 'ĐẠT' if ok else 'KHÔNG ĐẠT'


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--gia-lap', action='store_true')
    ap.add_argument('--chi-xem', action='store_true', help='chỉ in số liên tục, không làm 3 bước')
    a = ap.parse_args()
    with CongCu('imu_test', gia_lap=a.gia_lap) as c:
        imu, so = c.imu, c.so_imu
        ket = [('Trọng lực', f'{so["trong_luc"]:.2f} m/s2', TRONG_LUC[0] <= so['trong_luc'] <= TRONG_LUC[1]),
               ('Bias gyro z', f'{so["bias_gz"]:+.4f} rad/s', abs(so['bias_gz']) < BIAS_MAX),
               ('Rung gyro z', f'{so["rung_gz"]:.4f} rad/s', so['rung_gz'] < RUNG_MAX)]

        def cho(giay, chu):
            t0, cuoi, mx = time.monotonic(), -1, -1e9
            while time.monotonic() - t0 < giay and c.dang_chay():
                c.xe.ngu(0.05)
                mx = max(mx, math.degrees(imu.pitch))
                con = int(giay - (time.monotonic() - t0))
                if con != cuoi:
                    cuoi = con
                    log(f'{chu} | còn {con:2d} s | pitch {math.degrees(imu.pitch):+5.1f}  '
                        f'roll {math.degrees(imu.roll):+5.1f}  yaw {math.degrees(imu.yaw):+6.1f}')
                if c.xem:
                    c.xem.cap_nhat(giai_doan='imu_test', yaw=round(math.degrees(imu.yaw), 1),
                                   pitch=round(math.degrees(imu.pitch), 2), roll=round(math.degrees(imu.roll), 2),
                                   chi_tiet=chu)
            return mx

        if a.chi_xem:
            cho(1e9, 'xem IMU')
            return
        mx = cho(8, 'B1: NÂNG MŨI XE lên rồi đặt xuống')
        ket.append(('Dấu pitch (nâng mũi -> dương)', f'max {mx:+.1f} độ', mx > 5.0))
        cho(10, 'B2: XOAY XE sang TRÁI 90 độ rồi để yên')
        yaw2 = math.degrees(imu.yaw)
        ket.append(('Xoay trái 90 độ', f'yaw {yaw2:+.1f} độ', 85.0 <= yaw2 <= 95.0))
        cho(10, 'B3: để YÊN')
        troi = math.degrees(imu.yaw) - yaw2
        ket.append(('Trôi yaw 10 s', f'{troi:+.2f} độ', abs(troi) < 0.5))
        log('================ KẾT QUẢ IMU ================')
        for ten, gt, ok in ket:
            log(f'{ten:32s} {gt:22s} {dat(ok)}')
        if mx <= 0:
            log('=> pitch ÂM khi nâng mũi: sửa DAU_PITCH = -1.0 trong chung/imu.py')
        log('Chấm xong. Tiếp tục phát IMU TRỰC TIẾP lên mô hình xe 3D (nghiêng / xoay xe thử). Ctrl+C hoặc DỪNG để thoát.')
        cho(1e9, 'IMU trực tiếp')


if __name__ == '__main__':
    main()
