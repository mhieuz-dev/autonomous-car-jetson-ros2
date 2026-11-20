#!/usr/bin/env python3
"""Kiểm IMU (XE KHÔNG CHẠY). Hiệu chuẩn 2 s rồi làm theo 3 bước có đếm giờ:
  B1 (8 s)  NÂNG MŨI XE lên ~10 độ rồi đặt xuống  -> pitch lớn nhất phải DƯƠNG > 5 độ
  B2 (10 s) XOAY XE sang TRÁI đúng 90 độ (theo mép bàn / thước) rồi để yên -> yaw ~ +90
  B3 (10 s) để YÊN                                   -> yaw trôi < 0.5 độ
Trang xem (http://<ip-xe>:8097) và bảng điều khiển có mô hình xe 3D xoay theo IMU.
    bash scripts/car.sh imu_test            bash scripts/car.sh imu_test --view-only   (chỉ in số, không làm bước)
"""
import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.runtime import Tool  # noqa: E402
from core.log import log       # noqa: E402

GRAVITY = (9.70, 9.95)  # m/s2 chấp nhận
BIAS_MAX = 0.005        # rad/s
JITTER_MAX = 0.01       # rad/s (độ lệch chuẩn gyro z khi đứng yên)


def verdict(ok):
    return 'ĐẠT' if ok else 'KHÔNG ĐẠT'


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--sim', action='store_true')
    parser.add_argument('--view-only', action='store_true', help='chỉ in số liên tục, không làm 3 bước')
    a = parser.parse_args()
    with Tool('imu_test', sim=a.sim) as c:
        imu, count = c.imu, c.imu_stats
        checks = [('Trọng lực', f'{count["gravity"]:.2f} m/s2', GRAVITY[0] <= count['gravity'] <= GRAVITY[1]),
               ('Bias gyro z', f'{count["bias_gz"]:+.4f} rad/s', abs(count['bias_gz']) < BIAS_MAX),
               ('Rung gyro z', f'{count["jitter_gz"]:.4f} rad/s', count['jitter_gz'] < JITTER_MAX)]

        def wait(seconds, text):
            t0, last, mx = time.monotonic(), -1, -1e9
            while time.monotonic() - t0 < seconds and c.running():
                c.car.sleep(0.05)
                mx = max(mx, math.degrees(imu.pitch))
                left_s = int(seconds - (time.monotonic() - t0))
                if left_s != last:
                    last = left_s
                    log(f'{text} | còn {left_s:2d} s | pitch {math.degrees(imu.pitch):+5.1f}  '
                        f'roll {math.degrees(imu.roll):+5.1f}  yaw {math.degrees(imu.yaw):+6.1f}')
                if c.viewer:
                    c.viewer.update(program='imu_test', yaw=round(math.degrees(imu.yaw), 1),
                                   pitch=round(math.degrees(imu.pitch), 2), roll=round(math.degrees(imu.roll), 2),
                                   detail=text)
            return mx

        if a.view_only:
            wait(1e9, 'xem IMU')
            return
        mx = wait(8, 'B1: NÂNG MŨI XE lên rồi đặt xuống')
        checks.append(('Dấu pitch (nâng mũi -> dương)', f'max {mx:+.1f} độ', mx > 5.0))
        wait(10, 'B2: XOAY XE sang TRÁI 90 độ rồi để yên')
        yaw2 = math.degrees(imu.yaw)
        checks.append(('Xoay trái 90 độ', f'yaw {yaw2:+.1f} độ', 85.0 <= yaw2 <= 95.0))
        wait(10, 'B3: để YÊN')
        drift = math.degrees(imu.yaw) - yaw2
        checks.append(('Trôi yaw 10 s', f'{drift:+.2f} độ', abs(drift) < 0.5))
        log('================ KẾT QUẢ IMU ================')
        for name, gt, ok in checks:
            log(f'{name:32s} {gt:22s} {verdict(ok)}')
        if mx <= 0:
            log('=> pitch ÂM khi nâng mũi: sửa PITCH_SIGN = -1.0 trong core/imu.py')
        log('Chấm xong. Tiếp tục phát IMU TRỰC TIẾP lên mô hình xe 3D (nghiêng / xoay xe thử). Ctrl+C hoặc DỪNG để thoát.')
        wait(1e9, 'IMU trực tiếp')


if __name__ == '__main__':
    main()
