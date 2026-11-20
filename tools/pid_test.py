"""Kiểm BÁNH / PID (XE CHẠY THẬT, cần 2 m trống phía trước, có người đứng cạnh).

Chạy thẳng DISTANCE mét ở tốc độ V rồi dừng, báo tốc độ thật so với lệnh, độ
vọt lúc khởi động, và lệch hướng (IMU) -> bánh trái / phải có đều không.

PID TỪNG BÁNH nằm trong driver BTC (file ~/crc_car.env trên Jetson), KHÔNG phải
code Champion. Sửa số ở khối THAM SỐ bên dưới, Ctrl+S, bấm Xe (hoặc
`bash scripts/car.sh pid_test`): scripts/car.sh tự so với ~/crc_car.env trên xe, khác thì ghi
số mới vào (giữ bản cũ ~/crc_car.env.bak_champion_<giờ>), khởi động lại driver
rồi mới chạy test. Số này áp dụng cho MỌI lần chạy sau, kể cả full map.
Chạy giả lập thì các số này KHÔNG có tác dụng (xe giả có độ lệch bánh cố định).

    bash scripts/car.sh pid_test                          dùng số trong khối THAM SỐ
    bash scripts/car.sh pid_test --v 0.30                 đổi tốc độ cho lần này
    bash scripts/car.sh pid_test --hold-heading              có giữ hướng IMU + vòng gyro (như lúc chạy map)
    bash scripts/car.sh pid_test --arc 0.5               chạy vòng cung w = 0.5 rad/s (kiểm rẽ)
"""
import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import settings                                # noqa: E402
from core.runtime import Tool                            # noqa: E402
from core.log import log                                 # noqa: E402
from core.pid import RateLimiter, HeadingHold, GyroLoop  # noqa: E402

# ===== THAM SỐ PID BÁNH (driver BTC, ~/crc_car.env) - sửa, Ctrl+S, bấm Xe =====
# Xe tự quay TRÁI (lệch hướng dương) = bánh PHẢI nhanh hơn -> giảm FF_GAIN_RIGHT
# (hoặc tăng FF_GAIN_LEFT) từng chút 0.02-0.05, chạy lại tới khi lệch < 2 độ.
CRC_FF_GAIN_LEFT = 2.02   # feed-forward bánh trái (số 09/10, calib_wheels BTC)
CRC_FF_GAIN_RIGHT = 2.27  # feed-forward bánh phải
CRC_FF_OFFSET = 0.0
CRC_KP = 2.0               # PI tốc độ bánh trong driver
CRC_KI = 10.0
CRC_I_BAND = 6             # 09/10: 3 thì chạy thẳng chậm 18-24 %, 6 thì đúng

# ===== THAM SỐ BÀI TEST ======================================================
V = 0.15        # m/s
DISTANCE = 1.5  # m
# ===========================================================================

DRIVER = ['CRC_FF_GAIN_LEFT', 'CRC_FF_GAIN_RIGHT', 'CRC_FF_OFFSET', 'CRC_KP', 'CRC_KI', 'CRC_I_BAND']
MAX_HEADING_DEV = 2.0  # độ trên cả quãng đường
MAX_SPEED_ERR = 0.05   # 5 %


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--v', type=float, default=V)
    parser.add_argument('--distance', type=float, default=DISTANCE)
    parser.add_argument('--hold-heading', action='store_true')
    parser.add_argument('--arc', type=float, default=0.0, help='w (rad/s) cho bài vòng cung')
    parser.add_argument('--sim', action='store_true')
    parser.add_argument('--in-env', action='store_true', help='chỉ in KEY=VALUE của PID bánh (scripts/car.sh dùng)')
    a = parser.parse_args()
    if a.in_env:
        for k in DRIVER:
            print(f'{k}={globals()[k]}')
        return
    log('PID bánh đang đặt: ' + '  '.join(f'{k[4:]}={globals()[k]}' for k in DRIVER)
        + ('  (giả lập: KHÔNG áp dụng)' if a.sim else ''))
    with Tool('pid_test', sim=a.sim) as c:
        car, imu = c.car, c.imu
        gh = RateLimiter(accel=0.6)
        heading = HeadingHold(1.5, 0.3, 0.1, 0.8)
        gyro = GyroLoop(settings.GYRO_KP, settings.GYRO_KI, settings.GYRO_I_MAX)
        dist0, t0 = car.odom.distance, car.now()
        samples, v_max_seen = [], 0.0
        log(f'CHẠY: v {a.v} m/s, {a.distance} m, {"giữ hướng" if a.hold_heading else "THÔ"}'
            f'{f", cung w {a.arc}" if a.arc else ""}')
        t_log = 0.0
        while c.running():
            dist = car.odom.distance - dist0
            if dist >= a.distance or car.now() - t0 > 4 * a.distance / max(a.v, 0.05) + 3:
                break
            dt = 0.05
            w = a.arc
            if a.hold_heading and not a.arc:
                w, _ = heading.compute(imu, 0.0, dt)
            v, w = gh.parser(a.v, w, dt)
            if a.hold_heading:
                w = gyro.parser(v, w, imu.gz, dt)
            car.send(v, w)
            v_max_seen = max(v_max_seen, car.odom.v)
            if dist > 0.3 * a.distance:
                samples.append((car.odom.v, imu.gz))
            if car.now() - t_log > 0.5:
                t_log = car.now()
                log(f'qd {dist:4.2f} m | v odom {car.odom.v:.3f} | yaw {math.degrees(imu.yaw):+6.2f} độ')
            car.sleep(dt)
        car.stop()
        time.sleep(0.5) if not a.sim else car.sleep(0.5)
        if not samples:
            log('Không đủ dữ liệu (dừng sớm?)')
            return
        v_mean = sum(m[0] for m in samples) / len(samples)
        gz_mean = sum(m[1] for m in samples) / len(samples)
        yaw = math.degrees(imu.yaw)
        log('================ KẾT QUẢ ================')
        err = (v_mean - a.v) / a.v
        log(f'Tốc độ thật {v_mean:.3f} m/s (lệnh {a.v}) sai {err * 100:+.1f} %   '
            f'{"ĐẠT" if abs(err) < MAX_SPEED_ERR else "KHÔNG ĐẠT"}')
        log(f'Vọt lúc khởi động: max {v_max_seen:.3f} m/s ({(v_max_seen / a.v - 1) * 100:+.0f} %)')
        if a.arc:
            r_real = v_mean / gz_mean if abs(gz_mean) > 1e-3 else math.inf
            log(f'Vòng cung: w thật {gz_mean:+.3f} (lệnh {a.arc:+.3f}) | bán kính thật {r_real:.2f} m '
                f'(mong {a.v / a.arc:.2f} m)')
        else:
            log(f'Lệch hướng sau {car.odom.distance - dist0:.2f} m: {yaw:+.2f} độ   '
                f'{"ĐẠT" if abs(yaw) < MAX_HEADING_DEV else "KHÔNG ĐẠT"}')
            if abs(yaw) >= MAX_HEADING_DEV and not a.hold_heading:
                side = 'TRÁI' if yaw > 0 else 'PHẢI'
                log(f'=> xe tự quay {side}: bánh {"PHẢI" if yaw > 0 else "TRÁI"} đang nhanh hơn. '
                    f'Giảm CRC_FF_GAIN_{"RIGHT" if yaw > 0 else "LEFT"} 0.02-0.05 ở đầu file pid_test.py, Ctrl+S, chạy lại.')


if __name__ == '__main__':
    main()
