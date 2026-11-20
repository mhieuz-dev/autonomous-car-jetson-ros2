"""Kiểm BÁNH / PID (XE CHẠY THẬT, cần 2 m trống phía trước, có người đứng cạnh).

Chạy thẳng QUANG_DUONG mét ở tốc độ V rồi dừng, báo tốc độ thật so với lệnh, độ
vọt lúc khởi động, và lệch hướng (IMU) -> bánh trái / phải có đều không.

PID TỪNG BÁNH nằm trong driver BTC (file ~/crc_car.env trên Jetson), KHÔNG phải
code Champion. Sửa số ở khối THAM SỐ bên dưới, Ctrl+S, bấm Xe (hoặc
`bash scripts/car.sh pid_test`): scripts/car.sh tự so với ~/crc_car.env trên xe, khác thì ghi
số mới vào (giữ bản cũ ~/crc_car.env.bak_champion_<giờ>), khởi động lại driver
rồi mới chạy test. Số này áp dụng cho MỌI lần chạy sau, kể cả full map.
Chạy giả lập thì các số này KHÔNG có tác dụng (xe giả có độ lệch bánh cố định).

    bash scripts/car.sh pid_test                          dùng số trong khối THAM SỐ
    bash scripts/car.sh pid_test --v 0.30                 đổi tốc độ cho lần này
    bash scripts/car.sh pid_test --giu-huong              có giữ hướng IMU + vòng gyro (như lúc chạy map)
    bash scripts/car.sh pid_test --cung 0.5               chạy vòng cung w = 0.5 rad/s (kiểm rẽ)
"""
import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import settings  # noqa: E402
from core.runtime import CongCu  # noqa: E402
from core.log import log  # noqa: E402
from core.pid import GioiHan, GiuHuong, VongGyro  # noqa: E402

# ===== THAM SỐ PID BÁNH (driver BTC, ~/crc_car.env) - sửa, Ctrl+S, bấm Xe =====
# Xe tự quay TRÁI (lệch hướng dương) = bánh PHẢI nhanh hơn -> giảm FF_GAIN_RIGHT
# (hoặc tăng FF_GAIN_LEFT) từng chút 0.02-0.05, chạy lại tới khi lệch < 2 độ.
CRC_FF_GAIN_LEFT = 2.02    # feed-forward bánh trái (số 09/10, calib_wheels BTC)
CRC_FF_GAIN_RIGHT = 2.27   # feed-forward bánh phải
CRC_FF_OFFSET = 0.0
CRC_KP = 2.0               # PI tốc độ bánh trong driver
CRC_KI = 10.0
CRC_I_BAND = 6             # 09/10: 3 thì chạy thẳng chậm 18-24 %, 6 thì đúng

# ===== THAM SỐ BÀI TEST ======================================================
V = 0.15                   # m/s
QUANG_DUONG = 1.5          # m
# ===========================================================================

DRIVER = ['CRC_FF_GAIN_LEFT', 'CRC_FF_GAIN_RIGHT', 'CRC_FF_OFFSET', 'CRC_KP', 'CRC_KI', 'CRC_I_BAND']
LECH_HUONG_MAX = 2.0      # độ trên cả quãng đường
SAI_TOC_DO_MAX = 0.05     # 5 %


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--v', type=float, default=V)
    ap.add_argument('--quang-duong', type=float, default=QUANG_DUONG)
    ap.add_argument('--giu-huong', action='store_true')
    ap.add_argument('--cung', type=float, default=0.0, help='w (rad/s) cho bài vòng cung')
    ap.add_argument('--gia-lap', action='store_true')
    ap.add_argument('--in-env', action='store_true', help='chỉ in KEY=VALUE của PID bánh (scripts/car.sh dùng)')
    a = ap.parse_args()
    if a.in_env:
        for k in DRIVER:
            print(f'{k}={globals()[k]}')
        return
    log('PID bánh đang đặt: ' + '  '.join(f'{k[4:]}={globals()[k]}' for k in DRIVER)
        + ('  (giả lập: KHÔNG áp dụng)' if a.gia_lap else ''))
    with CongCu('pid_test', gia_lap=a.gia_lap) as c:
        xe, imu = c.xe, c.imu
        gh = GioiHan(gia_toc=0.6)
        huong = GiuHuong(1.5, 0.3, 0.1, 0.8)
        gyro = VongGyro(settings.GYRO_KP, settings.GYRO_KI, settings.GYRO_I_MAX)
        qd0, t0 = xe.odom.quang_duong, xe.gio()
        mau, v_max_thay = [], 0.0
        log(f'CHẠY: v {a.v} m/s, {a.quang_duong} m, {"giữ hướng" if a.giu_huong else "THÔ"}'
            f'{f", cung w {a.cung}" if a.cung else ""}')
        t_log = 0.0
        while c.dang_chay():
            qd = xe.odom.quang_duong - qd0
            if qd >= a.quang_duong or xe.gio() - t0 > 4 * a.quang_duong / max(a.v, 0.05) + 3:
                break
            dt = 0.05
            w = a.cung
            if a.giu_huong and not a.cung:
                w, _ = huong.tinh(imu, 0.0, dt)
            v, w = gh.ap(a.v, w, dt)
            if a.giu_huong:
                w = gyro.ap(v, w, imu.gz, dt)
            xe.gui(v, w)
            v_max_thay = max(v_max_thay, xe.odom.v)
            if qd > 0.3 * a.quang_duong:
                mau.append((xe.odom.v, imu.gz))
            if xe.gio() - t_log > 0.5:
                t_log = xe.gio()
                log(f'qd {qd:4.2f} m | v odom {xe.odom.v:.3f} | yaw {math.degrees(imu.yaw):+6.2f} độ')
            xe.ngu(dt)
        xe.dung()
        time.sleep(0.5) if not a.gia_lap else xe.ngu(0.5)
        if not mau:
            log('Không đủ dữ liệu (dừng sớm?)')
            return
        v_tb = sum(m[0] for m in mau) / len(mau)
        gz_tb = sum(m[1] for m in mau) / len(mau)
        yaw = math.degrees(imu.yaw)
        log('================ KẾT QUẢ ================')
        sai = (v_tb - a.v) / a.v
        log(f'Tốc độ thật {v_tb:.3f} m/s (lệnh {a.v}) sai {sai * 100:+.1f} %   '
            f'{"ĐẠT" if abs(sai) < SAI_TOC_DO_MAX else "KHÔNG ĐẠT"}')
        log(f'Vọt lúc khởi động: max {v_max_thay:.3f} m/s ({(v_max_thay / a.v - 1) * 100:+.0f} %)')
        if a.cung:
            r_that = v_tb / gz_tb if abs(gz_tb) > 1e-3 else math.inf
            log(f'Vòng cung: w thật {gz_tb:+.3f} (lệnh {a.cung:+.3f}) | bán kính thật {r_that:.2f} m '
                f'(mong {a.v / a.cung:.2f} m)')
        else:
            log(f'Lệch hướng sau {xe.odom.quang_duong - qd0:.2f} m: {yaw:+.2f} độ   '
                f'{"ĐẠT" if abs(yaw) < LECH_HUONG_MAX else "KHÔNG ĐẠT"}')
            if abs(yaw) >= LECH_HUONG_MAX and not a.giu_huong:
                phia = 'TRÁI' if yaw > 0 else 'PHẢI'
                log(f'=> xe tự quay {phia}: bánh {"PHẢI" if yaw > 0 else "TRÁI"} đang nhanh hơn. '
                    f'Giảm CRC_FF_GAIN_{"RIGHT" if yaw > 0 else "LEFT"} 0.02-0.05 ở đầu file pid_test.py, Ctrl+S, chạy lại.')


if __name__ == '__main__':
    main()
