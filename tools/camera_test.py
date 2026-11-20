#!/usr/bin/env python3
"""Kiểm CAMERA (XE KHÔNG CHẠY). Hình trực tiếp: trái = ảnh GỐC, phải = ảnh đã NẮN bằng
config/camera.yaml (có lưới + dấu thập giữa ảnh để kiểm đường thẳng ngoài đời có thẳng không).
Xem: http://<ip-xe>:8097/img/camera.mjpg (hoặc bảng điều khiển). In mỗi giây: fps, kích thước
khung, độ sáng, độ nét, độ trễ, số khung rớt, kèm ĐẠT / KHÔNG ĐẠT.

Kiểm bằng mắt: đặt thước / mép bàn / vạch sơn trước camera, bên phải (ảnh nắn) phải THẲNG
dù nằm sát mép ảnh; còn bên trái (ảnh gốc) bị cong là bình thường.
    bash scripts/car.sh camera_test
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import settings             # noqa: E402
from core.camera import decode_color  # noqa: E402
from core.runtime import Tool         # noqa: E402
from core.log import log              # noqa: E402

FPS_MIN = 25.0              # camera 30 Hz; dưới số này là rớt khung
BRIGHTNESS = (40.0, 215.0)  # độ sáng trung bình 0-255 chấp nhận
SHARP_MIN = 60.0            # phương sai Laplacian: dưới = ảnh nhoè / mất nét
LAG_MAX = 0.5               # s: khung cũ hơn thì coi là đứng hình


def load_calib():
    f = settings.CONFIG_DIR / 'camera.yaml'
    if not f.exists():
        raise SystemExit(f'CHƯA CALIB CAMERA (không có {f}). Calib nội: đặt config/camera.yaml')
    c = yaml.safe_load(f.read_text())
    return (np.array(c['camera_matrix']['data'], float).reshape(3, 3),
            np.array(c['distortion_coefficients']['data'], float).ravel())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--sim', action='store_true')
    a = parser.parse_args()
    K, D = load_calib()
    with Tool('camera_test', need_camera=True, sim=a.sim, calibrate=False) as c:
        maps = None
        t_log, old_count, first_count = time.monotonic(), 0, 0
        n, kb, brightness, net = 0, 0.0, 0.0, 0.0
        t_frame, t_draw = time.monotonic(), 0.0
        log('Camera: trái = ảnh gốc, phải = ảnh nắn bằng config/camera.yaml (hình gửi về thu nhỏ cho nhẹ Wi-Fi)')
        while c.running():
            jpeg, count = c.car.camera_jpeg()
            if jpeg is None or count == old_count:
                c.car.sleep(0.005)
                if time.monotonic() - t_frame > LAG_MAX and time.monotonic() - t_log >= 1.0:
                    t_log = time.monotonic()
                    log(f'KHÔNG CÓ KHUNG MỚI {time.monotonic() - t_frame:.1f} s: KHÔNG ĐẠT (camera đứng hình?)')
                continue
            old_count, t_frame = count, time.monotonic()
            if not first_count:
                first_count = count - 1
            # Xử lý + gửi hình tối đa ~8 lần/giây cho nhẹ Jetson và Wi-Fi; fps THẬT của camera đo bằng bộ đếm khung.
            if time.monotonic() - t_draw >= 1 / 8:
                t_draw = time.monotonic()
                bgr = decode_color(jpeg)
                h, w = bgr.shape[:2]
                if maps is None:
                    maps = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_16SC2)
                    log(f'Khung {w}x{h}, calib {int(K[0, 0])}/{int(K[1, 1])} px, tâm ({K[0, 2]:.0f}, {K[1, 2]:.0f})')
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                n += 1
                kb += len(jpeg) / 1024
                brightness += float(gray.mean())
                net += float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if c.viewer:
                    nan = cv2.remap(bgr, maps[0], maps[1], cv2.INTER_LINEAR)
                    for i in range(1, 4):                       # lưới để kiểm đường thẳng
                        cv2.line(nan, (i * w // 4, 0), (i * w // 4, h), (0, 255, 255), 1)
                        cv2.line(nan, (0, i * h // 4), (w, i * h // 4), (0, 255, 255), 1)
                    cv2.drawMarker(nan, (int(K[0, 2]), int(K[1, 2])), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
                    cv2.putText(bgr, 'ROOT', (8, 22), 0, 0.7, (0, 255, 255), 2)
                    cv2.putText(nan, 'NAN (config/camera.yaml)', (8, 22), 0, 0.7, (0, 255, 255), 2)
                    to = np.hstack([bgr, nan])
                    to = cv2.resize(to, (to.shape[1] // 2, to.shape[0] // 2), interpolation=cv2.INTER_AREA)
                    c.viewer.set_image('camera', to, quality=60, always=True)
            dt = time.monotonic() - t_log
            if dt >= 1.0 and n:
                fps = (count - first_count) / dt
                first_count = count
                kbk, sg, nt = kb / n, brightness / n, net / n
                checks = [('fps', fps >= FPS_MIN), ('sáng', BRIGHTNESS[0] <= sg <= BRIGHTNESS[1]), ('nét', nt >= SHARP_MIN)]
                log(f'{fps:4.1f} fps (camera) | {kbk:4.0f} kB/khung | sáng {sg:3.0f} | nét {nt:4.0f} | '
                    + ' '.join(f'{t}:{"ĐẠT" if ok else "KHÔNG ĐẠT"}' for t, ok in checks))
                if c.viewer:
                    c.viewer.update(program='camera_test',
                                   detail=f'{fps:.1f} fps, sáng {sg:.0f}, nét {nt:.0f}')
                t_log, n, kb, brightness, net = time.monotonic(), 0, 0.0, 0.0, 0.0


if __name__ == '__main__':
    main()
