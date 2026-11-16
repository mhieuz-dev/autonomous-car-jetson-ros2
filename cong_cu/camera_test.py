#!/usr/bin/env python3
"""Kiểm CAMERA (XE KHÔNG CHẠY). Hình trực tiếp: trái = ảnh GỐC, phải = ảnh đã NẮN bằng
calib/camera.yaml (có lưới + dấu thập giữa ảnh để kiểm đường thẳng ngoài đời có thẳng không).
Xem: http://<ip-xe>:8097/anh/camera.mjpg (hoặc bảng điều khiển). In mỗi giây: fps, kích thước
khung, độ sáng, độ nét, độ trễ, số khung rớt, kèm ĐẠT / KHÔNG ĐẠT.

Kiểm bằng mắt: đặt thước / mép bàn / vạch sơn trước camera, bên phải (ảnh nắn) phải THẲNG
dù nằm sát mép ảnh; còn bên trái (ảnh gốc) bị cong là bình thường.
    bash chay.sh camera_test
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chung import cau_hinh  # noqa: E402
from chung.camera import giai_nen_mau  # noqa: E402
from chung.khung import CongCu  # noqa: E402
from chung.log import log  # noqa: E402

FPS_MIN = 25.0                 # camera 30 Hz; dưới số này là rớt khung
SANG = (40.0, 215.0)           # độ sáng trung bình 0-255 chấp nhận
NET_MIN = 60.0                 # phương sai Laplacian: dưới = ảnh nhoè / mất nét
TRE_MAX = 0.5                  # s: khung cũ hơn thì coi là đứng hình


def nap_calib():
    f = cau_hinh.CALIB / 'camera.yaml'
    if not f.exists():
        raise SystemExit(f'CHƯA CALIB CAMERA (không có {f}). Calib nội: đặt calib/camera.yaml')
    c = yaml.safe_load(f.read_text())
    return (np.array(c['camera_matrix']['data'], float).reshape(3, 3),
            np.array(c['distortion_coefficients']['data'], float).ravel())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--gia-lap', action='store_true')
    a = ap.parse_args()
    K, D = nap_calib()
    with CongCu('camera_test', can_camera=True, gia_lap=a.gia_lap, hieu_chuan=False) as c:
        maps = None
        t_log, so_cu, so_dau = time.monotonic(), 0, 0
        n, kb, sang, net = 0, 0.0, 0.0, 0.0
        t_khung, t_ve = time.monotonic(), 0.0
        log('Camera: trái = ảnh gốc, phải = ảnh nắn bằng calib/camera.yaml (hình gửi về thu nhỏ cho nhẹ Wi-Fi)')
        while c.dang_chay():
            jpeg, so = c.xe.anh_jpeg()
            if jpeg is None or so == so_cu:
                c.xe.ngu(0.005)
                if time.monotonic() - t_khung > TRE_MAX and time.monotonic() - t_log >= 1.0:
                    t_log = time.monotonic()
                    log(f'KHÔNG CÓ KHUNG MỚI {time.monotonic() - t_khung:.1f} s: KHÔNG ĐẠT (camera đứng hình?)')
                continue
            so_cu, t_khung = so, time.monotonic()
            if not so_dau:
                so_dau = so - 1
            # Xử lý + gửi hình tối đa ~8 lần/giây cho nhẹ Jetson và Wi-Fi; fps THẬT của camera đo bằng bộ đếm khung.
            if time.monotonic() - t_ve >= 1 / 8:
                t_ve = time.monotonic()
                bgr = giai_nen_mau(jpeg)
                h, w = bgr.shape[:2]
                if maps is None:
                    maps = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_16SC2)
                    log(f'Khung {w}x{h}, calib {int(K[0, 0])}/{int(K[1, 1])} px, tâm ({K[0, 2]:.0f}, {K[1, 2]:.0f})')
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                n += 1
                kb += len(jpeg) / 1024
                sang += float(gray.mean())
                net += float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if c.xem:
                    nan = cv2.remap(bgr, maps[0], maps[1], cv2.INTER_LINEAR)
                    for i in range(1, 4):                       # lưới để kiểm đường thẳng
                        cv2.line(nan, (i * w // 4, 0), (i * w // 4, h), (0, 255, 255), 1)
                        cv2.line(nan, (0, i * h // 4), (w, i * h // 4), (0, 255, 255), 1)
                    cv2.drawMarker(nan, (int(K[0, 2]), int(K[1, 2])), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
                    cv2.putText(bgr, 'GOC', (8, 22), 0, 0.7, (0, 255, 255), 2)
                    cv2.putText(nan, 'NAN (calib/camera.yaml)', (8, 22), 0, 0.7, (0, 255, 255), 2)
                    to = np.hstack([bgr, nan])
                    to = cv2.resize(to, (to.shape[1] // 2, to.shape[0] // 2), interpolation=cv2.INTER_AREA)
                    c.xem.dat_anh('camera', to, chat_luong=60, luon=True)
            dt = time.monotonic() - t_log
            if dt >= 1.0 and n:
                fps = (so - so_dau) / dt
                so_dau = so
                kbk, sg, nt = kb / n, sang / n, net / n
                ket = [('fps', fps >= FPS_MIN), ('sáng', SANG[0] <= sg <= SANG[1]), ('nét', nt >= NET_MIN)]
                log(f'{fps:4.1f} fps (camera) | {kbk:4.0f} kB/khung | sáng {sg:3.0f} | nét {nt:4.0f} | '
                    + ' '.join(f'{t}:{"ĐẠT" if ok else "KHÔNG ĐẠT"}' for t, ok in ket))
                if c.xem:
                    c.xem.cap_nhat(giai_doan='camera_test',
                                   chi_tiet=f'{fps:.1f} fps, sáng {sg:.0f}, nét {nt:.0f}')
                t_log, n, kb, sang, net = time.monotonic(), 0, 0.0, 0.0, 0.0


if __name__ == '__main__':
    main()
