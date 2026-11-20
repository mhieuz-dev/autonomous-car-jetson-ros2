#!/usr/bin/env python3
"""Kiểm model ONNX (best.onnx, YOLO11n) trên xe: nhận ĐỦ 14 LỚP (XE KHÔNG CHẠY).
In model chạy bằng gì (TensorRT / onnxruntime / cv2.dnn), mấy ms mỗi khung, thấy lớp nào
với độ tin bao nhiêu, và đếm số khung thấy từng lớp. Hình có khung đủ mọi lớp:
http://<ip-xe>:8097/img/onnx.mjpg (hoặc bảng điều khiển).
14 lớp: bien cao toc, bien co doc, bien do lan xe bus, bien het cao toc, bien vao ham,
den do, den vang, den xanh, di thang, lan xe bus, nguoi di bo, re phai, re trai, stop.
    bash scripts/car.sh onnx_test                         vùng ROI đã lưu (config/sign_roi.yaml, chỉnh trên bảng điều khiển)
    bash scripts/car.sh onnx_test --on-car                chạy model ngay trên Jetson (chậm), mặc định model chạy trên laptop
    bash scripts/car.sh onnx_test --full-image                cả ảnh
    bash scripts/car.sh onnx_test --zone 380 60 560 260   chỉ nhận trong vùng này
    python3 tools/onnx_test.py --image anh.jpg     thử 1 ảnh trên laptop
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.camera import decode_color  # noqa: E402
from collections import Counter

from core.detector import Detector, crop_zone, read_roi, draw_detections  # noqa: E402
from core.log import log                                                  # noqa: E402


def run_on_laptop():
    """Model chạy trên LAPTOP (detector_laptop.py, bảng điều khiển tự bật): laptop nhận ảnh camera nén qua DDS và gửi kết
    quả về topic /champion/den. Công cụ này chỉ ghi log kết quả xe nhận được; hình xem trên bảng điều khiển."""
    from core.runtime import Tool
    with Tool('onnx_test', need_camera=False, calibrate=False) as c:
        log('Model chạy trên LAPTOP: bảng điều khiển phải Bật "Nhận biển bằng laptop". Dưới đây là kết quả XE nhận được qua DDS.')
        t_log, t_old = 0.0, None
        while c.running():
            n = getattr(c.car, 'remote_detections', None)
            if time.monotonic() - t_log > 1.0:
                t_log = time.monotonic()
                if n is None or time.monotonic() - n['t'] > 2.0:
                    log('Chưa nhận kết quả từ laptop (bật "Nhận biển bằng laptop" trên bảng điều khiển; kiểm ROS_DOMAIN_ID=2)')
                else:
                    log(f'laptop {n["ms"]:4.0f} ms | ' + (', '.join(f'{d["name"]} {d["conf"]:.2f}' for d in n['result']) or 'không thấy gì'))
            time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--zone', type=int, nargs=4, metavar=('X1', 'Y1', 'X2', 'Y2'),
                    help='chỉ nhận trong vùng này (mặc định cả ảnh)')
    parser.add_argument('--on-car', action='store_true', help='chạy model ngay trên Jetson (chậm ~1 s/khung); mặc định model chạy trên LAPTOP (detector_laptop.py)')
    parser.add_argument('--full-image', action='store_true', help='nhận trên cả ảnh, bỏ qua ROI đã lưu')
    parser.add_argument('--image', help='thử trên 1 file ảnh (không cần xe)')
    parser.add_argument('--only', nargs='*', default=None, help='backend thử, ví dụ: --only cv2')
    a = parser.parse_args()
    zone = a.zone or (None if a.full_image else read_roi())
    if not a.image and not a.on_car:
        return run_on_laptop()
    backend = Detector(tuple(a.only) if a.only else ('tensorrt', 'onnxruntime', 'cv2'))
    log(f'Model chạy bằng: {backend.name}')
    for l in backend.error:
        log(f'  (không dùng được {l})')
    if not backend.ready:
        log('=> Không backend nào chạy được trên máy này: cài onnxruntime hoặc build best.engine (trtexec).')
        return
    if a.image:
        bgr = cv2.imread(a.image)
        crop, origin = crop_zone(bgr, zone)
        result, ms = backend.detect(crop)
        log(f'{ms:.0f} ms | ' + (', '.join(f'{d["name"]} {d["conf"]:.2f}' for d in result) or 'không thấy gì'))
        from core import settings
        settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
        out = settings.LOG_DIR / (Path(a.image).stem + '_onnx.jpg')
        cv2.imwrite(str(out), draw_detections(bgr, result, origin, zone))
        log(f'Hình có khung: {out}')
        return
    from core.runtime import Tool
    import threading
    with Tool('onnx_test', need_camera=True, calibrate=False) as c:
        log(f'Vùng: {zone or "cả ảnh"}')
        # Model chậm (~1 s/khung trên Jetson CPU) nên chạy ở luồng riêng; vòng chính chỉ giải nén + phát hình
        # theo nhịp camera, vẽ khung của kết quả gần nhất => hình mượt dù model chậm.
        chung = {'bgr': None, 'result': [], 'ms': 0.0, 'n': 0, 'counts': Counter(), 'new': 0}
        mutex = threading.Lock()

        def model_thread():
            count = 0
            while c.running():
                with mutex:
                    bgr, new = chung['bgr'], chung['new']
                if bgr is None or new == count:
                    time.sleep(0.02)
                    continue
                count = new
                crop, origin = crop_zone(bgr, zone)
                result, ms = backend.detect(crop)
                for d in result:
                    d['box'][0] += origin[0]
                    d['box'][2] += origin[0]
                    d['box'][1] += origin[1]
                    d['box'][3] += origin[1]
                with mutex:
                    chung['result'], chung['ms'] = result, ms
                    chung['n'] += 1
                    chung['counts'].update({d['name'] for d in result})

        threading.Thread(target=model_thread, daemon=True).start()
        old_count, t_log, n_log = 0, 0.0, 0
        while c.running():
            jpeg, count = c.car.camera_jpeg()
            if jpeg is None or count == old_count:
                time.sleep(0.02)
                continue
            old_count = count
            bgr = decode_color(jpeg)
            with mutex:
                chung['bgr'], chung['new'] = bgr, count
                result, ms, n = chung['result'], chung['ms'], chung['n']
            if c.viewer:
                c.viewer.set_jpeg('cam', jpeg)   # ảnh thô cho ô ROI trên bảng điều khiển
                c.viewer.set_image('onnx', draw_detections(bgr, result, (0, 0), zone, zone_only=True), always=True)
                c.viewer.update(program='onnx_test', detail=f'{backend.name} {ms:.0f} ms')
            if n != n_log and time.monotonic() - t_log > 1.0:
                t_log, n_log = time.monotonic(), n
                log(f'{ms:4.0f} ms | ' + (', '.join(f'{d["name"]} {d["conf"]:.2f}' for d in result) or 'không thấy gì'))
                if n % 20 < 3:
                    with mutex:
                        counts = dict(chung['counts'])
                    log(f'  đếm {n} khung: ' + (', '.join(f'{k} {v}' for k, v in sorted(counts.items(), key=lambda x: -x[1])) or 'chưa thấy lớp nào'))


if __name__ == '__main__':
    main()
