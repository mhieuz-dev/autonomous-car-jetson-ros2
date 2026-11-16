#!/usr/bin/env python3
"""Kiểm model ONNX (best.onnx, YOLO11n) trên xe: nhận ĐỦ 14 LỚP (XE KHÔNG CHẠY).
In model chạy bằng gì (TensorRT / onnxruntime / cv2.dnn), mấy ms mỗi khung, thấy lớp nào
với độ tin bao nhiêu, và đếm số khung thấy từng lớp. Hình có khung đủ mọi lớp:
http://<ip-xe>:8097/anh/onnx.mjpg (hoặc bảng điều khiển).
14 lớp: bien cao toc, bien co doc, bien do lan xe bus, bien het cao toc, bien vao ham,
den do, den vang, den xanh, di thang, lan xe bus, nguoi di bo, re phai, re trai, stop.
    bash chay.sh onnx_test                         vùng ROI đã lưu (calib/roi_bien.yaml, chỉnh trên bảng điều khiển)
    bash chay.sh onnx_test --tai-xe                chạy model ngay trên Jetson (chậm), mặc định model chạy trên laptop
    bash chay.sh onnx_test --ca-anh                cả ảnh
    bash chay.sh onnx_test --vung 380 60 560 260   chỉ nhận trong vùng này
    python3 cong_cu/onnx_test.py --anh anh.jpg     thử 1 ảnh trên laptop
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chung.camera import giai_nen_mau  # noqa: E402
from collections import Counter

from chung.den_onnx import NhanDen, cat_vung, doc_roi, ve_ket_qua  # noqa: E402
from chung.log import log  # noqa: E402


def chay_laptop():
    """Model chạy trên LAPTOP (den_laptop.py, bảng điều khiển tự bật): laptop nhận ảnh camera nén qua DDS và gửi kết
    quả về topic /champion/den. Công cụ này chỉ ghi log kết quả xe nhận được; hình xem trên bảng điều khiển."""
    from chung.khung import CongCu
    with CongCu('onnx_test', can_camera=False, hieu_chuan=False) as c:
        log('Model chạy trên LAPTOP: bảng điều khiển phải Bật "Nhận biển bằng laptop". Dưới đây là kết quả XE nhận được qua DDS.')
        t_log, t_cu = 0.0, None
        while c.dang_chay():
            n = getattr(c.xe, 'den_ngoai', None)
            if time.monotonic() - t_log > 1.0:
                t_log = time.monotonic()
                if n is None or time.monotonic() - n['t'] > 2.0:
                    log('Chưa nhận kết quả từ laptop (bật "Nhận biển bằng laptop" trên bảng điều khiển; kiểm ROS_DOMAIN_ID=2)')
                else:
                    log(f'laptop {n["ms"]:4.0f} ms | ' + (', '.join(f'{d["ten"]} {d["tin"]:.2f}' for d in n['kq']) or 'không thấy gì'))
            time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--vung', type=int, nargs=4, metavar=('X1', 'Y1', 'X2', 'Y2'),
                    help='chỉ nhận trong vùng này (mặc định cả ảnh)')
    ap.add_argument('--tai-xe', action='store_true', help='chạy model ngay trên Jetson (chậm ~1 s/khung); mặc định model chạy trên LAPTOP (den_laptop.py)')
    ap.add_argument('--ca-anh', action='store_true', help='nhận trên cả ảnh, bỏ qua ROI đã lưu')
    ap.add_argument('--anh', help='thử trên 1 file ảnh (không cần xe)')
    ap.add_argument('--chi', nargs='*', default=None, help='backend thử, ví dụ: --chi cv2')
    a = ap.parse_args()
    vung = a.vung or (None if a.ca_anh else doc_roi())
    if not a.anh and not a.tai_xe:
        return chay_laptop()
    bo = NhanDen(tuple(a.chi) if a.chi else ('tensorrt', 'onnxruntime', 'cv2'))
    log(f'Model chạy bằng: {bo.ten}')
    for l in bo.loi:
        log(f'  (không dùng được {l})')
    if not bo.san_sang:
        log('=> Không backend nào chạy được trên máy này: cài onnxruntime hoặc build best.engine (trtexec).')
        return
    if a.anh:
        bgr = cv2.imread(a.anh)
        cat, goc = cat_vung(bgr, vung)
        kq, ms = bo.nhan(cat)
        log(f'{ms:.0f} ms | ' + (', '.join(f'{d["ten"]} {d["tin"]:.2f}' for d in kq) or 'không thấy gì'))
        from chung import cau_hinh
        cau_hinh.THU_MUC_LOG.mkdir(parents=True, exist_ok=True)
        out = cau_hinh.THU_MUC_LOG / (Path(a.anh).stem + '_onnx.jpg')
        cv2.imwrite(str(out), ve_ket_qua(bgr, kq, goc, vung))
        log(f'Hình có khung: {out}')
        return
    from chung.khung import CongCu
    import threading
    with CongCu('onnx_test', can_camera=True, hieu_chuan=False) as c:
        log(f'Vùng: {vung or "cả ảnh"}')
        # Model chậm (~1 s/khung trên Jetson CPU) nên chạy ở luồng riêng; vòng chính chỉ giải nén + phát hình
        # theo nhịp camera, vẽ khung của kết quả gần nhất => hình mượt dù model chậm.
        chung = {'bgr': None, 'kq': [], 'ms': 0.0, 'n': 0, 'dem': Counter(), 'moi': 0}
        khoa = threading.Lock()

        def luong_model():
            so = 0
            while c.dang_chay():
                with khoa:
                    bgr, moi = chung['bgr'], chung['moi']
                if bgr is None or moi == so:
                    time.sleep(0.02)
                    continue
                so = moi
                cat, goc = cat_vung(bgr, vung)
                kq, ms = bo.nhan(cat)
                for d in kq:
                    d['hop'][0] += goc[0]
                    d['hop'][2] += goc[0]
                    d['hop'][1] += goc[1]
                    d['hop'][3] += goc[1]
                with khoa:
                    chung['kq'], chung['ms'] = kq, ms
                    chung['n'] += 1
                    chung['dem'].update({d['ten'] for d in kq})

        threading.Thread(target=luong_model, daemon=True).start()
        so_cu, t_log, n_log = 0, 0.0, 0
        while c.dang_chay():
            jpeg, so = c.xe.anh_jpeg()
            if jpeg is None or so == so_cu:
                time.sleep(0.02)
                continue
            so_cu = so
            bgr = giai_nen_mau(jpeg)
            with khoa:
                chung['bgr'], chung['moi'] = bgr, so
                kq, ms, n = chung['kq'], chung['ms'], chung['n']
            if c.xem:
                c.xem.dat_jpeg('cam', jpeg)   # ảnh thô cho ô ROI trên bảng điều khiển
                c.xem.dat_anh('onnx', ve_ket_qua(bgr, kq, (0, 0), vung, chi_vung=True), luon=True)
                c.xem.cap_nhat(chuong_trinh='onnx_test', chi_tiet=f'{bo.ten} {ms:.0f} ms')
            if n != n_log and time.monotonic() - t_log > 1.0:
                t_log, n_log = time.monotonic(), n
                log(f'{ms:4.0f} ms | ' + (', '.join(f'{d["ten"]} {d["tin"]:.2f}' for d in kq) or 'không thấy gì'))
                if n % 20 < 3:
                    with khoa:
                        dem = dict(chung['dem'])
                    log(f'  đếm {n} khung: ' + (', '.join(f'{k} {v}' for k, v in sorted(dem.items(), key=lambda x: -x[1])) or 'chưa thấy lớp nào'))


if __name__ == '__main__':
    main()
