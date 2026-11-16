#!/usr/bin/env python3
"""Xem bộ dò vạch làm việc TRỰC TIẾP trên xe (XE KHÔNG CHẠY, đẩy xe bằng tay được).
Dùng đúng tham số DO_VACH trong chung/bam_vach.py: sửa ở đó, Ctrl+S, chạy lại.
Hình 8 ô (ảnh gốc, bird's eye, ROI, top-hat, ngưỡng, morph, thành phần, fit):
    http://<ip-xe>:8097/anh/vach.mjpg   (hoặc bảng điều khiển)
In mỗi 0,5 s: khoảng cách tới vạch, sai so với D_DICH, góc, số điểm, ms/khung, Hz.
    bash chay.sh line_debug
"""
import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chung import bam_vach as BV  # noqa: E402
from chung.khung import CongCu  # noqa: E402
from chung.log import log  # noqa: E402
from chung.ve import luoi  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--gia-lap', action='store_true')
    a = ap.parse_args()
    with CongCu('line_debug', can_camera=True, gia_lap=a.gia_lap) as c:
        if a.gia_lap:
            c.xe.x, c.xe.y, c.xe.yaw = 5.0, 1.6, math.pi / 2
        bam = BV.BamVach(a.gia_lap)
        log(f'horizon {bam.do.be.calib.horizon} | '
            f'bird\'s eye {bam.do.be.so_cot}x{bam.do.be.so_hang} px')
        t_log, so_khung, thay, ms = time.monotonic(), 0, 0, 0.0
        while c.dang_chay():
            kq, moi = bam.cap_nhat(c, debug=True)
            if not moi:
                c.xe.ngu(0.01)
                continue
            so_khung += 1
            thay += kq.do_moi
            ms += kq.ms
            if c.xem:
                c.xem.dat_anh('vach', luoi(list(kq.anh.items())), luon=True)
            if time.monotonic() - t_log >= 0.5:
                dt = time.monotonic() - t_log
                t_log = time.monotonic()
                d = (f'd={kq.khoang_cach_truoc:.3f} m (sai {kq.khoang_cach_truoc - BV.D_DICH:+.3f}) '
                     f'psi={math.degrees(kq.psi):+.1f} độ k={kq.kappa:+.2f}' if kq.thay else f'KHÔNG THẤY ({kq.ly_do})')
                log(f'{d} | điểm {kq.so_diem} | bắt {thay}/{so_khung} khung | '
                    f'{ms / max(so_khung, 1):.1f} ms/khung | {so_khung / dt:.1f} Hz')
                if c.xem:
                    c.xem.cap_nhat(chuong_trinh='line_debug', chi_tiet=d)
                so_khung, thay, ms = 0, 0, 0.0


if __name__ == '__main__':
    main()
