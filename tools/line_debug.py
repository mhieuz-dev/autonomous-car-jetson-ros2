#!/usr/bin/env python3
"""Xem bộ dò vạch làm việc TRỰC TIẾP trên xe (XE KHÔNG CHẠY, đẩy xe bằng tay được).
Dùng đúng tham số DETECTOR_PARAMS trong core/lane_follow.py: sửa ở đó, Ctrl+S, chạy lại.
Hình 8 ô (ảnh gốc, bird's eye, ROI, top-hat, ngưỡng, morph, thành phần, fit):
    http://<ip-xe>:8097/img/vach.mjpg   (hoặc bảng điều khiển)
In mỗi 0,5 s: khoảng cách tới vạch, sai so với D_TARGET, góc, số điểm, ms/khung, Hz.
    bash scripts/car.sh line_debug
"""
import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import lane_follow as BV  # noqa: E402
from core.runtime import Tool       # noqa: E402
from core.log import log            # noqa: E402
from core.draw import grid          # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--sim', action='store_true')
    a = parser.parse_args()
    with Tool('line_debug', need_camera=True, sim=a.sim) as c:
        if a.sim:
            c.car.x, c.car.y, c.car.yaw = 5.0, 1.6, math.pi / 2
        tracker = BV.LaneTracker(a.sim)
        log(f'horizon {tracker.det.birdeye.calib.horizon} | '
            f'bird\'s eye {tracker.det.birdeye.n_cols}x{tracker.det.birdeye.n_rows} px')
        t_log, n_frames, found, ms = time.monotonic(), 0, 0, 0.0
        while c.running():
            result, new = tracker.update(c, debug=True)
            if not new:
                c.car.sleep(0.01)
                continue
            n_frames += 1
            found += result.fresh
            ms += result.ms
            if c.viewer:
                c.viewer.set_image('line', grid(list(result.image.items())), always=True)
            if time.monotonic() - t_log >= 0.5:
                dt = time.monotonic() - t_log
                t_log = time.monotonic()
                d = (f'd={result.dist_ahead:.3f} m (sai {result.dist_ahead - BV.D_TARGET:+.3f}) '
                     f'psi={math.degrees(result.psi):+.1f} độ k={result.kappa:+.2f}' if result.found else f'KHÔNG THẤY ({result.reason})')
                log(f'{d} | điểm {result.n_points} | bắt {found}/{n_frames} khung | '
                    f'{ms / max(n_frames, 1):.1f} ms/khung | {n_frames / dt:.1f} Hz')
                if c.viewer:
                    c.viewer.update(program='line_debug', detail=d)
                n_frames, found, ms = 0, 0, 0.0


if __name__ == '__main__':
    main()
