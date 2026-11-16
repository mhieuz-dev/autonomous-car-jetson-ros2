#!/usr/bin/env python3
"""XE TỰ HÀNH: đặt xe ở đâu trên sa bàn cũng được, xe tự nhận ra tình huống và chọn cách lái.

    bash chay.sh xe_tu_hanh                       (trên xe, Ctrl+C = dừng)
    python3 xe_tu_hanh.py --gia-lap --nhanh       (thử trên sa bàn giả, không cần xe)
Tham số: tu_hanh/tham_so.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chung import khung  # noqa: E402
from tu_hanh import tham_so  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--gia-lap', action='store_true', help='chạy trên sa bàn giả (không cần xe)')
    ap.add_argument('--nhanh', action='store_true', help='giả lập không chờ giờ thật')
    ap.add_argument('--khong-ghi', action='store_true', help='không ghi bag')
    ap.add_argument('--toi-da', type=float, default=tham_so.THOI_GIAN_TOI_DA, help='tự dừng sau bấy nhiêu giây')
    a = ap.parse_args()
    tham_so.THOI_GIAN_TOI_DA = a.toi_da
    ma, _ = khung.chay('tu_hanh.lai_xe', gia_lap=a.gia_lap, ghi_bag=not a.khong_ghi, nhanh=a.nhanh)
    sys.exit(ma)
