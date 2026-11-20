#!/usr/bin/env python3
"""XE TỰ HÀNH: đặt xe ở đâu trên sa bàn cũng được, xe tự nhận ra tình huống và chọn cách lái.

    bash scripts/car.sh run_car                       (trên xe, Ctrl+C = dừng)
    python3 run_car.py --gia-lap --nhanh       (thử trên sa bàn giả, không cần xe)
Tham số: autonomy/params.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import runtime  # noqa: E402
from autonomy import params  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--gia-lap', action='store_true', help='chạy trên sa bàn giả (không cần xe)')
    ap.add_argument('--nhanh', action='store_true', help='giả lập không chờ giờ thật')
    ap.add_argument('--khong-ghi', action='store_true', help='không ghi bag')
    ap.add_argument('--toi-da', type=float, default=params.THOI_GIAN_TOI_DA, help='tự dừng sau bấy nhiêu giây')
    a = ap.parse_args()
    params.THOI_GIAN_TOI_DA = a.toi_da
    ma, _ = runtime.chay('autonomy.driver', gia_lap=a.gia_lap, ghi_bag=not a.khong_ghi, nhanh=a.nhanh)
    sys.exit(ma)
