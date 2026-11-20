#!/usr/bin/env python3
"""XE TỰ HÀNH: đặt xe ở đâu trên sa bàn cũng được, xe tự nhận ra tình huống và chọn cách lái.

    bash scripts/car.sh run_car                       (trên xe, Ctrl+C = dừng)
    python3 run_car.py --sim --fast       (thử trên sa bàn giả, không cần xe)
Tham số: autonomy/params.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import runtime     # noqa: E402
from autonomy import params  # noqa: E402

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--sim', action='store_true', help='chạy trên sa bàn giả (không cần xe)')
    parser.add_argument('--fast', action='store_true', help='giả lập không chờ giờ thật')
    parser.add_argument('--no-record', action='store_true', help='không ghi bag')
    parser.add_argument('--max-time', type=float, default=params.MAX_RUN_TIME, help='tự dừng sau bấy nhiêu giây')
    a = parser.parse_args()
    params.MAX_RUN_TIME = a.max_time
    code, _ = runtime.run('autonomy.driver', sim=a.sim, record_bag=not a.no_record, fast=a.fast)
    sys.exit(code)
