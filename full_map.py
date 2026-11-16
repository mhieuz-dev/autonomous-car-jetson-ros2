#!/usr/bin/env python3
"""FULL MAP: S0 đèn xanh -> S1 xuất phát -> S2 lên dốc -> S3 đỉnh -> S4 xuống ->
S5 hầm -> S6 ra hầm -> S7 bám vạch phải. Tham số nằm trong từng file giai_doan/.

    bash chay.sh full_map                 (trên xe)
    python3 full_map.py --gia-lap         (thử trên xe giả lập)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chung import cau_hinh, khung  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--gia-lap', action='store_true')
    ap.add_argument('--nhanh', action='store_true', help='giả lập không chờ giờ thật')
    ap.add_argument('--khong-ghi', action='store_true', help='không ghi bag')
    ap.add_argument('--tu', default='s1_xuat_phat', choices=cau_hinh.THU_TU,
                    help='bắt đầu từ giai đoạn nào (mặc định S1)')
    a = ap.parse_args()
    ds = cau_hinh.THU_TU[cau_hinh.THU_TU.index(a.tu):]
    ma, _ = khung.chay(ds, gia_lap=a.gia_lap, ghi_bag=not a.khong_ghi, nhanh=a.nhanh)
    sys.exit(ma)
