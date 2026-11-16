#!/usr/bin/env python3
"""Đo TẢI JETSON: CPU từng nhân, GPU, RAM, nhiệt độ (tegrastats), tiến trình nặng nhất.
CHAY_TREN: host

Chạy NGAY LÚC script khác đang chạy (ví dụ trong lúc xe_tu_hanh) để biết
xe có quá tải không:   bash chay.sh tai_jetson [--giay 15]
Ngưỡng tham khảo: CPU trung bình < 70 %, nhiệt < 70 độ C, RAM còn trống > 300 MB.
"""
import argparse
import re
import shutil
import subprocess
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--giay', type=int, default=15)
    a = ap.parse_args()
    if not shutil.which('tegrastats'):
        raise SystemExit('Không có tegrastats: script này chạy trên Jetson (ngoài docker).')
    print(f'Đo {a.giay} s bằng tegrastats...', flush=True)
    p = subprocess.Popen(['tegrastats', '--interval', '1000'], stdout=subprocess.PIPE, text=True)
    cpu, gpu, ram, ram_max, nhiet = [], [], [], 0, {}
    t0 = time.monotonic()
    try:
        for dong in p.stdout:
            m = re.search(r'RAM (\d+)/(\d+)MB', dong)
            if m:
                ram.append(int(m.group(1)))
                ram_max = int(m.group(2))
            m = re.search(r'CPU \[([^\]]+)\]', dong)
            if m:
                nhan = [int(x.split('%')[0]) for x in m.group(1).split(',') if '%' in x]
                if nhan:
                    cpu.append(nhan)
            m = re.search(r'GR3D_FREQ (\d+)%', dong)
            if m:
                gpu.append(int(m.group(1)))
            for ten, gt in re.findall(r'(\w+)@([\d.]+)C', dong):
                nhiet[ten] = max(nhiet.get(ten, 0.0), float(gt))
            print(dong.strip()[:160], flush=True)
            if time.monotonic() - t0 > a.giay:
                break
    finally:
        p.terminate()
    if not cpu:
        raise SystemExit('Không đọc được số nào từ tegrastats')
    tb = [sum(c[i] for c in cpu) / len(cpu) for i in range(len(cpu[0]))]
    print('================ TẢI JETSON ================')
    print(f'CPU từng nhân (TB): {" ".join(f"{x:.0f}%" for x in tb)} | TB chung {sum(tb) / len(tb):.0f} %'
          f'   {"ỔN" if sum(tb) / len(tb) < 70 else "CAO"}')
    if gpu:
        print(f'GPU: TB {sum(gpu) / len(gpu):.0f} %, max {max(gpu)} %')
    if ram:
        print(f'RAM: {max(ram)}/{ram_max} MB (còn {ram_max - max(ram)} MB)   '
              f'{"ỔN" if ram_max - max(ram) > 300 else "THIẾU"}')
    if nhiet:
        print('Nhiệt cao nhất: ' + ', '.join(f'{k} {v:.0f}C' for k, v in sorted(nhiet.items())))
    print('Tiến trình nặng nhất:')
    out = subprocess.run(['ps', '-eo', 'pcpu,pmem,comm,args', '--sort=-pcpu'], capture_output=True, text=True).stdout
    for dong in out.splitlines()[:7]:
        print('  ' + dong[:150])


if __name__ == '__main__':
    main()
