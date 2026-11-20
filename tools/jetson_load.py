#!/usr/bin/env python3
"""Đo TẢI JETSON: CPU từng nhân, GPU, RAM, nhiệt độ (tegrastats), tiến trình nặng nhất.
RUNS_ON: host

Chạy NGAY LÚC script khác đang chạy (ví dụ trong lúc run_car) để biết
xe có quá tải không:   bash scripts/car.sh jetson_load [--seconds 15]
Ngưỡng tham khảo: CPU trung bình < 70 %, nhiệt < 70 độ C, RAM còn trống > 300 MB.
"""
import argparse
import re
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--seconds', type=int, default=15)
    a = parser.parse_args()
    if not shutil.which('tegrastats'):
        raise SystemExit('Không có tegrastats: script này chạy trên Jetson (ngoài docker).')
    print(f'Đo {a.seconds} s bằng tegrastats...', flush=True)
    p = subprocess.Popen(['tegrastats', '--interval', '1000'], stdout=subprocess.PIPE, text=True)
    cpu, gpu, ram, ram_max, temp = [], [], [], 0, {}
    t0 = time.monotonic()
    try:
        for line in p.stdout:
            m = re.search(r'RAM (\d+)/(\d+)MB', line)
            if m:
                ram.append(int(m.group(1)))
                ram_max = int(m.group(2))
            m = re.search(r'CPU \[([^\]]+)\]', line)
            if m:
                labels = [int(x.split('%')[0]) for x in m.group(1).split(',') if '%' in x]
                if labels:
                    cpu.append(labels)
            m = re.search(r'GR3D_FREQ (\d+)%', line)
            if m:
                gpu.append(int(m.group(1)))
            for name, gt in re.findall(r'(\w+)@([\d.]+)C', line):
                temp[name] = max(temp.get(name, 0.0), float(gt))
            print(line.strip()[:160], flush=True)
            if time.monotonic() - t0 > a.seconds:
                break
    finally:
        p.terminate()
    if not cpu:
        raise SystemExit('Không đọc được số nào từ tegrastats')
    mean = [sum(c[i] for c in cpu) / len(cpu) for i in range(len(cpu[0]))]
    print('================ TẢI JETSON ================')
    print(f'CPU từng nhân (TB): {" ".join(f"{x:.0f}%" for x in mean)} | TB chung {sum(mean) / len(mean):.0f} %'
          f'   {"ỔN" if sum(mean) / len(mean) < 70 else "CAO"}')
    if gpu:
        print(f'GPU: TB {sum(gpu) / len(gpu):.0f} %, max {max(gpu)} %')
    if ram:
        print(f'RAM: {max(ram)}/{ram_max} MB (còn {ram_max - max(ram)} MB)   '
              f'{"ỔN" if ram_max - max(ram) > 300 else "THIẾU"}')
    if temp:
        print('Nhiệt cao nhất: ' + ', '.join(f'{k} {v:.0f}C' for k, v in sorted(temp.items())))
    print('Tiến trình nặng nhất:')
    out = subprocess.run(['ps', '-eo', 'pcpu,pmem,comm,args', '--sort=-pcpu'], capture_output=True, text=True).stdout
    for line in out.splitlines()[:7]:
        print('  ' + line[:150])


if __name__ == '__main__':
    main()
