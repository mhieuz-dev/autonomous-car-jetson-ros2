#!/usr/bin/env python3
"""Đo PITCH trên DỐC: xe chạy thẳng từ chân dốc, lên, giữa dốc, xuống, tới hầm; biết pitch bao nhiêu là LÊN / GIỮA / XUỐNG.

Đặt xe ở CHÂN DỐC, mũi xe thẳng hướng hầm. Xe giữ yaw = 0 bằng IMU, chạy V_RUN m/s,
ghi pitch theo quãng đường (odom), vẽ đồ thị trực tiếp ở bảng điều khiển (hình "pitch": nền màu theo phần
dốc), mô hình xe 3D xoay theo IMU. Dừng khi LiDAR thấy hầm (cổng 2 tường) rồi chạy thêm EXTRA_DIST_M, hoặc quá
MAX_DIST. Cuối cùng in BẢNG pitch từng phần (trung vị, p10-p90, nhỏ nhất - lớn nhất) và GỢI Ý
ngưỡng dốc cho autonomy/params.py; lưu CSV + bảng vào logs/ (bash scripts/car.sh pull để kéo về).
Pitch ghi lại đã trừ phần xe ngóc mũi theo tốc độ (NOSE_UP_DEG * v^2), cùng cách xe tự hành so ngưỡng.

    bash scripts/car.sh imu_slope               xe TỰ CHẠY (cần ~5 m trống phía trước)
    bash scripts/car.sh imu_slope --manual         xe ĐỨNG YÊN, bạn ĐẨY BẰNG TAY qua dốc (odom đo quãng đường), Ctrl+C khi xong
    python3 tools/imu_slope.py --sim
Các phần: PHẲNG (trước dốc) -> LÊN -> GIỮA (đỉnh, pitch gần 0) -> XUỐNG -> HẦM/PHẲNG (sau dốc).
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import settings, tunnel, runtime  # noqa: E402
from core.log import log                    # noqa: E402
from core.pid import HeadingHold            # noqa: E402
from autonomy import params as TS           # noqa: E402

# ===== THAM SỐ (sửa, Ctrl+S, chạy lại) =====================================
V_RUN = 0.20         # m/s khi tự chạy
EXTRA_DIST_M = 0.25  # thấy hầm rồi chạy thêm bấy nhiêu mét thì dừng
MAX_DIST = 5.0       # m
MAX_RUN_TIME = 90.0  # s
KP_HEADING = 1.5
KI_HEADING = 0.3
KD_HEADING = 0.10
W_MAX = 0.8

# Cách chia phần (chỉ để TÊN phần; các ngưỡng thật của xe là gợi ý ở cuối, bạn tự chọn)
SMOOTH_WINDOW = 9    # số mẫu lấy trung vị cho mượt (20 Hz -> 0,45 s)
UP_DEG = 0.5         # pitch trung vị > số này (độ) = đang LÊN
DOWN_DEG = 0.5       # pitch trung vị < -số này = đang XUỐNG
FLAT_DEG = 0.3       # |pitch| < số này = về phẳng / đỉnh
SWITCH_SAMPLES = 6   # phải giữ liên tục bấy nhiêu mẫu mới đổi phần
MAX_PITCH_DEG = 8.0  # |pitch| lớn hơn số này = xe đang bị NHẤC / LẬT (dốc thật đo 10/10 chỉ tới ~5,5): BỎ mẫu, không tính
STILL_TAIL_S = 1.0   # đuôi dữ liệu mà xe KHÔNG lăn thêm (odom đứng) dài hơn số này (giây) = bạn đang cầm xe: cắt bỏ
# ===========================================================================

PART_NAMES = {'FLAT': 'PHẲNG (trước dốc)', 'UP': 'LÊN', 'MIDDLE': 'GIỮA / ĐỈNH', 'DOWN': 'XUỐNG', 'AFTER': 'PHẲNG (sau dốc)'}
PART_COLORS = {'FLAT': (70, 70, 70), 'UP': (40, 110, 40), 'MIDDLE': (40, 90, 130), 'DOWN': (40, 50, 140), 'AFTER': (70, 70, 70)}
ORDER = ['FLAT', 'UP', 'MIDDLE', 'DOWN', 'AFTER']
PART_LABELS = {'FLAT': 'FLAT', 'UP': 'UP', 'MIDDLE': 'GIUA/DINH', 'DOWN': 'DOWN', 'AFTER': 'PHANG (sau)'}   # OpenCV không vẽ dấu


def smooth(p, k=SMOOTH_WINDOW):
    p = np.asarray(p, float)
    if len(p) < 3:
        return p
    h = k // 2
    pad = np.concatenate([np.full(h, p[0]), p, np.full(h, p[-1])])
    return np.array([np.median(pad[i:i + k]) for i in range(len(p))])


def split_phases(pitch, up_deg=UP_DEG, down_deg=DOWN_DEG, flat_deg=FLAT_DEG, n_switch=SWITCH_SAMPLES):
    """Nhãn phần cho từng mẫu (máy trạng thái có trễ): PHANG -> LEN -> GIUA -> XUONG -> SAU.
    Mỗi điều kiện có bộ đếm riêng; phải giữ liên tục n_switch mẫu mới đổi phần."""
    s = smooth(pitch)
    labels, phase = [], 'FLAT'
    counts = {'up': 0, 'flat': 0, 'down': 0}
    for v in s:
        counts['up'] = counts['up'] + 1 if v > up_deg else 0
        counts['flat'] = counts['flat'] + 1 if abs(v) < flat_deg else 0
        counts['down'] = counts['down'] + 1 if v < -down_deg else 0
        new = phase
        if phase == 'FLAT' and counts['up'] >= n_switch:
            new = 'UP'
        elif phase == 'UP':
            new = 'MIDDLE' if counts['flat'] >= n_switch else ('DOWN' if counts['down'] >= n_switch else phase)
        elif phase == 'MIDDLE' and counts['down'] >= n_switch:
            new = 'DOWN'
        elif phase == 'DOWN' and counts['flat'] >= n_switch:
            new = 'AFTER'
        if new != phase:
            phase = new
            counts = {'up': 0, 'flat': 0, 'down': 0}
        labels.append(phase)
    return labels


def trim_still_tail(t, dist, pitch, seconds=STILL_TAIL_S, stop=0.01):
    """Bỏ đuôi mà quãng đường không tăng thêm (> dung mét) trong hơn `seconds` giây: xe đã tới cuối và bị nhấc / cầm."""
    t, dist, pitch = (np.asarray(x) for x in (t, dist, pitch))
    if len(t) < 5:
        return t, dist, pitch
    last = int(np.argmax(dist >= dist.max() - stop))              # lần đầu quãng đường chạm mức cuối
    if t[-1] - t[last] > seconds:
        k = last + 1
        return t[:k], dist[:k], pitch[:k]
    return t, dist, pitch


def summarize(t, dist, pitch, labels, trim_tail=12):
    """Mỗi phần: số mẫu, giây, mét, trung vị, p5-p95, nhỏ nhất, lớn nhất của pitch (độ).
    Nhãn đổi phần chậm khoảng 0,5 s (phải giữ liên tục mới đổi), nên mỗi phần (trừ phần cuối) bỏ trim_tail mẫu
    CUỐI khi tính pitch: đó là đoạn đã sang phần mới mà nhãn chưa kịp đổi."""
    t, dist, pitch, labels = (np.asarray(x) for x in (t, dist, pitch, labels))
    out = {}
    size = [ph for ph in ORDER if (labels == ph).sum() >= 3]
    for ph in size:
        m = labels == ph
        if m.sum() < 3:
            continue
        idx = np.nonzero(m)[0]
        stop = idx[:-trim_tail] if (ph != size[-1] and len(idx) > trim_tail + 5) else idx
        v = pitch[stop]
        out[ph] = {'n': int(m.sum()), 'seconds': float(t[m][-1] - t[m][0]), 'meters': float(dist[m][-1] - dist[m][0]),
                  'tv': float(np.median(v)), 'p5': float(np.percentile(v, 5)), 'p10': float(np.percentile(v, 10)),
                  'p25': float(np.percentile(v, 25)), 'p75': float(np.percentile(v, 75)),
                  'p90': float(np.percentile(v, 90)), 'p95': float(np.percentile(v, 95)),
                  'min': float(v.min()), 'max': float(v.max())}
    return out


def hint(stats):
    """Ngưỡng gợi ý = điểm giữa hai phần liền nhau, dùng p25/p75/p5/p95 (không dùng nhỏ nhất/lớn nhất vì mẫu ở
    chỗ chuyển phần bị lẫn). Trả list (param_name, file, value, reason)."""
    gy = []
    flat = stats.get('FLAT')
    if flat and 'UP' in stats:
        noise = max(abs(flat['p5']), abs(flat['p95']))
        gy.append(('SLOPE_UP_DEG', 'autonomy/params.py', round(max((noise + stats['UP']['p25']) / 2, noise + 0.15), 2),
                   f'giữa nhiễu phẳng (±{noise:.2f}) và LÊN (p25 {stats["UP"]["p25"]:+.2f})'))
    if 'UP' in stats:
        after = stats['MIDDLE']['p90'] if 'MIDDLE' in stats else 0.0
        gy.append(('CREST_DEG', 'autonomy/params.py', round(max((stats['UP']['p25'] + max(after, 0.0)) / 2, 0.15), 2),
                   f'giữa LÊN (p25 {stats["UP"]["p25"]:+.2f}) và đỉnh ({max(after, 0.0):+.2f})'))
    if 'DOWN' in stats:
        front = stats['MIDDLE']['p10'] if 'MIDDLE' in stats else 0.0
        gy.append(('SLOPE_DOWN_DEG', 'autonomy/params.py', round(min((stats['DOWN']['p75'] + min(front, 0.0)) / 2, -0.15), 2),
                   f'giữa đỉnh ({min(front, 0.0):+.2f}) và XUỐNG (p75 {stats["DOWN"]["p75"]:+.2f})'))
    return gy


def text_table(stats):
    lines = [f'{"PHẦN":22s} {"giây":>6s} {"mét":>6s} {"trung vị":>9s} {"p10":>7s} {"p90":>7s} {"nhỏ nhất":>9s} {"lớn nhất":>9s}']
    for ph in ORDER:
        if ph in stats:
            x = stats[ph]
            lines.append(f'{PART_NAMES[ph]:22s} {x["seconds"]:6.1f} {x["meters"]:6.2f} {x["tv"]:+9.2f} {x["p10"]:+7.2f} '
                        f'{x["p90"]:+7.2f} {x["min"]:+9.2f} {x["max"]:+9.2f}')
    return lines


class Program(runtime.ProgramBase):
    NAME = 'imu_slope'
    HZ = 20
    NEED_CAMERA = False
    MANUAL = False

    def __init__(self):
        self.MAX_RUN_TIME = MAX_RUN_TIME
        self.NO_SEND = self.MANUAL        # đẩy tay: khung KHÔNG gửi /cmd_vel (driver khỏi giữ bánh, tay đẩy nhẹ)
        self.heading = HeadingHold(KP_HEADING, KI_HEADING, KD_HEADING, W_MAX)
        self.tunnel = tunnel.TunnelGate()
        self.tunnel_dist = None
        self.h = {'t': [], 'dist': [], 'pitch': [], 'roll': []}
        self.phase = 'FLAT'
        self.reported = False
        self.drop = 0

    def start(self, ctx):
        if self.MANUAL:
            log('imu_slope --manual: xe KHÔNG nhận lệnh nào (bánh quay tự do). Đẩy xe bằng tay từ chân dốc qua đỉnh, xuống dốc, tới hầm; Ctrl+C khi xong.')
        else:
            log(f'imu_slope: xe TỰ CHẠY {V_RUN} m/s giữ yaw 0 từ chân dốc tới hầm (dừng khi thấy hầm + {EXTRA_DIST_M} m).')

    def step(self, ctx, dt):
        h = self.h
        p_raw = math.degrees(ctx.imu.pitch)
        if abs(p_raw) > MAX_PITCH_DEG:
            self.drop += 1
            if self.drop == 1 or self.drop % 40 == 0:
                log(f'CẢNH BÁO: pitch {p_raw:+.1f} độ vượt ±{MAX_PITCH_DEG:.0f}: xe đang bị NHẤC / LẬT, BỎ các mẫu này '
                    '(đặt xe xuống sàn / dốc để đo tiếp)')
            if self.MANUAL:
                return 0.0, 0.0
            return runtime.Finish(f'xe bị nhấc / lật (pitch {p_raw:+.1f} độ)')
        h['t'].append(ctx.elapsed())
        h['dist'].append(ctx.distance())
        h['pitch'].append(p_raw - TS.NOSE_UP_DEG * ctx.car.odom.v ** 2)
        h['roll'].append(math.degrees(ctx.imu.roll))
        labels = split_phases(h['pitch'])[-1]
        if labels != self.phase:
            self.phase = labels
            log(f'>>> PHẦN: {PART_NAMES[labels]}  (pitch {h["pitch"][-1]:+.2f} độ, đã đi {h["dist"][-1]:.2f} m)')
        if self.MANUAL:
            return 0.0, 0.0
        if self.tunnel.update(ctx.car.scan) >= tunnel.ENTER_FRAMES and self.tunnel_dist is None:
            self.tunnel_dist = ctx.distance()
            log(f'>>> LiDAR thấy HẦM tại {self.tunnel_dist:.2f} m')
        if self.tunnel_dist is not None and ctx.distance() - self.tunnel_dist >= EXTRA_DIST_M:
            return runtime.Finish('vào hầm (xong đoạn đo)')
        if ctx.distance() > MAX_DIST:
            return runtime.Finish(f'đi quá {MAX_DIST} m mà chưa thấy hầm')
        w, _ = self.heading.compute(ctx.imu, 0.0, dt)
        return V_RUN, w

    def status(self):
        p = self.h['pitch'][-1] if self.h['pitch'] else 0.0
        return f'{PART_NAMES[self.phase]} | pitch {p:+.2f} độ' + (' | ĐÃ THẤY HẦM' if self.tunnel_dist is not None else '')

    def draw(self, ctx):
        if ctx.viewer is None:
            return
        ctx.viewer.set_image('pitch', draw_plot(self.h), quality=70, always=True)

    def finish(self, ctx, reason):
        if self.reported or len(self.h['t']) < 10:
            return
        self.reported = True
        h = self.h
        t_, dist_, p_ = trim_still_tail(h['t'], h['dist'], h['pitch'])
        if len(t_) < len(h['t']):
            log(f'(cắt {len(h["t"]) - len(t_)} mẫu đuôi: xe không lăn thêm, coi như bạn đang cầm xe)')
        labels = split_phases(p_)
        stats = summarize(t_, dist_, p_, labels)
        lines = ['================ PITCH THEO TỪNG PHẦN DỐC ================'] + text_table(stats)
        lines.append('PITCH THEO QUÃNG ĐƯỜNG (trung vị mỗi 0,25 m):')
        qa, pa = np.asarray(dist_), np.asarray(p_)
        for a in np.arange(0.0, float(qa.max()) if len(qa) else 0.0, 0.25):
            m = (qa >= a) & (qa < a + 0.25)
            if m.sum() > 3:
                lines.append(f'  {a:4.2f}-{a + 0.25:4.2f} m: {np.median(pa[m]):+6.2f} độ   (nhỏ nhất {pa[m].min():+5.2f}, lớn nhất {pa[m].max():+5.2f})')
        gy = hint(stats)
        if gy:
            lines.append('GỢI Ý ngưỡng (bạn tự chọn, sửa trong khối THAM SỐ của file):')
            lines += [f'  {name} = {gt:+.2f}   ({f})  <- {why}' for name, f, gt, why in gy]
        else:
            lines.append('Chưa đủ dữ liệu để gợi ý: chưa thấy phần LÊN rõ ràng. Chạy lại từ chân dốc.')
        if self.drop:
            lines.append(f'(đã BỎ {self.drop} mẫu vì pitch quá ±{MAX_PITCH_DEG:.0f} độ: xe bị nhấc / lật)')
        for d in lines:
            log(d)
        settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
        samples = settings.LOG_DIR / f'imu_doc_{__import__("time").strftime("%Y%m%d_%H%M%S")}'
        with open(str(samples) + '.csv', 'w') as f:
            f.write('t_s,quang_duong_m,pitch_deg,roll_deg,phan\n')
            for i in range(len(t_)):
                f.write(f'{t_[i]:.3f},{dist_[i]:.4f},{p_[i]:.3f},{h["roll"][i]:.3f},{labels[i]}\n')
        Path(str(samples) + '.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        import cv2
        cv2.imwrite(str(samples) + '.png', draw_plot({'t': t_, 'dist': dist_, 'pitch': p_, 'roll': h['roll'][:len(t_)]}, labels))
        log(f'Đã lưu {samples.name}.csv / .txt / .png trong logs/ (bash scripts/car.sh pull để kéo về laptop).')


def draw_plot(h, labels=None, width=640, height=280):
    """Đồ thị pitch theo quãng đường (hoặc theo giờ nếu xe đứng yên), nền màu theo phần dốc."""
    import cv2
    img = np.full((height, width, 3), 24, np.uint8)
    n = len(h['pitch'])
    cv2.putText(img, 'PITCH (do) theo quang duong (m)', (8, 16), 0, 0.5, (255, 255, 255), 1)
    if n < 2:
        return img
    p = np.asarray(h['pitch'])
    dist = np.asarray(h['dist'])
    x = dist if dist[-1] - dist[0] > 0.1 else np.asarray(h['t'])
    if x is not dist:
        cv2.putText(img, '(xe dung yen: truc ngang = giay)', (8, 34), 0, 0.42, (200, 200, 200), 1)
    labels = labels or split_phases(p)
    m = 44
    x0, x1 = float(x[0]), max(float(x[-1]), float(x[0]) + 0.5)
    pm = max(2.0, float(np.abs(p).max()) * 1.15)
    X = lambda v: int(m + (v - x0) / (x1 - x0) * (width - m - 10))        # noqa: E731
    Y = lambda v: int(height - 24 - (v + pm) / (2 * pm) * (height - 52))  # noqa: E731
    i = 0
    while i < n:                                                            # nền theo phần
        j = i
        while j + 1 < n and labels[j + 1] == labels[i]:
            j += 1
        cv2.rectangle(img, (X(x[i]), 24), (X(x[j]) + 1, height - 24), PART_COLORS[labels[i]], -1)
        cv2.putText(img, PART_LABELS[labels[i]], (X(x[i]) + 3, 40), 0, 0.4, (230, 230, 230), 1)
        i = j + 1
    for g in range(-int(pm), int(pm) + 1):
        cv2.line(img, (m, Y(g)), (width - 10, Y(g)), (90, 90, 90) if g else (200, 200, 200), 1)
        cv2.putText(img, f'{g:+d}', (4, Y(g) + 4), 0, 0.4, (200, 200, 200), 1)
    pts = np.array([[X(a), Y(b)] for a, b in zip(x, p)], np.int32)
    cv2.polylines(img, [pts], False, (0, 220, 255), 1)
    cv2.polylines(img, [np.array([[X(a), Y(b)] for a, b in zip(x, smooth(p))], np.int32)], False, (255, 255, 255), 2)
    cv2.putText(img, f'{p[-1]:+.2f} do', (width - 110, height - 6), 0, 0.5, (0, 255, 255), 1)
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--manual', action='store_true', help='xe đứng yên, bạn đẩy tay')
    parser.add_argument('--sim', action='store_true')
    parser.add_argument('--fast', action='store_true', help='giả lập chạy nhanh hết cỡ')
    a = parser.parse_args()
    import tools.imu_slope as modul          # chay() nạp 'tools.imu_slope': đặt cờ ở đúng bản đó, không phải bản __main__
    modul.Program.MANUAL = a.manual
    code, _ = runtime.run('tools.imu_slope', sim=a.sim, record_bag=False, fast=a.fast)
    sys.exit(code)


if __name__ == '__main__':
    main()
