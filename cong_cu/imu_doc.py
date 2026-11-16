#!/usr/bin/env python3
"""Đo PITCH trên DỐC: xe chạy thẳng từ chân dốc, lên, giữa dốc, xuống, tới hầm; biết pitch bao nhiêu là LÊN / GIỮA / XUỐNG.

Đặt xe ở CHÂN DỐC, mũi xe thẳng hướng hầm. Xe giữ yaw = 0 bằng IMU, chạy V_CHAY m/s,
ghi pitch theo quãng đường (odom), vẽ đồ thị trực tiếp ở bảng điều khiển (hình "pitch": nền màu theo phần
dốc), mô hình xe 3D xoay theo IMU. Dừng khi LiDAR thấy hầm (cổng 2 tường) rồi chạy thêm XA_THEM_M, hoặc quá
QUANG_DUONG_TOI_DA. Cuối cùng in BẢNG pitch từng phần (trung vị, p10-p90, nhỏ nhất - lớn nhất) và GỢI Ý
ngưỡng dốc cho tu_hanh/tham_so.py; lưu CSV + bảng vào logs/ (bash chay.sh keo để kéo về).
Pitch ghi lại đã trừ phần xe ngóc mũi theo tốc độ (NGOC_MUI_DO * v^2), cùng cách xe tự hành so ngưỡng.

    bash chay.sh imu_doc               xe TỰ CHẠY (cần ~5 m trống phía trước)
    bash chay.sh imu_doc --tay         xe ĐỨNG YÊN, bạn ĐẨY BẰNG TAY qua dốc (odom đo quãng đường), Ctrl+C khi xong
    python3 cong_cu/imu_doc.py --gia-lap
Các phần: PHẲNG (trước dốc) -> LÊN -> GIỮA (đỉnh, pitch gần 0) -> XUỐNG -> HẦM/PHẲNG (sau dốc).
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chung import cau_hinh, ham, khung  # noqa: E402
from chung.log import log  # noqa: E402
from chung.pid import GiuHuong  # noqa: E402
from tu_hanh import tham_so as TS  # noqa: E402

# ===== THAM SỐ (sửa, Ctrl+S, chạy lại) =====================================
V_CHAY = 0.20              # m/s khi tự chạy
XA_THEM_M = 0.25           # thấy hầm rồi chạy thêm bấy nhiêu mét thì dừng
QUANG_DUONG_TOI_DA = 5.0   # m
THOI_GIAN_TOI_DA = 90.0    # s
KP_HUONG = 1.5
KI_HUONG = 0.3
KD_HUONG = 0.10
W_MAX = 0.8

# Cách chia phần (chỉ để TÊN phần; các ngưỡng thật của xe là gợi ý ở cuối, bạn tự chọn)
CUA_SO_MUOT = 9            # số mẫu lấy trung vị cho mượt (20 Hz -> 0,45 s)
SANG_LEN_DO = 0.5          # pitch trung vị > số này (độ) = đang LÊN
SANG_XUONG_DO = 0.5        # pitch trung vị < -số này = đang XUỐNG
VE_PHANG_DO = 0.3          # |pitch| < số này = về phẳng / đỉnh
SO_MAU_CHUYEN = 6          # phải giữ liên tục bấy nhiêu mẫu mới đổi phần
PITCH_TOI_DA_DO = 8.0      # |pitch| lớn hơn số này = xe đang bị NHẤC / LẬT (dốc thật đo 10/10 chỉ tới ~5,5): BỎ mẫu, không tính
DUOI_DUNG_YEN_S = 1.0      # đuôi dữ liệu mà xe KHÔNG lăn thêm (odom đứng) dài hơn số này (giây) = bạn đang cầm xe: cắt bỏ
# ===========================================================================

TEN_PHAN = {'PHANG': 'PHẲNG (trước dốc)', 'LEN': 'LÊN', 'GIUA': 'GIỮA / ĐỈNH', 'XUONG': 'XUỐNG', 'SAU': 'PHẲNG (sau dốc)'}
MAU_PHAN = {'PHANG': (70, 70, 70), 'LEN': (40, 110, 40), 'GIUA': (40, 90, 130), 'XUONG': (40, 50, 140), 'SAU': (70, 70, 70)}
THU_TU = ['PHANG', 'LEN', 'GIUA', 'XUONG', 'SAU']
NHAN_HINH = {'PHANG': 'PHANG', 'LEN': 'LEN', 'GIUA': 'GIUA/DINH', 'XUONG': 'XUONG', 'SAU': 'PHANG (sau)'}   # OpenCV không vẽ dấu


def muot(p, k=CUA_SO_MUOT):
    p = np.asarray(p, float)
    if len(p) < 3:
        return p
    h = k // 2
    pad = np.concatenate([np.full(h, p[0]), p, np.full(h, p[-1])])
    return np.array([np.median(pad[i:i + k]) for i in range(len(p))])


def chia_phan(pitch, len_do=SANG_LEN_DO, xuong_do=SANG_XUONG_DO, phang_do=VE_PHANG_DO, n_chuyen=SO_MAU_CHUYEN):
    """Nhãn phần cho từng mẫu (máy trạng thái có trễ): PHANG -> LEN -> GIUA -> XUONG -> SAU.
    Mỗi điều kiện có bộ đếm riêng; phải giữ liên tục n_chuyen mẫu mới đổi phần."""
    s = muot(pitch)
    nhan, pha = [], 'PHANG'
    dem = {'len': 0, 'phang': 0, 'xuong': 0}
    for v in s:
        dem['len'] = dem['len'] + 1 if v > len_do else 0
        dem['phang'] = dem['phang'] + 1 if abs(v) < phang_do else 0
        dem['xuong'] = dem['xuong'] + 1 if v < -xuong_do else 0
        moi = pha
        if pha == 'PHANG' and dem['len'] >= n_chuyen:
            moi = 'LEN'
        elif pha == 'LEN':
            moi = 'GIUA' if dem['phang'] >= n_chuyen else ('XUONG' if dem['xuong'] >= n_chuyen else pha)
        elif pha == 'GIUA' and dem['xuong'] >= n_chuyen:
            moi = 'XUONG'
        elif pha == 'XUONG' and dem['phang'] >= n_chuyen:
            moi = 'SAU'
        if moi != pha:
            pha = moi
            dem = {'len': 0, 'phang': 0, 'xuong': 0}
        nhan.append(pha)
    return nhan


def cat_duoi_dung_yen(t, qd, pitch, giay=DUOI_DUNG_YEN_S, dung=0.01):
    """Bỏ đuôi mà quãng đường không tăng thêm (> dung mét) trong hơn `giay` giây: xe đã tới cuối và bị nhấc / cầm."""
    t, qd, pitch = (np.asarray(x) for x in (t, qd, pitch))
    if len(t) < 5:
        return t, qd, pitch
    cuoi = int(np.argmax(qd >= qd.max() - dung))              # lần đầu quãng đường chạm mức cuối
    if t[-1] - t[cuoi] > giay:
        k = cuoi + 1
        return t[:k], qd[:k], pitch[:k]
    return t, qd, pitch


def thong_ke(t, qd, pitch, nhan, cat_duoi=12):
    """Mỗi phần: số mẫu, giây, mét, trung vị, p5-p95, nhỏ nhất, lớn nhất của pitch (độ).
    Nhãn đổi phần chậm khoảng 0,5 s (phải giữ liên tục mới đổi), nên mỗi phần (trừ phần cuối) bỏ cat_duoi mẫu
    CUỐI khi tính pitch: đó là đoạn đã sang phần mới mà nhãn chưa kịp đổi."""
    t, qd, pitch, nhan = (np.asarray(x) for x in (t, qd, pitch, nhan))
    ra = {}
    co = [ph for ph in THU_TU if (nhan == ph).sum() >= 3]
    for ph in co:
        m = nhan == ph
        if m.sum() < 3:
            continue
        idx = np.nonzero(m)[0]
        dung = idx[:-cat_duoi] if (ph != co[-1] and len(idx) > cat_duoi + 5) else idx
        v = pitch[dung]
        ra[ph] = {'n': int(m.sum()), 'giay': float(t[m][-1] - t[m][0]), 'met': float(qd[m][-1] - qd[m][0]),
                  'tv': float(np.median(v)), 'p5': float(np.percentile(v, 5)), 'p10': float(np.percentile(v, 10)),
                  'p25': float(np.percentile(v, 25)), 'p75': float(np.percentile(v, 75)),
                  'p90': float(np.percentile(v, 90)), 'p95': float(np.percentile(v, 95)),
                  'min': float(v.min()), 'max': float(v.max())}
    return ra


def goi_y(st):
    """Ngưỡng gợi ý = điểm giữa hai phần liền nhau, dùng p25/p75/p5/p95 (không dùng nhỏ nhất/lớn nhất vì mẫu ở
    chỗ chuyển phần bị lẫn). Trả list (ten_tham_so, file, gia_tri, ly_do)."""
    gy = []
    phang = st.get('PHANG')
    if phang and 'LEN' in st:
        nhieu = max(abs(phang['p5']), abs(phang['p95']))
        gy.append(('DOC_LEN_DO', 'tu_hanh/tham_so.py', round(max((nhieu + st['LEN']['p25']) / 2, nhieu + 0.15), 2),
                   f'giữa nhiễu phẳng (±{nhieu:.2f}) và LÊN (p25 {st["LEN"]["p25"]:+.2f})'))
    if 'LEN' in st:
        sau = st['GIUA']['p90'] if 'GIUA' in st else 0.0
        gy.append(('DOC_DINH_DO', 'tu_hanh/tham_so.py', round(max((st['LEN']['p25'] + max(sau, 0.0)) / 2, 0.15), 2),
                   f'giữa LÊN (p25 {st["LEN"]["p25"]:+.2f}) và đỉnh ({max(sau, 0.0):+.2f})'))
    if 'XUONG' in st:
        truoc = st['GIUA']['p10'] if 'GIUA' in st else 0.0
        gy.append(('DOC_XUONG_DO', 'tu_hanh/tham_so.py', round(min((st['XUONG']['p75'] + min(truoc, 0.0)) / 2, -0.15), 2),
                   f'giữa đỉnh ({min(truoc, 0.0):+.2f}) và XUỐNG (p75 {st["XUONG"]["p75"]:+.2f})'))
    return gy


def bang_chu(st):
    dong = [f'{"PHẦN":22s} {"giây":>6s} {"mét":>6s} {"trung vị":>9s} {"p10":>7s} {"p90":>7s} {"nhỏ nhất":>9s} {"lớn nhất":>9s}']
    for ph in THU_TU:
        if ph in st:
            x = st[ph]
            dong.append(f'{TEN_PHAN[ph]:22s} {x["giay"]:6.1f} {x["met"]:6.2f} {x["tv"]:+9.2f} {x["p10"]:+7.2f} '
                        f'{x["p90"]:+7.2f} {x["min"]:+9.2f} {x["max"]:+9.2f}')
    return dong


class ChuongTrinh(khung.ChuongTrinhCoSo):
    TEN = 'imu_doc'
    HZ = 20
    CAN_CAMERA = False
    TAY = False

    def __init__(self):
        self.THOI_GIAN_TOI_DA = THOI_GIAN_TOI_DA
        self.KHONG_GUI = self.TAY        # đẩy tay: khung KHÔNG gửi /cmd_vel (driver khỏi giữ bánh, tay đẩy nhẹ)
        self.huong = GiuHuong(KP_HUONG, KI_HUONG, KD_HUONG, W_MAX)
        self.ham = ham.DemHam()
        self.qd_ham = None
        self.h = {'t': [], 'qd': [], 'pitch': [], 'roll': []}
        self.pha = 'PHANG'
        self.da_bao = False
        self.bo = 0

    def bat_dau(self, ctx):
        if self.TAY:
            log('imu_doc --tay: xe KHÔNG nhận lệnh nào (bánh quay tự do). Đẩy xe bằng tay từ chân dốc qua đỉnh, xuống dốc, tới hầm; Ctrl+C khi xong.')
        else:
            log(f'imu_doc: xe TỰ CHẠY {V_CHAY} m/s giữ yaw 0 từ chân dốc tới hầm (dừng khi thấy hầm + {XA_THEM_M} m).')

    def buoc(self, ctx, dt):
        h = self.h
        p_tho = math.degrees(ctx.imu.pitch)
        if abs(p_tho) > PITCH_TOI_DA_DO:
            self.bo += 1
            if self.bo == 1 or self.bo % 40 == 0:
                log(f'CẢNH BÁO: pitch {p_tho:+.1f} độ vượt ±{PITCH_TOI_DA_DO:.0f}: xe đang bị NHẤC / LẬT, BỎ các mẫu này '
                    '(đặt xe xuống sàn / dốc để đo tiếp)')
            if self.TAY:
                return 0.0, 0.0
            return khung.KetThuc(f'xe bị nhấc / lật (pitch {p_tho:+.1f} độ)')
        h['t'].append(ctx.thoi_gian())
        h['qd'].append(ctx.quang_duong())
        h['pitch'].append(p_tho - TS.NGOC_MUI_DO * ctx.xe.odom.v ** 2)
        h['roll'].append(math.degrees(ctx.imu.roll))
        nhan = chia_phan(h['pitch'])[-1]
        if nhan != self.pha:
            self.pha = nhan
            log(f'>>> PHẦN: {TEN_PHAN[nhan]}  (pitch {h["pitch"][-1]:+.2f} độ, đã đi {h["qd"][-1]:.2f} m)')
        if self.TAY:
            return 0.0, 0.0
        if self.ham.cap_nhat(ctx.xe.scan) >= ham.SO_KHUNG_VAO and self.qd_ham is None:
            self.qd_ham = ctx.quang_duong()
            log(f'>>> LiDAR thấy HẦM tại {self.qd_ham:.2f} m')
        if self.qd_ham is not None and ctx.quang_duong() - self.qd_ham >= XA_THEM_M:
            return khung.KetThuc('vào hầm (xong đoạn đo)')
        if ctx.quang_duong() > QUANG_DUONG_TOI_DA:
            return khung.KetThuc(f'đi quá {QUANG_DUONG_TOI_DA} m mà chưa thấy hầm')
        w, _ = self.huong.tinh(ctx.imu, 0.0, dt)
        return V_CHAY, w

    def trang_thai(self):
        p = self.h['pitch'][-1] if self.h['pitch'] else 0.0
        return f'{TEN_PHAN[self.pha]} | pitch {p:+.2f} độ' + (' | ĐÃ THẤY HẦM' if self.qd_ham is not None else '')

    def ve(self, ctx):
        if ctx.xem is None:
            return
        ctx.xem.dat_anh('pitch', ve_do_thi(self.h), chat_luong=70, luon=True)

    def ket_thuc(self, ctx, ly_do):
        if self.da_bao or len(self.h['t']) < 10:
            return
        self.da_bao = True
        h = self.h
        t_, qd_, p_ = cat_duoi_dung_yen(h['t'], h['qd'], h['pitch'])
        if len(t_) < len(h['t']):
            log(f'(cắt {len(h["t"]) - len(t_)} mẫu đuôi: xe không lăn thêm, coi như bạn đang cầm xe)')
        nhan = chia_phan(p_)
        st = thong_ke(t_, qd_, p_, nhan)
        dong = ['================ PITCH THEO TỪNG PHẦN DỐC ================'] + bang_chu(st)
        dong.append('PITCH THEO QUÃNG ĐƯỜNG (trung vị mỗi 0,25 m):')
        qa, pa = np.asarray(qd_), np.asarray(p_)
        for a in np.arange(0.0, float(qa.max()) if len(qa) else 0.0, 0.25):
            m = (qa >= a) & (qa < a + 0.25)
            if m.sum() > 3:
                dong.append(f'  {a:4.2f}-{a + 0.25:4.2f} m: {np.median(pa[m]):+6.2f} độ   (nhỏ nhất {pa[m].min():+5.2f}, lớn nhất {pa[m].max():+5.2f})')
        gy = goi_y(st)
        if gy:
            dong.append('GỢI Ý ngưỡng (bạn tự chọn, sửa trong khối THAM SỐ của file):')
            dong += [f'  {ten} = {gt:+.2f}   ({f})  <- {ly}' for ten, f, gt, ly in gy]
        else:
            dong.append('Chưa đủ dữ liệu để gợi ý: chưa thấy phần LÊN rõ ràng. Chạy lại từ chân dốc.')
        if self.bo:
            dong.append(f'(đã BỎ {self.bo} mẫu vì pitch quá ±{PITCH_TOI_DA_DO:.0f} độ: xe bị nhấc / lật)')
        for d in dong:
            log(d)
        cau_hinh.THU_MUC_LOG.mkdir(parents=True, exist_ok=True)
        mau = cau_hinh.THU_MUC_LOG / f'imu_doc_{__import__("time").strftime("%Y%m%d_%H%M%S")}'
        with open(str(mau) + '.csv', 'w') as f:
            f.write('t_s,quang_duong_m,pitch_do,roll_do,phan\n')
            for i in range(len(t_)):
                f.write(f'{t_[i]:.3f},{qd_[i]:.4f},{p_[i]:.3f},{h["roll"][i]:.3f},{nhan[i]}\n')
        Path(str(mau) + '.txt').write_text('\n'.join(dong) + '\n', encoding='utf-8')
        import cv2
        cv2.imwrite(str(mau) + '.png', ve_do_thi({'t': t_, 'qd': qd_, 'pitch': p_, 'roll': h['roll'][:len(t_)]}, nhan))
        log(f'Đã lưu {mau.name}.csv / .txt / .png trong logs/ (bash chay.sh keo để kéo về laptop).')


def ve_do_thi(h, nhan=None, rong=640, cao=280):
    """Đồ thị pitch theo quãng đường (hoặc theo giờ nếu xe đứng yên), nền màu theo phần dốc."""
    import cv2
    img = np.full((cao, rong, 3), 24, np.uint8)
    n = len(h['pitch'])
    cv2.putText(img, 'PITCH (do) theo quang duong (m)', (8, 16), 0, 0.5, (255, 255, 255), 1)
    if n < 2:
        return img
    p = np.asarray(h['pitch'])
    qd = np.asarray(h['qd'])
    x = qd if qd[-1] - qd[0] > 0.1 else np.asarray(h['t'])
    if x is not qd:
        cv2.putText(img, '(xe dung yen: truc ngang = giay)', (8, 34), 0, 0.42, (200, 200, 200), 1)
    nhan = nhan or chia_phan(p)
    m = 44
    x0, x1 = float(x[0]), max(float(x[-1]), float(x[0]) + 0.5)
    pm = max(2.0, float(np.abs(p).max()) * 1.15)
    X = lambda v: int(m + (v - x0) / (x1 - x0) * (rong - m - 10))          # noqa: E731
    Y = lambda v: int(cao - 24 - (v + pm) / (2 * pm) * (cao - 52))         # noqa: E731
    i = 0
    while i < n:                                                            # nền theo phần
        j = i
        while j + 1 < n and nhan[j + 1] == nhan[i]:
            j += 1
        cv2.rectangle(img, (X(x[i]), 24), (X(x[j]) + 1, cao - 24), MAU_PHAN[nhan[i]], -1)
        cv2.putText(img, NHAN_HINH[nhan[i]], (X(x[i]) + 3, 40), 0, 0.4, (230, 230, 230), 1)
        i = j + 1
    for g in range(-int(pm), int(pm) + 1):
        cv2.line(img, (m, Y(g)), (rong - 10, Y(g)), (90, 90, 90) if g else (200, 200, 200), 1)
        cv2.putText(img, f'{g:+d}', (4, Y(g) + 4), 0, 0.4, (200, 200, 200), 1)
    pts = np.array([[X(a), Y(b)] for a, b in zip(x, p)], np.int32)
    cv2.polylines(img, [pts], False, (0, 220, 255), 1)
    cv2.polylines(img, [np.array([[X(a), Y(b)] for a, b in zip(x, muot(p))], np.int32)], False, (255, 255, 255), 2)
    cv2.putText(img, f'{p[-1]:+.2f} do', (rong - 110, cao - 6), 0, 0.5, (0, 255, 255), 1)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--tay', action='store_true', help='xe đứng yên, bạn đẩy tay')
    ap.add_argument('--gia-lap', action='store_true')
    ap.add_argument('--nhanh', action='store_true', help='giả lập chạy nhanh hết cỡ')
    a = ap.parse_args()
    import cong_cu.imu_doc as modul          # chay() nạp 'cong_cu.imu_doc': đặt cờ ở đúng bản đó, không phải bản __main__
    modul.ChuongTrinh.TAY = a.tay
    ma, _ = khung.chay('cong_cu.imu_doc', gia_lap=a.gia_lap, ghi_bag=False, nhanh=a.nhanh)
    sys.exit(ma)


if __name__ == '__main__':
    main()
