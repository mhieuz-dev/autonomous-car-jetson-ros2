"""Chạy FULL MAP trên xe giả lập, đo độ lệch so với đường đúng ở từng đoạn."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chung import cau_hinh, khung  # noqa: E402
from chung.gia_lap import XeGiaLap  # noqa: E402


def chay_va_do(ds=cau_hinh.THU_TU, xe=None):
    xe = xe or XeGiaLap(can_camera=True, nhanh=True)
    ghi = []
    gui_goc = xe.gui
    het = [False]

    def gui(v, w):
        gui_goc(v, w)
        if het[0]:          # tới cuối đường giả rồi: S7 xoay tìm vạch, xe đi lung tung, không tính vào độ lệch
            return
        i = int(np.argmin(np.hypot(xe.tim[:, 0] - xe.x, xe.tim[:, 1] - xe.y)))
        het[0] = xe.s[i] >= xe.s[-1] - 0.3
        n = xe.phap[i]
        ghi.append((xe.s[i], (xe.x - xe.tim[i, 0]) * n[0] + (xe.y - xe.tim[i, 1]) * n[1]))

    xe.gui = gui
    ma, ctx = khung.chay(list(ds), gia_lap=True, nhanh=True, xem_cong=None, xe=xe)
    return ma, ctx, xe, np.array(ghi)


def lech(ghi, s0, s1, dich):
    m = (ghi[:, 0] >= s0) & (ghi[:, 0] < s1)
    e = ghi[m, 1] - dich
    return float(np.sqrt(np.mean(e ** 2))), float(np.abs(e).max())


def test_full_map_qua_het_giai_doan():
    # sa bàn giả khác đường thật đã ghi (calib/duong_s7.csv) nên luôn kiểm S7 bám vạch ở đây
    ds = cau_hinh.THU_TU[:-1] + ['s7_bam_vach']
    ma, ctx, xe, ghi = chay_va_do(ds)
    ten = [g for g, _, _ in ctx.du_lieu['lich_su']]
    assert ten == ds
    assert xe.va_cham is None, f'va chạm {xe.va_cham}'
    # Hết đường giả thì mất vạch -> S7 đứng xoay phải tìm vạch; không thấy thì hết 360 độ hoặc hết giờ là bình thường
    ly_do = ctx.du_lieu['lich_su'][-1][1]
    assert 'không thấy vạch' in ly_do or 'quá thời gian' in ly_do, ly_do
    rms_truoc, _ = lech(ghi, 0.5, 3.9, -0.20)       # trước hầm: giữ làn phải bằng IMU
    rms_ham, max_ham = lech(ghi, 4.1, 6.4, 0.0)     # trong hầm: giữa 2 tường
    rms_sau, max_sau = lech(ghi, 7.5, 12.5, -0.20)  # bám vạch: tâm làn phải
    print(f'RMS trước hầm {rms_truoc:.3f} | hầm {rms_ham:.3f} (max {max_ham:.3f}) | '
          f'bám vạch {rms_sau:.3f} (max {max_sau:.3f})')
    assert rms_truoc < 0.04
    assert max_ham < 0.25            # hầm rộng 0.77, xe rộng ~0.16: lệch 0.25 vẫn chưa chạm
    assert rms_sau < 0.05


def test_dung_cho_nguoi_di_bo():
    ma, ctx, xe, _ = chay_va_do(['s0_den_xanh', 's1_xuat_phat'])
    assert xe.va_cham is None
    assert ctx.du_lieu['lich_su'][1][1].startswith('tới chân dốc')
