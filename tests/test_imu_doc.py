"""imu_doc: chia phần dốc + ngưỡng gợi ý + chạy thử trên xe giả lập (dốc nhẹ ~1,2 độ như dốc thật)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chung import khung  # noqa: E402
from chung.gia_lap import XeGiaLap  # noqa: E402
from cong_cu import imu_doc as D  # noqa: E402


def _profile():
    rng = np.random.default_rng(0)
    seg = [(0.0, 40), (1.2, 70), (0.05, 25), (-1.2, 70), (0.0, 40)]          # phẳng, lên, giữa, xuống, phẳng (20 Hz)
    p = np.concatenate([v + rng.normal(0, 0.15, n) for v, n in seg])
    t = np.arange(len(p)) / 20.0
    return t, t * 0.2, p


def test_chia_phan_dung_thu_tu_va_so_do():
    t, qd, p = _profile()
    nhan = D.chia_phan(p)
    thu_tu = [n for i, n in enumerate(nhan) if i == 0 or n != nhan[i - 1]]
    assert thu_tu == ['PHANG', 'LEN', 'GIUA', 'XUONG', 'SAU']
    st = D.thong_ke(t, qd, p, nhan)
    assert abs(st['LEN']['tv'] - 1.2) < 0.15 and abs(st['XUONG']['tv'] + 1.2) < 0.15
    assert abs(st['GIUA']['tv']) < 0.3 and abs(st['PHANG']['tv']) < 0.15


def test_goi_y_nam_giua_cac_phan():
    t, qd, p = _profile()
    gy = {ten: gt for ten, _, gt, _ in D.goi_y(D.thong_ke(t, qd, p, D.chia_phan(p)))}
    assert 0.3 < gy['NGUONG_LEN_DOC_DO'] < 1.0         # trên nhiễu phẳng, dưới dốc lên (~1,2)
    assert 0.1 < gy['NGUONG_DINH_DO'] < 0.9
    assert -1.0 < gy['NGUONG_XUONG_DO'] < -0.1


def test_xe_gia_lap_tu_chay_qua_doc_toi_ham():
    xe = XeGiaLap(can_camera=False, nhanh=True, co_nguoi=False, doc_len=1.3, doc_xuong=-1.2)
    ma, ctx = khung.chay(['cong_cu.imu_doc'], gia_lap=True, ghi_bag=False, nhanh=True, xem_cong=None, xe=xe)
    assert ctx.du_lieu['lich_su'][-1][1].startswith('vào hầm'), ctx.du_lieu['lich_su']
    assert xe.va_cham is None
