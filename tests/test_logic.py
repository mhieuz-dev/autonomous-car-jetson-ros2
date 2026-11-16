"""Test phần logic thuần (không cần xe, không cần ROS)."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chung.imu import IMU  # noqa: E402
from chung.lidar import Scan, cong_ham, range_in_cone, tunnel_pose  # noqa: E402
from chung.pid import GioiHan, GiuHuong, VongGyro, goc  # noqa: E402
from giai_doan import s1_xuat_phat as S1  # noqa: E402
from giai_doan import s5_ham as S5  # noqa: E402

G = 9.81


def _scan_tu_tuong(tuong, nhieu=0.0, rng=None):
    """360 tia 1 độ từ gốc (0,0) tới các đoạn [(x1,y1,x2,y2)]."""
    a = np.radians(np.arange(-180, 180, 1.0))
    r = np.full(a.shape, np.inf)
    for x1, y1, x2, y2 in tuong:
        for k, ang in enumerate(a):
            dx, dy = math.cos(ang), math.sin(ang)
            ex, ey = x2 - x1, y2 - y1
            den = dx * ey - dy * ex
            if abs(den) < 1e-9:
                continue
            t = (x1 * ey - y1 * ex) / den
            u = (x1 * dy - y1 * dx) / den
            if t > 0 and 0 <= u <= 1:
                r[k] = min(r[k], t)
    if nhieu and rng is not None:
        r = np.where(np.isfinite(r), r + rng.normal(0, nhieu, r.shape), r)
    return Scan(-math.pi, math.radians(1.0), r, 0.05)


# ------------------------------------------------------------------- IMU ---
def test_imu_hieu_chuan_va_dau_pitch():
    imu = IMU()
    imu.bat_dau_gom()
    for i in range(100):                       # đứng yên, gyro lệch 0.01 rad/s
        imu.cap_nhat(i * 0.01, 0.0, 0.0, G, 0.0, 0.0, 0.01)
    so = imu.ket_thuc_gom()
    assert abs(so['trong_luc'] - G) < 1e-6 and abs(so['bias_gz'] - 0.01) < 1e-9
    t = 1.0
    for i in range(300):                       # mũi ngóc lên 6 độ
        q = math.radians(6)
        t += 0.01
        imu.cap_nhat(t, -G * math.sin(q), 0.0, G * math.cos(q), 0.0, 0.0, 0.01)
    assert imu.pitch > math.radians(5)         # DƯƠNG khi leo dốc
    assert abs(imu.yaw) < 1e-6                 # bias đã trừ: không trôi


def test_imu_yaw_trai_la_duong():
    imu = IMU()
    imu.bat_dau_gom()
    for i in range(20):
        imu.cap_nhat(i * 0.01, 0, 0, G, 0, 0, 0)
    imu.ket_thuc_gom()
    for i in range(100):                       # quay trái 0.5 rad/s trong 1 s
        imu.cap_nhat(0.2 + i * 0.01, 0, 0, G, 0, 0, 0.5)
    assert abs(imu.yaw - 0.5) < 0.02


# ------------------------------------------------------------------- PID ---
def test_giu_huong_quay_ve_yaw_0():
    imu = SimpleNamespace(yaw=math.radians(10), gz=0.0)
    w, sai = GiuHuong(1.5).tinh(imu, 0.0, 0.05)
    assert w < 0 and abs(sai + math.radians(10)) < 1e-9     # lệch trái -> quẹo phải


def test_goc_wrap():
    assert abs(goc(math.radians(350)) - math.radians(-10)) < 1e-9


def test_gioi_han_gia_toc():
    gh = GioiHan(gia_toc=0.6)
    v, _ = gh.ap(0.3, 0.0, 0.1)
    assert abs(v - 0.06) < 1e-9


def test_vong_gyro_bu_lech():
    vg = VongGyro(kp=0.5, ki=1.0)
    w_that, troi = 0.0, 0.05                   # bánh lệch làm xe tự quay 0.05 rad/s
    for _ in range(400):
        w_gui = vg.ap(0.15, 0.0, w_that, 0.02)
        w_that = w_gui + troi
    assert abs(w_that) < 0.005


# ------------------------------------------------------------ người đi bộ ---
def test_nguoi_di_bo_trong_vung():
    nguong = S1.nguong_dung()
    gan = nguong - 0.05
    scan = _scan_tu_tuong([(gan, -0.05, gan, 0.05)])
    co, d = S1.co_nguoi(scan)
    assert co and abs(d - gan) < 0.01


def test_vat_ben_le_khong_phai_nguoi():
    scan = _scan_tu_tuong([(0.30, 0.30, 0.35, 0.30)])  # cột biển bên trái, ngoài vùng
    co, _ = S1.co_nguoi(scan)
    assert not co


# -------------------------------------------------------- hầm (logic 04a) ---
def test_cong_ham_va_tunnel_pose_thang():
    # Xe lệch phải 0.15 m trong hành lang thẳng rộng 0.77
    tuong = [(-1.0, 0.385 + 0.15, 1.5, 0.385 + 0.15), (-1.0, -0.385 + 0.15, 1.5, -0.385 + 0.15)]
    scan = _scan_tu_tuong(tuong)
    assert cong_ham(scan, S5.P) is not None
    off, psi, kappa = tunnel_pose(scan, S5.P)
    assert abs(off - 0.15) < 0.01 and abs(psi) < 0.02 and kappa == 0.0
    w = S5.lenh_lai(off, psi, kappa, 0.15)
    assert w > 0                              # tâm hầm bên trái -> quẹo trái


def test_khong_phai_ham_khi_rong_qua():
    tuong = [(-1.0, 0.6, 1.5, 0.6), (-1.0, -0.6, 1.5, -0.6)]   # tổng 1.2 m
    assert cong_ham(_scan_tu_tuong(tuong), S5.P) is None


def test_range_in_cone_bo_diem_le():
    r = np.full(360, np.inf)
    r[180] = 0.15                             # 1 tia lẻ (sượt mặt dốc)
    assert range_in_cone(Scan(-math.pi, math.radians(1), r, 0.05), 0.0, 25.0) is None
    r[178:183] = 0.40                         # vật thật: 5 tia liền
    assert abs(range_in_cone(Scan(-math.pi, math.radians(1), r, 0.05), 0.0, 25.0) - 0.40) < 1e-9


def test_khuc_cua_ham_quay_ve_phia_mo():
    p = S5.P
    tuong = [(0.45, -0.6, 0.45, 0.3),          # tường chắn phía trước
             (-0.5, -0.385, 0.45, -0.385),     # tường phải chạy tới góc
             (-0.5, 0.385, 0.0, 0.385)]        # bên trái mở ra phía trước
    from chung.lidar import CuaHam
    kq = CuaHam().tinh(_scan_tu_tuong(tuong), p, 0.15)
    assert kq is not None and kq[1] > 0 and kq[0] < 0.15


# ------------------------------------------------------------------ đèn ---
def test_s0_nghe_den_to_nhat_va_phai_du_lon():
    from giai_doan.s0_den_xanh import mau_den
    d = lambda ten, cao=60, tin=0.9: {'ten': ten, 'tin': tin, 'hop': [10, 10, 40, 10 + cao]}  # noqa: E731
    assert mau_den([d('den xanh')], cao_min_px=40) == 'xanh'
    assert mau_den([d('den xanh', 60), d('den do', 90)], cao_min_px=40) == 'do'      # 2 đèn: nghe đèn TO hơn
    assert mau_den([d('den xanh', 90), d('den do', 60)], cao_min_px=40) == 'xanh'
    assert mau_den([d('den xanh', 30), d('den do', 25)], cao_min_px=40) is None      # đèn to nhất vẫn nhỏ: chưa tính
    assert mau_den([d('den xanh', 30)], cao_min_px=20) == 'xanh'                     # hạ ngưỡng thì tính
    assert mau_den([d('den do', 80, 0.2)], cao_min_px=40) is None                    # dưới ngưỡng tin
    assert mau_den([d('stop'), d('nguoi di bo')], cao_min_px=40) is None


def test_ghi_doc_nguong_px(tmp_path, monkeypatch):
    from chung import den_onnx as D
    monkeypatch.setattr(D, 'NGUONG_FILE', tmp_path / 'ng.yaml')
    assert D.doc_nguong('den_xanh_s0', 40) == 40
    D.ghi_nguong('den_xanh_s0', 55)
    assert D.doc_nguong('den_xanh_s0', 40) == 55
    try:
        D.ghi_nguong('den_xanh_s0', 1000)
        assert False
    except ValueError:
        pass


def test_s0_do_roi_xanh_moi_chay():
    from giai_doan.s0_den_xanh import GiaiDoan, SO_LAN_XANH
    gd = GiaiDoan()
    for mau in ['do'] * 3 + ['xanh'] * (SO_LAN_XANH - 1) + ['do'] + ['xanh'] * (SO_LAN_XANH - 1):
        gd._cap_nhat(mau)
        assert gd.dem_xanh < SO_LAN_XANH       # đỏ chen giữa thì đếm xanh về 0
    gd._cap_nhat('xanh')
    assert gd.dem_xanh >= SO_LAN_XANH
