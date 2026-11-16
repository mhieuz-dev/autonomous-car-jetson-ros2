"""Chọn tham số bird's eye SAU KHI đã tính xong (bước 4b trên trang calib, chạy trên laptop).

birdeye_banco (trên xe) mỗi lần chốt chỉ lưu một ỨNG VIÊN vào calib_moi/ung_vien/:
    lan_NN.yaml  tư thế camera + số đo của lần đó,  lan_NN.png  ảnh xám lúc chốt.
Ở đây bạn tự chọn: những lần nào dùng (nhiều lần -> trung vị), VÙNG NHÌN (x gần/xa, y trái/phải)
và ĐỘ PHÂN GIẢI (mm/pixel), xem thử bird's eye với đúng các số đó, rồi mới ghi
calib_moi/ngoai.yaml (tư thế camera) + calib_moi/bev.yaml (vùng nhìn + độ phân giải).
"""
import glob
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from . import cau_hinh
from .camera import BirdEye, Calib

MM_HANG_ANH_TOI_DA = 15.0        # gợi ý x xa: nơi 1 hàng ảnh còn ứng với <= 15 mm mặt sàn
MAX_PIXEL = 4_000_000


def thu_muc_ung_vien(goc=None):
    return Path(goc or cau_hinh.CALIB_MOI) / 'ung_vien'


def luu_ung_vien(res, rvec, tvec, gray, ket, truoc, lech, sz, o_mm, aspect, co, goc=None, them=None):
    """Chạy trên XE: lưu một lần chốt. Trả đường dẫn file yaml."""
    d = thu_muc_ung_vien(goc)
    d.mkdir(parents=True, exist_ok=True)
    n = len(list(d.glob('lan_*.yaml'))) + 1
    f = d / f'lan_{n:02d}.yaml'
    f.write_text(yaml.safe_dump({
        'n': n, 'gio': time.strftime('%d/%m %H:%M:%S'),
        'cao_m': round(float(res['cao']), 5), 'pitch_do': round(float(res['pitch']), 3),
        'roll_do': round(float(res['roll']), 3), 'yaw_do': round(float(res['yaw']), 3),
        'rms_px': round(float(res['rms']), 3),
        'o_vuong': None if aspect is None else round(float(aspect), 4),
        'co_o': None if co is None else round(float(co), 4),
        'truoc_tam_m': truoc, 'lech_ngang_m': lech, 'ban_co': f'{sz[0]}x{sz[1]}', 'o_mm': o_mm,
        'ket': [[t, g, bool(ok)] for t, g, ok in ket],
        'rvec': [float(v) for v in rvec], 'tvec': [float(v) for v in tvec], **(them or {})},
        sort_keys=False, allow_unicode=True))
    cv2.imwrite(str(d / f'lan_{n:02d}.png'), gray)
    return f


def liet_ke(goc=None):
    ds = []
    for f in sorted(glob.glob(str(thu_muc_ung_vien(goc) / 'lan_*.yaml'))):
        y = yaml.safe_load(Path(f).read_text())
        y['anh'] = f[:-5] + '.png'
        y['dat'] = sum(1 for k in y['ket'] if k[2])
        y['tong'] = len(y['ket'])
        ds.append(y)
    return ds


def _chon(ds_n, goc=None):
    ds = [u for u in liet_ke(goc) if u['n'] in set(ds_n)]
    if not ds:
        raise ValueError('chưa chọn lần nào (hoặc chưa kéo kết quả từ xe: bấm "Kéo kết quả từ xe")')
    return ds


def tu_the_chon(ds_n, goc=None):
    """Trung vị tư thế của các lần được chọn (1 lần = chính nó). -> (rvec, tvec)."""
    ds = _chon(ds_n, goc)
    return (np.median([u['rvec'] for u in ds], axis=0), np.median([u['tvec'] for u in ds], axis=0))


def _camera_kd():
    c = yaml.safe_load((cau_hinh.CALIB / 'camera.yaml').read_text())
    return (np.array(c['camera_matrix']['data'], float).reshape(3, 3),
            np.array(c['distortion_coefficients']['data'], float).ravel())


def de_xuat(ds_n, goc=None):
    """Gợi ý vùng nhìn theo phần sàn camera thật sự thấy rõ + bảng mm/hàng ảnh để bạn tự cân nhắc."""
    K, D = _camera_kd()
    rvec, tvec = tu_the_chon(ds_n, goc)
    c = Calib.tu_pose(K, D, rvec, tvec)
    xs = np.arange(0.02, 3.0, 0.005)
    u, v = c.mat_dat_sang_goc(xs, np.zeros_like(xs))
    thay = (v >= 0) & (v <= c.cao_anh - 1) & (u >= 0) & (u <= c.rong - 1)
    x_min = float(xs[thay][0]) if thay.any() else 0.15
    dv = np.abs(np.gradient(v, xs))                                   # pixel hàng ảnh trên mỗi mét
    mm_hang = np.where(dv > 1e-6, 1000.0 / np.maximum(dv, 1e-6), np.inf)
    ok = thay & (mm_hang <= MM_HANG_ANH_TOI_DA)
    x_xa = float(xs[ok][-1]) if ok.any() else 1.0
    bang = []
    for x in np.arange(0.2, 1.61, 0.1):
        i = int(np.argmin(np.abs(xs - x)))
        ys = np.linspace(-1.2, 1.2, 481)
        uu, vv = c.mat_dat_sang_goc(np.full_like(ys, x), ys)
        vis = (uu >= 0) & (uu <= c.rong - 1) & (vv >= 0) & (vv <= c.cao_anh - 1)
        rong = float(ys[vis][-1] - ys[vis][0]) if vis.any() else 0.0
        bang.append({'x': round(float(x), 2), 'mm_hang': round(float(mm_hang[i]), 1), 'rong': round(rong, 2)})
    return {'x_thay_gan_nhat': round(x_min, 3), 'x_gan': round(float(np.ceil((x_min + 0.01) * 100) / 100), 2),
            'x_xa': round(float(np.floor(min(x_xa, 1.5) * 20) / 20), 2),
            'y_trai': 0.5, 'y_phai': -0.5, 'mm_px': 5.0, 'bang': bang}


def _kiem_tham_so(p):
    xg, xx, yt, yp, mm = (float(p[k]) for k in ('x_gan', 'x_xa', 'y_trai', 'y_phai', 'mm_px'))
    if not (xx > xg):
        raise ValueError('x xa phải lớn hơn x gần')
    if not (yt > yp):
        raise ValueError('y trái phải lớn hơn y phải')
    if not (0.5 <= mm <= 20):
        raise ValueError('độ phân giải phải trong khoảng 0,5 tới 20 mm/pixel')
    if ((xx - xg) / (mm / 1000) + 1) * ((yt - yp) / (mm / 1000) + 1) > MAX_PIXEL:
        raise ValueError('ảnh bird\'s eye quá lớn (> 4 triệu pixel): tăng mm/pixel hoặc thu hẹp vùng nhìn')
    return xg, xx, yt, yp, mm


def ve_thu(ds_n, p, goc=None):
    """Bird's eye của ảnh lần ĐẦU trong ds_n với tham số p; có lưới 10 cm. -> ảnh BGR."""
    xg, xx, yt, yp, mm = _kiem_tham_so(p)
    K, D = _camera_kd()
    rvec, tvec = tu_the_chon(ds_n, goc)
    ung = _chon(ds_n, goc)[0]
    gray = cv2.imread(ung['anh'], cv2.IMREAD_GRAYSCALE)
    c = Calib.tu_pose(K, D, rvec, tvec, (gray.shape[1], gray.shape[0]))
    be = BirdEye(c, xg, xx, yt, yp, mm / 1000.0, ti_le_anh=1.0)
    bev = be.chieu(gray)
    k = max(1, int(round(560.0 / bev.shape[1])))                     # ảnh nhỏ thì phóng to cho dễ nhìn
    img = cv2.cvtColor(cv2.resize(bev, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    for x in np.arange(np.ceil(xg * 10) / 10, xx + 1e-9, 0.1):
        r = int(round(be.m_sang_px(x, 0)[0] * k))
        cv2.line(img, (0, r), (img.shape[1], r), (0, 200, 255), 1)
        cv2.putText(img, f'x {x:.1f}', (4, max(12, r - 3)), 0, 0.42, (0, 255, 255), 1)
    for y in np.arange(np.ceil(yp * 10) / 10, yt + 1e-9, 0.1):
        c_ = int(round(be.m_sang_px(0, y)[1] * k))
        cv2.line(img, (c_, 0), (c_, img.shape[0]), (255, 200, 0) if abs(y) > 1e-9 else (0, 0, 255), 1)
        cv2.putText(img, f'{y:+.1f}', (c_ + 2, img.shape[0] - 24), 0, 0.4, (255, 255, 0), 1)
    cv2.putText(img, f"bird's eye {bev.shape[1]}x{bev.shape[0]} px, {mm:g} mm/px (x tien, y trai; do do = y 0)",
                (6, img.shape[0] - 8), 0, 0.45, (255, 255, 255), 1)
    if img.shape[1] > 900:
        f = 900.0 / img.shape[1]
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return img


def ghi_lua_chon(ds_n, p, goc=None, dich=None):
    """Ghi calib_moi/ngoai.yaml (tư thế = trung vị các lần chọn) + bev.yaml (vùng nhìn, độ phân giải)."""
    xg, xx, yt, yp, mm = _kiem_tham_so(p)
    ds = _chon(ds_n, goc)
    rvec, tvec = tu_the_chon(ds_n, goc)
    K, D = _camera_kd()
    c = Calib.tu_pose(K, D, rvec, tvec)
    dich = Path(dich or cau_hinh.CALIB_MOI)
    dich.mkdir(parents=True, exist_ok=True)
    (dich / 'ngoai.yaml').write_text(
        f'# chon tay {time.strftime("%d/%m %H:%M")} tu cac lan {sorted(u["n"] for u in ds)} (trung vi)\n' + yaml.safe_dump({
            'rvec': [round(float(v), 6) for v in rvec], 'tvec': [round(float(v), 6) for v in tvec],
            'cao_m': round(c.cao_m, 4), 'truoc_tam_m': round(c.truoc_tam, 4), 'lech_ngang_m': round(c.lech_ngang, 4),
            'lan_chon': sorted(u['n'] for u in ds)}, sort_keys=False))
    (dich / 'bev.yaml').write_text(
        f'# vung nhin + do phan giai bird\'s eye, chon tay {time.strftime("%d/%m %H:%M")}\n' + yaml.safe_dump({
            'x_gan': xg, 'x_xa': xx, 'y_trai': yt, 'y_phai': yp, 'm_px': round(mm / 1000.0, 6)}, sort_keys=False))
    return {'cao_m': c.cao_m, 'truoc_tam_m': c.truoc_tam, 'lan': sorted(u['n'] for u in ds)}
