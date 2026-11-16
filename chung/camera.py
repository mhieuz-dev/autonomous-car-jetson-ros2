"""Camera: đọc calib, giải nén ảnh, chiếu bird's eye bằng MỘT bảng remap tính sẵn.

Mô hình: mặt đường phẳng (z = 0), hệ toạ độ xe: x phía trước, y bên trái, gốc trên mặt
đường ngay dưới TÂM xe. Camera có tư thế đầy đủ (rvec, tvec: hệ xe -> hệ camera) lấy từ
calib bàn cờ nằm trên sàn (calib/birdeye_banco.py), nên đúng cả khi camera chúi xuống, nghiêng
ngang hay lệch hướng. Không có số mặc định nào: thiếu calib thì báo ChuaCalib.
Bảng remap đi thẳng từ ảnh GỐC (còn méo) sang bird's eye, mỗi khung chỉ tốn 1 lần cv2.remap.
"""
from pathlib import Path

import cv2
import numpy as np
import yaml

from . import cau_hinh


class ChuaCalib(RuntimeError):
    pass


def tu_pitch(pitch_rad, cao_m, truoc_m, lech_m=0.0):
    """(rvec, tvec) cho camera chúi xuống pitch_rad, cao cao_m, đặt trước tâm xe truoc_m (chỉ để dựng xe GIẢ)."""
    sp, cp = np.sin(pitch_rad), np.cos(pitch_rad)
    R = np.array([[0, -1, 0], [-sp, 0, -cp], [cp, 0, -sp]], float)
    C = np.array([truoc_m, lech_m, cao_m], float)
    return cv2.Rodrigues(R)[0].ravel(), (-R @ C)


class Calib:
    """Calib camera. Không có file calib thì báo ChuaCalib (KHÔNG dùng số cũ)."""

    def __init__(self, thu_muc=None):
        thu_muc = Path(thu_muc or cau_hinh.CALIB)
        f_cam, f_ngoai = thu_muc / 'camera.yaml', thu_muc / 'ngoai.yaml'
        if not f_cam.exists():
            raise ChuaCalib(f'CHƯA CALIB CAMERA (không có {f_cam}). Calib nội: đặt calib/camera.yaml')
        with open(f_cam) as f:
            c = yaml.safe_load(f)
        n = {}
        if f_ngoai.exists():
            with open(f_ngoai) as f:
                n = yaml.safe_load(f) or {}
        if n.get('rvec') is None or n.get('tvec') is None:
            raise ChuaCalib("CHƯA CÓ BIRD'S EYE (không có rvec/tvec trong "
                            f"{f_ngoai}). Chạy birdeye_banco rồi chọn ứng viên (bảng điều khiển)")
        self._nap(c, n['rvec'], n['tvec'])
        self.thu_muc = thu_muc

    def _nap(self, c, rvec, tvec):
        self.K = np.array(c['camera_matrix']['data'], float).reshape(3, 3)
        self.D = np.array(c['distortion_coefficients']['data'], float).ravel()
        self.rong, self.cao_anh = int(c['image_width']), int(c['image_height'])
        self.rvec = np.asarray(rvec, float).ravel()
        self.tvec = np.asarray(tvec, float).ravel()
        self.R = cv2.Rodrigues(self.rvec)[0]
        C = -self.R.T @ self.tvec                     # vị trí camera trong hệ xe
        self.truoc_tam, self.lech_ngang, self.cao_m = float(C[0]), float(C[1]), float(C[2])
        d = self.R[:, 0]                              # hướng "thẳng tiến" trong hệ camera
        self.horizon = float(self.K[1, 1] * d[1] / d[2] + self.K[1, 2]) if d[2] > 1e-6 else float('nan')

    @classmethod
    def tu_pose(cls, K, D, rvec, tvec, size=(640, 480)):
        """Calib dựng trong bộ nhớ (công cụ birdeye_banco, test)."""
        o = cls.__new__(cls)
        o._nap({'camera_matrix': {'data': list(np.asarray(K, float).ravel())},
                'distortion_coefficients': {'data': list(np.asarray(D, float).ravel())},
                'image_width': size[0], 'image_height': size[1]}, rvec, tvec)
        o.thu_muc = None
        return o

    @classmethod
    def gia_lap(cls):
        """Camera LÝ TƯỞNG cho xe giả lập (không méo). Không liên quan xe thật."""
        rvec, tvec = tu_pitch(np.arctan((240.0 - 244.0) / 420.0), 0.14, 0.069)
        return cls.tu_pose(np.array([[420, 0, 320], [0, 420, 240], [0, 0, 1.0]]), np.zeros(5), rvec, tvec)

    def mat_dat_sang_goc(self, x, y):
        """Điểm mặt đường (x trước, y trái, m) -> pixel trên ảnh GỐC còn méo. Phía sau camera: -1e4."""
        x = np.asarray(x, float).ravel()
        y = np.asarray(y, float).ravel()
        P = np.column_stack([x, y, np.zeros_like(x)])
        z = (P @ self.R.T + self.tvec)[:, 2]
        ok = z > 0.02
        px = np.full((len(x), 2), -1e4)
        if ok.any():
            px[ok] = cv2.projectPoints(P[ok].reshape(-1, 1, 3), self.rvec, self.tvec, self.K, self.D)[0][:, 0]
        return px[:, 0], px[:, 1]

    def raw_sang_mat_dat(self, u, v):
        """Pixel trên ảnh GỐC -> mặt đường (x, y) trong hệ xe. Tia không chạm đất (trên chân trời): nan."""
        pts = np.column_stack([np.asarray(u, float).ravel(), np.asarray(v, float).ravel()])
        n = cv2.undistortPoints(pts.reshape(-1, 1, 2), self.K, self.D)[:, 0]
        ray = np.column_stack([n, np.ones(len(n))]) @ self.R              # R^T r
        C = -self.R.T @ self.tvec
        with np.errstate(divide='ignore', invalid='ignore'):
            s = np.where(ray[:, 2] < -1e-6, -C[2] / ray[:, 2], np.nan)
        s = np.where(s > 0, s, np.nan)
        return C[0] + s * ray[:, 0], C[1] + s * ray[:, 1]


def giai_nen_xam_nua(jpeg):
    """JPEG -> ảnh xám NỬA độ phân giải. Giải nén thẳng ở 1/2 nên nhanh hơn nhiều."""
    buf = np.frombuffer(jpeg, np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_REDUCED_GRAYSCALE_2)


def giai_nen_mau(jpeg):
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


class BirdEye:
    """Ảnh nhìn từ trên xuống: hàng 0 = xa nhất (X_XA), cột 0 = bên trái nhất (Y_TRAI)."""

    def __init__(self, calib, x_gan=0.15, x_xa=1.0, y_trai=0.50, y_phai=-0.50, m_px=0.005, ti_le_anh=0.5):
        self.calib = calib
        self.x_gan, self.x_xa = x_gan, x_xa
        self.y_trai, self.y_phai = y_trai, y_phai
        self.m = m_px
        self.so_hang = int(round((x_xa - x_gan) / m_px)) + 1
        self.so_cot = int(round((y_trai - y_phai) / m_px)) + 1
        hang, cot = np.mgrid[0:self.so_hang, 0:self.so_cot]
        x, y = self.px_sang_m(hang, cot)
        ur, vr = calib.mat_dat_sang_goc(x, y)
        # Ảnh đầu vào có thể nhỏ hơn ảnh gốc (mặc định giải nén ở 1/2): tâm điểm ảnh (i + 0.5) * ti_le - 0.5
        mx = ((ur + 0.5) * ti_le_anh - 0.5).reshape(hang.shape).astype(np.float32)
        my = ((vr + 0.5) * ti_le_anh - 0.5).reshape(hang.shape).astype(np.float32)
        self.map1, self.map2 = cv2.convertMaps(mx, my, cv2.CV_16SC2)

    def px_sang_m(self, hang, cot):
        return self.x_xa - np.asarray(hang) * self.m, self.y_trai - np.asarray(cot) * self.m

    def m_sang_px(self, x, y):
        return (self.x_xa - np.asarray(x)) / self.m, (self.y_trai - np.asarray(y)) / self.m

    def chieu(self, xam_nua):
        return cv2.remap(xam_nua, self.map1, self.map2, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def doc_bev(thu_muc=None):
    """calib/bev.yaml (vùng nhìn + độ phân giải bird's eye do người dùng chọn). Không có thì None."""
    f = Path(thu_muc or cau_hinh.CALIB) / 'bev.yaml'
    if not f.exists():
        return None
    b = yaml.safe_load(f.read_text()) or {}
    return b if all(k in b for k in ('x_gan', 'x_xa', 'y_trai', 'y_phai', 'm_px')) else None


def calib_dung(gia_lap=False):
    return Calib.gia_lap() if gia_lap else Calib()
