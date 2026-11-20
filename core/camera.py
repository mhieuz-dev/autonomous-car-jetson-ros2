"""Camera: đọc calib, giải nén ảnh, chiếu bird's eye bằng MỘT bảng remap tính sẵn.

Mô hình: mặt đường phẳng (z = 0), hệ toạ độ xe: x phía trước, y bên trái, gốc trên mặt
đường ngay dưới TÂM xe. Camera có tư thế đầy đủ (rvec, tvec: hệ xe -> hệ camera) lấy từ
calib bàn cờ nằm trên sàn (tools/birdeye_chessboard.py), nên đúng cả khi camera chúi xuống, nghiêng
ngang hay lệch hướng. Không có số mặc định nào: thiếu calib thì báo NotCalibrated.
Bảng remap đi thẳng từ ảnh GỐC (còn méo) sang bird's eye, mỗi khung chỉ tốn 1 lần cv2.remap.
"""
from pathlib import Path

import cv2
import numpy as np
import yaml

from . import settings


class NotCalibrated(RuntimeError):
    pass


def from_pitch(pitch_rad, height_m, front_m, lateral_m=0.0):
    """(rvec, tvec) cho camera chúi xuống pitch_rad, cao height_m, đặt trước tâm xe front_m (chỉ để dựng xe GIẢ)."""
    sp, cp = np.sin(pitch_rad), np.cos(pitch_rad)
    R = np.array([[0, -1, 0], [-sp, 0, -cp], [cp, 0, -sp]], float)
    C = np.array([front_m, lateral_m, height_m], float)
    return cv2.Rodrigues(R)[0].ravel(), (-R @ C)


class Calib:
    """Calib camera. Không có file calib thì báo NotCalibrated (KHÔNG dùng số cũ)."""

    def __init__(self, folder=None):
        folder = Path(folder or settings.CONFIG_DIR)
        f_cam, f_extr = folder / 'camera.yaml', folder / 'extrinsics.yaml'
        if not f_cam.exists():
            raise NotCalibrated(f'CHƯA CALIB CAMERA (không có {f_cam}). Calib nội: đặt config/camera.yaml')
        with open(f_cam) as f:
            c = yaml.safe_load(f)
        n = {}
        if f_extr.exists():
            with open(f_extr) as f:
                n = yaml.safe_load(f) or {}
        if n.get('rvec') is None or n.get('tvec') is None:
            raise NotCalibrated("CHƯA CÓ BIRD'S EYE (không có rvec/tvec trong "
                            f"{f_extr}). Chạy birdeye_chessboard rồi chọn ứng viên (bảng điều khiển)")
        self._load(c, n['rvec'], n['tvec'])
        self.folder = folder

    def _load(self, c, rvec, tvec):
        self.K = np.array(c['camera_matrix']['data'], float).reshape(3, 3)
        self.D = np.array(c['distortion_coefficients']['data'], float).ravel()
        self.width, self.img_h = int(c['image_width']), int(c['image_height'])
        self.rvec = np.asarray(rvec, float).ravel()
        self.tvec = np.asarray(tvec, float).ravel()
        self.R = cv2.Rodrigues(self.rvec)[0]
        C = -self.R.T @ self.tvec                     # vị trí camera trong hệ xe
        self.front_of_center, self.lateral, self.height_m = float(C[0]), float(C[1]), float(C[2])
        d = self.R[:, 0]                              # hướng "thẳng tiến" trong hệ camera
        self.horizon = float(self.K[1, 1] * d[1] / d[2] + self.K[1, 2]) if d[2] > 1e-6 else float('nan')

    @classmethod
    def from_pose(cls, K, D, rvec, tvec, size=(640, 480)):
        """Calib dựng trong bộ nhớ (công cụ birdeye_chessboard, test)."""
        o = cls.__new__(cls)
        o._load({'camera_matrix': {'data': list(np.asarray(K, float).ravel())},
                'distortion_coefficients': {'data': list(np.asarray(D, float).ravel())},
                'image_width': size[0], 'image_height': size[1]}, rvec, tvec)
        o.folder = None
        return o

    @classmethod
    def sim(cls):
        """Camera LÝ TƯỞNG cho xe giả lập (không méo). Không liên quan xe thật."""
        rvec, tvec = from_pitch(np.arctan((240.0 - 244.0) / 420.0), 0.14, 0.069)
        return cls.from_pose(np.array([[420, 0, 320], [0, 420, 240], [0, 0, 1.0]]), np.zeros(5), rvec, tvec)

    def ground_to_pixel(self, x, y):
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

    def pixel_to_ground(self, u, v):
        """Pixel trên ảnh GỐC -> mặt đường (x, y) trong hệ xe. Tia không chạm đất (trên chân trời): nan."""
        pts = np.column_stack([np.asarray(u, float).ravel(), np.asarray(v, float).ravel()])
        n = cv2.undistortPoints(pts.reshape(-1, 1, 2), self.K, self.D)[:, 0]
        ray = np.column_stack([n, np.ones(len(n))]) @ self.R              # R^T r
        C = -self.R.T @ self.tvec
        with np.errstate(divide='ignore', invalid='ignore'):
            s = np.where(ray[:, 2] < -1e-6, -C[2] / ray[:, 2], np.nan)
        s = np.where(s > 0, s, np.nan)
        return C[0] + s * ray[:, 0], C[1] + s * ray[:, 1]


def decode_gray_half(jpeg):
    """JPEG -> ảnh xám NỬA độ phân giải. Giải nén thẳng ở 1/2 nên nhanh hơn nhiều."""
    buf = np.frombuffer(jpeg, np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_REDUCED_GRAYSCALE_2)


def decode_color(jpeg):
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


class BirdEye:
    """Ảnh nhìn từ trên xuống: hàng 0 = xa nhất (X_FAR), cột 0 = bên trái nhất (Y_TRAI)."""

    def __init__(self, calib, x_near=0.15, x_far=1.0, y_left=0.50, y_right=-0.50, m_px=0.005, img_scale=0.5):
        self.calib = calib
        self.x_near, self.x_far = x_near, x_far
        self.y_left, self.y_right = y_left, y_right
        self.m = m_px
        self.n_rows = int(round((x_far - x_near) / m_px)) + 1
        self.n_cols = int(round((y_left - y_right) / m_px)) + 1
        row, col = np.mgrid[0:self.n_rows, 0:self.n_cols]
        x, y = self.px_to_m(row, col)
        ur, vr = calib.ground_to_pixel(x, y)
        # Ảnh đầu vào có thể nhỏ hơn ảnh gốc (mặc định giải nén ở 1/2): tâm điểm ảnh (i + 0.5) * img_scale - 0.5
        mx = ((ur + 0.5) * img_scale - 0.5).reshape(row.shape).astype(np.float32)
        my = ((vr + 0.5) * img_scale - 0.5).reshape(row.shape).astype(np.float32)
        self.map1, self.map2 = cv2.convertMaps(mx, my, cv2.CV_16SC2)

    def px_to_m(self, row, col):
        return self.x_far - np.asarray(row) * self.m, self.y_left - np.asarray(col) * self.m

    def m_to_px(self, x, y):
        return (self.x_far - np.asarray(x)) / self.m, (self.y_left - np.asarray(y)) / self.m

    def warp(self, gray_half):
        return cv2.remap(gray_half, self.map1, self.map2, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def read_bev(folder=None):
    """config/bev.yaml (vùng nhìn + độ phân giải bird's eye do người dùng chọn). Không có thì None."""
    f = Path(folder or settings.CONFIG_DIR) / 'bev.yaml'
    if not f.exists():
        return None
    b = yaml.safe_load(f.read_text()) or {}
    return b if all(k in b for k in ('x_near', 'x_far', 'y_left', 'y_right', 'm_px')) else None


def active_calib(sim=False):
    return Calib.sim() if sim else Calib()
