"""Nhận đèn giao thông bằng model YOLO11n `models/best.onnx` (14 lớp, có 'den xanh').

ONNX chỉ là file model trung gian. Để chạy nhanh trên một card NVIDIA, build
ra file engine TensorRT NGAY TRÊN MÁY ĐÓ (engine gắn với loại GPU + bản TensorRT):
    /usr/src/tensorrt/bin/trtexec --onnx=models/best.onnx --saveEngine=models/best.engine --fp16
Thứ tự thử: TensorRT (máy chủ cong_cu/trt_may_chu.py trên Jetson, GPU) -> onnxruntime (CUDA rồi CPU) -> cv2.dnn.
Tiền / hậu xử lý giống bản crc_solutionn (letterbox nền 114, output [1, 18, 8400], NMS).
"""
import ast
import time

import cv2
import numpy as np

from . import cau_hinh

TEN_LOP = {0: 'bien cao toc', 1: 'bien co doc', 2: 'bien do lan xe bus', 3: 'bien het cao toc',
           4: 'bien vao ham', 5: 'den do', 6: 'den vang', 7: 'den xanh', 8: 'di thang',
           9: 'lan xe bus', 10: 'nguoi di bo', 11: 're phai', 12: 're trai', 13: 'stop'}
CO = 640


def letterbox(rgb):
    h, w = rgb.shape[:2]
    k = min(CO / w, CO / h)
    nw, nh = round(w * k), round(h * k)
    trai, tren = (CO - nw) // 2, (CO - nh) // 2
    nen = np.full((CO, CO, 3), 114, np.uint8)
    nen[tren:tren + nh, trai:trai + nw] = cv2.resize(rgb, (nw, nh))
    x = np.ascontiguousarray(nen.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return x, k, trai, tren


def hau_xu_ly(out, k, trai, tren, w, h, nguong=0.25, ten_lop=TEN_LOP):
    rows = np.asarray(out, np.float32).reshape(4 + len(ten_lop), -1).T     # (8400, 18)
    lop = rows[:, 4:].argmax(axis=1)
    diem = rows[np.arange(len(rows)), lop + 4]
    m = diem >= nguong
    rows, lop, diem = rows[m], lop[m], diem[m]
    hop = rows[:, :4].copy()
    hop[:, :2] -= hop[:, 2:] / 2
    kq = []
    for c in np.unique(lop):
        idx = np.flatnonzero(lop == c)
        giu = cv2.dnn.NMSBoxes(hop[idx].tolist(), diem[idx].tolist(), nguong, 0.45)
        for j in np.asarray(giu).reshape(-1):
            i = idx[j]
            x, y, bw, bh = hop[i]
            xyxy = [(x - trai) / k, (y - tren) / k, (x + bw - trai) / k, (y + bh - tren) / k]
            xyxy = [float(np.clip(v, 0, w if n % 2 == 0 else h)) for n, v in enumerate(xyxy)]
            kq.append({'ten': ten_lop.get(int(c), str(c)), 'lop': int(c),
                       'tin': float(diem[i]), 'hop': xyxy})
    return sorted(kq, key=lambda d: -d['tin'])


class _OnnxRuntime:
    def __init__(self, duong_dan, luong=2):
        import onnxruntime as ort
        co = ort.get_available_providers()
        ds = [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider') if p in co]
        o = ort.SessionOptions()
        o.intra_op_num_threads = luong
        self.s = ort.InferenceSession(str(duong_dan), sess_options=o, providers=ds)
        self.vao = self.s.get_inputs()[0].name
        self.ten = f'onnxruntime ({self.s.get_providers()[0]})'
        meta = self.s.get_modelmeta().custom_metadata_map
        self.ten_lop = ast.literal_eval(meta['names']) if 'names' in meta else TEN_LOP

    def chay(self, x):
        return self.s.run(None, {self.vao: x})[0]


class _CvDnn:
    def __init__(self, duong_dan):
        self.net = cv2.dnn.readNetFromONNX(str(duong_dan))
        self.ten = f'cv2.dnn (OpenCV {cv2.__version__}, CPU)'
        self.ten_lop = TEN_LOP

    def chay(self, x):
        self.net.setInput(x)
        return self.net.forward()


class _TensorRT:
    """Gọi máy chủ TensorRT trên Jetson (cong_cu/trt_may_chu.py chạy NGOÀI container, GPU, models/best.engine)
    qua TCP 127.0.0.1:8095. Container không có TensorRT cho python nên không nạp engine trực tiếp được."""

    CONG = 8095

    def __init__(self, cho_giay=15.0):     # lần đầu máy chủ nạp engine mất ~8 s
        import socket
        t = time.monotonic()
        while True:
            try:
                self.s = socket.create_connection(('127.0.0.1', self.CONG), timeout=3.0)
                break
            except OSError:
                if time.monotonic() - t > cho_giay:
                    raise ConnectionError('không nối được máy chủ TensorRT 127.0.0.1:8095 (chay.sh tự bật cong_cu/trt_may_chu.py)')
                time.sleep(0.5)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.ten = 'TensorRT GPU (best.engine, máy chủ trên Jetson)'
        self.ten_lop = TEN_LOP

    def _doc(self, n):
        buf = bytearray(n)
        v = memoryview(buf)
        i = 0
        while i < n:
            k = self.s.recv_into(v[i:], n - i)
            if k == 0:
                raise ConnectionError('máy chủ TensorRT đóng kết nối')
            i += k
        return buf

    def chay(self, x):
        import struct
        u8 = np.ascontiguousarray((x[0] * 255.0 + 0.5).astype(np.uint8))      # CHW 3x640x640
        self.s.sendall(struct.pack('<I', u8.nbytes) + u8.tobytes())
        n = struct.unpack('<I', self._doc(4))[0]
        return np.frombuffer(self._doc(n), np.float32).reshape(1, 4 + len(TEN_LOP), -1)


class NhanDen:
    def __init__(self, thu=('tensorrt', 'onnxruntime', 'cv2'), nguong=0.25, luong=2):
        self.loi = []
        self.bo = None
        self.nguong = nguong
        for ten in thu:
            try:
                if ten == 'tensorrt':
                    if not cau_hinh.MODEL_ENGINE.exists():
                        raise FileNotFoundError('chưa có models/best.engine (build bằng trtexec)')
                    self.bo = _TensorRT()
                elif ten == 'onnxruntime':
                    self.bo = _OnnxRuntime(cau_hinh.MODEL_ONNX, luong)
                elif ten == 'cv2':
                    self.bo = _CvDnn(cau_hinh.MODEL_ONNX)
                    self.bo.chay(letterbox(np.zeros((480, 640, 3), np.uint8))[0])   # thử 1 lần
                break
            except Exception as e:      # noqa: BLE001 - ghi lại để báo cho người dùng
                self.loi.append(f'{ten}: {str(e).splitlines()[0][:120]}')
                self.bo = None
        self.ten = self.bo.ten if self.bo else 'KHÔNG CÓ'

    @property
    def san_sang(self):
        return self.bo is not None

    def nhan(self, bgr):
        """-> (danh sách {'ten','tin','hop'}, mili giây)."""
        t = time.monotonic()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x, k, trai, tren = letterbox(rgb)
        out = self.bo.chay(x)
        h, w = bgr.shape[:2]
        kq = hau_xu_ly(out, k, trai, tren, w, h, self.nguong, self.bo.ten_lop)
        return kq, (time.monotonic() - t) * 1000


def cat_vung(bgr, vung):
    """Cắt vùng đặt đèn (x1, y1, x2, y2) trên ảnh gốc. vung = None -> cả ảnh."""
    if not vung:
        return bgr, (0, 0)
    x1, y1, x2, y2 = [int(v) for v in vung]
    return bgr[max(0, y1):y2, max(0, x1):x2], (max(0, x1), max(0, y1))


def ve_ket_qua(bgr, kq, goc=(0, 0), vung=None, chi_vung=False):
    """chi_vung=True: trả riêng phần ảnh trong `vung` (bỏ hẳn phần ngoài ROI)."""
    img = bgr.copy()
    if vung:
        cv2.rectangle(img, tuple(int(v) for v in vung[:2]), tuple(int(v) for v in vung[2:]), (255, 255, 0), 1)
    mau = {'den xanh': (0, 255, 0), 'den do': (0, 0, 255), 'den vang': (0, 255, 255),
           'stop': (60, 60, 255), 'nguoi di bo': (255, 0, 255), 'bien vao ham': (0, 200, 255),
           'bien co doc': (255, 200, 0), 're trai': (255, 120, 0), 're phai': (255, 120, 0),
           'di thang': (255, 120, 0), 'lan xe bus': (0, 160, 0), 'bien do lan xe bus': (0, 160, 0),
           'bien cao toc': (200, 200, 200), 'bien het cao toc': (200, 200, 200)}
    for d in kq:
        x1, y1, x2, y2 = d['hop']
        p1 = (int(x1 + goc[0]), int(y1 + goc[1]))
        p2 = (int(x2 + goc[0]), int(y2 + goc[1]))
        c = mau.get(d['ten'], (255, 255, 255))
        cv2.rectangle(img, p1, p2, c, 2)
        cv2.putText(img, f'{d["ten"]} {d["tin"]:.2f}', (p1[0], max(12, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
    if vung and chi_vung:
        x1, y1, x2, y2 = [int(v) for v in vung]
        return img[max(0, y1):y2, max(0, x1):x2]
    return img


def mau_den_lon(ket_qua, cao_min_px, tin_min=0.5, tin_do_min=None):
    """Màu đèn ĐỦ LỚN trong ảnh ('do' > 'vang' > 'xanh') hoặc None, và chiều cao (px) của đèn lớn nhất thấy được.

    'Đủ gần' = khung bao quanh đèn cao >= cao_min_px (camera không đo được mét, nhưng đèn càng gần càng to).
    tin_do_min: độ tin riêng cho đỏ/vàng (None = dùng tin_min); đặt cao hơn để đèn đỏ bớt nhạy.
    Trả (mau, cao_lon_nhat_bat_ky, tin_cua_no)."""
    tot = {'den do': 'do', 'den vang': 'vang', 'den xanh': 'xanh'}
    lon, cao_max = {}, 0.0
    for d in ket_qua:
        m = tot.get(d['ten'])
        if m is None or d['tin'] < (tin_do_min if m != 'xanh' and tin_do_min is not None else tin_min):
            continue
        cao = d['hop'][3] - d['hop'][1]
        cao_max = max(cao_max, cao)
        if cao >= cao_min_px:
            lon[m] = max(lon.get(m, 0.0), d['tin'])
    for m in ('do', 'vang', 'xanh'):
        if m in lon:
            return m, cao_max, lon[m]
    return None, cao_max, 0.0


# ----------------------------------------------------------------- ROI biển ---
ROI_FILE = cau_hinh.CALIB / 'roi_bien.yaml'
ROI_MAC_DINH = {'y1': 240, 'y2': 480}      # chỉ nhận biển báo/đèn trong dải hàng pixel y1..y2 (nửa DƯỚI ảnh 480)


def doc_roi(w=640, h=480):
    """(x1, y1, x2, y2) vùng nhận biển; đọc calib/roi_bien.yaml (bảng điều khiển ghi), thiếu thì nửa dưới ảnh."""
    import yaml
    d = dict(ROI_MAC_DINH)
    try:
        d.update(yaml.safe_load(ROI_FILE.read_text()) or {})
    except (OSError, yaml.YAMLError):
        pass
    y1 = int(np.clip(d['y1'], 0, h - 2))
    y2 = int(np.clip(d['y2'], y1 + 1, h))
    return 0, y1, w, y2


def ghi_roi(y1, y2):
    y1, y2 = int(y1), int(y2)
    if not 0 <= y1 < y2 <= 480:
        raise ValueError(f'ROI không hợp lệ: y1={y1}, y2={y2} (cần 0 <= y1 < y2 <= 480)')
    ROI_FILE.write_text(f'# vung nhan bien bao + den (hang pixel, anh 480 cao), sua tren bang dieu khien\ny1: {y1}\ny2: {y2}\n')


def nhan_trong_roi(bo, bgr, roi):
    """Chạy model CHỈ trên vùng ROI, trả toạ độ khung theo ảnh gốc."""
    cat, goc = cat_vung(bgr, roi)
    kq, ms = bo.nhan(cat)
    for d in kq:
        d['hop'][0] += goc[0]
        d['hop'][2] += goc[0]
        d['hop'][1] += goc[1]
        d['hop'][3] += goc[1]
    return kq, ms


def bien_lon(ket_qua, ten, cao_min_px, tin_min=0.5):
    """(đủ lớn?, chiều cao lớn nhất px) của biển `ten` (vd 'stop')."""
    cao_max, ok = 0.0, False
    for d in ket_qua:
        if d['ten'] != ten or d['tin'] < tin_min:
            continue
        cao = d['hop'][3] - d['hop'][1]
        cao_max = max(cao_max, cao)
        ok = ok or cao >= cao_min_px
    return ok, cao_max
