"""Nhận đèn giao thông bằng model YOLO11n `models/best.onnx` (14 lớp, có 'green_light').

ONNX chỉ là file model trung gian. Để chạy nhanh trên một card NVIDIA, build
ra file engine TensorRT NGAY TRÊN MÁY ĐÓ (engine gắn với loại GPU + bản TensorRT):
    /usr/src/tensorrt/bin/trtexec --onnx=models/best.onnx --saveEngine=models/best.engine --fp16
Thứ tự thử: TensorRT (máy chủ tools/trt_server.py trên Jetson, GPU) -> onnxruntime (CUDA rồi CPU) -> cv2.dnn.
Tiền / hậu xử lý giống bản crc_solutionn (letterbox nền 114, output [1, 18, 8400], NMS).
"""
import ast
import time

import cv2
import numpy as np

from . import settings

CLASS_NAMES = {0: 'expressway', 1: 'slope_ahead', 2: 'bus_lane_sign', 3: 'expressway_end',
           4: 'tunnel_entrance', 5: 'red_light', 6: 'yellow_light', 7: 'green_light', 8: 'go_straight',
           9: 'bus_lane', 10: 'pedestrian', 11: 'turn_right', 12: 'turn_left', 13: 'stop'}
INPUT_SIZE = 640


def letterbox(rgb):
    h, w = rgb.shape[:2]
    k = min(INPUT_SIZE / w, INPUT_SIZE / h)
    nw, nh = round(w * k), round(h * k)
    pad_left, pad_top = (INPUT_SIZE - nw) // 2, (INPUT_SIZE - nh) // 2
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, np.uint8)
    canvas[pad_top:pad_top + nh, pad_left:pad_left + nw] = cv2.resize(rgb, (nw, nh))
    x = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return x, k, pad_left, pad_top


def postprocess(out, k, pad_left, pad_top, w, h, thresh=0.25, class_names=CLASS_NAMES):
    rows = np.asarray(out, np.float32).reshape(4 + len(class_names), -1).T     # (8400, 18)
    cls_ids = rows[:, 4:].argmax(axis=1)
    score = rows[np.arange(len(rows)), cls_ids + 4]
    m = score >= thresh
    rows, cls_ids, score = rows[m], cls_ids[m], score[m]
    box = rows[:, :4].copy()
    box[:, :2] -= box[:, 2:] / 2
    result = []
    for c in np.unique(cls_ids):
        idx = np.flatnonzero(cls_ids == c)
        keep = cv2.dnn.NMSBoxes(box[idx].tolist(), score[idx].tolist(), thresh, 0.45)
        for j in np.asarray(keep).reshape(-1):
            i = idx[j]
            x, y, bw, bh = box[i]
            xyxy = [(x - pad_left) / k, (y - pad_top) / k, (x + bw - pad_left) / k, (y + bh - pad_top) / k]
            xyxy = [float(np.clip(v, 0, w if n % 2 == 0 else h)) for n, v in enumerate(xyxy)]
            result.append({'name': class_names.get(int(c), str(c)), 'cls': int(c),
                       'conf': float(score[i]), 'box': xyxy})
    return sorted(result, key=lambda d: -d['conf'])


class _OnnxRuntime:
    def __init__(self, model_path, threads=2):
        import onnxruntime as ort
        size = ort.get_available_providers()
        ds = [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider') if p in size]
        o = ort.SessionOptions()
        o.intra_op_num_threads = threads
        self.s = ort.InferenceSession(str(model_path), sess_options=o, providers=ds)
        self.inp = self.s.get_inputs()[0].name
        self.name = f'onnxruntime ({self.s.get_providers()[0]})'
        meta = self.s.get_modelmeta().custom_metadata_map
        self.class_names = ast.literal_eval(meta['names']) if 'names' in meta else CLASS_NAMES

    def run(self, x):
        return self.s.run(None, {self.inp: x})[0]


class _CvDnn:
    def __init__(self, model_path):
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.name = f'cv2.dnn (OpenCV {cv2.__version__}, CPU)'
        self.class_names = CLASS_NAMES

    def run(self, x):
        self.net.setInput(x)
        return self.net.forward()


class _TensorRT:
    """Gọi máy chủ TensorRT trên Jetson (tools/trt_server.py chạy NGOÀI container, GPU, models/best.engine)
    qua TCP 127.0.0.1:8095. Container không có TensorRT cho python nên không nạp engine trực tiếp được."""

    PORT = 8095

    def __init__(self, wait_s=15.0):     # lần đầu máy chủ nạp engine mất ~8 s
        import socket
        t = time.monotonic()
        while True:
            try:
                self.s = socket.create_connection(('127.0.0.1', self.PORT), timeout=3.0)
                break
            except OSError:
                if time.monotonic() - t > wait_s:
                    raise ConnectionError('không nối được máy chủ TensorRT 127.0.0.1:8095 (scripts/car.sh tự bật tools/trt_server.py)')
                time.sleep(0.5)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.name = 'TensorRT GPU (best.engine, máy chủ trên Jetson)'
        self.class_names = CLASS_NAMES

    def _read(self, n):
        buf = bytearray(n)
        v = memoryview(buf)
        i = 0
        while i < n:
            k = self.s.recv_into(v[i:], n - i)
            if k == 0:
                raise ConnectionError('máy chủ TensorRT đóng kết nối')
            i += k
        return buf

    def run(self, x):
        import struct
        u8 = np.ascontiguousarray((x[0] * 255.0 + 0.5).astype(np.uint8))      # CHW 3x640x640
        self.s.sendall(struct.pack('<I', u8.nbytes) + u8.tobytes())
        n = struct.unpack('<I', self._read(4))[0]
        return np.frombuffer(self._read(n), np.float32).reshape(1, 4 + len(CLASS_NAMES), -1)


class Detector:
    def __init__(self, try_order=('tensorrt', 'onnxruntime', 'cv2'), thresh=0.25, threads=2):
        self.error = []
        self.backend = None
        self.thresh = thresh
        for name in try_order:
            try:
                if name == 'tensorrt':
                    if not settings.MODEL_ENGINE.exists():
                        raise FileNotFoundError('chưa có models/best.engine (build bằng trtexec)')
                    self.backend = _TensorRT()
                elif name == 'onnxruntime':
                    self.backend = _OnnxRuntime(settings.MODEL_ONNX, threads)
                elif name == 'cv2':
                    self.backend = _CvDnn(settings.MODEL_ONNX)
                    self.backend.run(letterbox(np.zeros((480, 640, 3), np.uint8))[0])   # thử 1 lần
                break
            except Exception as e:      # noqa: BLE001 - ghi lại để báo cho người dùng
                self.error.append(f'{name}: {str(e).splitlines()[0][:120]}')
                self.backend = None
        self.name = self.backend.name if self.backend else 'KHÔNG CÓ'

    @property
    def ready(self):
        return self.backend is not None

    def detect(self, bgr):
        """-> (danh sách {'name','conf','box'}, mili giây)."""
        t = time.monotonic()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x, k, pad_left, pad_top = letterbox(rgb)
        out = self.backend.run(x)
        h, w = bgr.shape[:2]
        result = postprocess(out, k, pad_left, pad_top, w, h, self.thresh, self.backend.class_names)
        return result, (time.monotonic() - t) * 1000


def crop_zone(bgr, zone):
    """Cắt vùng đặt đèn (x1, y1, x2, y2) trên ảnh gốc. vung = None -> cả ảnh."""
    if not zone:
        return bgr, (0, 0)
    x1, y1, x2, y2 = [int(v) for v in zone]
    return bgr[max(0, y1):y2, max(0, x1):x2], (max(0, x1), max(0, y1))


def draw_detections(bgr, result, origin=(0, 0), zone=None, zone_only=False):
    """zone_only=True: trả riêng phần ảnh trong `zone` (bỏ hẳn phần ngoài ROI)."""
    img = bgr.copy()
    if zone:
        cv2.rectangle(img, tuple(int(v) for v in zone[:2]), tuple(int(v) for v in zone[2:]), (255, 255, 0), 1)
    colors = {'green_light': (0, 255, 0), 'red_light': (0, 0, 255), 'yellow_light': (0, 255, 255),
           'stop': (60, 60, 255), 'pedestrian': (255, 0, 255), 'tunnel_entrance': (0, 200, 255),
           'slope_ahead': (255, 200, 0), 'turn_left': (255, 120, 0), 'turn_right': (255, 120, 0),
           'go_straight': (255, 120, 0), 'bus_lane': (0, 160, 0), 'bus_lane_sign': (0, 160, 0),
           'expressway': (200, 200, 200), 'expressway_end': (200, 200, 200)}
    for d in result:
        x1, y1, x2, y2 = d['box']
        p1 = (int(x1 + origin[0]), int(y1 + origin[1]))
        p2 = (int(x2 + origin[0]), int(y2 + origin[1]))
        c = colors.get(d['name'], (255, 255, 255))
        cv2.rectangle(img, p1, p2, c, 2)
        cv2.putText(img, f'{d["name"]} {d["conf"]:.2f}', (p1[0], max(12, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
    if zone and zone_only:
        x1, y1, x2, y2 = [int(v) for v in zone]
        return img[max(0, y1):y2, max(0, x1):x2]
    return img


def largest_light(result, min_height_px, min_conf=0.5, min_conf_red=None):
    """Màu đèn ĐỦ LỚN trong ảnh ('red' > 'yellow' > 'green') hoặc None, và chiều cao (px) của đèn lớn nhất thấy được.

    'Đủ gần' = khung bao quanh đèn cao >= min_height_px (camera không đo được mét, nhưng đèn càng gần càng to).
    min_conf_red: độ tin riêng cho đỏ/vàng (None = dùng min_conf); đặt cao hơn để đèn đỏ bớt nhạy.
    Trả (color, max_height, conf)."""
    good = {'red_light': 'red', 'yellow_light': 'yellow', 'green_light': 'green'}
    best_conf, max_height = {}, 0.0
    for d in result:
        m = good.get(d['name'])
        if m is None or d['conf'] < (min_conf_red if m != 'green' and min_conf_red is not None else min_conf):
            continue
        height = d['box'][3] - d['box'][1]
        max_height = max(max_height, height)
        if height >= min_height_px:
            best_conf[m] = max(best_conf.get(m, 0.0), d['conf'])
    for m in ('red', 'yellow', 'green'):
        if m in best_conf:
            return m, max_height, best_conf[m]
    return None, max_height, 0.0


# ----------------------------------------------------------------- ROI biển ---
ROI_FILE = settings.CONFIG_DIR / 'sign_roi.yaml'
ROI_DEFAULT = {'y1': 240, 'y2': 480}      # chỉ nhận biển báo/đèn trong dải hàng pixel y1..y2 (nửa DƯỚI ảnh 480)


def read_roi(w=640, h=480):
    """(x1, y1, x2, y2) vùng nhận biển; đọc config/sign_roi.yaml (bảng điều khiển ghi), thiếu thì nửa dưới ảnh."""
    import yaml
    d = dict(ROI_DEFAULT)
    try:
        d.update(yaml.safe_load(ROI_FILE.read_text()) or {})
    except (OSError, yaml.YAMLError):
        pass
    y1 = int(np.clip(d['y1'], 0, h - 2))
    y2 = int(np.clip(d['y2'], y1 + 1, h))
    return 0, y1, w, y2


def save_roi(y1, y2):
    y1, y2 = int(y1), int(y2)
    if not 0 <= y1 < y2 <= 480:
        raise ValueError(f'ROI không hợp lệ: y1={y1}, y2={y2} (cần 0 <= y1 < y2 <= 480)')
    ROI_FILE.write_text(f'# vung nhan bien bao + den (hang pixel, anh 480 cao), sua tren bang dieu khien\ny1: {y1}\ny2: {y2}\n')


def detect_in_roi(backend, bgr, roi):
    """Chạy model CHỈ trên vùng ROI, trả toạ độ khung theo ảnh gốc."""
    crop, origin = crop_zone(bgr, roi)
    result, ms = backend.detect(crop)
    for d in result:
        d['box'][0] += origin[0]
        d['box'][2] += origin[0]
        d['box'][1] += origin[1]
        d['box'][3] += origin[1]
    return result, ms


def large_sign(result, name, min_height_px, min_conf=0.5):
    """(đủ lớn?, chiều cao lớn nhất px) của biển `name` (vd 'stop')."""
    max_height, ok = 0.0, False
    for d in result:
        if d['name'] != name or d['conf'] < min_conf:
            continue
        height = d['box'][3] - d['box'][1]
        max_height = max(max_height, height)
        ok = ok or height >= min_height_px
    return ok, max_height
