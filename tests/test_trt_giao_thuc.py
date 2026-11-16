"""Giao thức container -> máy chủ TensorRT (cong_cu/trt_may_chu.py): dựng máy chủ giả chạy onnxruntime cùng giao
thức, kết quả nhận qua socket phải giống chạy onnxruntime trực tiếp (chỉ khác do ảnh đổi sang uint8)."""
import socket
import struct
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chung import den_onnx as D  # noqa: E402


def test_giao_thuc_may_chu_trt(monkeypatch):
    ort = pytest.importorskip('onnxruntime')
    that = D._OnnxRuntime(D.cau_hinh.MODEL_ONNX)
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', 0))
    srv.listen(1)
    monkeypatch.setattr(D._TensorRT, 'CONG', srv.getsockname()[1])

    def may_chu():                      # giống vòng lặp của trt_may_chu.py, thay engine bằng onnxruntime
        conn, _ = srv.accept()
        def doc(n):
            b = b''
            while len(b) < n:
                b += conn.recv(n - len(b))
            return b
        for _ in range(2):
            n = struct.unpack('<I', doc(4))[0]
            x = np.frombuffer(doc(n), np.uint8).reshape(1, 3, 640, 640).astype(np.float32) / 255.0
            out = that.chay(x).astype(np.float32).tobytes()
            conn.sendall(struct.pack('<I', len(out)) + out)
        conn.close()
    threading.Thread(target=may_chu, daemon=True).start()

    trt = D._TensorRT(cho_giay=2.0)
    rng = np.random.default_rng(0)
    for _ in range(2):
        img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        x = D.letterbox(img)[0]
        a = trt.chay(x)
        b = that.chay(x)
        assert a.shape == (1, 18, 8400) and np.asarray(b).reshape(a.shape).shape == a.shape
        assert np.abs(a - np.asarray(b).reshape(a.shape)).max() < 2.0          # chỉ sai do làm tròn uint8


def test_khong_co_may_chu_thi_bao_loi_nhanh(monkeypatch):
    monkeypatch.setattr(D._TensorRT, 'CONG', 1)          # cổng không ai nghe
    with pytest.raises(ConnectionError):
        D._TensorRT(cho_giay=0.6)
