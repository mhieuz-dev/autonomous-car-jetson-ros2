"""Giao thức container -> máy chủ TensorRT (tools/trt_server.py): dựng máy chủ giả chạy onnxruntime cùng giao
thức, kết quả nhận qua socket phải giống chạy onnxruntime trực tiếp (chỉ khác do ảnh đổi sang uint8)."""
import socket
import struct
import threading

import numpy as np
import pytest

from core import detector as D


def test_trt_server_protocol(monkeypatch):
    ort = pytest.importorskip('onnxruntime')
    real = D._OnnxRuntime(D.settings.MODEL_ONNX)
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', 0))
    srv.listen(1)
    monkeypatch.setattr(D._TensorRT, 'PORT', srv.getsockname()[1])

    def srv_proc():                      # giống vòng lặp của trt_server.py, thay engine bằng onnxruntime
        conn, _ = srv.accept()
        def read(n):
            b = b''
            while len(b) < n:
                b += conn.recv(n - len(b))
            return b
        for _ in range(2):
            n = struct.unpack('<I', read(4))[0]
            x = np.frombuffer(read(n), np.uint8).reshape(1, 3, 640, 640).astype(np.float32) / 255.0
            out = real.run(x).astype(np.float32).tobytes()
            conn.sendall(struct.pack('<I', len(out)) + out)
        conn.close()
    threading.Thread(target=srv_proc, daemon=True).start()

    trt = D._TensorRT(wait_s=2.0)
    rng = np.random.default_rng(0)
    for _ in range(2):
        img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        x = D.letterbox(img)[0]
        a = trt.run(x)
        b = real.run(x)
        assert a.shape == (1, 18, 8400) and np.asarray(b).reshape(a.shape).shape == a.shape
        assert np.abs(a - np.asarray(b).reshape(a.shape)).max() < 2.0          # chỉ sai do làm tròn uint8


def test_no_server_fails_fast(monkeypatch):
    monkeypatch.setattr(D._TensorRT, 'PORT', 1)          # cổng không ai nghe
    with pytest.raises(ConnectionError):
        D._TensorRT(wait_s=0.6)
