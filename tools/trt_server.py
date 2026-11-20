#!/usr/bin/env python3
# RUNS_ON: host
"""Máy chủ TensorRT chạy trên Jetson (NGOÀI container, python3.6 của JetPack): nhận ảnh đã letterbox từ container
qua TCP 127.0.0.1:8095, chạy models/best.engine trên GPU, trả output thô [1, 18, 8400] để container hậu xử lý.

Vì sao chạy ngoài container: container crc_car (Ubuntu 22.04, python3.10) không có TensorRT/CUDA cho python;
JetPack trên Jetson có sẵn TensorRT 8.2 (python3-libnvinfer). scripts/car.sh tự bật script này trước khi chạy xe.
Giao thức: [4 byte độ dài][uint8 CHW 3x640x640]  ->  [4 byte độ dài][float32 output].
Build engine (1 lần, ~10 phút):  /usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=best.engine --fp16
"""
import ctypes
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import tensorrt as trt

PORT = 8095
ENGINE = Path(__file__).resolve().parents[1] / 'models' / 'best.engine'

_cuda = ctypes.CDLL('libcudart.so.10.2')
H2D, D2H = 1, 2


def cuda_malloc(n):
    p = ctypes.c_void_p()
    if _cuda.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(n)) != 0:
        raise RuntimeError('cudaMalloc lỗi')
    return p


def cuda_copy(dst, src, n, kind):
    if _cuda.cudaMemcpy(dst, src, ctypes.c_size_t(n), kind) != 0:
        raise RuntimeError('cudaMemcpy lỗi')


class Engine:
    def __init__(self, model_path):
        log = trt.Logger(trt.Logger.WARNING)
        with open(str(model_path), 'rb') as f, trt.Runtime(log) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.inp = self.out = None
        self.bind = []
        for i in range(self.engine.num_bindings):
            shape = tuple(self.engine.get_binding_shape(i))
            n = int(np.prod(shape)) * 4
            d = cuda_malloc(n)
            self.bind.append(d.value)
            if self.engine.binding_is_input(i):
                self.inp = (d, shape, n)
            else:
                self.out = (d, shape, n)

    def run(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        cuda_copy(self.inp[0], x.ctypes.data_as(ctypes.c_void_p), self.inp[2], H2D)
        self.ctx.execute_v2(self.bind)
        out = np.empty(self.out[1], np.float32)
        cuda_copy(out.ctypes.data_as(ctypes.c_void_p), self.out[0], self.out[2], D2H)
        return out


def read_exact(conn, n):
    buf = bytearray(n)
    v = memoryview(buf)
    i = 0
    while i < n:
        k = conn.recv_into(v[i:], n - i)
        if k == 0:
            raise ConnectionError('đóng kết nối')
        i += k
    return buf


def main():
    if not ENGINE.exists():
        print('[trt] chưa có ' + str(ENGINE) + ' (build bằng trtexec)', flush=True)
        sys.exit(1)
    t = time.time()
    e = Engine(ENGINE)
    print('[trt] nạp engine %.1f s, vào %s, ra %s' % (time.time() - t, e.inp[1], e.out[1]), flush=True)
    e.run(np.zeros(e.inp[1], np.float32))  # chạy nháp cho GPU nóng máy
    mutex = threading.Lock()               # 1 engine dùng chung: mỗi lúc 1 lần chạy GPU

    def serve_client(conn):
        """Mỗi kết nối 1 luồng: xe tự hành, onnx_test (kể cả bản cũ chưa tắt) không chặn nhau."""
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while True:
                n = struct.unpack('<I', read_exact(conn, 4))[0]
                x = np.frombuffer(read_exact(conn, n), np.uint8).reshape(e.inp[1]).astype(np.float32) / 255.0
                with mutex:
                    out = e.run(x).tobytes()
                conn.sendall(struct.pack('<I', len(out)) + out)
        except (ConnectionError, OSError, ValueError):
            pass
        finally:
            conn.close()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', PORT))
    srv.listen(8)
    print('[trt] sẵn sàng ở 127.0.0.1:%d' % PORT, flush=True)
    while True:
        conn, _ = srv.accept()
        t = threading.Thread(target=serve_client, args=(conn,))
        t.daemon = True
        t.start()

if __name__ == '__main__':
    main()
