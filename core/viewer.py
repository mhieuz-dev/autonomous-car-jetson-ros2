"""Trang xem hình + trạng thái của script đang chạy (cổng 8097, mở từ laptop).

    http://<ip-xe>:8097/                 trang có mọi hình đang phát
    http://<ip-xe>:8097/img/<ten>.mjpg   luồng hình (MJPEG)
    http://<ip-xe>:8097/img/<ten>.jpg    một khung
    http://<ip-xe>:8097/trang_thai.json  số liệu (chương trình, v, w, yaw, pitch, roll...)
Đi bằng TCP nên qua được hotspot. Chỉ nén JPEG khi CÓ người đang xem hình đó,
để Jetson không tốn CPU vô ích.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from . import settings


class Xem:
    def __init__(self, cong=settings.CONG_XEM):
        self.anh = {}              # tên -> jpeg bytes
        self.can = {}              # tên -> giờ có người xem gần nhất
        self.trang_thai = {}
        self.lock = threading.Lock()
        self.server = None
        try:
            self.server = ThreadingHTTPServer(('0.0.0.0', cong), self._handler())
            self.server.daemon_threads = True
            threading.Thread(target=self.server.serve_forever, daemon=True).start()
            self.cong = cong
        except OSError as e:
            print(f'[xem] không mở được cổng {cong}: {e} (vẫn chạy, chỉ không xem hình)', flush=True)

    def dang_xem(self, ten):
        return time.monotonic() - self.can.get(ten, -1e9) < 2.0

    def dat_anh(self, ten, bgr, chat_luong=70, luon=False):
        """Đặt hình `ten`. Chỉ nén nếu có người xem (hoặc luon=True)."""
        with self.lock:
            if ten not in self.anh:
                self.anh[ten] = None
        if not (luon or self.dang_xem(ten)):
            return
        ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, chat_luong])
        if ok:
            with self.lock:
                self.anh[ten] = buf.tobytes()

    def dat_jpeg(self, ten, jpeg):
        with self.lock:
            self.anh[ten] = jpeg

    def cap_nhat(self, **kv):
        with self.lock:
            self.trang_thai.update(kv)

    def dong(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()

    def _handler(self):
        xem = self

        class H(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            disable_nagle_algorithm = True     # đầu + thân trả lời đi ngay, khỏi chờ ACK trễ ~200 ms
            wbufsize = 1 << 16                 # gom đầu + thân thành một lần gửi (handle_one_request tự flush)

            def log_message(self, *a):
                pass

            def _gui(self, body, kieu, ma=200):
                self.send_response(ma)
                self.send_header('Content-Type', kieu)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split('?')[0]
                if path == '/trang_thai.json':
                    with xem.lock:
                        body = json.dumps({**xem.trang_thai, 'anh': sorted(xem.anh)}).encode()
                    return self._gui(body, 'application/json')
                if path.startswith('/img/') and path.endswith('.jpg'):
                    ten = path[5:-4]
                    xem.can[ten] = time.monotonic()
                    j = xem.anh.get(ten)
                    return self._gui(j or b'', 'image/jpeg', 200 if j else 404)
                if path.startswith('/img/') and path.endswith('.mjpg'):
                    return self._mjpg(path[5:-5])
                if path == '/':
                    the = ''.join(f'<h3>{t}</h3><img src="/img/{t}.mjpg" style="max-width:100%">'
                                  for t in sorted(xem.anh))
                    body = (f'<html><meta charset="utf-8"><body style="background:#111;color:#eee">'
                            f'<h2>Champion: hình đang phát</h2>{the}</body></html>').encode()
                    return self._gui(body, 'text/html; charset=utf-8')
                self._gui(b'not found', 'text/plain', 404)

            def do_OPTIONS(self):
                self.send_response(204)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                self.send_header('Content-Length', '0')
                self.end_headers()

            def _mjpg(self, ten):
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=f')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.flush()
                cu = None
                try:
                    while True:
                        xem.can[ten] = time.monotonic()
                        j = xem.anh.get(ten)
                        if j and j is not cu:
                            self.wfile.write(b'--f\r\nContent-Type: image/jpeg\r\nContent-Length: '
                                             + str(len(j)).encode() + b'\r\n\r\n' + j + b'\r\n')
                            self.wfile.flush()
                            cu = j
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        return H
