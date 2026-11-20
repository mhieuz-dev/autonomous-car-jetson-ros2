"""Trang xem hình + trạng thái của script đang chạy (cổng 8097, mở từ laptop).

    http://<ip-xe>:8097/                 trang có mọi hình đang phát
    http://<ip-xe>:8097/img/<ten>.mjpg   luồng hình (MJPEG)
    http://<ip-xe>:8097/img/<ten>.jpg    một khung
    http://<ip-xe>:8097/status.json  số liệu (chương trình, v, w, yaw, pitch, roll...)
Đi bằng TCP nên qua được hotspot. Chỉ nén JPEG khi CÓ người đang xem hình đó,
để Jetson không tốn CPU vô ích.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from . import settings


class Viewer:
    def __init__(self, port=settings.VIEW_PORT):
        self.image = {}   # tên -> jpeg bytes
        self.needed = {}  # tên -> giờ có người xem gần nhất
        self.status = {}
        self.lock = threading.Lock()
        self.server = None
        try:
            self.server = ThreadingHTTPServer(('0.0.0.0', port), self._handler())
            self.server.daemon_threads = True
            threading.Thread(target=self.server.serve_forever, daemon=True).start()
            self.port = port
        except OSError as e:
            print(f'[xem] không mở được cổng {port}: {e} (vẫn chạy, chỉ không xem hình)', flush=True)

    def is_watched(self, name):
        return time.monotonic() - self.needed.get(name, -1e9) < 2.0

    def set_image(self, name, bgr, quality=70, always=False):
        """Đặt hình `name`. Chỉ nén nếu có người xem (hoặc luon=True)."""
        with self.lock:
            if name not in self.image:
                self.image[name] = None
        if not (always or self.is_watched(name)):
            return
        ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            with self.lock:
                self.image[name] = buf.tobytes()

    def set_jpeg(self, name, jpeg):
        with self.lock:
            self.image[name] = jpeg

    def update(self, **kv):
        with self.lock:
            self.status.update(kv)

    def close(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()

    def _handler(self):
        viewer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            disable_nagle_algorithm = True  # đầu + thân trả lời đi ngay, khỏi chờ ACK trễ ~200 ms
            wbufsize = 1 << 16              # gom đầu + thân thành một lần gửi (handle_one_request tự flush)

            def log_message(self, *a):
                pass

            def _send(self, body, kind, code=200):
                self.send_response(code)
                self.send_header('Content-Type', kind)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split('?')[0]
                if path == '/status.json':
                    with viewer.lock:
                        body = json.dumps({**viewer.status, 'image': sorted(viewer.image)}).encode()
                    return self._send(body, 'application/json')
                if path.startswith('/img/') and path.endswith('.jpg'):
                    name = path[5:-4]
                    viewer.needed[name] = time.monotonic()
                    j = viewer.image.get(name)
                    return self._send(j or b'', 'image/jpeg', 200 if j else 404)
                if path.startswith('/img/') and path.endswith('.mjpg'):
                    return self._mjpg(path[5:-5])
                if path == '/':
                    html = ''.join(f'<h3>{t}</h3><img src="/img/{t}.mjpg" style="max-width:100%">'
                                  for t in sorted(viewer.image))
                    body = (f'<html><meta charset="utf-8"><body style="background:#111;color:#eee">'
                            f'<h2>Champion: hình đang phát</h2>{html}</body></html>').encode()
                    return self._send(body, 'text/html; charset=utf-8')
                self._send(b'not found', 'text/plain', 404)

            def do_OPTIONS(self):
                self.send_response(204)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                self.send_header('Content-Length', '0')
                self.end_headers()

            def _mjpg(self, name):
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=f')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.flush()
                old = None
                try:
                    while True:
                        viewer.needed[name] = time.monotonic()
                        j = viewer.image.get(name)
                        if j and j is not old:
                            self.wfile.write(b'--f\r\nContent-Type: image/jpeg\r\nContent-Length: '
                                             + str(len(j)).encode() + b'\r\n\r\n' + j + b'\r\n')
                            self.wfile.flush()
                            old = j
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        return H
