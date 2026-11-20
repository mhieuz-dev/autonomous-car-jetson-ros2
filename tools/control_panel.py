#!/usr/bin/env python3
"""BẢNG ĐIỀU KHIỂN Champion trên laptop: http://127.0.0.1:8099
CHAY_TREN: laptop

    python3 tools/control_panel.py            (IP xe lấy từ CRC_HOST, mặc định jetson.local, đổi bằng CRC_HOST=jetson@<ip>)

Bên TRÁI: mọi script (xe tự hành, công cụ IMU / PID / LiDAR / calib / đèn / dò vạch /
tải Jetson). Bấm "Xe" = chạy trên xe qua `scripts/car.sh` (y như gõ tay), "Giả lập" = chạy trên
laptop với xe giả.
Bên PHẢI: log của script đang chạy (realtime), hình của nó (camera / bird's eye / LiDAR...)
và mô hình xe 3D theo IMU. Muốn sửa tham số: bấm "Sửa" (mở file), sửa khối THAM SỐ,
Ctrl+S, bấm chạy lại: lần chạy mới dùng đúng số vừa lưu.
Chỉ dùng thư viện chuẩn Python.
"""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

GOC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC))
CONG = 8099
XE_IP = os.environ.get('CRC_HOST', 'jetson@jetson.local').split('@')[-1]
GIA_LAP_DUOC = {'imu_slope', 'imu_test', 'pid_test', 'lidar_test', 'line_debug', 'camera_test', 'run_car'}


BUOC_CALIB = {
    'birdeye_chessboard': "BƯỚC 4: bird's eye bằng BÀN CỜ nằm trên sàn, sát mũi xe. Tự tính chiều cao + độ chúi camera, ô cờ phải thành ô vuông. Hỏi quang tâm trước tâm xe (đo bằng thước).",
    'line_debug': 'BƯỚC 5: kiểm bằng mắt. Đặt xe giữa làn: ô "6 fit" phải có đường ĐỎ nằm đúng vạch phải, d ≈ 0,20 m.',
}


def ds_script():
    nhom = [('Xe tự hành', [GOC / 'run_car.py']),
            ("Bird's eye (bước 4 và 5)",
             [GOC / 'tools/birdeye_chessboard.py', GOC / 'tools/line_debug.py']),
            ('Công cụ hiệu chuẩn / kiểm tra',
             [f for f in sorted(GOC.glob('tools/*.py')) if f.stem not in ('__init__', 'line_debug', 'detector_laptop', 'trt_server', 'control_panel', 'birdeye_chessboard')])]
    ra = []
    for ten_nhom, files in nhom:
        muc = []
        for f in files:
            txt = f.read_text(encoding='utf-8')
            doc = re.search(r'"""(.*?)"""', txt, re.S)
            mo_ta = doc.group(1).strip().splitlines()[0] if doc else ''
            mo_ta = BUOC_CALIB.get(f.stem, mo_ta)
            noi = re.search(r'CHAY_TREN: (host|laptop)', txt)
            sua = GOC / 'autonomy/params.py' if f.stem == 'run_car' else f     # tham số xe tự hành gom ở đây
            ts = re.search(r'(# ===== THAM SỐ.*?)\n# =+\n', sua.read_text(encoding='utf-8'), re.S)
            muc.append({'ten': f.stem, 'file': str(sua.relative_to(GOC)), 'mo_ta': mo_ta,
                        'noi': noi.group(1) if noi else 'xe', 'tham_so': ts.group(1) if ts else '',
                        'tay': '--tay' in txt, 'gia_lap': f.stem in GIA_LAP_DUOC})
        ra.append({'nhom': ten_nhom, 'muc': muc})
    return ra


class TienTrinh:
    def __init__(self):
        self.proc = None
        self.dong = []
        self.ten = ''
        self.kieu = ''
        self.ma = None
        self.lock = threading.Lock()

    def them(self, s):
        with self.lock:
            self.dong.append(s)
            if len(self.dong) > 20000:
                del self.dong[:5000]

    def chay_lenh(self, ten, cmd):
        """Chạy một lệnh bất kỳ (dùng cho trang calib), log đổ vào cùng chỗ."""
        if self.proc and self.proc.poll() is None:
            return False, f'Đang chạy {self.ten}: bấm DỪNG trước'
        self.ten, self.kieu, self.ma = ten, 'xe', None
        self.them('')
        self.them(f'########## {time.strftime("%H:%M:%S")}  $ {cmd[-1] if cmd[0] == "bash" and cmd[1] == "-c" else " ".join(cmd)}')
        self.proc = subprocess.Popen(cmd, cwd=GOC, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, start_new_session=True,
                                     env=dict(os.environ, PYTHONUNBUFFERED='1'))
        threading.Thread(target=self._doc, args=(self.proc,), daemon=True).start()
        return True, 'ok'

    def chay(self, ten, kieu, them):
        if self.proc and self.proc.poll() is None:
            return False, f'Đang chạy {self.ten}: bấm DỪNG trước'
        tim = [GOC / 'tools' / f'{ten}.py', GOC / f'{ten}.py']
        f = next((p for p in tim if p.exists()), None)
        if f is None:
            return False, f'không có script {ten}'
        if kieu == 'gia_lap':
            cmd = [sys.executable, '-u', str(f), '--gia-lap', *them]
        else:
            cmd = ['bash', str(GOC / 'scripts/car.sh'), ten, *them]
        self.ten, self.kieu, self.ma = ten, kieu, None
        self.them('')
        self.them(f'########## {time.strftime("%H:%M:%S")}  $ {" ".join(cmd[1:] if kieu != "gia_lap" else cmd[2:])}')
        env = dict(os.environ, PYTHONUNBUFFERED='1')
        self.proc = subprocess.Popen(cmd, cwd=GOC, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, start_new_session=True, env=env)
        threading.Thread(target=self._doc, args=(self.proc,), daemon=True).start()
        return True, 'ok'

    def _doc(self, proc):
        for raw in iter(proc.stdout.readline, b''):
            self.them(raw.decode('utf-8', 'replace').replace('\r', '').rstrip('\n'))
        self.ma = proc.wait()
        self.them(f'########## kết thúc, mã {self.ma}')

    def dung(self):
        p = self.proc
        if p and p.poll() is None:
            self.them('########## DỪNG KHẨN')
            try:
                if self.kieu == 'gia_lap':
                    os.killpg(p.pid, signal.SIGINT)
                else:
                    p.stdin.write(b'\x03')          # Ctrl+C qua ssh -tt tới script trên xe
                    p.stdin.flush()
            except OSError:
                pass
        if self.kieu != 'gia_lap':
            subprocess.Popen(['bash', str(GOC / 'scripts/car.sh'), 'dung'], cwd=GOC,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        def chot():
            time.sleep(4)
            if p and p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except OSError:
                    pass
        threading.Thread(target=chot, daemon=True).start()


TT = TienTrinh()


def mo_file(rel):
    f = GOC / rel
    for ed in ('code', 'gedit', 'xdg-open'):
        if shutil.which(ed):
            subprocess.Popen([ed, str(f)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return ed
    return None


def _chon_bev():
    """Khối 4b (chọn tham số bird's eye): dùng chung cho bảng điều khiển chính và trang calib."""
    return (GOC / 'tools/birdeye_select.html').read_text(encoding='utf-8')


def _tham_so_bev(d):
    return {'x_gan': float(d['xg']), 'x_xa': float(d['xx']), 'y_trai': float(d['yt']),
            'y_phai': float(d['yp']), 'mm_px': float(d['mm'])}


class DenLaptop:
    """Giữ tiến trình tools/detector_laptop.py (model nhận biển/đèn chạy trên laptop, kết quả gửi về xe)."""

    def __init__(self):
        self.proc = None
        self.dong = 'chưa bật'

    def chay(self):
        return self.proc is not None and self.proc.poll() is None

    def bat(self):
        if self.chay():
            return
        # dọn bản cũ còn sống (bảng điều khiển trước bị tắt đột ngột) kẻo giữ cổng 8096
        for pid in subprocess.run(['pgrep', '-f', '^python3 -u tools/detector_laptop.py'], capture_output=True, text=True).stdout.split():
            try:
                os.kill(int(pid), signal.SIGTERM)
            except OSError:
                pass
        time.sleep(0.5)

        def con_chet_theo_cha():       # bảng điều khiển chết (kể cả bị kill) thì detector_laptop cũng tự tắt
            import ctypes
            ctypes.CDLL('libc.so.6').prctl(1, signal.SIGTERM)   # PR_SET_PDEATHSIG
        # cần môi trường ROS 2 (rclpy) của hệ thống để đi bằng DDS
        self.proc = subprocess.Popen(['bash', '-c', 'source /opt/ros/humble/setup.bash && exec python3 -u tools/detector_laptop.py'],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(GOC), preexec_fn=con_chet_theo_cha)
        self.dong = 'đang khởi động model...'

        def doc(p):
            for l in p.stdout:
                self.dong = l.strip()[:200]
            self.dong = 'đã tắt' if self.dong.startswith('[detector_laptop]') or not self.dong else self.dong
        threading.Thread(target=doc, args=(self.proc,), daemon=True).start()

    def tat(self):
        if self.chay():
            self.proc.terminate()
        self.dong = 'đã tắt'


DL = DenLaptop()


class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        pass

    def _gui(self, body, kieu='application/json', ma=200):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(ma)
        self.send_header('Content-Type', kieu)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _bev_get(self, path, q):
        """Chọn tham số bird's eye (bước 4b): danh sách lần chốt, gợi ý vùng nhìn, xem thử."""
        from urllib.parse import parse_qs
        a = {k: v[0] for k, v in parse_qs(q).items()}
        try:
            from core import birdeye_select as B
            ds = [int(v) for v in a.get('ds', '').split(',') if v]
            if path == '/api/bev/ung_vien':
                return self._gui([{k: u[k] for k in ('n', 'gio', 'cao_m', 'pitch_do', 'roll_do', 'yaw_do', 'rms_px',
                                                      'o_vuong', 'co_o', 'dat', 'tong', 'ket')} for u in B.liet_ke()])
            if path == '/api/bev/de_xuat':
                return self._gui(B.de_xuat(ds))
            if path == '/api/bev/xem.jpg':
                import cv2
                ok, buf = cv2.imencode('.jpg', B.ve_thu(ds, _tham_so_bev(a)), [cv2.IMWRITE_JPEG_QUALITY, 85])
                return self._gui(buf.tobytes(), 'image/jpeg')
        except Exception as e:  # noqa: BLE001
            return self._gui({'loi': str(e)}, 'application/json', 400)
        self._gui(b'not found', 'text/plain', 404)

    def do_GET(self):
        path, _, q = self.path.partition('?')
        if path == '/':
            return self._gui(TRANG.replace('__XE_IP__', XE_IP).replace('__CHON_BEV__', _chon_bev()).encode(),
                             'text/html; charset=utf-8')
        if path.startswith('/calib_new/') and '..' not in path:
            f = GOC / path.lstrip('/')
            if f.is_file():
                return self._gui(f.read_bytes(), 'image/jpeg')
            return self._gui(b'', 'image/jpeg', 404)
        if path.startswith('/api/bev/'):
            return self._bev_get(path, q)
        if path == '/api/calib_trang_thai':
            return self._gui({'co_calib': (GOC / 'config/camera.yaml').exists(),
                              'co_ngoai': (GOC / 'config/extrinsics.yaml').exists(),
                              'anh': sorted(p.name for p in (GOC / 'calib_new').glob('anh_*'))[-1:] if (GOC / 'calib_new').exists() else [],
                              'moi_cam': (GOC / 'calib_new/camera.yaml').exists(),
                              'moi_ngoai': (GOC / 'calib_new/extrinsics.yaml').exists()})
        if path == '/api/detector_laptop':
            return self._gui({'chay': DL.chay(), 'dong': DL.dong})
        if path == '/api/roi':
            from core.detector import doc_roi
            r = doc_roi()
            return self._gui({'y1': r[1], 'y2': r[3]})
        if path == '/api/ds':
            return self._gui(ds_script())
        if path == '/api/log':
            tu = int(dict(x.split('=') for x in q.split('&') if '=' in x).get('tu', 0))
            with TT.lock:
                n = len(TT.dong)
                dong = TT.dong[tu:] if tu <= n else TT.dong[:]
            chay = TT.proc is not None and TT.proc.poll() is None
            return self._gui({'tu': n, 'dong': dong, 'chay': chay, 'ten': TT.ten,
                              'kieu': TT.kieu, 'ma': TT.ma})
        self._gui(b'not found', 'text/plain', 404)

    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        d = json.loads(self.rfile.read(n) or b'{}')
        if self.path == '/api/chay':
            them = [x for x in d.get('them', '').split() if x]
            ok, tb = TT.chay(d['ten'], d.get('kieu', 'xe'), them)
            return self._gui({'ok': ok, 'tb': tb})
        if self.path == '/api/bev/luu':
            try:
                from core import birdeye_select as B
                tom = B.ghi_lua_chon([int(v) for v in d.get('ds', [])], _tham_so_bev(d))
                return self._gui({'ok': True, 'tb': f"Đã lưu calib_new/extrinsics.yaml + bev.yaml (lần {tom['lan']}, "
                                  f"cao {tom['cao_m'] * 100:.1f} cm). Bấm 'Dùng số này' để đưa vào config/."})
            except Exception as e:  # noqa: BLE001
                return self._gui({'ok': False, 'tb': str(e)})
        if self.path == '/api/calib':
            lenh = {'keo': ['bash', 'scripts/car.sh', 'keo'],
                    'nhan_ngoai': ['bash', '-c', 'cp -v calib_new/extrinsics.yaml config/extrinsics.yaml && '
                                   'cp -v calib_new/bev.yaml config/bev.yaml']}.get(d.get('buoc'))
            if lenh is None:
                return self._gui({'ok': False, 'tb': 'lệnh lạ'})
            ok, tb = TT.chay_lenh(f'bird\'s eye: {d["buoc"]}', lenh)
            return self._gui({'ok': ok, 'tb': tb})
        if self.path == '/api/detector_laptop':
            DL.bat() if d.get('bat') else DL.tat()
            return self._gui({'chay': DL.chay(), 'dong': DL.dong})
        if self.path == '/api/roi':
            try:
                from core.detector import ghi_roi
                ghi_roi(d['y1'], d['y2'])
                return self._gui({'ok': True, 'tb': f"Đã lưu ROI hàng {int(d['y1'])}..{int(d['y2'])}. Chạy lại xe là có hiệu lực."})
            except (ValueError, KeyError) as e:
                return self._gui({'ok': False, 'tb': str(e)})
        if self.path == '/api/dung':
            TT.dung()
            return self._gui({'ok': True})
        if self.path == '/api/mo':
            ed = mo_file(d['file'])
            return self._gui({'ok': bool(ed), 'tb': f'mở bằng {ed}' if ed else 'không có trình sửa'})
        self._gui(b'not found', 'text/plain', 404)


TRANG = r'''<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Champion - bảng điều khiển</title>
<style>
:root{--nen:#0f1115;--o:#181b22;--o2:#20242d;--chu:#e6e8ec;--mo:#9aa3b2;--xanh:#3ddc84;--do:#ff4d4f;--vang:#ffc53d;--lam:#4ea1ff}
*{box-sizing:border-box}body{margin:0;background:var(--nen);color:var(--chu);font:14px system-ui,sans-serif;height:100vh;display:flex}
#trai{width:400px;min-width:340px;overflow:auto;border-right:1px solid #2a2f3a;padding:12px}
#phai{flex:1;display:flex;flex-direction:column;min-width:0}
h1{font-size:18px;margin:0 0 4px}.mo{color:var(--mo);font-size:12px}
.nut{border:0;border-radius:6px;padding:5px 9px;cursor:pointer;font-weight:600;font-size:12px;background:var(--o2);color:var(--chu)}
.nut:hover{filter:brightness(1.3)}.xe{background:#1f5f3a}.gl{background:#2b3a5c}.tiep{background:#5c4a1f}
#dung{width:100%;background:var(--do);color:#fff;font-size:18px;padding:12px;margin:8px 0}
#full{width:100%;background:#1f7a46;color:#fff;font-size:16px;padding:10px}
.nhom{margin-top:14px;font-weight:700;color:var(--lam)}
.muc{background:var(--o);border-radius:8px;padding:8px;margin-top:6px}
.muc .ten{font-weight:700;cursor:pointer}.muc .ten:hover{color:var(--lam)}.muc .nuts{margin-top:6px;display:flex;gap:5px;flex-wrap:wrap}
pre.ts{background:#0b0d11;color:#cfd6e4;font-size:11px;padding:8px;border-radius:6px;white-space:pre-wrap;margin:6px 0 0;display:none}
#tren{padding:10px;border-bottom:1px solid #2a2f3a;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
#trangthai{font-weight:700}.so{font-family:ui-monospace,monospace;background:var(--o);padding:4px 8px;border-radius:6px}
#hinh{display:flex;gap:8px;padding:8px;overflow-x:auto;min-height:220px;align-items:flex-start}
#hinh figure{margin:0;background:var(--o);border-radius:8px;padding:4px}#hinh figcaption{font-size:11px;color:var(--mo)}
#hinh img{max-height:300px;max-width:640px;display:block}
#xe3d{width:250px;height:230px;perspective:700px;display:flex;align-items:center;justify-content:center}
#nen{width:150px;height:150px;position:relative;transform-style:preserve-3d;transform:rotateX(62deg) rotateZ(-38deg)}
#san{position:absolute;inset:0;transform:translateZ(-18px);border:1px solid #3a4352;background:repeating-linear-gradient(0deg,#2a303b 0 1px,transparent 1px 25px),repeating-linear-gradient(90deg,#2a303b 0 1px,transparent 1px 25px),rgba(60,70,90,.15)}
#hop{width:90px;height:50px;position:absolute;left:30px;top:50px;transform-style:preserve-3d;transition:transform .08s linear}
#hop div{position:absolute;border:1px solid #6fa8ff;background:rgba(78,161,255,.25)}
#log{flex:1;overflow:auto;background:#07080b;font:12px ui-monospace,monospace;padding:8px;white-space:pre-wrap;margin:0}
.buoc{background:var(--o);border-radius:10px;padding:10px;margin-top:8px}.buoc h2{font-size:14px;margin:0 0 6px}.buoc ol{margin:6px 0;padding-left:20px;line-height:1.5}.nut.phu{background:var(--o2)}
#panelcalib{background:#12151b;border-bottom:1px solid #2a2f3a;padding:6px 10px;max-height:46vh;overflow:auto}#panelcalib summary{cursor:pointer;font-weight:700;color:var(--lam)}
.loi{color:var(--do)}.vao{color:var(--xanh)}.dat{color:var(--xanh)}.kdat{color:var(--vang)}
input{background:var(--o2);color:var(--chu);border:1px solid #333;border-radius:6px;padding:5px;width:100%}
</style></head><body>
<div id="trai">
 <h1>CHAMPION</h1><div class="mo">Xe: __XE_IP__ · log realtime bên phải · sửa tham số = bấm "Sửa", Ctrl+S, chạy lại</div>
 <button id="dung" class="nut" onclick="dung()">■ DỪNG KHẨN</button>
 <button id="full" class="nut" onclick="chay('run_car','xe')">▶ CHẠY TỰ HÀNH (trên xe)</button>
 <div style="margin-top:8px"><div class="mo">Tham số thêm cho lần bấm tới (ví dụ <code>--v 0.30</code>, <code>--vung 380 60 560 260</code>):</div>
 <input id="them" placeholder="để trống nếu không cần"></div>
 <div class="muc" style="margin-top:8px"><b>Nhận biển + đèn bằng LAPTOP</b> <button id="dlnut" class="nut gl" onclick="dlDoi()">...</button><div id="dltt" class="mo" style="margin-top:4px"></div><div class="mo">Model chạy trên laptop (~50 ms/khung, Jetson ~850 ms). Xe tự hành (DEN_NGUON = "laptop") và onnx_test lấy kết quả từ đây. ROI Lưu là có hiệu lực ngay.</div></div>
 <div id="ds"></div>
</div>
<div id="phai">
 <div id="tren"><span id="trangthai">Chưa chạy gì</span>
  <span class="so" id="gd">đang chạy -</span><span class="so" id="vw">v - w -</span>
  <span class="so" id="goc">yaw - pitch - roll -</span><span class="so" id="qd">- m</span></div>
 <div id="hinh"><figure><div id="xe3d"><div id="nen"><div id="san"></div><div id="hop"></div></div></div><figcaption id="cap3d">Mô hình xe theo IMU (mũi xe màu cam; mặt sàn đứng yên)</figcaption></figure></div>
 <details id="panelcalib" open><summary>Bird's eye: chọn ứng viên tốt nhất (sau khi chạy birdeye_chessboard)</summary>
  <div class="mo" style="margin:6px 0">Chạy <b>birdeye_chessboard</b> ở nhóm "Bird's eye" bên trái (đặt bàn cờ nhiều chỗ, mỗi chỗ một lần chốt), rồi chọn ở đây.</div>
  __CHON_BEV__
 </details>
 <pre id="log"></pre>
</div>
<script>
let tu=0, kieu='xe', hinhDangCo=new Set();
const $=id=>document.getElementById(id);
function hop(){const W=90,D=50,H=30,f=[['translateZ(15px)',W,D,'#6fa8ff'],['rotateX(180deg) translateZ(15px)',W,D,'#6fa8ff'],
 ['rotateY(90deg) translateZ(45px)',H,D,'#ff9f43'],['rotateY(-90deg) translateZ(45px)',H,D,'#6fa8ff'],
 ['rotateX(90deg) translateZ(25px)',W,H,'#6fa8ff'],['rotateX(-90deg) translateZ(25px)',W,H,'#6fa8ff']];
 $('hop').innerHTML=f.map(([t,w,h,c])=>`<div style="width:${w}px;height:${h}px;left:${(90-w)/2}px;top:${(50-h)/2}px;transform:${t};background:${c}${c==='#ff9f43'?'dd':'55'};border-color:${c}"></div>`).join('');}
function eje(c,txt,css,lb){return `<div style="position:absolute;border:0;background:${c};${css}"><b style="position:absolute;border:0;color:${c};font:700 12px sans-serif;white-space:nowrap;${lb}">${txt}</b></div>`;}
function truc(){
 // trục THÂN XE (xoay theo xe): x = tiến (đỏ), y = trái (xanh lá), z = lên (xanh dương); gốc ở tâm xe
 $('hop').insertAdjacentHTML('beforeend',
  eje('#ff5d5d','x','left:45px;top:23.5px;width:80px;height:3px;','left:82px;top:-7px;')+
  eje('#3ddc84','y','left:43.5px;top:-35px;width:3px;height:60px;transform:none;','left:6px;top:-14px;')+
  eje('#4ea1ff','z','left:43.5px;top:25px;width:3px;height:70px;transform-origin:50% 0;transform:rotateX(90deg);','left:6px;top:70px;'));
 // trục SÀN (đứng yên, xám) để thấy xe nghiêng so với mặt đất
 $('san').insertAdjacentHTML('beforeend',
  eje('#8d96a5','X0','left:75px;top:74px;width:65px;height:2px;opacity:.7;','left:67px;top:-7px;')+
  eje('#8d96a5','Y0','left:74px;top:10px;width:2px;height:65px;opacity:.7;','left:5px;top:-14px;')+
  eje('#8d96a5','Z0','left:74px;top:75px;width:2px;height:60px;transform-origin:50% 0;transform:rotateX(90deg);opacity:.7;','left:5px;top:60px;'));
}
hop();truc();
function xe3d(p,r,y){ // yaw + = quay trái, pitch + = ngóc mũi lên, roll + = nghiêng sang phải (đúng quy ước IMU của xe)
 $('hop').style.transform=`rotateZ(${-y}deg) rotateY(${-p}deg) rotateX(${-r}deg)`;
 const f=v=>(v>=0?'+':'')+(+v).toFixed(1)+'°';
 $('cap3d').innerHTML=`<span style="color:#ff5d5d">x · roll ${f(r)}</span> &nbsp; <span style="color:#3ddc84">y · pitch ${f(p)}</span> &nbsp; <span style="color:#4ea1ff">z · yaw ${f(y)}</span><br>trục màu = gắn vào xe, trục xám = mặt sàn đứng yên`;}
const ROI_HTML=`<div class="roi" style="margin-top:8px;padding-top:8px;border-top:1px solid #2a2f3a"><b>ROI nhận biển báo + đèn</b>
  <div class="mo" style="margin:4px 0">Chỉ nhận trong dải hàng pixel y1..y2 của ảnh 640x480 (0 = mép trên, 480 = mép dưới). Mặc định 240..480. Camera hiện trực tiếp từ laptop (bật "Nhận biển + đèn bằng LAPTOP"); Lưu xong hình "den" bên phải đổi ngay, Lần chạy xe sau dùng ROI này.</div>
  <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
   <label>y1<input id="roi1" type="number" min="0" max="478" step="5" style="width:64px" oninput="roiVe()"></label>
   <label>y2<input id="roi2" type="number" min="1" max="480" step="5" style="width:64px" oninput="roiVe()"></label>
   <button class="nut gl" onclick="roiLuu()">Lưu ROI</button>
   <div style="position:relative;width:64px;height:48px;background:#222;border:1px solid #444"><div id="roihop" style="position:absolute;left:0;right:0;background:rgba(78,161,255,.45)"></div></div>
  </div>
  <div id="roicamkhung" style="position:relative;width:100%;max-width:370px;aspect-ratio:4/3;background:#000;margin-top:6px;cursor:crosshair" onclick="roiClick(event)">
   <img id="roicam" onerror="this.removeAttribute('src');$('roicho').style.display='block'" style="width:100%;height:100%;display:block;pointer-events:none">
   <div id="roitren" style="position:absolute;left:0;right:0;top:0;background:rgba(0,0,0,.55);pointer-events:none"></div>
   <div id="roiduoi" style="position:absolute;left:0;right:0;bottom:0;background:rgba(0,0,0,.55);pointer-events:none"></div>
   <div id="roivien" style="position:absolute;left:0;right:0;border:2px solid #4ea1ff;pointer-events:none"></div>
   <div id="roicho" class="mo" style="position:absolute;left:6px;top:6px">Chưa có hình: bấm Xe ở onnx_test để bật camera</div>
  </div>
  <div class="mo">Bấm lên hình = đặt y1 (mép trên). Giữ Shift + bấm = đặt y2 (mép dưới). Vùng sáng là vùng được nhận.</div>
  <input id="roir" type="range" min="0" max="478" step="5" style="margin-top:6px" oninput="$('roi1').value=this.value;roiVe()">
  <div id="roitb" class="mo"></div></div>`;
async function ds(){const r=await (await fetch('/api/ds')).json();$('ds').innerHTML=r.map(n=>`<div class="nhom">${n.nhom}</div>`+n.muc.map(m=>`
 <div class="muc"><div class="ten" onclick="ts('${m.ten}')">${m.ten}</div><div class="mo">${m.mo_ta}</div>
 <div class="nuts">${m.noi==='laptop'?`<button class="nut xe" onclick="chay('${m.ten}','xe')">Chạy (laptop)</button>`:`<button class="nut xe" onclick="chay('${m.ten}','xe')">Xe</button>`}
 ${m.tay?`<button class="nut tiep" onclick="chay('${m.ten}','xe','--tay')">Tay (đẩy xe)</button>`:''}
 ${m.gia_lap?`<button class="nut gl" onclick="chay('${m.ten}','gia_lap')">Giả lập</button>`:''}
 <button class="nut" onclick="mo('${m.file}')">Sửa</button></div>
 <pre class="ts" id="ts_${m.ten}">${(m.tham_so||'(không có khối THAM SỐ)').replace(/</g,'&lt;')}</pre></div>`).join('')).join('');const o=$('ts_onnx_test');if(o){o.parentElement.insertAdjacentHTML('beforeend',ROI_HTML);roiTai();}}
function ts(t){const e=$('ts_'+t);e.style.display=e.style.display==='block'?'none':'block';}
function thamCalib(ten){ // bước 4 cần 2 số đo bằng thước: hỏi, nhớ lại lần sau
 if(ten!=='birdeye_chessboard'||$('them').value.trim())return null;
 const cao=prompt('Chiều cao camera so với mặt đường (m), đo bằng thước (chỉ để đối chiếu với số tự tính từ bàn cờ; để trống = bỏ qua):',localStorage.getItem('cao')||'0.110');if(cao===null)return false;
 const truoc=prompt('Quang tâm camera nằm TRƯỚC tâm xe bao nhiêu m (sau tâm xe thì nhập số âm), đo bằng thước:',localStorage.getItem('truoc')||'');if(!truoc)return false;
 localStorage.setItem('cao',cao);localStorage.setItem('truoc',truoc);return `--truoc ${truoc}`+(cao?` --cao ${cao}`:'');}
async function chay(ten,k,extra=''){kieu=k;let them=($('them').value+' '+extra).trim();const tc=thamCalib(ten);if(tc===false)return;if(tc)them=tc;const r=await (await fetch('/api/chay',{method:'POST',body:JSON.stringify({ten,kieu:k,them})})).json();
 if(!r.ok)alert(r.tb);hinhDangCo.clear();document.querySelectorAll('#hinh figure.dong').forEach(e=>e.remove());}
function roiCam(){if($('roicam')&&!$('roicam').src){$('roicam').src='http://127.0.0.1:8096/img/cam.mjpg';$('roicho').style.display='none';}}
function roiClick(e){const r=$('roicamkhung').getBoundingClientRect();const y=Math.round((e.clientY-r.top)/r.height*480/5)*5;if(e.shiftKey)$('roi2').value=Math.min(480,Math.max(y,1));else $('roi1').value=Math.min(478,Math.max(y,0));roiVe();}
function roiVe(){const a=+$('roi1').value,b=+$('roi2').value;$('roir').value=a;$('roihop').style.top=(a/480*48)+'px';$('roihop').style.height=(Math.max(b-a,0)/480*48)+'px';if($('roitren')){$('roitren').style.height=(a/4.8)+'%';$('roiduoi').style.height=(Math.max(480-b,0)/4.8)+'%';$('roivien').style.top=(a/4.8)+'%';$('roivien').style.height=(Math.max(b-a,0)/4.8)+'%';}}
async function roiTai(){try{const r=await (await fetch('/api/roi')).json();$('roi1').value=r.y1;$('roi2').value=r.y2;roiVe();}catch(e){}}
async function roiLuu(){const r=await (await fetch('/api/roi',{method:'POST',body:JSON.stringify({y1:+$('roi1').value,y2:+$('roi2').value})})).json();$('roitb').textContent=r.tb;$('roitb').style.color=r.ok?'var(--xanh)':'var(--do)';}
async function dung(){await fetch('/api/dung',{method:'POST',body:'{}'});}
async function mo(f){const r=await (await fetch('/api/mo',{method:'POST',body:JSON.stringify({file:f})})).json();if(!r.ok)alert(r.tb);}
function to(s){const e=s.replace(/&/g,'&amp;').replace(/</g,'&lt;');
 if(/LỖI|MẤT|KHÔNG ĐẠT|Traceback|Error|!!/.test(s))return `<span class="loi">${e}</span>`;
 if(/==>|<==|ĐẠT|XONG|ĐÈN XANH|#####/.test(s))return `<span class="vao">${e}</span>`;return e;}
let dlChay=false;
function dlHinh(){ // hình từ LAPTOP (nhanh, không qua xe): ảnh camera nén trong ô ROI + hình đèn/biển trong khung hình
 if(!dlChay)return;roiCam();
 for(const [k,t] of [['den','den (laptop, ROI)'],['lidar','lidar (laptop)'],['vach','vach: 8 ô dò vạch (laptop)'],['birdeye','bird\'s eye + vạch (laptop)']]){
  if(!$('hl_'+k))$('hinh').insertAdjacentHTML('beforeend',`<figure id="hl_${k}"><img src="http://127.0.0.1:8096/img/${k}.mjpg" onerror="const i=this;setTimeout(()=>{i.src='http://127.0.0.1:8096/img/${k}.mjpg?t='+Date.now()},1000)"><figcaption>${t}</figcaption></figure>`);}}
async function dlTai(){try{const r=await (await fetch('/api/detector_laptop')).json();dlChay=r.chay;dlHinh();$('dlnut').textContent=r.chay?'Tắt':'Bật';$('dltt').textContent=(r.chay?'● ':'○ ')+r.dong;$('dltt').style.color=r.chay?'var(--xanh)':'var(--mo)';}catch(e){}setTimeout(dlTai,2000);}
async function dlDoi(){await fetch('/api/detector_laptop',{method:'POST',body:JSON.stringify({bat:!dlChay})});}
async function vongLog(){try{const r=await (await fetch('/api/log?tu='+tu)).json();
 if(r.dong.length){const L=$('log'),cuoi=L.scrollTop+L.clientHeight>=L.scrollHeight-30;L.insertAdjacentHTML('beforeend',r.dong.map(to).join('\n')+'\n');tu=r.tu;if(cuoi)L.scrollTop=L.scrollHeight;}
 $('trangthai').textContent=r.chay?`Đang chạy: ${r.ten} (${r.kieu==='gia_lap'?'giả lập':'trên xe'})`:(r.ten?`Đã xong: ${r.ten} (mã ${r.ma})`:'Chưa chạy gì');
 $('trangthai').style.color=r.chay?'var(--xanh)':'var(--mo)';kieu=r.kieu||kieu;}catch(e){} setTimeout(vongLog,400);}
async function vongXe(){const host=kieu==='gia_lap'?'127.0.0.1':'__XE_IP__';
 try{const c=new AbortController();setTimeout(()=>c.abort(),800);const t=await (await fetch(`http://${host}:8097/trang_thai.json`,{signal:c.signal})).json();
  $('gd').textContent='đang chạy '+(t.chuong_trinh||'-')+(t.chi_tiet?' | '+t.chi_tiet:'');
  $('vw').textContent=`v ${t.v??'-'} w ${t.w??'-'}`;$('goc').textContent=`yaw ${t.yaw??'-'}° pitch ${t.pitch??'-'}° roll ${t.roll??'-'}°`;
  $('qd').textContent=(t.quang_duong??'-')+' m';
  xe3d(t.pitch||0,t.roll||0,t.yaw||0);
  for(const a of (t.anh||[])){if(a==='roi_cam'||a==='cam'||(dlChay&&['den','lidar','vach','birdeye'].includes(a))){continue;}if(hinhDangCo.has(a))continue;hinhDangCo.add(a);
   $('hinh').insertAdjacentHTML('beforeend',`<figure class="dong"><img src="http://${host}:8097/img/${a}.mjpg"><figcaption>${a}</figcaption></figure>`);}
 }catch(e){} setTimeout(vongXe,120);}
async function buoc(b){const r=await (await fetch('/api/calib',{method:'POST',body:JSON.stringify({buoc:b})})).json();if(!r.ok)alert(r.tb);}
ds();dlTai();vongLog();vongXe();
</script></body></html>'''



def main():
    trang = ''
    try:
        srv = ThreadingHTTPServer(('127.0.0.1', CONG), H)
    except OSError:
        # Cổng đang bị bản bảng điều khiển khác giữ: không văng lỗi, mở trình duyệt vào bản đó.
        print(f'Bảng điều khiển ĐÃ đang chạy ở http://127.0.0.1:{CONG}{trang}, mở trình duyệt vào đó.\n'
              f'(Muốn khởi động lại hẳn: pkill -f tools/control_panel.py rồi chạy lại.)', flush=True)
        if shutil.which('xdg-open') and os.environ.get('DISPLAY'):
            subprocess.Popen(['xdg-open', f'http://127.0.0.1:{CONG}{trang}'],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    print(f'Bảng điều khiển: http://127.0.0.1:{CONG}{trang}  (xe {XE_IP}). Ctrl+C để tắt.', flush=True)
    # detector_laptop KHÔNG tự bật nữa: model đang chạy trên Jetson (DEN_NGUON = 'xe'); bấm Bật ở cột trái nếu muốn thử bản laptop
    if shutil.which('xdg-open') and os.environ.get('DISPLAY'):
        subprocess.Popen(['xdg-open', f'http://127.0.0.1:{CONG}{trang}'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    signal.signal(signal.SIGTERM, lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))   # kill thường cũng dọn detector_laptop
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        TT.dung()
    finally:
        DL.tat()


if __name__ == '__main__':
    main()
