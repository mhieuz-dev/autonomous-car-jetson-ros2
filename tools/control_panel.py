#!/usr/bin/env python3
"""BẢNG ĐIỀU KHIỂN Champion trên laptop: http://127.0.0.1:8099
RUNS_ON: laptop

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PORT = 8099
CAR_IP = os.environ.get('CRC_HOST', 'jetson@jetson.local').split('@')[-1]
SIM_CAPABLE = {'imu_slope', 'imu_test', 'pid_test', 'lidar_test', 'line_debug', 'camera_test', 'run_car'}


CALIB_STEPS = {
    'birdeye_chessboard': "BƯỚC 4: bird's eye bằng BÀN CỜ nằm trên sàn, sát mũi xe. Tự tính chiều cao + độ chúi camera, ô cờ phải thành ô vuông. Hỏi quang tâm trước tâm xe (đo bằng thước).",
    'line_debug': 'BƯỚC 5: kiểm bằng mắt. Đặt xe giữa làn: ô "6 fit" phải có đường ĐỎ nằm đúng vạch phải, d ≈ 0,20 m.',
}


def list_scripts():
    groups = [('Xe tự hành', [ROOT / 'run_car.py']),
            ("Bird's eye (bước 4 và 5)",
             [ROOT / 'tools/birdeye_chessboard.py', ROOT / 'tools/line_debug.py']),
            ('Công cụ hiệu chuẩn / kiểm tra',
             [f for f in sorted(ROOT.glob('tools/*.py')) if f.stem not in ('__init__', 'line_debug', 'detector_laptop', 'trt_server', 'control_panel', 'birdeye_chessboard')])]
    out = []
    for group_name, files in groups:
        entries = []
        for f in files:
            txt = f.read_text(encoding='utf-8')
            doc = re.search(r'"""(.*?)"""', txt, re.S)
            desc = doc.group(1).strip().splitlines()[0] if doc else ''
            desc = CALIB_STEPS.get(f.stem, desc)
            runs_on = re.search(r'RUNS_ON: (host|laptop)', txt)
            edit_path = ROOT / 'autonomy/params.py' if f.stem == 'run_car' else f     # tham số xe tự hành gom ở đây
            ts = re.search(r'(# ===== THAM SỐ.*?)\n# =+\n', edit_path.read_text(encoding='utf-8'), re.S)
            entries.append({'name': f.stem, 'file': str(edit_path.relative_to(ROOT)), 'desc': desc,
                        'runs_on': runs_on.group(1) if runs_on else 'car', 'params': ts.group(1) if ts else '',
                        'manual': '--manual' in txt, 'sim': f.stem in SIM_CAPABLE})
        out.append({'group': group_name, 'items': entries})
    return out


class ProcessRunner:
    def __init__(self):
        self.proc = None
        self.lines = []
        self.name = ''
        self.kind = ''
        self.code = None
        self.lock = threading.Lock()

    def add_line(self, s):
        with self.lock:
            self.lines.append(s)
            if len(self.lines) > 20000:
                del self.lines[:5000]

    def run_cmd(self, name, cmd):
        """Chạy một lệnh bất kỳ (dùng cho trang calib), log đổ vào cùng chỗ."""
        if self.proc and self.proc.poll() is None:
            return False, f'Đang chạy {self.name}: bấm DỪNG trước'
        self.name, self.kind, self.code = name, 'car', None
        self.add_line('')
        self.add_line(f'########## {time.strftime("%H:%M:%S")}  $ {cmd[-1] if cmd[0] == "bash" and cmd[1] == "-c" else " ".join(cmd)}')
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, start_new_session=True,
                                     env=dict(os.environ, PYTHONUNBUFFERED='1'))
        threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()
        return True, 'ok'

    def run(self, name, kind, extra):
        if self.proc and self.proc.poll() is None:
            return False, f'Đang chạy {self.name}: bấm DỪNG trước'
        candidates = [ROOT / 'tools' / f'{name}.py', ROOT / f'{name}.py']
        f = next((p for p in candidates if p.exists()), None)
        if f is None:
            return False, f'không có script {name}'
        if kind == 'sim':
            cmd = [sys.executable, '-u', str(f), '--sim', *extra]
        else:
            cmd = ['bash', str(ROOT / 'scripts/car.sh'), name, *extra]
        self.name, self.kind, self.code = name, kind, None
        self.add_line('')
        self.add_line(f'########## {time.strftime("%H:%M:%S")}  $ {" ".join(cmd[1:] if kind != "sim" else cmd[2:])}')
        env = dict(os.environ, PYTHONUNBUFFERED='1')
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, start_new_session=True, env=env)
        threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()
        return True, 'ok'

    def _read(self, proc):
        for raw in iter(proc.stdout.readline, b''):
            self.add_line(raw.decode('utf-8', 'replace').replace('\r', '').rstrip('\n'))
        self.code = proc.wait()
        self.add_line(f'########## kết thúc, mã {self.code}')

    def stop(self):
        p = self.proc
        if p and p.poll() is None:
            self.add_line('########## DỪNG KHẨN')
            try:
                if self.kind == 'sim':
                    os.killpg(p.pid, signal.SIGINT)
                else:
                    p.stdin.write(b'\x03')          # Ctrl+C qua ssh -tt tới script trên xe
                    p.stdin.flush()
            except OSError:
                pass
        if self.kind != 'sim':
            subprocess.Popen(['bash', str(ROOT / 'scripts/car.sh'), 'stop'], cwd=ROOT,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        def finish_up():
            time.sleep(4)
            if p and p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except OSError:
                    pass
        threading.Thread(target=finish_up, daemon=True).start()


PROC = ProcessRunner()


def open_in_editor(rel):
    f = ROOT / rel
    for ed in ('code', 'gedit', 'xdg-open'):
        if shutil.which(ed):
            subprocess.Popen([ed, str(f)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return ed
    return None


def _bev_block():
    """Khối 4b (chọn tham số bird's eye): dùng chung cho bảng điều khiển chính và trang calib."""
    return (ROOT / 'tools/birdeye_select.html').read_text(encoding='utf-8')


def _bev_params(d):
    return {k: float(d[k]) for k in ('x_near', 'x_far', 'y_left', 'y_right', 'mm_px')}


class LaptopDetector:
    """Giữ tiến trình tools/detector_laptop.py (model nhận biển/đèn chạy trên laptop, kết quả gửi về xe)."""

    def __init__(self):
        self.proc = None
        self.last_line = 'chưa bật'

    def is_running(self):
        return self.proc is not None and self.proc.poll() is None

    def turn_on(self):
        if self.is_running():
            return
        # dọn bản cũ còn sống (bảng điều khiển trước bị tắt đột ngột) kẻo giữ cổng 8096
        for pid in subprocess.run(['pgrep', '-f', '^python3 -u tools/detector_laptop.py'], capture_output=True, text=True).stdout.split():
            try:
                os.kill(int(pid), signal.SIGTERM)
            except OSError:
                pass
        time.sleep(0.5)

        def die_with_parent():       # bảng điều khiển chết (kể cả bị kill) thì detector_laptop cũng tự tắt
            import ctypes
            ctypes.CDLL('libc.so.6').prctl(1, signal.SIGTERM)   # PR_SET_PDEATHSIG
        # cần môi trường ROS 2 (rclpy) của hệ thống để đi bằng DDS
        self.proc = subprocess.Popen(['bash', '-c', 'source /opt/ros/humble/setup.bash && exec python3 -u tools/detector_laptop.py'],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(ROOT), preexec_fn=die_with_parent)
        self.last_line = 'đang khởi động model...'

        def pump(p):
            for l in p.stdout:
                self.last_line = l.strip()[:200]
            self.last_line = 'đã tắt' if self.last_line.startswith('[detector_laptop]') or not self.last_line else self.last_line
        threading.Thread(target=pump, args=(self.proc,), daemon=True).start()

    def turn_off(self):
        if self.is_running():
            self.proc.terminate()
        self.last_line = 'đã tắt'


LAPTOP_DET = LaptopDetector()


class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        pass

    def _send(self, body, kind='application/json', code=200):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', kind)
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
            items = [int(v) for v in a.get('items', '').split(',') if v]
            if path == '/api/bev/candidates':
                return self._send([{k: u[k] for k in ('n', 'time', 'height_m', 'pitch_deg', 'roll_deg', 'yaw_deg', 'rms_px',
                                                      'square', 'has_squares', 'passed', 'total', 'checks')} for u in B.list_candidates()])
            if path == '/api/bev/suggest':
                return self._send(B.suggest(items))
            if path == '/api/bev/preview.jpg':
                import cv2
                ok, buf = cv2.imencode('.jpg', B.preview(items, _bev_params(a)), [cv2.IMWRITE_JPEG_QUALITY, 85])
                return self._send(buf.tobytes(), 'image/jpeg')
        except Exception as e:  # noqa: BLE001
            return self._send({'error': str(e)}, 'application/json', 400)
        self._send(b'not found', 'text/plain', 404)

    def do_GET(self):
        path, _, q = self.path.partition('?')
        if path == '/':
            return self._send(PAGE.replace('__CAR_IP__', CAR_IP).replace('__BEV_BLOCK__', _bev_block()).encode(),
                             'text/html; charset=utf-8')
        if path.startswith('/calib_new/') and '..' not in path:
            f = ROOT / path.lstrip('/')
            if f.is_file():
                return self._send(f.read_bytes(), 'image/jpeg')
            return self._send(b'', 'image/jpeg', 404)
        if path.startswith('/api/bev/'):
            return self._bev_get(path, q)
        if path == '/api/calib_status':
            return self._send({'has_calib': (ROOT / 'config/camera.yaml').exists(),
                              'has_extr': (ROOT / 'config/extrinsics.yaml').exists(),
                              'image': sorted(p.name for p in (ROOT / 'calib_new').glob('anh_*'))[-1:] if (ROOT / 'calib_new').exists() else [],
                              'new_cam': (ROOT / 'calib_new/camera.yaml').exists(),
                              'new_extr': (ROOT / 'calib_new/extrinsics.yaml').exists()})
        if path == '/api/detector_laptop':
            return self._send({'running': LAPTOP_DET.is_running(), 'last_line': LAPTOP_DET.last_line})
        if path == '/api/roi':
            from core.detector import read_roi
            r = read_roi()
            return self._send({'y1': r[1], 'y2': r[3]})
        if path == '/api/scripts':
            return self._send(list_scripts())
        if path == '/api/log':
            start = int(dict(x.split('=') for x in q.split('&') if '=' in x).get('start', 0))
            with PROC.lock:
                n = len(PROC.lines)
                lines = PROC.lines[start:] if start <= n else PROC.lines[:]
            run = PROC.proc is not None and PROC.proc.poll() is None
            return self._send({'start': n, 'lines': lines, 'running': run, 'name': PROC.name,
                              'kind': PROC.kind, 'code': PROC.code})
        self._send(b'not found', 'text/plain', 404)

    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        d = json.loads(self.rfile.read(n) or b'{}')
        if self.path == '/api/run':
            extra = [x for x in d.get('extra', '').split() if x]
            ok, msg = PROC.run(d['name'], d.get('kind', 'car'), extra)
            return self._send({'ok': ok, 'msg': msg})
        if self.path == '/api/bev/save':
            try:
                from core import birdeye_select as B
                summary = B.save_selection([int(v) for v in d.get('items', [])], _bev_params(d))
                return self._send({'ok': True, 'msg': f"Đã lưu calib_new/extrinsics.yaml + bev.yaml (lần {summary['times']}, "
                                  f"cao {summary['height_m'] * 100:.1f} cm). Bấm 'Dùng số này' để đưa vào config/."})
            except Exception as e:  # noqa: BLE001
                return self._send({'ok': False, 'msg': str(e)})
        if self.path == '/api/calib':
            command = {'pull': ['bash', 'scripts/car.sh', 'pull'],
                    'accept_extr': ['bash', '-c', 'cp -v calib_new/extrinsics.yaml config/extrinsics.yaml && '
                                   'cp -v calib_new/bev.yaml config/bev.yaml']}.get(d.get('step'))
            if command is None:
                return self._send({'ok': False, 'msg': 'lệnh lạ'})
            ok, msg = PROC.run_cmd(f'bird\'s eye: {d["step"]}', command)
            return self._send({'ok': ok, 'msg': msg})
        if self.path == '/api/detector_laptop':
            LAPTOP_DET.turn_on() if d.get('on') else LAPTOP_DET.turn_off()
            return self._send({'running': LAPTOP_DET.is_running(), 'last_line': LAPTOP_DET.last_line})
        if self.path == '/api/roi':
            try:
                from core.detector import save_roi
                save_roi(d['y1'], d['y2'])
                return self._send({'ok': True, 'msg': f"Đã lưu ROI hàng {int(d['y1'])}..{int(d['y2'])}. Chạy lại xe là có hiệu lực."})
            except (ValueError, KeyError) as e:
                return self._send({'ok': False, 'msg': str(e)})
        if self.path == '/api/stop':
            PROC.stop()
            return self._send({'ok': True})
        if self.path == '/api/open':
            ed = open_in_editor(d['file'])
            return self._send({'ok': bool(ed), 'msg': f'mở bằng {ed}' if ed else 'không có trình sửa'})
        self._send(b'not found', 'text/plain', 404)


PAGE = r'''<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Champion - bảng điều khiển</title>
<style>
:root{--bg:#0f1115;--panel:#181b22;--panel2:#20242d;--text:#e6e8ec;--muted:#9aa3b2;--green:#3ddc84;--red:#ff4d4f;--yellow:#ffc53d;--blue:#4ea1ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif;height:100vh;display:flex}
#left{width:400px;min-width:340px;overflow:auto;border-right:1px solid #2a2f3a;padding:12px}
#right{flex:1;display:flex;flex-direction:column;min-width:0}
h1{font-size:18px;margin:0 0 4px}.muted{color:var(--muted);font-size:12px}
.btn{border:0;border-radius:6px;padding:5px 9px;cursor:pointer;font-weight:600;font-size:12px;background:var(--panel2);color:var(--text)}
.btn:hover{filter:brightness(1.3)}.car{background:#1f5f3a}.sim{background:#2b3a5c}.alt{background:#5c4a1f}
#stop{width:100%;background:var(--red);color:#fff;font-size:18px;padding:12px;margin:8px 0}
#full{width:100%;background:#1f7a46;color:#fff;font-size:16px;padding:10px}
.group{margin-top:14px;font-weight:700;color:var(--blue)}
.item{background:var(--panel);border-radius:8px;padding:8px;margin-top:6px}
.item .name{font-weight:700;cursor:pointer}.item .name:hover{color:var(--blue)}.item .btns{margin-top:6px;display:flex;gap:5px;flex-wrap:wrap}
pre.params{background:#0b0d11;color:#cfd6e4;font-size:11px;padding:8px;border-radius:6px;white-space:pre-wrap;margin:6px 0 0;display:none}
#top{padding:10px;border-bottom:1px solid #2a2f3a;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
#status{font-weight:700}.stat{font-family:ui-monospace,monospace;background:var(--panel);padding:4px 8px;border-radius:6px}
#views{display:flex;gap:8px;padding:8px;overflow-x:auto;min-height:220px;align-items:flex-start}
#views figure{margin:0;background:var(--panel);border-radius:8px;padding:4px}#views figcaption{font-size:11px;color:var(--muted)}
#views img{max-height:300px;max-width:640px;display:block}
#car3d{width:250px;height:230px;perspective:700px;display:flex;align-items:center;justify-content:center}
#stage{width:150px;height:150px;position:relative;transform-style:preserve-3d;transform:rotateX(62deg) rotateZ(-38deg)}
#floor{position:absolute;inset:0;transform:translateZ(-18px);border:1px solid #3a4352;background:repeating-linear-gradient(0deg,#2a303b 0 1px,transparent 1px 25px),repeating-linear-gradient(90deg,#2a303b 0 1px,transparent 1px 25px),rgba(60,70,90,.15)}
#box{width:90px;height:50px;position:absolute;left:30px;top:50px;transform-style:preserve-3d;transition:transform .08s linear}
#box div{position:absolute;border:1px solid #6fa8ff;background:rgba(78,161,255,.25)}
#log{flex:1;overflow:auto;background:#07080b;font:12px ui-monospace,monospace;padding:8px;white-space:pre-wrap;margin:0}
.step{background:var(--panel);border-radius:10px;padding:10px;margin-top:8px}.step h2{font-size:14px;margin:0 0 6px}.step ol{margin:6px 0;padding-left:20px;line-height:1.5}.btn.sub{background:var(--panel2)}
#calibPanel{background:#12151b;border-bottom:1px solid #2a2f3a;padding:6px 10px;max-height:46vh;overflow:auto}#calibPanel summary{cursor:pointer;font-weight:700;color:var(--blue)}
.err{color:var(--red)}.ok{color:var(--green)}.pass{color:var(--green)}.fail{color:var(--yellow)}
input{background:var(--panel2);color:var(--text);border:1px solid #333;border-radius:6px;padding:5px;width:100%}
</style></head><body>
<div id="left">
 <h1>CHAMPION</h1><div class="muted">Xe: __CAR_IP__ · log realtime bên phải · sửa tham số = bấm "Sửa", Ctrl+S, chạy lại</div>
 <button id="stop" class="btn" onclick="stopCar()">■ DỪNG KHẨN</button>
 <button id="full" class="btn" onclick="run('run_car','car')">▶ CHẠY TỰ HÀNH (trên xe)</button>
 <div style="margin-top:8px"><div class="muted">Tham số thêm cho lần bấm tới (ví dụ <code>--v 0.30</code>, <code>--zone 380 60 560 260</code>):</div>
 <input id="extra" placeholder="để trống nếu không cần"></div>
 <div class="item" style="margin-top:8px"><b>Nhận biển + đèn bằng LAPTOP</b> <button id="ldBtn" class="btn sim" onclick="ldToggle()">...</button><div id="ldStatus" class="muted" style="margin-top:4px"></div><div class="muted">Model chạy trên laptop (~50 ms/khung, Jetson ~850 ms). Xe tự hành (LIGHT_SOURCE = "laptop") và onnx_test lấy kết quả từ đây. ROI Lưu là có hiệu lực ngay.</div></div>
 <div id="scripts"></div>
</div>
<div id="right">
 <div id="top"><span id="status">Chưa chạy gì</span>
  <span class="stat" id="prog">đang chạy -</span><span class="stat" id="vw">v - w -</span>
  <span class="stat" id="att">yaw - pitch - roll -</span><span class="stat" id="dist">- m</span></div>
 <div id="views"><figure><div id="car3d"><div id="stage"><div id="floor"></div><div id="box"></div></div></div><figcaption id="cap3d">Mô hình xe theo IMU (mũi xe màu cam; mặt sàn đứng yên)</figcaption></figure></div>
 <details id="calibPanel" open><summary>Bird's eye: chọn ứng viên tốt nhất (sau khi chạy birdeye_chessboard)</summary>
  <div class="muted" style="margin:6px 0">Chạy <b>birdeye_chessboard</b> ở nhóm "Bird's eye" bên trái (đặt bàn cờ nhiều chỗ, mỗi chỗ một lần chốt), rồi chọn ở đây.</div>
  __BEV_BLOCK__
 </details>
 <pre id="log"></pre>
</div>
<script>
let logStart=0, mode='car', shownViews=new Set();
const $=id=>document.getElementById(id);
function buildBox(){const W=90,D=50,H=30,f=[['translateZ(15px)',W,D,'#6fa8ff'],['rotateX(180deg) translateZ(15px)',W,D,'#6fa8ff'],
 ['rotateY(90deg) translateZ(45px)',H,D,'#ff9f43'],['rotateY(-90deg) translateZ(45px)',H,D,'#6fa8ff'],
 ['rotateX(90deg) translateZ(25px)',W,H,'#6fa8ff'],['rotateX(-90deg) translateZ(25px)',W,H,'#6fa8ff']];
 $('box').innerHTML=f.map(([t,w,h,c])=>`<div style="width:${w}px;height:${h}px;left:${(90-w)/2}px;top:${(50-h)/2}px;transform:${t};background:${c}${c==='#ff9f43'?'dd':'55'};border-color:${c}"></div>`).join('');}
function axis(c,txt,css,lb){return `<div style="position:absolute;border:0;background:${c};${css}"><b style="position:absolute;border:0;color:${c};font:700 12px sans-serif;white-space:nowrap;${lb}">${txt}</b></div>`;}
function drawAxes(){
 // trục THÂN XE (xoay theo xe): x = tiến (đỏ), y = trái (xanh lá), z = lên (xanh dương); gốc ở tâm xe
 $('box').insertAdjacentHTML('beforeend',
  axis('#ff5d5d','x','left:45px;top:23.5px;width:80px;height:3px;','left:82px;top:-7px;')+
  axis('#3ddc84','y','left:43.5px;top:-35px;width:3px;height:60px;transform:none;','left:6px;top:-14px;')+
  axis('#4ea1ff','z','left:43.5px;top:25px;width:3px;height:70px;transform-origin:50% 0;transform:rotateX(90deg);','left:6px;top:70px;'));
 // trục SÀN (đứng yên, xám) để thấy xe nghiêng so với mặt đất
 $('floor').insertAdjacentHTML('beforeend',
  axis('#8d96a5','X0','left:75px;top:74px;width:65px;height:2px;opacity:.7;','left:67px;top:-7px;')+
  axis('#8d96a5','Y0','left:74px;top:10px;width:2px;height:65px;opacity:.7;','left:5px;top:-14px;')+
  axis('#8d96a5','Z0','left:74px;top:75px;width:2px;height:60px;transform-origin:50% 0;transform:rotateX(90deg);opacity:.7;','left:5px;top:60px;'));
}
buildBox();drawAxes();
function drawCar3d(p,r,y){ // yaw + = quay trái, pitch + = ngóc mũi lên, roll + = nghiêng sang phải (đúng quy ước IMU của xe)
 $('box').style.transform=`rotateZ(${-y}deg) rotateY(${-p}deg) rotateX(${-r}deg)`;
 const f=v=>(v>=0?'+':'')+(+v).toFixed(1)+'°';
 $('cap3d').innerHTML=`<span style="color:#ff5d5d">x · roll ${f(r)}</span> &nbsp; <span style="color:#3ddc84">y · pitch ${f(p)}</span> &nbsp; <span style="color:#4ea1ff">z · yaw ${f(y)}</span><br>trục màu = gắn vào xe, trục xám = mặt sàn đứng yên`;}
const ROI_HTML=`<div class="roi" style="margin-top:8px;padding-top:8px;border-top:1px solid #2a2f3a"><b>ROI nhận biển báo + đèn</b>
  <div class="muted" style="margin:4px 0">Chỉ nhận trong dải hàng pixel y1..y2 của ảnh 640x480 (0 = mép trên, 480 = mép dưới). Mặc định 240..480. Camera hiện trực tiếp từ laptop (bật "Nhận biển + đèn bằng LAPTOP"); Lưu xong hình "light" bên phải đổi ngay, Lần chạy xe sau dùng ROI này.</div>
  <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
   <label>y1<input id="roi1" type="number" min="0" max="478" step="5" style="width:64px" oninput="roiDraw()"></label>
   <label>y2<input id="roi2" type="number" min="1" max="480" step="5" style="width:64px" oninput="roiDraw()"></label>
   <button class="btn sim" onclick="roiSave()">Lưu ROI</button>
   <div style="position:relative;width:64px;height:48px;background:#222;border:1px solid #444"><div id="roiBox" style="position:absolute;left:0;right:0;background:rgba(78,161,255,.45)"></div></div>
  </div>
  <div id="roiFrame" style="position:relative;width:100%;max-width:370px;aspect-ratio:4/3;background:#000;margin-top:6px;cursor:crosshair" onclick="roiClick(event)">
   <img id="roiImg" onerror="this.removeAttribute('src');$('roiWait').style.display='block'" style="width:100%;height:100%;display:block;pointer-events:none">
   <div id="roiTop" style="position:absolute;left:0;right:0;top:0;background:rgba(0,0,0,.55);pointer-events:none"></div>
   <div id="roiBottom" style="position:absolute;left:0;right:0;bottom:0;background:rgba(0,0,0,.55);pointer-events:none"></div>
   <div id="roiBorder" style="position:absolute;left:0;right:0;border:2px solid #4ea1ff;pointer-events:none"></div>
   <div id="roiWait" class="muted" style="position:absolute;left:6px;top:6px">Chưa có hình: bấm Xe ở onnx_test để bật camera</div>
  </div>
  <div class="muted">Bấm lên hình = đặt y1 (mép trên). Giữ Shift + bấm = đặt y2 (mép dưới). Vùng sáng là vùng được nhận.</div>
  <input id="roiRange" type="range" min="0" max="478" step="5" style="margin-top:6px" oninput="$('roi1').value=this.value;roiDraw()">
  <div id="roiMsg" class="muted"></div></div>`;
async function loadScripts(){const r=await (await fetch('/api/scripts')).json();$('scripts').innerHTML=r.map(g=>`<div class="group">${g.group}</div>`+g.items.map(m=>`
 <div class="item"><div class="name" onclick="toggleParams('${m.name}')">${m.name}</div><div class="muted">${m.desc}</div>
 <div class="btns">${m.runs_on==='laptop'?`<button class="btn car" onclick="run('${m.name}','car')">Chạy (laptop)</button>`:`<button class="btn car" onclick="run('${m.name}','car')">Xe</button>`}
 ${m.manual?`<button class="btn alt" onclick="run('${m.name}','car','--manual')">Tay (đẩy xe)</button>`:''}
 ${m.sim?`<button class="btn sim" onclick="run('${m.name}','sim')">Giả lập</button>`:''}
 <button class="btn" onclick="openFile('${m.file}')">Sửa</button></div>
 <pre class="params" id="params_${m.name}">${(m.params||'(không có khối THAM SỐ)').replace(/</g,'&lt;')}</pre></div>`).join('')).join('');const o=$('params_onnx_test');if(o){o.parentElement.insertAdjacentHTML('beforeend',ROI_HTML);roiLoad();}}
function toggleParams(t){const e=$('params_'+t);e.style.display=e.style.display==='block'?'none':'block';}
function calibArgs(name){ // bước 4 cần 2 số đo bằng thước: hỏi, nhớ lại lần sau
 if(name!=='birdeye_chessboard'||$('extra').value.trim())return null;
 const height=prompt('Chiều cao camera so với mặt đường (m), đo bằng thước (chỉ để đối chiếu với số tự tính từ bàn cờ; để trống = bỏ qua):',localStorage.getItem('height')||'0.110');if(height===null)return false;
 const front=prompt('Quang tâm camera nằm TRƯỚC tâm xe bao nhiêu m (sau tâm xe thì nhập số âm), đo bằng thước:',localStorage.getItem('front')||'');if(!front)return false;
 localStorage.setItem('height',height);localStorage.setItem('front',front);return `--front ${front}`+(height?` --height ${height}`:'');}
async function run(name,k,more=''){mode=k;let extra=($('extra').value+' '+more).trim();const ca=calibArgs(name);if(ca===false)return;if(ca)extra=ca;const r=await (await fetch('/api/run',{method:'POST',body:JSON.stringify({name,kind:k,extra})})).json();
 if(!r.ok)alert(r.msg);shownViews.clear();document.querySelectorAll('#views figure.live').forEach(e=>e.remove());}
function roiCam(){if($('roiImg')&&!$('roiImg').src){$('roiImg').src='http://127.0.0.1:8096/img/cam.mjpg';$('roiWait').style.display='none';}}
function roiClick(e){const r=$('roiFrame').getBoundingClientRect();const y=Math.round((e.clientY-r.top)/r.height*480/5)*5;if(e.shiftKey)$('roi2').value=Math.min(480,Math.max(y,1));else $('roi1').value=Math.min(478,Math.max(y,0));roiDraw();}
function roiDraw(){const a=+$('roi1').value,b=+$('roi2').value;$('roiRange').value=a;$('roiBox').style.top=(a/480*48)+'px';$('roiBox').style.height=(Math.max(b-a,0)/480*48)+'px';if($('roiTop')){$('roiTop').style.height=(a/4.8)+'%';$('roiBottom').style.height=(Math.max(480-b,0)/4.8)+'%';$('roiBorder').style.top=(a/4.8)+'%';$('roiBorder').style.height=(Math.max(b-a,0)/4.8)+'%';}}
async function roiLoad(){try{const r=await (await fetch('/api/roi')).json();$('roi1').value=r.y1;$('roi2').value=r.y2;roiDraw();}catch(e){}}
async function roiSave(){const r=await (await fetch('/api/roi',{method:'POST',body:JSON.stringify({y1:+$('roi1').value,y2:+$('roi2').value})})).json();$('roiMsg').textContent=r.msg;$('roiMsg').style.color=r.ok?'var(--green)':'var(--red)';}
async function stopCar(){await fetch('/api/stop',{method:'POST',body:'{}'});}
async function openFile(f){const r=await (await fetch('/api/open',{method:'POST',body:JSON.stringify({file:f})})).json();if(!r.ok)alert(r.msg);}
function colorize(s){const e=s.replace(/&/g,'&amp;').replace(/</g,'&lt;');
 if(/LỖI|MẤT|KHÔNG ĐẠT|Traceback|Error|!!/.test(s))return `<span class="err">${e}</span>`;
 if(/==>|<==|ĐẠT|XONG|ĐÈN XANH|#####/.test(s))return `<span class="ok">${e}</span>`;return e;}
let ldRunning=false;
function ldViews(){ // hình từ LAPTOP (nhanh, không qua xe): ảnh camera nén trong ô ROI + hình đèn/biển trong khung hình
 if(!ldRunning)return;roiCam();
 for(const [k,t] of [['light','đèn/biển (laptop, ROI)'],['lidar','lidar (laptop)'],['line','vạch: 8 ô dò vạch (laptop)'],['birdeye','bird\'s eye + vạch (laptop)']]){
  if(!$('ld_'+k))$('views').insertAdjacentHTML('beforeend',`<figure id="ld_${k}"><img src="http://127.0.0.1:8096/img/${k}.mjpg" onerror="const i=this;setTimeout(()=>{i.src='http://127.0.0.1:8096/img/${k}.mjpg?t='+Date.now()},1000)"><figcaption>${t}</figcaption></figure>`);}}
async function ldPoll(){try{const r=await (await fetch('/api/detector_laptop')).json();ldRunning=r.running;ldViews();$('ldBtn').textContent=r.running?'Tắt':'Bật';$('ldStatus').textContent=(r.running?'● ':'○ ')+r.last_line;$('ldStatus').style.color=r.running?'var(--green)':'var(--muted)';}catch(e){}setTimeout(ldPoll,2000);}
async function ldToggle(){await fetch('/api/detector_laptop',{method:'POST',body:JSON.stringify({on:!ldRunning})});}
async function pollLog(){try{const r=await (await fetch('/api/log?start='+logStart)).json();
 if(r.lines.length){const L=$('log'),atEnd=L.scrollTop+L.clientHeight>=L.scrollHeight-30;L.insertAdjacentHTML('beforeend',r.lines.map(colorize).join('\n')+'\n');logStart=r.start;if(atEnd)L.scrollTop=L.scrollHeight;}
 $('status').textContent=r.running?`Đang chạy: ${r.name} (${r.kind==='sim'?'giả lập':'trên xe'})`:(r.name?`Đã xong: ${r.name} (mã ${r.code})`:'Chưa chạy gì');
 $('status').style.color=r.running?'var(--green)':'var(--muted)';mode=r.kind||mode;}catch(e){} setTimeout(pollLog,400);}
async function pollCar(){const host=mode==='sim'?'127.0.0.1':'__CAR_IP__';
 try{const c=new AbortController();setTimeout(()=>c.abort(),800);const t=await (await fetch(`http://${host}:8097/status.json`,{signal:c.signal})).json();
  $('prog').textContent='đang chạy '+(t.program||'-')+(t.detail?' | '+t.detail:'');
  $('vw').textContent=`v ${t.v??'-'} w ${t.w??'-'}`;$('att').textContent=`yaw ${t.yaw??'-'}° pitch ${t.pitch??'-'}° roll ${t.roll??'-'}°`;
  $('dist').textContent=(t.distance??'-')+' m';
  drawCar3d(t.pitch||0,t.roll||0,t.yaw||0);
  for(const a of (t.image||[])){if(a==='roi_cam'||a==='cam'||(ldRunning&&['light','lidar','line','birdeye'].includes(a))){continue;}if(shownViews.has(a))continue;shownViews.add(a);
   $('views').insertAdjacentHTML('beforeend',`<figure class="live"><img src="http://${host}:8097/img/${a}.mjpg"><figcaption>${a}</figcaption></figure>`);}
 }catch(e){} setTimeout(pollCar,120);}
async function calibStep(b){const r=await (await fetch('/api/calib',{method:'POST',body:JSON.stringify({step:b})})).json();if(!r.ok)alert(r.msg);}
loadScripts();ldPoll();pollLog();pollCar();
</script></body></html>'''



def main():
    page = ''
    try:
        srv = ThreadingHTTPServer(('127.0.0.1', PORT), H)
    except OSError:
        # Cổng đang bị bản bảng điều khiển khác giữ: không văng lỗi, mở trình duyệt vào bản đó.
        print(f'Bảng điều khiển ĐÃ đang chạy ở http://127.0.0.1:{PORT}{page}, mở trình duyệt vào đó.\n'
              f'(Muốn khởi động lại hẳn: pkill -f tools/control_panel.py rồi chạy lại.)', flush=True)
        if shutil.which('xdg-open') and os.environ.get('DISPLAY'):
            subprocess.Popen(['xdg-open', f'http://127.0.0.1:{PORT}{page}'],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    print(f'Bảng điều khiển: http://127.0.0.1:{PORT}{page}  (xe {CAR_IP}). Ctrl+C để tắt.', flush=True)
    # detector_laptop KHÔNG tự bật nữa: model đang chạy trên Jetson (LIGHT_SOURCE = 'car'); bấm Bật ở cột trái nếu muốn thử bản laptop
    if shutil.which('xdg-open') and os.environ.get('DISPLAY'):
        subprocess.Popen(['xdg-open', f'http://127.0.0.1:{PORT}{page}'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    signal.signal(signal.SIGTERM, lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))   # kill thường cũng dọn detector_laptop
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        PROC.stop()
    finally:
        LAPTOP_DET.turn_off()


if __name__ == '__main__':
    main()
