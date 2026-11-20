#!/usr/bin/env python3
"""CALIB CAMERA bước 4: BIRD'S EYE bằng BÀN CỜ đặt NẰM TRÊN SÀN (XE KHÔNG CHẠY).

Cách làm (như video RidgeRun "Chessboard Auto Calibration"):
  1. Đặt bàn cờ NẰM PHẲNG trên mặt sàn (cùng mặt sàn với đường đua), SÁT trước mũi xe: mép
     gần của bàn cờ cách camera khoảng 15-45 cm (camera rất thấp nên ô cờ bị ép mỏng; xa quá
     không dò được), thấy TRỌN bàn cờ trong khung hình, không bị cắt ở mép dưới (camera ngẩng
     lên thì sàn gần nhất chỉ thấy từ ~30-35 cm). Cạnh DÀI (10 góc) nằm NGANG. CẠNH bàn cờ vuông góc / song song với hướng xe (cạnh gần
     song song với cản trước). Giữ yên.
  2. Mở hình: http://<ip-xe>:8097/img/birdeye.mjpg (hoặc bảng điều khiển). Trái: ảnh camera,
     bàn cờ được dò có viền đỏ + 4 chấm xanh dương; viền XANH LÁ bao quanh khi đã ổn định.
     Phải: BIRD'S EYE của bàn cờ (1 mm/pixel): các ô phải thành Ô VUÔNG.
  3. Log và dòng chữ trên hình báo BÀN CỜ CÁCH CAMERA bao nhiêu cm (hàng góc gần nhất, hàng xa nhất,
     lệch ngang, cách tâm xe), tính theo toạ độ sàn. Hàng góc gần cách mép thật của hoa văn ~2 cm.
     Tự tính từ bàn cờ: CHIỀU CAO camera, độ CHÚI, NGHIÊNG NGANG, LỆCH HƯỚNG so với mặt sàn
     (thay cho cách đo đường chân trời). Ổn định thì LƯU MỘT ỨNG VIÊN (calib_new/candidates/lan_NN.*)
     và in bảng ĐẠT / KHÔNG ĐẠT; KHÔNG tự ghi kết quả. Nhấc bàn cờ đặt chỗ khác, ổn định lần nữa:
     chiều cao và độ chúi phải ra giống lần trước (camera không đổi) - phép kiểm nhất quán.
  4. Bạn TỰ CHỌN trên trang calib (bước 4b): lần nào dùng (nhiều lần = trung vị), vùng nhìn và độ
     phân giải bird's eye, xem thử, rồi mới lưu. Cuối cùng "Dùng số này".

    bash scripts/car.sh birdeye_chessboard --front 0.050 [--height 0.110] [--lateral 0.0]
  --front  quang tâm camera nằm TRƯỚC tâm xe bao nhiêu m (sau tâm xe: số âm). ĐO BẰNG THƯỚC
  --height    chiều cao camera đo bằng thước (m), chỉ để đối chiếu với số tính từ bàn cờ
  --lateral   camera lệch sang TRÁI so với tâm xe bao nhiêu m (mặc định 0)
  --grid / --square-mm  đọc từ config/intrinsics_raw.yaml (10x7 góc trong, ô 19 mm), có thể ghi đè
Thử trên 1 ảnh (laptop): python3 tools/birdeye_chessboard.py --image anh.jpg --front 0.05
"""
import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import settings                                                   # noqa: E402
from core.camera import BirdEye, Calib, decode_color                        # noqa: E402
from core.birdeye_select import save_candidate                              # noqa: E402
from core.camera_pose import solve_pose, board_corners_in_car, pose_in_car  # noqa: E402
from core.runtime import Tool                                               # noqa: E402
from core.log import log                                                    # noqa: E402

RMS_MAX = 1.5              # px: sai số chiếu lại các góc
HEIGHT_TOL = 0.015         # m: lệch so với số đo bằng thước
ROLL_MAX = 2.0             # độ
YAW_MAX = 3.0              # độ: lệch hướng so với cạnh bàn cờ
SQUARE_TOL = 0.04          # 4 %: ô cờ trên bird's eye phải vuông và đúng cỡ
STABLE_HEIGHT_STD = 0.003  # m: độ lệch chuẩn chiều cao qua N lần dò
STABLE_PITCH_STD = 0.5     # độ
STABLE_YAW_STD = 1.0
N_STABLE = 10           # số lần dò liên tiếp để coi là ổn định
CONSIST_HEIGHT = 0.005  # m: hai lần chốt (bàn cờ đặt chỗ khác) phải giống nhau
CONSIST_PITCH = 1.0     # độ


def load_camera():
    f = settings.CONFIG_DIR / 'camera.yaml'
    if not f.exists():
        raise SystemExit(f'CHƯA CALIB CAMERA (không có {f}). Calib nội: đặt config/camera.yaml')
    c = yaml.safe_load(f.read_text())
    return (np.array(c['camera_matrix']['data'], float).reshape(3, 3),
            np.array(c['distortion_coefficients']['data'], float).ravel())


def default_grid():
    f = settings.CONFIG_DIR / 'intrinsics_raw.yaml'
    if f.exists():
        y = yaml.safe_load(f.read_text())
        return int(y['board_cols']), int(y['board_rows']), float(y['square_m']) * 1000.0
    return 10, 7, 19.0


V_STRETCH = (2, 4, 3)     # các mức kéo giãn ảnh theo chiều DỌC khi dò (thử lần lượt)


def detect_board(gray, cols, rows, heading=None):
    """Dò góc bàn cờ bằng findChessboardCornersSB.
    Camera nhìn gần như dọc theo mặt sàn nên ô cờ bị ép mỏng theo chiều dọc (chỉ cao vài pixel): kéo giãn ảnh
    theo chiều dọc 2-4 lần cho ô gần vuông rồi dò, tính ngược toạ độ về ảnh gốc. Cách này vừa nhanh (~20-40 ms,
    dò không giãn mất ~2 s) vừa dò được bàn cờ xa tới ~45 cm trước camera.
    Chỉ thử 1 hướng: cạnh có `cols` góc nằm NGANG. Bàn cờ nằm trên sàn nên chỉ tìm ở 75 % phía DƯỚI ảnh.
    -> (corners, (cols,rows))."""
    y0 = int(0.25 * gray.shape[0])
    zone = np.ascontiguousarray(gray[y0:])
    for k in V_STRETCH:
        stretch = cv2.resize(zone, None, fx=1.0, fy=float(k), interpolation=cv2.INTER_CUBIC)
        ok, cor = cv2.findChessboardCornersSB(stretch, (cols, rows), cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            cor = cor.astype(np.float32)
            cor[:, 0, 1] = (cor[:, 0, 1] + 0.5) / k - 0.5 + y0          # về toạ độ ảnh gốc
            return cor, (cols, rows)
    return None, None


def check_squares(gray, K, D, res, sz, square_mm, front, lateral):
    """Dựng bird's eye 1 mm/px quanh bàn cờ; đo lại ô cờ TRÊN ẢNH bird's eye. -> (bev, aspect, cỡ so với thật, corners)."""
    cols, rows = sz
    x, y = board_corners_in_car(res)
    span = max(x.max() - x.min(), y.max() - y.min()) + 0.12
    m = max(0.001, span / 380.0)
    calib = Calib.from_pose(K, D, res['rvec'], res['tvec'], (gray.shape[1], gray.shape[0]))
    birdeye = BirdEye(calib, x.min() - 0.04, x.max() + 0.04, y.max() + 0.04, y.min() - 0.04, m, img_scale=1.0)
    bev = birdeye.warp(gray)
    border = cv2.copyMakeBorder(bev, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    ok, cor = cv2.findChessboardCorners(border, sz, cv2.CALIB_CB_ADAPTIVE_THRESH)
    if not ok:
        return bev, None, None, None, m
    cor = cor.reshape(rows, cols, 2) - 12
    horiz = float(np.linalg.norm(np.diff(cor, axis=1), axis=2).mean())
    slope = float(np.linalg.norm(np.diff(cor, axis=0), axis=2).mean())
    return bev, horiz / slope, ((horiz + slope) / 2 * m) / (square_mm / 1000.0), cor.reshape(-1, 2), m


def draw_frame(bgr, cor, sz, bev, bev_cor, stable, lines):
    """Hình gửi về: trái = camera + viền bàn cờ, phải = bird's eye. Thu nhỏ cho nhẹ Wi-Fi."""
    h, w = bgr.shape[:2]
    left_img = bgr.copy()
    if cor is not None:
        c = cor.reshape(-1, 2)
        cols, rows = sz
        corners = [c[0], c[cols - 1], c[-1], c[-cols]]
        cv2.polylines(left_img, [np.array(corners, np.int32)], True, (0, 0, 255), 2)
        for p in corners:
            cv2.circle(left_img, (int(p[0]), int(p[1])), 7, (255, 0, 0), 2)
    if stable:
        cv2.rectangle(left_img, (2, 2), (w - 3, h - 3), (0, 255, 0), 6)
    left_img = cv2.resize(left_img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    if bev is not None:
        right_img = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
        if bev_cor is not None:
            for p in bev_cor:
                cv2.circle(right_img, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)
        k = (h // 2) / right_img.shape[0]
        right_img = cv2.resize(right_img, (max(10, int(right_img.shape[1] * k)), h // 2), interpolation=cv2.INTER_AREA)
    else:
        right_img = np.zeros((h // 2, w // 3, 3), np.uint8)
    cv2.putText(left_img, 'CAMERA', (6, 16), 0, 0.5, (0, 255, 255), 1)
    cv2.putText(right_img, "BIRD'S EYE", (6, 16), 0, 0.5, (0, 255, 255), 1)
    img = np.hstack([left_img, right_img])
    cv2.rectangle(img, (0, img.shape[0] - 20), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    cv2.putText(img, lines[:70], (4, img.shape[0] - 6), 0, 0.42, (255, 255, 255), 1)
    return img


def results_table(res, mean, sz, args, aspect, measured, hist_filtered):
    checks = [('Sai số chiếu lại các góc', f'{res["rms"]:.2f} px', res['rms'] < RMS_MAX),
           ('Camera nghiêng ngang (roll)', f'{res["roll"]:+.2f} độ', abs(res['roll']) < ROLL_MAX),
           ('Lệch hướng so với cạnh bàn cờ (yaw)', f'{res["yaw"]:+.2f} độ', abs(res['yaw']) < YAW_MAX),
           ('Ổn định chiều cao (độ lệch chuẩn)', f'{mean["std_height"] * 1000:.1f} mm', mean['std_height'] < STABLE_HEIGHT_STD)]
    if args.height:
        checks.append(('Chiều cao so với thước', f'{res["height"] * 100:.1f} cm (thước {args.height * 100:.1f})',
                    abs(res['height'] - args.height) < HEIGHT_TOL))
    if aspect is not None:
        checks.append(('Ô cờ trên bird\'s eye vuông', f'tỉ lệ ngang/dọc {aspect:.3f}', abs(aspect - 1) < SQUARE_TOL))
        checks.append(('Ô cờ trên bird\'s eye đúng cỡ', f'{measured * 100:.1f} % so với {args.square_mm:.0f} mm',
                    abs(measured - 1) < SQUARE_TOL))
    else:
        checks.append(("Ô cờ trên bird's eye", 'không dò lại được bàn cờ trên ảnh bird\'s eye', False))
    if hist_filtered:
        old = hist_filtered[-1]
        checks.append(('Nhất quán với lần chốt trước (đặt bàn cờ chỗ khác)',
                    f'cao {abs(res["height"] - old["height"]) * 1000:.1f} mm, chúi {abs(res["pitch"] - old["pitch"]):.2f} độ',
                    abs(res['height'] - old['height']) < CONSIST_HEIGHT and abs(res['pitch'] - old['pitch']) < CONSIST_PITCH))
    return checks


def median(hist):
    return {k: float(np.median([h[k] for h in hist])) for k in ('height', 'pitch', 'roll', 'yaw', 'rms')} | {
        'rvec': np.median([h['rvec'] for h in hist], axis=0), 'tvec': np.median([h['tvec'] for h in hist], axis=0),
        'W': hist[-1]['W'], 'C_b': hist[-1]['C_b'], 'up': hist[-1]['up'], 'obj': hist[-1]['obj']}


def main():
    cols0, rows0, o0 = default_grid()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--front', type=float, required=True, help='quang tâm camera trước tâm xe (m), đo bằng thước')
    parser.add_argument('--height', type=float, default=0.0, help='chiều cao camera đo bằng thước (m), để đối chiếu')
    parser.add_argument('--lateral', type=float, default=0.0, help='camera lệch sang trái so với tâm xe (m)')
    parser.add_argument('--grid', default=f'{cols0}x{rows0}', help='số góc trong, ví dụ 10x7')
    parser.add_argument('--square-mm', type=float, default=o0, help='cạnh 1 ô cờ (mm)')
    parser.add_argument('--image', help='thử trên 1 file ảnh (không cần xe)')
    a = parser.parse_args()
    cols, rows = (int(v) for v in a.grid.lower().split('x'))
    K, D = load_camera()
    square_m = a.square_mm / 1000.0

    if a.image:
        bgr = cv2.imread(a.image)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        cor, sz = detect_board(gray, cols, rows)
        if cor is None:
            raise SystemExit('Không thấy bàn cờ trong ảnh.')
        res = solve_pose(cor, sz[0], sz[1], square_m, K, D)
        bev, asp, col, bc, m = check_squares(gray, K, D, res, sz, a.square_mm, a.front, a.lateral)
        print(f'cao {res["height"] * 100:.1f} cm | chúi {res["pitch"]:.1f} độ | roll {res["roll"]:+.1f} | '
              f'yaw {res["yaw"]:+.1f} | RMS {res["rms"]:.2f} px | ô vuông {asp} | cỡ {col}')
        settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
        out = settings.LOG_DIR / (Path(a.image).stem + '_birdeye.jpg')
        cv2.imwrite(str(out), draw_frame(bgr, cor, sz, bev, bc, False, 'thu 1 anh'))
        print(f'Hình: {out}')
        return

    import threading
    with Tool('birdeye_chessboard', need_camera=True, calibrate=False) as c:
        log(f"Bird's eye bằng bàn cờ {cols}x{rows} góc trong, ô {a.square_mm:.1f} mm. Đặt bàn cờ NẰM TRÊN SÀN, SÁT trước mũi xe.")
        log(f'Xem hình: http://<ip-xe>:{settings.VIEW_PORT}/img/birdeye.mjpg | camera trước tâm xe {a.front * 100:.1f} cm')
        state = {'job': None, 'shown': None}
        lock = threading.Lock()

        def measure_thread():
            """Dò bàn cờ + giải tư thế + chốt. Chậm (0,05-0,7 s/khung) nên nằm riêng, hình xem vẫn mượt."""
            hist, locs, written, t_log = deque(maxlen=N_STABLE), [], None, 0.0
            while c.running():
                with lock:
                    bgr, state['job'] = state['job'], None
                if bgr is None:
                    time.sleep(0.02)
                    continue
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                cor, sz = detect_board(gray, cols, rows)
                res = solve_pose(cor, sz[0], sz[1], square_m, K, D) if cor is not None else None
                if res is None:
                    hist.clear()
                    written = None
                    with lock:
                        state['shown'] = None
                    if time.monotonic() - t_log > 2:
                        t_log = time.monotonic()
                        log('Chưa thấy bàn cờ. Cạnh dài NẰM NGANG, thấy TRỌN cả bàn cờ trong khung (đừng để bị cắt ở mép dưới: nếu bị cắt thì đẩy '
                            'bàn cờ XA camera hơn; camera ngẩng lên thì sàn gần nhất chỉ thấy từ ~30-35 cm), đủ sáng')
                    continue
                hist.append(res)
                stable = (len(hist) >= N_STABLE
                           and np.std([h['height'] for h in hist]) < STABLE_HEIGHT_STD
                           and np.std([h['pitch'] for h in hist]) < STABLE_PITCH_STD
                           and np.std([h['yaw'] for h in hist]) < STABLE_YAW_STD)
                bev, asp, measured, bc, m = check_squares(gray, K, D, res, sz, a.square_mm, a.front, a.lateral)
                bx, by = board_corners_in_car(res)                 # toạ độ sàn, gốc ngay dưới camera
                near, far, horiz = float(bx.min()), float(bx.max()), float(by.mean())
                lines = (f'cao {res["height"] * 100:.1f} cm | chui {res["pitch"]:.1f} | roll {res["roll"]:+.1f} | '
                        f'yaw {res["yaw"]:+.1f} | rms {res["rms"]:.2f} | ban co cach {near * 100:.0f}-{far * 100:.0f} cm'
                        + (' | ON DINH' if stable else ''))
                with lock:
                    state['shown'] = (cor, sz, bev, bc, stable, lines)
                if c.viewer:
                    c.viewer.update(program='birdeye_chessboard', detail=lines)
                if time.monotonic() - t_log > 1:
                    t_log = time.monotonic()
                    log(f'cao {res["height"] * 100:5.1f} cm | chúi {res["pitch"]:5.1f}° | nghiêng ngang {res["roll"]:+5.1f}° | '
                        f'lệch hướng {res["yaw"]:+5.1f}° | RMS {res["rms"]:.2f} px | '
                        f'BÀN CỜ cách camera {near * 100:.0f} cm (hàng góc gần) tới {far * 100:.0f} cm (xa), '
                        f'lệch ngang {horiz * 100:+.0f} cm, cách tâm xe {(near + a.front) * 100:.0f} cm | '
                        + (f'ổn định {len(hist)}/{N_STABLE}' if not stable else 'ỔN ĐỊNH (giữ yên)'))
                if not stable:
                    continue
                fin = median(list(hist))
                if written is not None and abs(fin['height'] - written['height']) < 0.003 \
                        and abs(fin['pitch'] - written['pitch']) < 0.7:
                    continue                                # đã ghi đúng tư thế này rồi
                mean = {'std_height': float(np.std([h['height'] for h in hist]))}
                checks = results_table(fin, mean, sz, a, asp, measured, locs)
                rv_x, tv_x = pose_in_car(fin, a.front, a.lateral)
                f = save_candidate(fin, rv_x, tv_x, gray, checks, a.front, a.lateral, sz, a.square_mm, asp, measured,
                                 extra={'board_near_m': round(near, 3), 'board_far_m': round(far, 3),
                                       'board_lateral_m': round(horiz, 3)})
                locs.append(fin)
                written = fin
                log(f"================ CHỐT LẦN {len(locs)}: BIRD'S EYE BẰNG BÀN CỜ ================")
                for name, gt, ok in checks:
                    log(f'{name:48s} {gt:42s} {"ĐẠT" if ok else "KHÔNG ĐẠT"}')
                log(f'Đã lưu ỨNG VIÊN {len(locs)}: {f.name} (chưa dùng cho đến khi bạn chọn). Nhấc bàn cờ đặt CHỖ KHÁC để chốt '
                    'lần nữa (kiểm nhất quán) rồi Ctrl+C khi đủ 2-3 lần.')
                log('Tiếp: bấm "Kéo kết quả từ xe" trên trang calib, rồi chọn lần dùng + vùng nhìn + độ phân giải (bước 4b).')

        threading.Thread(target=measure_thread, daemon=True).start()
        old_count, t_draw = 0, 0.0
        while c.running():
            jpeg, count = c.car.camera_jpeg()
            if jpeg is None or count == old_count or time.monotonic() - t_draw < 1 / 10:
                c.car.sleep(0.005)
                continue
            old_count, t_draw = count, time.monotonic()
            bgr = decode_color(jpeg)
            with lock:
                if state['job'] is None:
                    state['job'] = bgr.copy()                  # luồng dò lấy khung mới nhất khi rảnh
                shown = state['shown']
            if c.viewer:
                if shown is None:
                    img = draw_frame(bgr, None, (cols, rows), None, None, False,
                                   'Chua thay ban co: thay TRON ca ban co (khong bi cat o mep duoi), canh dai NAM NGANG')
                else:
                    cor, sz, bev, bc, stable, lines = shown
                    img = draw_frame(bgr, cor, sz, bev, bc, stable, lines)
                c.viewer.set_image('birdeye', img, quality=60, always=True)


if __name__ == '__main__':
    main()
