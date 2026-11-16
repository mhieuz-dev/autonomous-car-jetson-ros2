#!/usr/bin/env python3
"""CALIB CAMERA bước 4: BIRD'S EYE bằng BÀN CỜ đặt NẰM TRÊN SÀN (XE KHÔNG CHẠY).

Cách làm (như video RidgeRun "Chessboard Auto Calibration"):
  1. Đặt bàn cờ NẰM PHẲNG trên mặt sàn (cùng mặt sàn với đường đua), SÁT trước mũi xe: mép
     gần của bàn cờ cách camera khoảng 15-45 cm (camera rất thấp nên ô cờ bị ép mỏng; xa quá
     không dò được), thấy TRỌN bàn cờ trong khung hình, không bị cắt ở mép dưới (camera ngẩng
     lên thì sàn gần nhất chỉ thấy từ ~30-35 cm). Cạnh DÀI (10 góc) nằm NGANG. CẠNH bàn cờ vuông góc / song song với hướng xe (cạnh gần
     song song với cản trước). Giữ yên.
  2. Mở hình: http://<ip-xe>:8097/anh/birdeye.mjpg (hoặc bảng điều khiển). Trái: ảnh camera,
     bàn cờ được dò có viền đỏ + 4 chấm xanh dương; viền XANH LÁ bao quanh khi đã ổn định.
     Phải: BIRD'S EYE của bàn cờ (1 mm/pixel): các ô phải thành Ô VUÔNG.
  3. Log và dòng chữ trên hình báo BÀN CỜ CÁCH CAMERA bao nhiêu cm (hàng góc gần nhất, hàng xa nhất,
     lệch ngang, cách tâm xe), tính theo toạ độ sàn. Hàng góc gần cách mép thật của hoa văn ~2 cm.
     Tự tính từ bàn cờ: CHIỀU CAO camera, độ CHÚI, NGHIÊNG NGANG, LỆCH HƯỚNG so với mặt sàn
     (thay cho cách đo đường chân trời). Ổn định thì LƯU MỘT ỨNG VIÊN (calib_moi/ung_vien/lan_NN.*)
     và in bảng ĐẠT / KHÔNG ĐẠT; KHÔNG tự ghi kết quả. Nhấc bàn cờ đặt chỗ khác, ổn định lần nữa:
     chiều cao và độ chúi phải ra giống lần trước (camera không đổi) - phép kiểm nhất quán.
  4. Bạn TỰ CHỌN trên trang calib (bước 4b): lần nào dùng (nhiều lần = trung vị), vùng nhìn và độ
     phân giải bird's eye, xem thử, rồi mới lưu. Cuối cùng "Dùng số này".

    bash chay.sh birdeye_banco --truoc 0.050 [--cao 0.110] [--lech 0.0]
  --truoc  quang tâm camera nằm TRƯỚC tâm xe bao nhiêu m (sau tâm xe: số âm). ĐO BẰNG THƯỚC
  --cao    chiều cao camera đo bằng thước (m), chỉ để đối chiếu với số tính từ bàn cờ
  --lech   camera lệch sang TRÁI so với tâm xe bao nhiêu m (mặc định 0)
  --luoi / --o-mm  đọc từ calib/intrinsics_goc.yaml (10x7 góc trong, ô 19 mm), có thể ghi đè
Thử trên 1 ảnh (laptop): python3 calib/birdeye_banco.py --anh anh.jpg --truoc 0.05
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
from chung import cau_hinh  # noqa: E402
from chung.camera import BirdEye, Calib, giai_nen_mau  # noqa: E402
from chung.birdeye_chon import luu_ung_vien  # noqa: E402
from chung.camera_pose import giai_pose, goc_ban_co_theo_xe, pose_theo_xe  # noqa: E402
from chung.khung import CongCu  # noqa: E402
from chung.log import log  # noqa: E402

RMS_MAX = 1.5            # px: sai số chiếu lại các góc
CAO_LECH_MAX = 0.015     # m: lệch so với số đo bằng thước
ROLL_MAX = 2.0           # độ
YAW_MAX = 3.0            # độ: lệch hướng so với cạnh bàn cờ
O_VUONG_MAX = 0.04       # 4 %: ô cờ trên bird's eye phải vuông và đúng cỡ
ON_DINH_CAO = 0.003      # m: độ lệch chuẩn chiều cao qua N lần dò
ON_DINH_PITCH = 0.5      # độ
ON_DINH_YAW = 1.0
N_ON_DINH = 10           # số lần dò liên tiếp để coi là ổn định
NHAT_QUAN_CAO = 0.005    # m: hai lần chốt (bàn cờ đặt chỗ khác) phải giống nhau
NHAT_QUAN_PITCH = 1.0    # độ


def nap_camera():
    f = cau_hinh.CALIB / 'camera.yaml'
    if not f.exists():
        raise SystemExit(f'CHƯA CALIB CAMERA (không có {f}). Calib nội: đặt calib/camera.yaml')
    c = yaml.safe_load(f.read_text())
    return (np.array(c['camera_matrix']['data'], float).reshape(3, 3),
            np.array(c['distortion_coefficients']['data'], float).ravel())


def luoi_mac_dinh():
    f = cau_hinh.CALIB / 'intrinsics_goc.yaml'
    if f.exists():
        y = yaml.safe_load(f.read_text())
        return int(y['board_cols']), int(y['board_rows']), float(y['square_m']) * 1000.0
    return 10, 7, 19.0


GIAN_DOC = (2, 4, 3)     # các mức kéo giãn ảnh theo chiều DỌC khi dò (thử lần lượt)


def do_ban_co(gray, cols, rows, huong=None):
    """Dò góc bàn cờ bằng findChessboardCornersSB.
    Camera nhìn gần như dọc theo mặt sàn nên ô cờ bị ép mỏng theo chiều dọc (chỉ cao vài pixel): kéo giãn ảnh
    theo chiều dọc 2-4 lần cho ô gần vuông rồi dò, tính ngược toạ độ về ảnh gốc. Cách này vừa nhanh (~20-40 ms,
    dò không giãn mất ~2 s) vừa dò được bàn cờ xa tới ~45 cm trước camera.
    Chỉ thử 1 hướng: cạnh có `cols` góc nằm NGANG. Bàn cờ nằm trên sàn nên chỉ tìm ở 75 % phía DƯỚI ảnh.
    -> (corners, (cols,rows))."""
    y0 = int(0.25 * gray.shape[0])
    vung = np.ascontiguousarray(gray[y0:])
    for k in GIAN_DOC:
        gian = cv2.resize(vung, None, fx=1.0, fy=float(k), interpolation=cv2.INTER_CUBIC)
        ok, cor = cv2.findChessboardCornersSB(gian, (cols, rows), cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            cor = cor.astype(np.float32)
            cor[:, 0, 1] = (cor[:, 0, 1] + 0.5) / k - 0.5 + y0          # về toạ độ ảnh gốc
            return cor, (cols, rows)
    return None, None


def kiem_o_vuong(gray, K, D, res, sz, o_mm, truoc, lech):
    """Dựng bird's eye 1 mm/px quanh bàn cờ; đo lại ô cờ TRÊN ẢNH bird's eye. -> (bev, aspect, cỡ so với thật, corners)."""
    cols, rows = sz
    x, y = goc_ban_co_theo_xe(res)
    span = max(x.max() - x.min(), y.max() - y.min()) + 0.12
    m = max(0.001, span / 380.0)
    calib = Calib.tu_pose(K, D, res['rvec'], res['tvec'], (gray.shape[1], gray.shape[0]))
    be = BirdEye(calib, x.min() - 0.04, x.max() + 0.04, y.max() + 0.04, y.min() - 0.04, m, ti_le_anh=1.0)
    bev = be.chieu(gray)
    vien = cv2.copyMakeBorder(bev, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    ok, cor = cv2.findChessboardCorners(vien, sz, cv2.CALIB_CB_ADAPTIVE_THRESH)
    if not ok:
        return bev, None, None, None, m
    cor = cor.reshape(rows, cols, 2) - 12
    ngang = float(np.linalg.norm(np.diff(cor, axis=1), axis=2).mean())
    doc = float(np.linalg.norm(np.diff(cor, axis=0), axis=2).mean())
    return bev, ngang / doc, ((ngang + doc) / 2 * m) / (o_mm / 1000.0), cor.reshape(-1, 2), m


def ve_khung(bgr, cor, sz, bev, bev_cor, on_dinh, dong):
    """Hình gửi về: trái = camera + viền bàn cờ, phải = bird's eye. Thu nhỏ cho nhẹ Wi-Fi."""
    h, w = bgr.shape[:2]
    trai = bgr.copy()
    if cor is not None:
        c = cor.reshape(-1, 2)
        cols, rows = sz
        goc = [c[0], c[cols - 1], c[-1], c[-cols]]
        cv2.polylines(trai, [np.array(goc, np.int32)], True, (0, 0, 255), 2)
        for p in goc:
            cv2.circle(trai, (int(p[0]), int(p[1])), 7, (255, 0, 0), 2)
    if on_dinh:
        cv2.rectangle(trai, (2, 2), (w - 3, h - 3), (0, 255, 0), 6)
    trai = cv2.resize(trai, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    if bev is not None:
        phai = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
        if bev_cor is not None:
            for p in bev_cor:
                cv2.circle(phai, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)
        k = (h // 2) / phai.shape[0]
        phai = cv2.resize(phai, (max(10, int(phai.shape[1] * k)), h // 2), interpolation=cv2.INTER_AREA)
    else:
        phai = np.zeros((h // 2, w // 3, 3), np.uint8)
    cv2.putText(trai, 'CAMERA', (6, 16), 0, 0.5, (0, 255, 255), 1)
    cv2.putText(phai, "BIRD'S EYE", (6, 16), 0, 0.5, (0, 255, 255), 1)
    img = np.hstack([trai, phai])
    cv2.rectangle(img, (0, img.shape[0] - 20), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    cv2.putText(img, dong[:70], (4, img.shape[0] - 6), 0, 0.42, (255, 255, 255), 1)
    return img


def bang_ket_qua(res, tb, sz, args, aspect, co_that, hist_loc):
    ket = [('Sai số chiếu lại các góc', f'{res["rms"]:.2f} px', res['rms'] < RMS_MAX),
           ('Camera nghiêng ngang (roll)', f'{res["roll"]:+.2f} độ', abs(res['roll']) < ROLL_MAX),
           ('Lệch hướng so với cạnh bàn cờ (yaw)', f'{res["yaw"]:+.2f} độ', abs(res['yaw']) < YAW_MAX),
           ('Ổn định chiều cao (độ lệch chuẩn)', f'{tb["std_cao"] * 1000:.1f} mm', tb['std_cao'] < ON_DINH_CAO)]
    if args.cao:
        ket.append(('Chiều cao so với thước', f'{res["cao"] * 100:.1f} cm (thước {args.cao * 100:.1f})',
                    abs(res['cao'] - args.cao) < CAO_LECH_MAX))
    if aspect is not None:
        ket.append(('Ô cờ trên bird\'s eye vuông', f'tỉ lệ ngang/dọc {aspect:.3f}', abs(aspect - 1) < O_VUONG_MAX))
        ket.append(('Ô cờ trên bird\'s eye đúng cỡ', f'{co_that * 100:.1f} % so với {args.o_mm:.0f} mm',
                    abs(co_that - 1) < O_VUONG_MAX))
    else:
        ket.append(("Ô cờ trên bird's eye", 'không dò lại được bàn cờ trên ảnh bird\'s eye', False))
    if hist_loc:
        cu = hist_loc[-1]
        ket.append(('Nhất quán với lần chốt trước (đặt bàn cờ chỗ khác)',
                    f'cao {abs(res["cao"] - cu["cao"]) * 1000:.1f} mm, chúi {abs(res["pitch"] - cu["pitch"]):.2f} độ',
                    abs(res['cao'] - cu['cao']) < NHAT_QUAN_CAO and abs(res['pitch'] - cu['pitch']) < NHAT_QUAN_PITCH))
    return ket


def trung_vi(hist):
    return {k: float(np.median([h[k] for h in hist])) for k in ('cao', 'pitch', 'roll', 'yaw', 'rms')} | {
        'rvec': np.median([h['rvec'] for h in hist], axis=0), 'tvec': np.median([h['tvec'] for h in hist], axis=0),
        'W': hist[-1]['W'], 'C_b': hist[-1]['C_b'], 'up': hist[-1]['up'], 'obj': hist[-1]['obj']}


def main():
    cols0, rows0, o0 = luoi_mac_dinh()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--truoc', type=float, required=True, help='quang tâm camera trước tâm xe (m), đo bằng thước')
    ap.add_argument('--cao', type=float, default=0.0, help='chiều cao camera đo bằng thước (m), để đối chiếu')
    ap.add_argument('--lech', type=float, default=0.0, help='camera lệch sang trái so với tâm xe (m)')
    ap.add_argument('--luoi', default=f'{cols0}x{rows0}', help='số góc trong, ví dụ 10x7')
    ap.add_argument('--o-mm', type=float, default=o0, help='cạnh 1 ô cờ (mm)')
    ap.add_argument('--anh', help='thử trên 1 file ảnh (không cần xe)')
    a = ap.parse_args()
    cols, rows = (int(v) for v in a.luoi.lower().split('x'))
    K, D = nap_camera()
    o_m = a.o_mm / 1000.0

    if a.anh:
        bgr = cv2.imread(a.anh)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        cor, sz = do_ban_co(gray, cols, rows)
        if cor is None:
            raise SystemExit('Không thấy bàn cờ trong ảnh.')
        res = giai_pose(cor, sz[0], sz[1], o_m, K, D)
        bev, asp, cot, bc, m = kiem_o_vuong(gray, K, D, res, sz, a.o_mm, a.truoc, a.lech)
        print(f'cao {res["cao"] * 100:.1f} cm | chúi {res["pitch"]:.1f} độ | roll {res["roll"]:+.1f} | '
              f'yaw {res["yaw"]:+.1f} | RMS {res["rms"]:.2f} px | ô vuông {asp} | cỡ {cot}')
        cau_hinh.THU_MUC_LOG.mkdir(parents=True, exist_ok=True)
        out = cau_hinh.THU_MUC_LOG / (Path(a.anh).stem + '_birdeye.jpg')
        cv2.imwrite(str(out), ve_khung(bgr, cor, sz, bev, bc, False, 'thu 1 anh'))
        print(f'Hình: {out}')
        return

    import threading
    with CongCu('birdeye_banco', can_camera=True, hieu_chuan=False) as c:
        log(f"Bird's eye bằng bàn cờ {cols}x{rows} góc trong, ô {a.o_mm:.1f} mm. Đặt bàn cờ NẰM TRÊN SÀN, SÁT trước mũi xe.")
        log(f'Xem hình: http://<ip-xe>:{cau_hinh.CONG_XEM}/anh/birdeye.mjpg | camera trước tâm xe {a.truoc * 100:.1f} cm')
        st = {'job': None, 'hien': None}
        lock = threading.Lock()

        def luong_do():
            """Dò bàn cờ + giải tư thế + chốt. Chậm (0,05-0,7 s/khung) nên nằm riêng, hình xem vẫn mượt."""
            hist, locs, da_ghi, t_log = deque(maxlen=N_ON_DINH), [], None, 0.0
            while c.dang_chay():
                with lock:
                    bgr, st['job'] = st['job'], None
                if bgr is None:
                    time.sleep(0.02)
                    continue
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                cor, sz = do_ban_co(gray, cols, rows)
                res = giai_pose(cor, sz[0], sz[1], o_m, K, D) if cor is not None else None
                if res is None:
                    hist.clear()
                    da_ghi = None
                    with lock:
                        st['hien'] = None
                    if time.monotonic() - t_log > 2:
                        t_log = time.monotonic()
                        log('Chưa thấy bàn cờ. Cạnh dài NẰM NGANG, thấy TRỌN cả bàn cờ trong khung (đừng để bị cắt ở mép dưới: nếu bị cắt thì đẩy '
                            'bàn cờ XA camera hơn; camera ngẩng lên thì sàn gần nhất chỉ thấy từ ~30-35 cm), đủ sáng')
                    continue
                hist.append(res)
                on_dinh = (len(hist) >= N_ON_DINH
                           and np.std([h['cao'] for h in hist]) < ON_DINH_CAO
                           and np.std([h['pitch'] for h in hist]) < ON_DINH_PITCH
                           and np.std([h['yaw'] for h in hist]) < ON_DINH_YAW)
                bev, asp, co_that, bc, m = kiem_o_vuong(gray, K, D, res, sz, a.o_mm, a.truoc, a.lech)
                bx, by = goc_ban_co_theo_xe(res)                 # toạ độ sàn, gốc ngay dưới camera
                gan, xa, ngang = float(bx.min()), float(bx.max()), float(by.mean())
                dong = (f'cao {res["cao"] * 100:.1f} cm | chui {res["pitch"]:.1f} | roll {res["roll"]:+.1f} | '
                        f'yaw {res["yaw"]:+.1f} | rms {res["rms"]:.2f} | ban co cach {gan * 100:.0f}-{xa * 100:.0f} cm'
                        + (' | ON DINH' if on_dinh else ''))
                with lock:
                    st['hien'] = (cor, sz, bev, bc, on_dinh, dong)
                if c.xem:
                    c.xem.cap_nhat(chuong_trinh='birdeye_banco', chi_tiet=dong)
                if time.monotonic() - t_log > 1:
                    t_log = time.monotonic()
                    log(f'cao {res["cao"] * 100:5.1f} cm | chúi {res["pitch"]:5.1f}° | nghiêng ngang {res["roll"]:+5.1f}° | '
                        f'lệch hướng {res["yaw"]:+5.1f}° | RMS {res["rms"]:.2f} px | '
                        f'BÀN CỜ cách camera {gan * 100:.0f} cm (hàng góc gần) tới {xa * 100:.0f} cm (xa), '
                        f'lệch ngang {ngang * 100:+.0f} cm, cách tâm xe {(gan + a.truoc) * 100:.0f} cm | '
                        + (f'ổn định {len(hist)}/{N_ON_DINH}' if not on_dinh else 'ỔN ĐỊNH (giữ yên)'))
                if not on_dinh:
                    continue
                fin = trung_vi(list(hist))
                if da_ghi is not None and abs(fin['cao'] - da_ghi['cao']) < 0.003 \
                        and abs(fin['pitch'] - da_ghi['pitch']) < 0.7:
                    continue                                # đã ghi đúng tư thế này rồi
                tb = {'std_cao': float(np.std([h['cao'] for h in hist]))}
                ket = bang_ket_qua(fin, tb, sz, a, asp, co_that, locs)
                rv_x, tv_x = pose_theo_xe(fin, a.truoc, a.lech)
                f = luu_ung_vien(fin, rv_x, tv_x, gray, ket, a.truoc, a.lech, sz, a.o_mm, asp, co_that,
                                 them={'ban_co_cach_camera_gan_m': round(gan, 3), 'ban_co_cach_camera_xa_m': round(xa, 3),
                                       'ban_co_lech_ngang_m': round(ngang, 3)})
                locs.append(fin)
                da_ghi = fin
                log(f"================ CHỐT LẦN {len(locs)}: BIRD'S EYE BẰNG BÀN CỜ ================")
                for ten, gt, ok in ket:
                    log(f'{ten:48s} {gt:42s} {"ĐẠT" if ok else "KHÔNG ĐẠT"}')
                log(f'Đã lưu ỨNG VIÊN {len(locs)}: {f.name} (chưa dùng cho đến khi bạn chọn). Nhấc bàn cờ đặt CHỖ KHÁC để chốt '
                    'lần nữa (kiểm nhất quán) rồi Ctrl+C khi đủ 2-3 lần.')
                log('Tiếp: bấm "Kéo kết quả từ xe" trên trang calib, rồi chọn lần dùng + vùng nhìn + độ phân giải (bước 4b).')

        threading.Thread(target=luong_do, daemon=True).start()
        so_cu, t_ve = 0, 0.0
        while c.dang_chay():
            jpeg, so = c.xe.anh_jpeg()
            if jpeg is None or so == so_cu or time.monotonic() - t_ve < 1 / 10:
                c.xe.ngu(0.005)
                continue
            so_cu, t_ve = so, time.monotonic()
            bgr = giai_nen_mau(jpeg)
            with lock:
                if st['job'] is None:
                    st['job'] = bgr.copy()                  # luồng dò lấy khung mới nhất khi rảnh
                hien = st['hien']
            if c.xem:
                if hien is None:
                    img = ve_khung(bgr, None, (cols, rows), None, None, False,
                                   'Chua thay ban co: thay TRON ca ban co (khong bi cat o mep duoi), canh dai NAM NGANG')
                else:
                    cor, sz, bev, bc, on_dinh, dong = hien
                    img = ve_khung(bgr, cor, sz, bev, bc, on_dinh, dong)
                c.xem.dat_anh('birdeye', img, chat_luong=60, luon=True)


if __name__ == '__main__':
    main()
