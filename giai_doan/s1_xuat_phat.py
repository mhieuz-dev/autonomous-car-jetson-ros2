#!/usr/bin/env python3
"""S1: từ ô xuất phát tới chân dốc: BÁM VẠCH PHẢI bằng BIRD'S EYE (như S7), dừng cho NGƯỜI ĐI BỘ (LiDAR), nhận ĐÈN
khi đèn đủ LỚN trong ảnh (đỏ / vàng dừng, xanh đi tiếp), và chuyển sang S2 khi IMU thấy mũi xe ngóc lên (chân dốc).

- Bám vạch: dùng chung bộ dò vạch + PID khoảng cách của S7 (tham số ở s7_bam_vach.py). Mất vạch thì giữ hướng
  chạy chậm, KHÔNG dừng (đường xuất phát có nhiều vạch ngang, dễ mất vạch một lúc).
- Người đi bộ: LiDAR, CHỈ ở S1 (qua S2 là hết, hầm không bao giờ dừng nhầm).
- Biển + đèn: model best.onnx chỉ trong ROI (mặc định nửa DƯỚI ảnh, hàng 240..480; chỉnh trên bảng điều khiển,
  lưu calib/roi_bien.yaml), CHỈ ở S1. Biển STOP đủ lớn: dừng STOP_DUNG_GIAY giây rồi đi tiếp; biển rẽ trái/phải đủ lớn thì ghi nhận (log + ctx.du_lieu['bien_re']), các biển khác bỏ qua.
- Đèn: logic như cũ. 'Đủ gần' = khung bao đèn cao >= DEN_CAO_MIN_PX pixel. Muốn chọn số
  này: đứng xe cách đèn đúng khoảng bạn muốn (~15 cm), chạy `bash chay.sh onnx_test`, đọc chiều cao khung đèn.
  Log luôn in 'ĐÈN ... cao N px' khi thấy đèn (kể cả chưa đủ lớn) để bạn chỉnh.
- Khi sang S2 đặt hướng giữ (yaw_doc): CANH_VE_0 = True thì là 0 (mũi xe lúc xuất phát), S2 canh IMU về 0 rồi leo.
Chạy riêng: bash chay.sh s1_xuat_phat   | tới hết: bash chay.sh s1_xuat_phat --tiep
"""
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chung import cau_hinh, khung  # noqa: E402
from chung.camera import giai_nen_mau  # noqa: E402
from chung.lidar import diem_xy  # noqa: E402
from chung.log import log  # noqa: E402
from chung.ve import lidar_tren_xuong, luoi  # noqa: E402
from giai_doan import s5_ham  # noqa: E402
from giai_doan.s7_bam_vach import DieuKhienLan  # noqa: E402

# ===== THAM SỐ (sửa, Ctrl+S, chạy lại) =====================================
V_CHAY = 0.5              # m/s khi bám vạch trước dốc

LIDAR_TOI_CAN_M = cau_hinh.LIDAR_TOI_CAN_M   # sửa trong chung/cau_hinh.py (đo bằng thước)
KHOANG_HO_M = 0.30         # dừng khi người cách CẢN TRƯỚC nhỏ hơn số này
THOI_GIAN_PHAN_UNG = 0.25 # s: cộng thêm V_CHAY * số này vào ngưỡng (xe cần thời gian dừng)
NUA_RONG_M = 0.15          # nửa bề rộng vùng xét phía trước (m): chỉ vật trong dải ±10 cm trước mũi mới tính là người
SO_DIEM_MIN = 3            # ít nhất bấy nhiêu điểm LiDAR trong vùng mới tính là có người
BO_NGUOI_KHI_THAY_HAM = True   # LiDAR đã thấy cổng hầm (2 tường) thì thôi dừng vì 'người' (tường/dốc hầm bị nhận nhầm)
THOI_GIAN_TRONG = 1.0      # vùng trống liên tục bao lâu thì chạy lại (s)

DEN_NGUON = 'xe'           # 'xe' = model chạy trên Jetson (đang dùng) | 'laptop' = model chạy trên laptop qua DDS (cong_cu/den_laptop.py, nút Bật trên bảng điều khiển); bản laptop từng làm xe mất cảm biến odom/scan
DEN_BAT = True             # False = không nhận đèn ở S1
DEN_CAO_MIN_PX = 40        # đèn 'đủ gần' khi khung bao cao >= số này (pixel, ảnh 640x480). CHỌN BẰNG onnx_test
DEN_TIN_MIN = 0.5          # độ tin tối thiểu của model (đèn xanh)
DEN_DO_TIN_MIN = 0.75      # độ tin tối thiểu riêng cho đèn ĐỎ/VÀNG (cao hơn = bớt nhạy, bớt dừng nhầm)
BIEN_BAT = True            # nhận biển trong ROI (ROI chỉnh trên bảng điều khiển, lưu calib/roi_bien.yaml)
BIEN_NHAN = ('stop', 're trai', 're phai')   # CHỈ nhận 3 biển này; STOP dừng, rẽ trái/phải ghi nhận (chưa có lệnh rẽ)
BIEN_TIN_MIN = 0.6         # độ tin tối thiểu của biển
STOP_CAO_MIN_PX = 60       # biển STOP 'đủ gần' khi khung cao >= số này (px)
STOP_DUNG_GIAY = 1.0       # đứng yên bấy nhiêu giây ở biển STOP
STOP_BO_QUA_GIAY = 6.0     # sau khi dừng xong, bỏ qua biển STOP bấy nhiêu giây (để xe chạy qua biển)
STOP_CA_MAP = ('s5_ham', 's6_ra_ham', 's7_bam_vach', 's7_phat_lai', 's7_stop_ghi')   # STOP ngẫu nhiên: ngoài S1 các giai đoạn này cũng dừng (dốc S2-S4 thì không)
RE_CAO_MIN_PX = 60         # biển rẽ trái/phải 'đủ gần' khi khung cao >= số này (px)
RE_BO_QUA_GIAY = 6.0       # ghi nhận 1 biển rẽ xong, bỏ qua cùng biển đó bấy nhiêu giây (khỏi đếm lặp)
DEN_CHU_KY = 0.5           # giây giữa 2 lần nhận đèn
DEN_XANH_LIEN = 2          # số lần nhận XANH liên tiếp mới đi tiếp sau khi dừng vì đỏ / vàng
DEN_CHO_TOI_DA = 20.0      # chờ đèn tối đa (s); hết giờ vẫn chưa xanh thì đi tiếp (tránh kẹt vì nhận nhầm)

CANH_VE_0 = True           # True: lên dốc thì S2-S4 giữ yaw = 0 (hướng mũi xe lúc xuất phát); False: giữ hướng lúc tới chân dốc
NGUONG_LEN_DOC_DO = 5.0    # pitch > số này (độ) ... TẠM: chạy 0,30 m/s trên sàn PHẲNG pitch đã +2,4..+4,0 (xe ngóc mũi theo tốc độ), dốc thật +3,6 + ~2,7
THOI_GIAN_LEN = 0.8        # ... liên tục bao lâu (s) thì coi là đã tới dốc -> S2
QUANG_DUONG_MIN = 0.20     # chạy ít nhất bấy nhiêu mét rồi mới xét dốc (tránh rung lúc tăng tốc)
THOI_GIAN_TOI_DA = 60.0
# ===========================================================================


def nguong_dung():
    return LIDAR_TOI_CAN_M + KHOANG_HO_M + V_CHAY * THOI_GIAN_PHAN_UNG


def co_nguoi(scan):
    """(có người?, khoảng cách gần nhất trong vùng tính từ LiDAR)."""
    p = diem_xy(scan, 2.0)
    if len(p) == 0:
        return False, math.inf
    vung = p[(p[:, 0] > 0.0) & (np.abs(p[:, 1]) < NUA_RONG_M)]
    gan = float(vung[:, 0].min()) if len(vung) else math.inf
    trong = vung[vung[:, 0] < nguong_dung()]
    return len(trong) >= SO_DIEM_MIN, gan


class GiaiDoan(khung.GiaiDoanCoSo):
    TEN = 's1_xuat_phat'
    HZ = 20
    CAN_CAMERA = True

    def __init__(self):
        self.ham = s5_ham.DemHam()
        self.bo_nguoi = False
        self.THOI_GIAN_TOI_DA = THOI_GIAN_TOI_DA
        self.dk = None
        self.cho = False                 # đang chờ người đi qua
        self.t_trong = 0.0
        self.t_len = 0.0
        self.gan = math.inf
        self.lan_dung = 0
        # đèn
        self.bo = None
        self.den = (None, 0.0, 0.0, 0.0)     # (màu đủ lớn, cao lớn nhất, tin, giờ nhận)
        self.den_cho = False                 # đang dừng chờ xanh
        self.den_t_dung = 0.0
        self.den_xanh_dem = 0
        self.den_so_lan = 0
        self._so_cam = -1
        self._t_ngoai = None
        self._t_bd = None
        self.kq = []                         # mọi khung nhận được trong ROI (biển + đèn)
        self.roi = None
        self._dang_stop = False              # đang đứng vì STOP (luật ở chung/bien_stop.DungStop, dùng chung cả map)
        self.re_bo_qua = {'trai': -1.0, 'phai': -1.0}
        self._bien_cu = ()
        self._chay_luong = True
        self._mau_cu = 'chua'

    # ---------------------------------------------------------------- đèn ---
    def bat_dau(self, ctx):
        self.dk = DieuKhienLan(ctx.gia_lap)
        ctx.du_lieu.pop('yaw_doc', None)
        log(f'S1: bám vạch phải bằng bird\'s eye {V_CHAY} m/s | người đi bộ: LiDAR dừng < {nguong_dung():.2f} m | '
            f'đèn: {"BẬT, đủ lớn khi cao >= " + str(DEN_CAO_MIN_PX) + " px" if DEN_BAT else "TẮT"}')
        self._t_bd = time.monotonic()
        if DEN_BAT and not ctx.gia_lap and DEN_NGUON == 'laptop':
            log('S1: nhận biển/đèn bằng model trên LAPTOP qua DDS (cần bảng điều khiển đang chạy, nút "Nhận biển bằng laptop" Bật)')
        elif DEN_BAT and not ctx.gia_lap:
            from chung.den_onnx import NhanDen, doc_roi
            self.roi = doc_roi()
            log(f'S1: ROI nhận biển/đèn: hàng {self.roi[1]}..{self.roi[3]} (sửa trên bảng điều khiển)')
            self.bo = ctx.du_lieu.get('bo_den') or NhanDen()     # có S0 trước thì dùng lại model đã nạp
            log(f'S1: nhận đèn bằng {self.bo.ten}' + (' (dùng lại model của S0)' if ctx.du_lieu.get('bo_den') else '')
                + (f' | lỗi đã gặp: {self.bo.loi}' if self.bo.loi else ''))
            if not self.bo.san_sang:
                log('CẢNH BÁO S1: không chạy được model đèn, S1 chạy KHÔNG nhận đèn')
            else:
                threading.Thread(target=self._vong_nhan, args=(ctx,), daemon=True).start()

    def _vong_nhan(self, ctx):
        from chung.den_onnx import mau_den_lon, nhan_trong_roi, ve_ket_qua
        so_cu = 0
        while self._chay_luong:
            jpeg, so = ctx.xe.anh_jpeg()
            if jpeg is None or so == so_cu:
                time.sleep(0.05)
                continue
            so_cu = so
            t_bd = time.monotonic()
            bgr = giai_nen_mau(jpeg)
            kq, _ = nhan_trong_roi(self.bo, bgr, self.roi)
            self.kq = kq
            self.den = mau_den_lon(kq, DEN_CAO_MIN_PX, DEN_TIN_MIN, DEN_DO_TIN_MIN) + (time.monotonic(),)
            if ctx.xem is not None:
                ctx.xem.dat_anh('den', ve_ket_qua(bgr, kq, vung=self.roi, chi_vung=True))
            con_lai = DEN_CHU_KY - (time.monotonic() - t_bd)
            if con_lai > 0:             # model chậm hơn chu kỳ thì chạy liên tục, không ngủ thêm
                time.sleep(con_lai)

    def _den_hien_tai(self, ctx):
        """(màu đủ lớn | None, cao lớn nhất, đã cũ?)."""
        if ctx.gia_lap:
            from chung.den_onnx import mau_den_lon
            m, cao, tin = mau_den_lon(ctx.xe.den_doc_duong(), DEN_CAO_MIN_PX, DEN_TIN_MIN, DEN_DO_TIN_MIN)
            return m, cao, False
        m, cao, tin, t = self.den
        return m, cao, (time.monotonic() - t) > 2.0

    def _cap_nhat_laptop(self, ctx):
        """DEN_NGUON='laptop': nhận kết quả đèn/biển laptop gửi qua DDS (topic /champion/den)."""
        if not DEN_BAT or ctx.gia_lap or DEN_NGUON != 'laptop':
            return
        from chung.den_onnx import mau_den_lon
        n = getattr(ctx.xe, 'den_ngoai', None)
        if n is None:
            if time.monotonic() - self._t_bd > 5.0 and self._t_ngoai is None:
                self._t_ngoai = 0
                log('S1: CẢNH BÁO 5 s chưa nhận kết quả đèn/biển từ laptop. Bật "Nhận biển bằng laptop" trên bảng điều khiển (hoặc DEN_NGUON = "xe")')
        elif n['t'] != self._t_ngoai:
            self._t_ngoai = n['t']
            self.kq = n['kq']
            self.den = mau_den_lon(self.kq, DEN_CAO_MIN_PX, DEN_TIN_MIN, DEN_DO_TIN_MIN) + (n['t'],)

    def _xu_ly_bien(self, ctx):
        """True nếu đang đứng vì biển STOP. Biển khác: chỉ ghi log (biển rẽ để vòng 2)."""
        if not BIEN_BAT:
            return False
        from chung.den_onnx import bien_lon
        kq = ctx.xe.den_doc_duong() if ctx.gia_lap else self.kq
        ten = tuple(sorted({d['ten'] for d in kq if d['tin'] >= BIEN_TIN_MIN and d['ten'] in BIEN_NHAN}))
        if ten != self._bien_cu:
            self._bien_cu = ten
            if ten:
                log('S1: biển trong ROI: ' + ', '.join(ten))
        t = ctx.thoi_gian()
        for ten_lop, huong in (('re trai', 'trai'), ('re phai', 'phai')):
            ok, cao = bien_lon(kq, ten_lop, RE_CAO_MIN_PX, BIEN_TIN_MIN)
            if ok and ten_lop in BIEN_NHAN and t >= self.re_bo_qua[huong]:
                self.re_bo_qua[huong] = t + RE_BO_QUA_GIAY
                ctx.du_lieu.setdefault('bien_re', []).append((round(t, 1), huong))
                log(f'S1: BIỂN RẼ {huong.upper()} (cao {cao:.0f} px >= {RE_CAO_MIN_PX}), đã ghi nhận (chưa có lệnh rẽ)')
        if ctx.dung_stop is None:
            from chung.bien_stop import DungStop
            ctx.dung_stop = DungStop()
        if ctx.dung_stop.buoc(kq, t, 'S1', STOP_CAO_MIN_PX, BIEN_TIN_MIN, STOP_DUNG_GIAY, STOP_BO_QUA_GIAY):
            if not self._dang_stop:
                self.dk.dat_lai()
            self._dang_stop = True
            return True
        self._dang_stop = False
        return False

    def _xu_ly_den(self, ctx, dt):
        """True nếu phải ĐỨNG (đèn đỏ / vàng đủ lớn, hoặc đang chờ xanh)."""
        if not DEN_BAT:
            return False
        mau, cao, cu = self._den_hien_tai(ctx)
        if cu:
            mau = None
        if mau != self._mau_cu:
            self._mau_cu = mau
            if mau:
                log(f'S1: ĐÈN {mau.upper()} đủ lớn (cao {cao:.0f} px >= {DEN_CAO_MIN_PX})')
            elif cao > 0:
                log(f'S1: thấy đèn nhưng chưa đủ lớn (cao {cao:.0f} px < {DEN_CAO_MIN_PX})')
        if not self.den_cho:
            if mau in ('do', 'vang'):
                self.den_cho, self.den_t_dung, self.den_xanh_dem = True, ctx.thoi_gian(), 0
                self.den_so_lan += 1
                self.dk.dat_lai()
                log(f'S1: ĐÈN {"ĐỎ" if mau == "do" else "VÀNG"} -> DỪNG chờ xanh (lần {self.den_so_lan})')
            return self.den_cho
        # đang chờ xanh
        self.den_xanh_dem = self.den_xanh_dem + 1 if mau == 'xanh' else 0
        if mau in ('do', 'vang'):
            self.den_xanh_dem = 0
        if self.den_xanh_dem >= DEN_XANH_LIEN * max(1, int(DEN_CHU_KY * self.HZ)):
            self.den_cho = False
            log('S1: ĐÈN XANH -> đi tiếp')
            return False
        if ctx.thoi_gian() - self.den_t_dung > DEN_CHO_TOI_DA:
            self.den_cho = False
            log(f'S1: CẢNH BÁO chờ đèn quá {DEN_CHO_TOI_DA:.0f} s chưa thấy xanh -> đi tiếp (kiểm tra nhận đèn)')
            return False
        return True

    # ------------------------------------------------------------- vòng lặp ---
    def buoc(self, ctx, dt):
        imu = ctx.imu
        self._cap_nhat_laptop(ctx)
        nguoi, self.gan = co_nguoi(ctx.xe.scan)
        thay_ham = self.ham.cap_nhat(ctx.xe.scan) >= 2
        if nguoi and BO_NGUOI_KHI_THAY_HAM and thay_ham:
            if not self.bo_nguoi:
                self.bo_nguoi = True
                log(f'S1: LiDAR thấy cổng hầm, bỏ qua "người" cách {self.gan:.2f} m (là tường/dốc hầm)')
            nguoi = False
        if nguoi:
            if not self.cho:
                self.lan_dung += 1
                log(f'S1: NGƯỜI ĐI BỘ cách LiDAR {self.gan:.2f} m -> DỪNG (lần {self.lan_dung})')
            self.cho, self.t_trong = True, 0.0
            self.dk.dat_lai()
            return 0.0, 0.0, 'gap'
        if self.cho:
            self.t_trong += dt
            if self.t_trong < THOI_GIAN_TRONG:
                return 0.0, 0.0, 'gap'
            self.cho = False
        if self._xu_ly_den(ctx, dt):
            return 0.0, 0.0, 'gap'
        if self._xu_ly_bien(ctx):
            return 0.0, 0.0, 'gap'
        if math.degrees(imu.pitch) > NGUONG_LEN_DOC_DO and ctx.quang_duong() > QUANG_DUONG_MIN:
            self.t_len += dt
            if self.t_len >= THOI_GIAN_LEN:
                ctx.du_lieu['yaw_doc'] = 0.0 if CANH_VE_0 else imu.yaw
                self._chay_luong = False
                return khung.KetThuc(f'tới chân dốc (pitch {math.degrees(imu.pitch):+.1f} độ, yaw lúc này '
                                     f'{math.degrees(imu.yaw):+.1f} độ, S2-S4 giữ '
                                     f'{math.degrees(ctx.du_lieu["yaw_doc"]):+.1f} độ)')
        else:
            self.t_len = 0.0
        v, w, _ = self.dk.buoc(ctx, dt, V_CHAY, debug=ctx.xem is not None and ctx.xem.dang_xem('vach'))
        return v, w

    def ket_thuc(self, ctx, ly_do):
        self._chay_luong = False

    def trang_thai(self):
        if self.dk is None:
            return ''
        den = f'ĐÈN {self._mau_cu.upper()}' if self._mau_cu not in (None, 'chua') else 'đèn -'
        return (f'{"CHỜ NGƯỜI" if self.cho else ("CHỜ ĐÈN" if self.den_cho else "chạy")} | trước {self.gan:.2f} m '
                f'(dừng < {nguong_dung():.2f}) | {den} | {self.dk.mo_ta()}')

    def ve(self, ctx):
        ctx.xem.dat_anh('lidar', lidar_tren_xuong(
            ctx.xe.scan, non=[(0, math.degrees(math.atan2(NUA_RONG_M, nguong_dung())), nguong_dung(),
                               (0, 0, 255) if self.cho else (0, 255, 0))],
            chu=('CHO NGUOI' if self.cho else 'chay')))
        if ctx.xem.dang_xem('vach'):
            ctx.xem.dat_anh('vach', luoi(list(self.dk.bam.anh.items())) if self.dk.bam.anh
                            else luoi([('chua co khung debug', None)]))
        else:
            ctx.xem.anh.setdefault('vach', None)


if __name__ == '__main__':
    khung.main(__file__)
