#!/usr/bin/env python3
"""S0: đứng yên (sau 2 s hiệu chuẩn IMU), chờ ĐÈN XANH (YOLO best.onnx, cả ảnh) rồi mới cho xe chạy.
Có 2 đèn giao thông trong ảnh thì CHỈ nghe đèn TO NHẤT (gần xe nhất). Đèn đó phải ĐỦ LỚN (khung cao >= CAO_MIN_PX,
chỉnh trên bảng điều khiển, ô "Ngưỡng pixel") mới tính. Đỏ / vàng / chưa thấy / còn nhỏ = tiếp tục đứng; xanh
phải SO_LAN_XANH khung liên tiếp mới chạy. Chờ mãi tới khi có xanh (không tự chạy).
Tự chạy trước S1 khi bấm `chay.sh s1_xuat_phat` (thêm --khong-cho-xanh để bỏ).

Chạy riêng:   bash chay.sh s0_den_xanh          (hoặc python3 giai_doan/s0_den_xanh.py)
Chạy tới hết: bash chay.sh s0_den_xanh --tiep   (= FULL MAP)
"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chung import khung  # noqa: E402
from chung.camera import giai_nen_mau  # noqa: E402
from chung.log import log  # noqa: E402

# ===== THAM SỐ (sửa, Ctrl+S, chạy lại) =====================================
VUNG_DEN = None            # (x1, y1, x2, y2) pixel trên ảnh gốc 640x480. None = cả ảnh (đang dùng)
CAO_MIN_PX = 40            # khung đèn cao >= số này mới 'đủ lớn' (px, ảnh 480 cao). Bảng điều khiển ghi đè ở calib/nguong_px.yaml
NGUONG_TIN = 0.50          # độ tin tối thiểu của 'den xanh'
SO_LAN_XANH = 3            # số lần nhận XANH liên tiếp mới cho chạy
CHU_KY_NHAN = 0.30         # giây giữa 2 lần nhận (xe đứng yên, không cần nhanh)
BO_QUA_DEN = False         # True = không chờ đèn, đi luôn (chỉ để tập)
THOI_GIAN_TOI_DA = 600.0   # chờ đèn tối đa (giây)
# ===========================================================================


def mau_den(ket_qua, nguong=None, cao_min_px=None):
    """'do' / 'vang' / 'xanh' / None: màu của đèn TO NHẤT nếu nó đủ lớn (xem den_to_nhat)."""
    from chung.den_onnx import den_to_nhat
    nguong = NGUONG_TIN if nguong is None else nguong
    return den_to_nhat(ket_qua, CAO_MIN_PX if cao_min_px is None else cao_min_px, nguong)[0]


class GiaiDoan(khung.GiaiDoanCoSo):
    TEN = 's0_den_xanh'
    HZ = 10
    CAN_CAMERA = True

    def __init__(self):
        self.THOI_GIAN_TOI_DA = THOI_GIAN_TOI_DA
        self.dem_xanh = 0
        self.ket_qua = []
        self.t_kq = 0.0
        self.cach = '?'
        self.bo = None
        self._chay = True
        self._t_dem = 0.0
        self.mau_cu = 'chua'
        self.cao_min = CAO_MIN_PX
        self.cao = 0.0

    def bat_dau(self, ctx):
        from chung.den_onnx import doc_nguong
        self.cao_min = doc_nguong('den_xanh_s0', CAO_MIN_PX)
        log(f'S0: đứng yên chờ đèn xanh (cả ảnh, nghe đèn to nhất, khung cao >= {self.cao_min:.0f} px, {SO_LAN_XANH} lần liên tiếp)')
        if BO_QUA_DEN:
            log('S0: BO_QUA_DEN = True -> không chờ đèn')
            return
        if ctx.gia_lap:
            self.cach = 'giả lập'
            return
        from chung.den_onnx import NhanDen
        self.bo = NhanDen()
        log(f'S0: nhận đèn bằng {self.bo.ten}' + (f' | lỗi đã gặp: {self.bo.loi}' if self.bo.loi else ''))
        if not self.bo.san_sang:
            raise RuntimeError('không chạy được model đèn trên Jetson (xem lỗi ở trên, chạy onnx_test)')
        self.cach = 'jetson'
        ctx.du_lieu['bo_den'] = self.bo          # S1 dùng lại, khỏi nạp model lần 2 ngay lúc vừa có đèn xanh
        threading.Thread(target=self._vong_nhan, args=(ctx,), daemon=True).start()

    def _vong_nhan(self, ctx):
        from chung.den_onnx import cat_vung, ve_ket_qua
        so_cu = 0
        while self._chay:
            jpeg, so = ctx.xe.anh_jpeg()
            if jpeg is None or so == so_cu:
                time.sleep(0.05)
                continue
            so_cu = so
            t_bd = time.monotonic()
            bgr = giai_nen_mau(jpeg)
            cat, goc = cat_vung(bgr, VUNG_DEN)
            kq, ms = self.bo.nhan(cat)
            self.ket_qua, self.t_kq = kq, time.monotonic()
            ctx.xem.dat_anh('den', ve_ket_qua(bgr, kq, goc, VUNG_DEN))
            con_lai = CHU_KY_NHAN - (time.monotonic() - t_bd)
            if con_lai > 0:             # model chậm hơn chu kỳ (Jetson ~0,85 s) thì nhận liên tục, không ngủ thêm
                time.sleep(con_lai)

    def buoc(self, ctx, dt):
        if BO_QUA_DEN:
            return khung.KetThuc('bỏ qua đèn (BO_QUA_DEN)')
        if ctx.gia_lap:
            self.ket_qua = [{'ten': ctx.xe.den_gia_lap(), 'tin': 1.0, 'hop': [0, 0, 40, 100]}]
            self._cap_nhat(mau_den(self.ket_qua, cao_min_px=self.cao_min))
        elif time.monotonic() - self.t_kq < 1.0 and self.t_kq > self._t_dem:
            self._t_dem = self.t_kq
            self._cap_nhat(mau_den(self.ket_qua, cao_min_px=self.cao_min))
        elif time.monotonic() - self.t_kq >= 1.0:
            self._cap_nhat(None)          # kết quả quá cũ = không thấy đèn = đứng yên
        if self.dem_xanh >= SO_LAN_XANH:
            self._chay = False
            return khung.KetThuc(f'ĐÈN XANH ({SO_LAN_XANH} lần liên tiếp)')
        return 0.0, 0.0

    def _cap_nhat(self, mau):
        self.dem_xanh = self.dem_xanh + 1 if mau == 'xanh' else 0
        if mau != self.mau_cu:
            self.mau_cu = mau
            log({'do': 'S0: ĐÈN ĐỎ -> đứng chờ', 'vang': 'S0: ĐÈN VÀNG -> đứng chờ',
                 'xanh': 'S0: thấy ĐÈN XANH, xác nhận...', None: 'S0: không thấy đèn -> đứng chờ'}[mau])

    def trang_thai(self):
        from chung.den_onnx import den_to_nhat
        _, cao, _ = den_to_nhat(self.ket_qua, self.cao_min, NGUONG_TIN)
        thay = ', '.join(f'{d["ten"]}:{d["tin"]:.2f}' for d in self.ket_qua[:3]) or 'chưa thấy đèn'
        return (f'[{self.cach}] {thay} | đèn to nhất cao {cao:.0f}/{self.cao_min:.0f} px | '
                f'xanh liên tiếp {self.dem_xanh}/{SO_LAN_XANH}')


if __name__ == '__main__':
    khung.main(__file__)
