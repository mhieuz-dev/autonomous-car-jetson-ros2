"""Biển STOP ở vị trí NGẪU NHIÊN trên map: nhận và dừng ở nhiều giai đoạn, không chỉ S1.

- DungStop: luật dừng dùng chung (S1 và khung): biển STOP trong ROI, khung cao >= cao_min (đủ GẦN) -> đứng yên
  dung_giay giây rồi đi tiếp, sau đó bỏ qua STOP bo_qua_giay giây để chạy qua biển.
- NhanNen: luồng nhận biển chạy nền cho các giai đoạn KHÔNG tự nhận biển (S5, S6, S7). Dùng lại model của S0/S1 nếu
  đã nạp (TensorRT trên Jetson), chỉ nhìn ROI chung (calib/roi_bien.yaml).
Tham số nằm ở đầu giai_doan/s1_xuat_phat.py (STOP_*, BIEN_TIN_MIN, STOP_CA_MAP).
"""
import threading
import time

from .den_onnx import bien_lon
from .log import log


class DungStop:
    def __init__(self):
        self.den = -1.0          # giờ hết dừng (giờ của xe)
        self.bo_qua = -1.0       # tới giờ này bỏ qua STOP
        self.so_lan = 0

    def buoc(self, kq, t, ten, cao_min, tin_min, dung_giay, bo_qua_giay):
        """True nếu xe phải đứng yên lúc này. kq = danh sách khung nhận được, t = giờ hiện tại (s)."""
        if t < self.den:
            return True
        if self.den > 0:
            self.den, self.bo_qua = -1.0, t + bo_qua_giay
            log(f'{ten}: hết {dung_giay:g} s dừng ở STOP -> đi tiếp')
        ok, cao = bien_lon(kq, 'stop', cao_min, tin_min)
        if ok and t >= self.bo_qua:
            self.den = t + dung_giay
            self.so_lan += 1
            log(f'{ten}: BIỂN STOP (cao {cao:.0f} px >= {cao_min:g}) -> DỪNG {dung_giay:g} s (lần {self.so_lan})')
            return True
        return False


class NhanNen:
    """Nhận biển ở luồng riêng trên ảnh camera mới nhất; kq() trả khung mới (cũ quá 2 s thì coi như không thấy)."""

    def __init__(self, ctx):
        self.ctx = ctx
        self._kq, self._t = [], 0.0
        self.chay = True
        if not ctx.gia_lap:
            threading.Thread(target=self._vong, daemon=True).start()

    def kq(self):
        if self.ctx.gia_lap:
            return self.ctx.xe.den_doc_duong()
        return self._kq if time.monotonic() - self._t < 2.0 else []

    def _vong(self):
        from .camera import giai_nen_mau
        from .den_onnx import NhanDen, doc_roi, nhan_trong_roi, ve_ket_qua
        bo = self.ctx.du_lieu.get('bo_den') or NhanDen()
        self.ctx.du_lieu['bo_den'] = bo
        if not bo.san_sang:
            log('CẢNH BÁO: không chạy được model biển -> các giai đoạn sau S1 KHÔNG nhận được biển STOP')
            return
        roi = doc_roi()
        log(f'Nhận biển STOP nền bằng {bo.ten}, ROI hàng {roi[1]}..{roi[3]}')
        so_cu = 0
        while self.chay:
            jpeg, so = self.ctx.xe.anh_jpeg()
            if jpeg is None or so == so_cu:
                time.sleep(0.03)
                continue
            so_cu = so
            bgr = giai_nen_mau(jpeg)
            kq, _ = nhan_trong_roi(bo, bgr, roi)
            self._kq, self._t = kq, time.monotonic()
            if self.ctx.xem is not None:
                self.ctx.xem.dat_anh('bien', ve_ket_qua(bgr, kq, vung=roi, chi_vung=True))
