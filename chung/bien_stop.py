"""Luật dừng ở biển STOP (biển có thể nằm bất kỳ đâu trên sa bàn).

DungStop: biển STOP trong ROI, khung cao >= cao_min (đủ GẦN) -> đứng yên dung_giay giây rồi đi tiếp, sau đó bỏ qua
STOP bo_qua_giay giây để chạy qua biển. Tham số ở tu_hanh/tham_so.py (STOP_*, BIEN_TIN_MIN).
"""
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
