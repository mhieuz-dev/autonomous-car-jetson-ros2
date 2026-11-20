"""Luật dừng ở biển STOP (biển có thể nằm bất kỳ đâu trên sa bàn).

StopSignLogic: biển STOP trong ROI, khung cao >= min_height (đủ GẦN) -> đứng yên stop_s giây rồi đi tiếp, sau đó bỏ qua
STOP ignore_s giây để chạy qua biển. Tham số ở autonomy/params.py (STOP_*, SIGN_MIN_CONF).
"""
from .detector import large_sign
from .log import log


class StopSignLogic:
    def __init__(self):
        self.light = -1.0         # giờ hết dừng (giờ của xe)
        self.ignore_until = -1.0  # tới giờ này bỏ qua STOP
        self.n_times = 0

    def step(self, result, t, name, min_height, min_conf, stop_s, ignore_s):
        """True nếu xe phải đứng yên lúc này. kq = danh sách khung nhận được, t = giờ hiện tại (s)."""
        if t < self.light:
            return True
        if self.light > 0:
            self.light, self.ignore_until = -1.0, t + ignore_s
            log(f'{name}: hết {stop_s:g} s dừng ở STOP -> đi tiếp')
        ok, height = large_sign(result, 'stop', min_height, min_conf)
        if ok and t >= self.ignore_until:
            self.light = t + stop_s
            self.n_times += 1
            log(f'{name}: BIỂN STOP (cao {height:.0f} px >= {min_height:g}) -> DỪNG {stop_s:g} s (lần {self.n_times})')
            return True
        return False
