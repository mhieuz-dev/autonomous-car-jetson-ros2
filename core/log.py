"""In log ra terminal (ngay lập tức) và ghi kèm vào logs/<giờ>_<tên>.log."""
import time

from . import settings

_t0 = time.monotonic()
_file = None


def mo_file(ten):
    """Mở file log cho lượt chạy này. Trả về đường dẫn."""
    global _file
    settings.THU_MUC_LOG.mkdir(parents=True, exist_ok=True)
    path = settings.THU_MUC_LOG / f'{time.strftime("%Y%m%d_%H%M%S")}_{ten}.log'
    _file = path.open('a', encoding='utf-8')
    return path


def log(*phan):
    dong = f'[{time.monotonic() - _t0:7.2f}] ' + ' '.join(str(p) for p in phan)
    print(dong, flush=True)
    if _file is not None:
        _file.write(dong + '\n')
        _file.flush()
