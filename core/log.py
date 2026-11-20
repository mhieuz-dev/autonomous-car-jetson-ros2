"""In log ra terminal (ngay lập tức) và ghi kèm vào logs/<giờ>_<tên>.log."""
import time

from . import settings

_t0 = time.monotonic()
_file = None


def open_log(name):
    """Mở file log cho lượt chạy này. Trả về đường dẫn."""
    global _file
    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = settings.LOG_DIR / f'{time.strftime("%Y%m%d_%H%M%S")}_{name}.log'
    _file = path.open('a', encoding='utf-8')
    return path


def log(*parts):
    line = f'[{time.monotonic() - _t0:7.2f}] ' + ' '.join(str(p) for p in parts)
    print(line, flush=True)
    if _file is not None:
        _file.write(line + '\n')
        _file.flush()
