from __future__ import annotations
from pathlib import Path
import time
from typing import Iterator
import cv2


class VideoStream:
    def __init__(
        self,
        source: str | int,
        open_retries: int = 3,
        retry_delay_s: float = 1.0,
        read_retries: int = 2,
    ) -> None:
        self.source = int(source) if str(source).isdigit() else source
        self.open_retries = max(1, int(open_retries))
        self.retry_delay_s = max(0.0, float(retry_delay_s))
        self.read_retries = max(0, int(read_retries))
        self.capture = self._open_capture()

    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        for attempt in range(self.read_retries + 1):
            ok, frame = self.capture.read()
            if ok:
                return frame
            if attempt < self.read_retries and self._reopen():
                time.sleep(self.retry_delay_s)
        self.release()
        raise StopIteration

    @property
    def fps(self) -> float:
        value = self.capture.get(cv2.CAP_PROP_FPS)
        return float(value if value and value > 1e-3 else 30.0)

    @property
    def is_live_source(self) -> bool:
        return isinstance(self.source, int)

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()

    def _open_capture(self):
        last_error = None
        for attempt in range(self.open_retries):
            capture = cv2.VideoCapture(self.source)
            if capture.isOpened():
                return capture
            capture.release()
            last_error = RuntimeError(
                f"Cannot open video source: {self.source} "
                f"(attempt {attempt + 1}/{self.open_retries})"
            )
            if attempt + 1 < self.open_retries:
                time.sleep(self.retry_delay_s)
        raise last_error or RuntimeError(f"Cannot open video source: {self.source}")

    def _reopen(self) -> bool:
        self.release()
        try:
            self.capture = self._open_capture()
        except RuntimeError:
            return False
        return True


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
