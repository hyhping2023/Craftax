"""CFR H.264 MP4 写入器（imageio + ffmpeg）。

- 固定帧率（CFR），禁止 VFR；
- yuv420p、libx264、CRF 适中（默认 18）；
- close() 后返回文件路径，frame_count 供 validator 核对解码帧数。
"""
from __future__ import annotations

from pathlib import Path

import imageio
import numpy as np

from craftax.contracts import VIDEO_FILENAME_PREFIX, VIDEO_FILENAME_SUFFIX

DEFAULT_CRF = 18
DEFAULT_CODEC = "libx264"


class VideoWriter:
    """累积 uint8 HWC RGB 帧，close() 时编码为 CFR MP4。"""

    def __init__(
        self,
        spool_dir,
        video_id: str,
        fps: int,
        width: int,
        height: int,
        *,
        crf: int = DEFAULT_CRF,
    ):
        if fps <= 0:
            raise ValueError("fps 必须为正")
        if width <= 0 or height <= 0:
            raise ValueError("width/height 必须为正")
        self.fps = int(fps)
        self.width = int(width)
        self.height = int(height)
        self.video_id = str(video_id)
        self.path = Path(spool_dir) / f"{VIDEO_FILENAME_PREFIX}{self.video_id}{VIDEO_FILENAME_SUFFIX}"
        self.frame_count = 0
        self._closed = False
        self._writer = imageio.get_writer(
            str(self.path),
            fps=self.fps,
            codec=DEFAULT_CODEC,
            pixelformat="yuv420p",
            output_params=["-crf", str(int(crf))],
            # 保持原始像素尺寸（130x110 等非 16 倍数）。默认 macro_block_size=16
            # 会把帧缩放为 16 的倍数，导致解码帧与 gold-frames/原始 RGB 尺寸不一致，
            # 破坏 frame_index 逐像素对齐契约。我们的 reader 用同一 imageio/ffmpeg
            # 解码，可正确读取非 16 倍数尺寸。
            macro_block_size=1,
        )

    def add_frame(self, frame: np.ndarray) -> None:
        """追加一帧 uint8 HWC RGB。尺寸必须与构造时一致。"""
        if self._closed:
            raise RuntimeError("VideoWriter 已关闭")
        arr = np.asarray(frame)
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"帧必须是 uint8 HWC RGB，got dtype={arr.dtype} shape={arr.shape}"
            )
        if arr.shape[0] != self.height or arr.shape[1] != self.width:
            raise ValueError(
                f"帧尺寸 ({arr.shape[1]},{arr.shape[0]}) 与构造尺寸 "
                f"({self.width},{self.height}) 不一致"
            )
        self._writer.append_data(arr)
        self.frame_count += 1

    def close(self) -> str:
        """结束编码并返回 MP4 文件路径。返回的帧数必须等于写入帧数。"""
        if not self._closed:
            self._writer.close()
            self._closed = True
        return str(self.path)

    @property
    def is_closed(self) -> bool:
        return self._closed

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
