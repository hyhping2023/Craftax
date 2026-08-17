"""RGB frame 编码。

把 host 层 uint8 HWC RGB numpy 数组编码为 PNG bytes。
仅依赖 numpy + Pillow，方便 service 各层复用。
"""
from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

# PNG 文件魔数，用于测试 / 校验。
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def encode_png(frame: np.ndarray) -> bytes:
    """把 uint8 HWC RGB numpy 数组编码为 PNG bytes。

    Args:
        frame: shape (H, W, 3) 的 uint8 RGB 数组。

    Returns:
        PNG 编码后的 bytes。

    Raises:
        ValueError: 输入 shape / dtype 不符合预期时。
    """
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"frame 必须为 HWC RGB 数组，实际 shape={arr.shape}"
        )
    if arr.dtype != np.uint8:
        # 容忍 float，先做量化，但调用方应传 uint8
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()
