"""
دالة تقييم الشبكة العصبية للمواقف، مستخدمة من agent.py (اللعب البسيط)
و search.py (بحث Minimax + Alpha-Beta) من غير ما نعمل استيراد دائري
بين الملفين.
"""

import numpy as np
import torch

from .encoding import board_to_tensor


@torch.no_grad()
def evaluate_positions(model, boards, device="cpu"):
    """يقيّم مجموعة رقع شطرنج دفعة واحدة (أسرع من واحدة واحدة)."""
    array = np.array([board_to_tensor(b) for b in boards], dtype=np.float32)
    tensors = torch.from_numpy(array).to(device)
    values = model(tensors)
    return values.cpu().tolist()


def evaluate_one(model, board, device="cpu") -> float:
    """يقيّم موقف واحد بس (شكل مختصر لـ evaluate_positions)."""
    return evaluate_positions(model, [board], device=device)[0]
