"""
عقل اللاعب: بياخد رقعة شطرنج ويختار أفضل حركة بناءً على تقييم الشبكة
العصبية للموقف اللي هيحصل بعد كل حركة ممكنة (بحث بعمق نقلة واحدة).

فيه احتمال عشوائي صغير (epsilon) لاختيار حركة عشوائية بدل الأفضل،
عشان اللاعب يجرب حاجات جديدة أثناء التدريب (استكشاف) بدل ما يكرر
نفس الأخطاء أو يعلق في نمط واحد.
"""

import random
import numpy as np
import torch
import chess

from .encoding import board_to_tensor


@torch.no_grad()
def evaluate_positions(model, boards, device="cpu"):
    """يقيّم مجموعة رقع شطرنج دفعة واحدة (أسرع من واحدة واحدة)."""
    array = np.array([board_to_tensor(b) for b in boards], dtype=np.float32)
    tensors = torch.from_numpy(array).to(device)
    values = model(tensors)
    return values.cpu().tolist()


def choose_move(model, board: chess.Board, epsilon: float = 0.0, device: str = "cpu"):
    """
    يختار حركة للاعب صاحب الدور الحالي.

    epsilon: نسبة العشوائية (0 = دايمًا أفضل حركة حسب الموديل،
             0.15 يعني 15% من الوقت يلعب حركة عشوائية للاستكشاف)
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    if random.random() < epsilon:
        return random.choice(legal_moves)

    # نقيّم كل حركة ممكنة عن طريق تجربتها ثم تقييم الموقف الناتج
    resulting_boards = []
    for move in legal_moves:
        b = board.copy(stack=False)
        b.push(move)
        resulting_boards.append(b)

    values = evaluate_positions(model, resulting_boards, device=device)

    # القيمة دايمًا من منظور الأبيض، فلو الدور الحالي للأسود
    # يبقى الأسود عايز أقل قيمة (أقرب لـ -1)
    if board.turn == chess.WHITE:
        best_idx = max(range(len(values)), key=lambda i: values[i])
    else:
        best_idx = min(range(len(values)), key=lambda i: values[i])

    return legal_moves[best_idx]
