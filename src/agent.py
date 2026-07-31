"""
عقل اللاعب: اختيار الحركة بناءً على تقييم الشبكة العصبية.

فيه طريقتين:
  1. choose_move_shallow: تقييم نقلة واحدة قدام بس (زي أول نسخة من
     المشروع). سريعة جدًا لكن "قصيرة النظر".
  2. choose_move: التخطيط الحقيقي، بيستخدم بحث Minimax + Alpha-Beta
     + Iterative Deepening (موجود في search.py) عشان يشوف قدام
     بعمق أكبر بكتير (مش بس 2-3 نقلات، لكن حسب ميزانية الوقت المديها).

فيه كمان احتمال عشوائي صغير (epsilon) لاختيار حركة عشوائية بدل
الأفضل، عشان اللاعب يجرب حاجات جديدة أثناء التدريب (استكشاف).
"""

import random
import chess

from .nn_eval import evaluate_positions, evaluate_one
from . import search as search_module


def choose_move_shallow(model, board: chess.Board, epsilon: float = 0.0, device: str = "cpu"):
    """
    اختيار قديم: تقييم نقلة واحدة قدام بس (سريع، لكن قصير النظر).
    محفوظة هنا للمقارنة ولو حبيت تشغل نسخة أسرع بدون بحث عميق.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    if random.random() < epsilon:
        return random.choice(legal_moves)

    resulting_boards = []
    for move in legal_moves:
        b = board.copy(stack=False)
        b.push(move)
        resulting_boards.append(b)

    values = evaluate_positions(model, resulting_boards, device=device)

    if board.turn == chess.WHITE:
        best_idx = max(range(len(values)), key=lambda i: values[i])
    else:
        best_idx = min(range(len(values)), key=lambda i: values[i])

    return legal_moves[best_idx]


def choose_move(model, board: chess.Board, epsilon: float = 0.0, device: str = "cpu",
                 time_limit: float = 2.0, max_depth: int = 12, verbose: bool = False):
    """
    اختيار الحركة "بالتخطيط": بحث Minimax + Alpha-Beta + Iterative
    Deepening بميزانية وقت time_limit (بالثواني) وعمق أقصى max_depth.

    epsilon: نسبة صغيرة لاختيار حركة عشوائية بدل نتيجة البحث، للاستكشاف
             أثناء التدريب. في اللعب الجاد (ضد إنسان) خليها 0.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    if random.random() < epsilon:
        return random.choice(legal_moves)

    move, score, depth_reached = search_module.search_best_move(
        model, board,
        time_limit=time_limit,
        max_depth=max_depth,
        device=device,
        verbose=verbose,
    )
    return move
                     
