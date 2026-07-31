"""
بحث حقيقي (Planning) بدل تقييم نقلة واحدة قدام.

الفكرة:
  - Minimax + Alpha-Beta Pruning: بيبحث كل الاحتمالات لعمق معيّن، لكن
    "يقصّ" (يقلّم) الفروع اللي أكيد مش هتتلعب، فبيوصل لعمق أكبر بكتير
    من نفس عدد العمليات الحسابية.
  - Iterative Deepening: بدل ما تحدد عمق ثابت (3 نقلات مثلاً)، بنديله
    "ميزانية وقت" (مثلاً ثانيتين)، وهو بيبحث بعمق 1، بعدين 2، بعدين 3...
    لحد ما الوقت يخلص. ده معناه إنه في المواضع البسيطة (قليل قطع، قليل
    احتمالات) هيوصل بعمق كبير جدًا (6-8 نقلات وأكتر)، وفي المواضع
    المعقدة هيبحث أقل عمق لكن هيفضل يديله أفضل حركة لقاها لحد دلوقتي.
  - ترتيب الحركات (Move Ordering): بنجرب الحركات "الواعدة" (الأكل
    الأول، وبالذات أكل قطعة كبيرة بقطعة صغيرة) قبل باقي الحركات، عشان
    الـ Alpha-Beta يقدر يقص فروع أكتر بسرعة.
  - Quiescence Search: في نهاية البحث، لو آخر حركة كانت أكل، بنكمّل
    شوية إضافية في خطوط الأكل بس، عشان نتجنب "وهم الأفق" (مثلاً نوقف
    البحث في لحظة إحنا فيها آخدين وزير لكن هيتاكل تاني في النقلة اللي
    بعدها لو كملنا).
"""

import time
import chess

from .nn_eval import evaluate_one as leaf_value_raw

# قيم القطع التقريبية، مستخدمة بس في ترتيب الحركات (مش في التقييم النهائي)
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

MATE_SCORE = 1000.0  # قيمة كبيرة لتمثيل "كش ملك" في البحث


def order_moves(board: chess.Board, moves):
    """يرتب الحركات: الأكل الجيد (MVV-LVA) الأول، بعدين الكش، بعدين الباقي."""

    def move_score(move: chess.Move) -> int:
        score = 0
        if board.is_capture(move):
            victim = board.piece_type_at(move.to_square)
            attacker = board.piece_type_at(move.from_square)
            victim_value = PIECE_VALUES.get(victim, 0) if victim else 1  # en passant
            attacker_value = PIECE_VALUES.get(attacker, 0)
            # MVV-LVA: نفضّل أكل قطعة غالية بقطعة رخيصة
            score += 1000 + victim_value * 10 - attacker_value
        if move.promotion:
            score += 900
        if board.gives_check(move):
            score += 50
        return score

    return sorted(moves, key=move_score, reverse=True)


def leaf_value(model, board: chess.Board, device: str = "cpu") -> float:
    """تقييم الشبكة العصبية لموقف نهائي (ورقة الشجرة)."""
    return leaf_value_raw(model, board, device=device)


def quiescence(model, board: chess.Board, alpha: float, beta: float,
               device: str, qdepth: int = 0, max_qdepth: int = 4) -> float:
    """
    بحث "هدوء": بيكمل بس في حركات الأكل (والكش) عشان يتجنب وهم الأفق.
    مثال: لو آخر حركة في البحث العادي كانت "ناخد وزير"، من غير الجزء
    ده كنا ممكن نفتكر إننا كسبنا وزير مجانًا، مع إن الحقيقة إن القطعة
    اللي أكلت هتتاكل هي كمان في النقلة اللي بعدها.
    """
    stand_pat = leaf_value(model, board, device)

    if board.turn == chess.WHITE:
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)
    else:
        if stand_pat <= alpha:
            return alpha
        beta = min(beta, stand_pat)

    if qdepth >= max_qdepth:
        return stand_pat

    capture_moves = [m for m in board.legal_moves if board.is_capture(m)]
    if not capture_moves:
        return stand_pat

    capture_moves = order_moves(board, capture_moves)

    for move in capture_moves:
        board.push(move)
        score = quiescence(model, board, alpha, beta, device, qdepth + 1, max_qdepth)
        board.pop()

        if board.turn == chess.WHITE:  # يعني إحنا كنا بنلعب بالأسود قبل الحركة
            beta = min(beta, score)
            if beta <= alpha:
                return alpha
        else:
            alpha = max(alpha, score)
            if alpha >= beta:
                return beta

    return alpha if board.turn == chess.BLACK else beta


def alphabeta(model, board: chess.Board, depth: int, alpha: float, beta: float,
              device: str, deadline: float, use_quiescence: bool = True):
    """
    Minimax + Alpha-Beta. القيمة دايمًا من منظور الأبيض (زي الشبكة).
    يرجع (score, best_move). best_move ممكن يكون None لو depth == 0
    أو مفيش وقت.
    """
    if time.time() > deadline:
        return leaf_value(model, board, device), None

    if board.is_checkmate():
        # كش ملك: أسوأ حاجة ممكنة لصاحب الدور الحالي
        score = -MATE_SCORE - depth if board.turn == chess.WHITE else MATE_SCORE + depth
        return score, None

    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0.0, None

    if depth == 0:
        if use_quiescence:
            return quiescence(model, board, alpha, beta, device), None
        return leaf_value(model, board, device), None

    legal_moves = order_moves(board, list(board.legal_moves))
    best_move = legal_moves[0]

    if board.turn == chess.WHITE:
        best_score = float("-inf")
        for move in legal_moves:
            board.push(move)
            score, _ = alphabeta(model, board, depth - 1, alpha, beta, device, deadline, use_quiescence)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break  # تقليم Beta: الخصم مش هيسمحلنا نوصل هنا أصلاً
        return best_score, best_move
    else:
        best_score = float("inf")
        for move in legal_moves:
            board.push(move)
            score, _ = alphabeta(model, board, depth - 1, alpha, beta, device, deadline, use_quiescence)
            board.pop()

            if score < best_score:
                best_score = score
                best_move = move
            beta = min(beta, best_score)
            if alpha >= beta:
                break  # تقليم Alpha
        return best_score, best_move


def search_best_move(model, board: chess.Board, time_limit: float = 2.0,
                      max_depth: int = 12, device: str = "cpu", verbose: bool = False):
    """
    التخطيط الرئيسي: Iterative Deepening.
    بيبحث بعمق 1، بعدين 2، بعدين 3... لحد ما ميزانية الوقت (time_limit
    بالثواني) تخلص أو يوصل max_depth. كل مرة بيبدأ عمق جديد، بيستخدم
    أفضل حركة من العمق اللي فات كأول حركة يجرّبها (بتحسن كفاءة التقليم).
    """
    deadline = time.time() + time_limit
    best_move = None
    best_score = None
    reached_depth = 0

    for depth in range(1, max_depth + 1):
        if time.time() > deadline:
            break

        score, move = alphabeta(
            model, board, depth,
            alpha=float("-inf"), beta=float("inf"),
            device=device, deadline=deadline,
        )

        # نأخذ نتيجة العمق ده بس لو خلص فعلًا قبل انتهاء الوقت
        if time.time() <= deadline and move is not None:
            best_move, best_score = move, score
            reached_depth = depth

        # لو لقينا مسار كش ملك أكيد، مفيش داعي نكمل نبحث أعمق
        if best_score is not None and abs(best_score) > MATE_SCORE / 2:
            break

    if best_move is None:
        # حالة نادرة: خلص الوقت قبل حتى عمق 1 -- ارجع لأول حركة قانونية
        legal_moves = list(board.legal_moves)
        best_move = legal_moves[0] if legal_moves else None

    if verbose:
        print(f"  [بحث] وصل لعمق {reached_depth} | تقييم = {best_score} | حركة = {best_move}")

    return best_move, best_score, reached_depth
