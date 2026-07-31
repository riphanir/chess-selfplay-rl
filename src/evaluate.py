"""
عشان نعرف هل النموذج فعلاً "بيتحسن" مع الوقت، بنخلي النسخة الحالية
تلعب عدد مباريات ضد نسخة "أساسية" (baseline) اتجمدت من أول تشغيل
ومابتتحدثش. نسبة فوز النسخة الحالية على الأساسية = مؤشر تقدّم.
"""

import argparse
import os
import shutil

import chess

from .model import load_or_create_model
from .agent import choose_move


def play_match(model_a, model_b, num_games: int, max_moves: int, device: str,
               time_limit: float, max_depth: int):
    """model_a يلعب ضد model_b، بالتبادل بالأبيض والأسود. يرجع (فوز_a, فوز_b, تعادل)."""
    wins_a, wins_b, draws = 0, 0, 0

    for i in range(num_games):
        board = chess.Board()
        a_is_white = i % 2 == 0
        move_count = 0

        while not board.is_game_over(claim_draw=True) and move_count < max_moves:
            current_model = (
                model_a if (board.turn == chess.WHITE) == a_is_white else model_b
            )
            move = choose_move(current_model, board, epsilon=0.0, device=device,
                                time_limit=time_limit, max_depth=max_depth)
            if move is None:
                break
            board.push(move)
            move_count += 1

        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            draws += 1
        elif outcome.winner == a_is_white:
            wins_a += 1
        else:
            wins_b += 1

    return wins_a, wins_b, draws


def main():
    parser = argparse.ArgumentParser(description="تقييم النموذج الحالي ضد النسخة الأساسية")
    parser.add_argument("--model", default="checkpoints/model.pt")
    parser.add_argument("--baseline", default="checkpoints/baseline.pt")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--max-moves", type=int, default=60)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--time-limit", type=float, default=0.5)
    parser.add_argument("--search-depth", type=int, default=8)
    args = parser.parse_args()

    device = args.device

    # لو مفيش baseline لسه، ننسخ الموديل الحالي كأول نسخة أساسية
    if not os.path.exists(args.baseline):
        os.makedirs(os.path.dirname(args.baseline), exist_ok=True)
        if os.path.exists(args.model):
            shutil.copy(args.model, args.baseline)
            print(f"تم إنشاء baseline جديدة من {args.model}")
        else:
            print("لا يوجد موديل بعد، هيتم إنشاء baseline في أول تشغيل تدريب")
            return

    current = load_or_create_model(args.model, device=device)
    baseline = load_or_create_model(args.baseline, device=device)
    current.eval()
    baseline.eval()

    wins_current, wins_baseline, draws = play_match(
        current, baseline, args.games, args.max_moves, device,
        args.time_limit, args.search_depth,
    )

    win_rate = (wins_current + 0.5 * draws) / max(args.games, 1) * 100

    print("\n--- نتيجة المباراة ضد النسخة الأساسية ---")
    print(f"فوز النسخة الحالية: {wins_current}")
    print(f"فوز النسخة الأساسية: {wins_baseline}")
    print(f"تعادل: {draws}")
    print(f"نسبة نقاط النسخة الحالية: {win_rate:.1f}%  (فوق 50% يعني في تحسّن عن البداية)")

    with open("data/last_eval_winrate.txt", "w", encoding="utf-8") as f:
        f.write(f"{win_rate:.1f}\n")


if __name__ == "__main__":
    main()
    
