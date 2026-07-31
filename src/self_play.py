"""
اللاعب بيلعب ضد نفسه (نفس الموديل بيلعب بالأبيض والأسود) عدد من
المباريات، وبيسجل كل موقف مرّ بيه في اللعبة مع نتيجة اللعبة النهائية
(فوز الأبيض = 1، فوز الأسود = -1، تعادل = 0) عشان تُستخدم كبيانات تدريب.
"""

import argparse
import time

import chess
import numpy as np
import torch

from .encoding import board_to_tensor
from .model import load_or_create_model
from .agent import choose_move


def play_one_game(model, max_moves: int, epsilon: float, device: str):
    board = chess.Board()
    states = []

    move_count = 0
    while not board.is_game_over(claim_draw=True) and move_count < max_moves:
        states.append(board_to_tensor(board))
        move = choose_move(model, board, epsilon=epsilon, device=device)
        if move is None:
            break
        board.push(move)
        move_count += 1

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        # وصلنا لحد أقصى عدد النقلات من غير نتيجة حاسمة => تعادل
        result_value = 0.0
    elif outcome.winner is True:
        result_value = 1.0
    elif outcome.winner is False:
        result_value = -1.0
    else:
        result_value = 0.0

    return states, result_value, move_count


def main():
    parser = argparse.ArgumentParser(description="توليد مباريات لعب ضد النفس")
    parser.add_argument("--model", default="checkpoints/model.pt")
    parser.add_argument("--out", default="data/latest_run.npz")
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--max-moves", type=int, default=60)
    parser.add_argument("--epsilon", type=float, default=0.12)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = args.device
    model = load_or_create_model(args.model, device=device)
    model.eval()

    all_states = []
    all_targets = []
    results = {"white": 0, "black": 0, "draw": 0}
    total_moves = 0

    start = time.time()
    for i in range(args.games):
        states, result_value, move_count = play_one_game(
            model, args.max_moves, args.epsilon, device
        )
        all_states.extend(states)
        all_targets.extend([result_value] * len(states))
        total_moves += move_count

        if result_value > 0:
            results["white"] += 1
        elif result_value < 0:
            results["black"] += 1
        else:
            results["draw"] += 1

        print(f"  مباراة {i + 1}/{args.games}: نتيجة={result_value:+.0f} نقلات={move_count}")

    elapsed = time.time() - start
    avg_moves = total_moves / max(args.games, 1)

    np.savez_compressed(
        args.out,
        states=np.array(all_states, dtype=np.float32),
        targets=np.array(all_targets, dtype=np.float32),
    )

    print("\n--- ملخص اللعب ضد النفس ---")
    print(f"عدد المباريات: {args.games} | الوقت: {elapsed:.1f} ثانية")
    print(f"فوز الأبيض: {results['white']} | فوز الأسود: {results['black']} | تعادل: {results['draw']}")
    print(f"متوسط عدد النقلات لكل مباراة: {avg_moves:.1f}")
    print(f"تم حفظ {len(all_states)} موقف تدريبي في {args.out}")

    # نحفظ الملخص في ملف عشان train.py يقدر يستخدمه في stats.csv
    with open("data/last_selfplay_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"{args.games},{results['white']},{results['black']},{results['draw']},{avg_moves:.2f}\n")


if __name__ == "__main__":
    main()
