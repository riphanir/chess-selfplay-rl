"""
العب ضد الموديل من التيرمينال، عشان تشوف بنفسك بيلعب إزاي.
استخدام:  python -m src.play_cli --model checkpoints/model.pt
اكتب حركاتك بصيغة UCI مثل: e2e4  أو  g1f3
"""

import argparse
import chess

from .model import load_or_create_model
from .agent import choose_move


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/model.pt")
    parser.add_argument("--human-color", choices=["white", "black"], default="white")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = load_or_create_model(args.model, device=args.device)
    model.eval()

    board = chess.Board()
    human_is_white = args.human_color == "white"

    while not board.is_game_over(claim_draw=True):
        print("\n" + str(board))
        if board.turn == chess.WHITE:
            print("(دور الأبيض)")
        else:
            print("(دور الأسود)")

        human_turn = (board.turn == chess.WHITE) == human_is_white
        if human_turn:
            move_str = input("حركتك (UCI مثل e2e4): ").strip()
            try:
                move = chess.Move.from_uci(move_str)
                if move not in board.legal_moves:
                    print("حركة غير مسموحة، جرب تاني")
                    continue
            except ValueError:
                print("صيغة غير صحيحة، جرب تاني")
                continue
        else:
            move = choose_move(model, board, epsilon=0.0, device=args.device)
            print(f"الموديل لعب: {move.uci()}")

        board.push(move)

    print("\n" + str(board))
    print("النتيجة النهائية:", board.outcome(claim_draw=True))


if __name__ == "__main__":
    main()
