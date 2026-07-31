"""
تحويل رقعة شطرنج (chess.Board) إلى تنسور رقمي (numpy array)
تقدر الشبكة العصبية تفهمه.

التمثيل: 13 "طبقة" (channel) بحجم 8x8:
  - 12 طبقة لكل نوع قطعة x لون (بيدق/حصان/فيل/رخ/وزير/ملك × أبيض/أسود)
  - طبقة إضافية: 1 لو الدور على الأبيض، 0 لو على الأسود (منتشرة على الرقعة كلها)
"""

import numpy as np
import chess

PIECE_TO_PLANE = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}

NUM_PLANES = 13


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """يحول رقعة شطرنج إلى مصفوفة numpy بشكل (13, 8, 8) من نوع float32."""
    tensor = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        rank = chess.square_rank(square)  # 0..7
        file = chess.square_file(square)  # 0..7
        plane = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
        tensor[plane, rank, file] = 1.0

    # طبقة الدور: 1 لو دور الأبيض، 0 لو دور الأسود
    tensor[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    return tensor
  
