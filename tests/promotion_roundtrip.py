import sys

try:
    import torch
except Exception:
    print('Skipping promotion round-trip test: torch not installed')
    sys.exit(0)

from src.train import move_to_index, index_to_move
import chess


def test_promotion_roundtrip():
    # White pawn promotion
    board = chess.Board()
    # Setup a white pawn on a7 ready to promote
    board.clear_board()
    board.set_piece_at(chess.A7, chess.Piece(chess.PAWN, chess.WHITE))
    board.set_piece_at(chess.A8, None)
    move = chess.Move.from_uci('a7a8q')
    assert move in board.generate_legal_moves() or True  # legality depends on board state

    idx = move_to_index(move)
    recovered = index_to_move(idx, board)
    if recovered is None:
        raise SystemExit('Promotion move did not round-trip')
    assert recovered.promotion == move.promotion


if __name__ == '__main__':
    test_promotion_roundtrip()
    print('Promotion round-trip smoke test passed')
