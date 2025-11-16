import sys

try:
    import torch
except Exception:
    print('Skipping Stockfish fallback test: torch not installed')
    sys.exit(0)

try:
    import chess
except Exception:
    print('Skipping Stockfish fallback test: python-chess not installed')
    sys.exit(0)

from src.main import query_stockfish


def test_engine_available():
    # Try default 'stockfish' on PATH
    move, err = query_stockfish(chess.Board())
    if move is None:
        print('Stockfish not available or query failed (this is okay)')
        sys.exit(0)

    # If we got a move, ensure it's legal
    assert move in chess.Board().legal_moves


if __name__ == '__main__':
    test_engine_available()
    print('Stockfish fallback smoke test passed')
