"""
Automated evaluation gauntlet.

Plays two players against each other over N games and reports a score and an
Elo-difference estimate. Each player is either one of our own checkpoints
(driven by our own MCTS) or Stockfish (driven over UCI).

Use this to check whether a training change actually helped before trusting it:
    python gauntlet.py --engine-a weights/new.pt --engine-b weights/previous.pt

Or to get an absolute strength anchor:
    python gauntlet.py --engine-a weights/new.pt --engine-b stockfish --stockfish-skill 0
"""
import argparse
import math
import os
import random
import sys
import time

import chess
import chess.engine
import chess.pgn
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import Config
from model import ChessNet, ParallelMCTS, decode_move, load_checkpoint

DEFAULT_STOCKFISH = os.path.join(SCRIPT_DIR, "CuteChess Support", "stockfish.exe")

# Inference is fully deterministic (greedy visit-count selection, no Dirichlet
# noise, no temperature), so two engines always play the exact same game
# twice in a row from the startline. Opening every game from one of these
# short, named lines is what actually makes an N-game match measure anything.
# Each line is a space-separated sequence of UCI moves (4-8 plies).
OPENING_BOOK = [
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5",              # Italian Game
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6",              # Ruy Lopez
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6",    # Sicilian Najdorf-ish
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g7g6",    # Sicilian Accelerated Dragon-ish
    "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6",              # French Defense
    "e2e4 c7c6 d2d4 d7d5 b1d2 g8f6",              # Caro-Kann Classical
    "e2e4 c7c6 d2d4 d7d5 e4e5 c8f5",              # Caro-Kann Advance
    "e2e4 d7d5 e4d5 d8d5 b1c3 d5a5",              # Scandinavian
    "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6",              # Pirc Defense
    "e2e4 g7g6 d2d4 f8g7 b1c3 d7d6",              # Modern Defense
    "e2e4 g8f6 e4e5 f6d5 d2d4 d7d6",              # Alekhine's Defense
    "e2e4 e7e5 f2f4 e5f4 g1f3 g7g5",              # King's Gambit
    "e2e4 e7e5 b1c3 g8f6 f2f4 d7d5",              # Vienna Game
    "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 g8f6",    # Scotch Game
    "e2e4 e7e5 g1f3 b8c6 b1c3 g8f6",              # Four Knights
    "e2e4 e7e5 g1f3 g8f6 f3e5 d7d6",              # Petrov's Defense
    "e2e4 e7e5 g1f3 d7d6 d2d4 e5d4",              # Philidor Defense
    "c2c4 e7e5 b1c3 g8f6 g1f3 b8c6",              # English vs 1...e5
    "c2c4 c7c5 g1f3 g8f6 b1c3 b8c6",              # English Symmetric
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6",              # Queen's Gambit Declined
    "d2d4 d7d5 c2c4 d5c4 g1f3 g8f6",              # Queen's Gambit Accepted
    "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6",              # Slav Defense
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4",              # Nimzo-Indian
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7",              # King's Indian Defense
    "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5",              # Grunfeld Defense
    "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6",              # Queen's Indian
    "d2d4 g8f6 c2c4 e7e6 g2g3 d7d5",              # Catalan
    "d2d4 f7f5 g2g3 g8f6 f1g2 e7e6",              # Dutch Defense
    "d2d4 g8f6 c2c4 c7c5 d4d5 e7e6",              # Benoni Defense
    "d2d4 d7d5 g1f3 g8f6 c1f4 e7e6",              # London System
    "g1f3 d7d5 c2c4 e7e6 g2g3 g8f6",              # Reti Opening
    "f2f4 d7d5 g1f3 g8f6 e2e3 g7g6",              # Bird's Opening
    "b2b3 e7e5 c1b2 b8c6 e2e3 g8f6",              # Nimzo-Larsen
]


class NeuralPlayer:
    """Wraps a checkpoint as a move-picking player using our own MCTS."""

    def __init__(self, checkpoint_path, device, simulations):
        self.device = device
        self.simulations = simulations
        self.net = ChessNet().to(device)

        state, metadata = load_checkpoint(checkpoint_path, device)
        self.net.load_state_dict(state)
        self.net.eval()

        self.name = os.path.basename(checkpoint_path)
        self.metadata = metadata

    def play(self, board):
        Config.device = self.device
        Config.num_simulations = self.simulations
        mcts = ParallelMCTS(self.net)
        root = mcts.search([board])[0]
        if not root.children:
            return None
        best_idx = max(root.children, key=lambda idx: root.children[idx].n)
        return decode_move(best_idx, board)

    def close(self):
        pass


class StockfishPlayer:
    def __init__(self, path, movetime_ms, skill_level=None, target_elo=None):
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        if target_elo is not None:
            # UCI_Elo is Stockfish's own calibrated strength target - unlike
            # "Skill Level" (a vague weakening dial with no documented Elo
            # correspondence), Stockfish is tuned to actually play at
            # approximately this rating. This is what you want for an
            # absolute strength estimate, not Skill Level.
            self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": target_elo})
            self.name = f"Stockfish(elo={target_elo})"
        else:
            self.engine.configure({"Skill Level": skill_level})
            self.name = f"Stockfish(skill={skill_level})"
        self.movetime_ms = movetime_ms

    def play(self, board):
        result = self.engine.play(board, chess.engine.Limit(time=self.movetime_ms / 1000))
        return result.move

    def close(self):
        self.engine.quit()


def make_player(spec, device, simulations, stockfish_path, stockfish_skill, stockfish_movetime_ms,
                 stockfish_elo=None):
    if spec.lower() == "stockfish":
        return StockfishPlayer(stockfish_path, stockfish_movetime_ms,
                                skill_level=stockfish_skill, target_elo=stockfish_elo)
    return NeuralPlayer(spec, device, simulations)


def play_game(white, black, max_plies=200, opening=None):
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = white.name
    game.headers["Black"] = black.name
    node = game

    if opening:
        game.headers["Opening"] = opening
        for uci in opening.split():
            move = chess.Move.from_uci(uci)
            board.push(move)
            node = node.add_variation(move)

    while not board.is_game_over() and len(board.move_stack) < max_plies:
        mover = white if board.turn == chess.WHITE else black
        move = mover.play(board)
        if move is None or move not in board.legal_moves:
            result = "0-1" if board.turn == chess.WHITE else "1-0"
            game.headers["Result"] = result
            game.headers["Termination"] = "illegal move / no move returned"
            return result, game
        board.push(move)
        node = node.add_variation(move)

    if board.is_game_over():
        result = board.result(claim_draw=True)
    else:
        result = "1/2-1/2"
        game.headers["Termination"] = "max plies reached"
    game.headers["Result"] = result
    return result, game


def elo_diff(score, games):
    if games == 0:
        return 0.0
    p = min(max(score / games, 1e-6), 1 - 1e-6)
    return 400 * math.log10(p / (1 - p))


def run_match(player_a, player_b, games=20, max_plies=200, seed=None, no_book=False,
              pgn_out=None, verbose=True):
    """Play a paired-opening match between two already-constructed players
    and return the result. Shared by the CLI (main(), below) and callers
    that need to run a gauntlet programmatically - e.g. training.py's
    gauntlet-gated promotion step.

    Returns a dict: {a_wins, a_losses, draws, a_score, games, elo_diff}.
    """
    if verbose:
        print(f"A: {player_a.name}  vs  B: {player_b.name}  ({games} games)")
        for label, p in (("A", player_a), ("B", player_b)):
            meta = getattr(p, "metadata", None)
            if meta:
                print(f"  {label} metadata: {meta}")

    rng = random.Random(seed)
    book_cycle = []

    def next_opening():
        nonlocal book_cycle
        if no_book:
            return None
        if not book_cycle:
            book_cycle = OPENING_BOOK.copy()
            rng.shuffle(book_cycle)
        return book_cycle.pop()

    a_score = 0.0
    a_wins = a_losses = draws = 0
    pgn_games = []
    pair_opening = None

    for i in range(games):
        a_is_white = (i % 2 == 0)
        white, black = (player_a, player_b) if a_is_white else (player_b, player_a)
        if a_is_white:
            pair_opening = next_opening()
        opening = pair_opening

        start = time.time()
        result, game = play_game(white, black, max_plies=max_plies, opening=opening)
        elapsed = time.time() - start

        if result == "1-0":
            winner = white
        elif result == "0-1":
            winner = black
        else:
            winner = None

        if winner is player_a:
            a_score += 1.0
            a_wins += 1
        elif winner is player_b:
            a_losses += 1
        else:
            a_score += 0.5
            draws += 1

        pgn_games.append(game)
        if verbose:
            print(f"  Game {i + 1}/{games}: {white.name} vs {black.name} -> {result}  ({elapsed:.1f}s)")

    diff = elo_diff(a_score, games)
    if verbose:
        print("\n" + "=" * 50)
        print(f"{player_a.name}: {a_wins}W {a_losses}L {draws}D  (score {a_score}/{games})")
        print(f"Estimated Elo diff (A - B): {diff:+.0f}")
        print("=" * 50)

    if pgn_out:
        with open(pgn_out, "w", encoding="utf-8") as f:
            for g in pgn_games:
                print(g, file=f, end="\n\n")
        if verbose:
            print(f"PGN saved to {pgn_out}")

    return {
        "a_wins": a_wins, "a_losses": a_losses, "draws": draws,
        "a_score": a_score, "games": games, "elo_diff": diff,
    }


def main():
    parser = argparse.ArgumentParser(description="Play a gauntlet between two engines")
    parser.add_argument("--engine-a", required=True, help="Checkpoint path, or 'stockfish'")
    parser.add_argument("--engine-b", required=True, help="Checkpoint path, or 'stockfish'")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--sims", type=int, default=200, help="MCTS simulations/move for neural players")
    parser.add_argument("--stockfish-path", default=DEFAULT_STOCKFISH)
    parser.add_argument("--stockfish-skill", type=int, default=0, help="0-20 (vague weakening dial, no documented Elo)")
    parser.add_argument("--stockfish-elo", type=int, default=None,
                         help="Calibrated strength target via UCI_Elo (1320-3190). Overrides --stockfish-skill "
                              "when set - use this for an actual absolute Elo estimate.")
    parser.add_argument("--stockfish-movetime-ms", type=int, default=100)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--pgn-out", default=None)
    parser.add_argument("--seed", type=int, default=None, help="Seed for opening selection (reproducible matches)")
    parser.add_argument("--no-book", action="store_true",
                         help="Disable the opening book and always start from the initial position "
                              "(inference is deterministic, so this makes every game with a given "
                              "color assignment identical - only use this for debugging)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    player_a = make_player(args.engine_a, device, args.sims, args.stockfish_path,
                            args.stockfish_skill, args.stockfish_movetime_ms, args.stockfish_elo)
    player_b = make_player(args.engine_b, device, args.sims, args.stockfish_path,
                            args.stockfish_skill, args.stockfish_movetime_ms, args.stockfish_elo)

    # Pair games two-at-a-time on the same opening line, with colors swapped
    # each half of the pair. This cancels out opening-related luck (one side
    # getting the objectively better line) far more effectively than picking
    # a fresh random opening independently for every game.
    run_match(player_a, player_b, games=args.games, max_plies=args.max_plies,
              seed=args.seed, no_book=args.no_book, pgn_out=args.pgn_out)

    player_a.close()
    player_b.close()


if __name__ == "__main__":
    main()
