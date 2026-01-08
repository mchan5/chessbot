from datasets import load_dataset, Dataset
import json
import os
from tqdm import tqdm
import argparse


def _normalize_moves(value):
    """Return a flat list of move strings from a raw field.

    Handles cases where the dataset provides:
    - a single whitespace-delimited string
    - a list of strings
    - nested lists (rare, but defensive)
    - None or other types -> returns empty list
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [m for m in value.strip().split() if m]
    if isinstance(value, list):
        flat = []
        for item in value:
            if isinstance(item, str):
                if item:
                    flat.append(item)
            elif isinstance(item, list):
                for inner in item:
                    if isinstance(inner, str) and inner:
                        flat.append(inner)
                    elif inner is not None:
                        flat.append(str(inner))
            elif item is not None:
                flat.append(str(item))
        return flat
    # Fallback: coerce to string
    return [str(value)]

parser = argparse.ArgumentParser(description="Stream and preprocess chess games and puzzles")
parser.add_argument("--games", type=int, default=20000, help="Number of games to process (default: 20000)")
parser.add_argument("--puzzles", type=int, default=10000, help="Number of puzzles to process (default: 10000)")
args = parser.parse_args()

games_target = max(0, args.games)
puzzles_target = max(0, args.puzzles)

os.makedirs("../data", exist_ok=True)

# ---- Load chess games dataset ----
print("Loading chess games dataset using streaming mode...")
print("This will download data as needed (much faster startup!)")

# Use streaming to avoid downloading entire dataset
games_ds = load_dataset("angeluriot/chess_games", split="train", streaming=True)

# Take first N games using streaming
processed_games = []
print(f"Processing {games_target} games...")
for i, game in enumerate(tqdm(games_ds, total=games_target, desc="Games")):
    if i >= games_target:
        break
    try:
        raw_moves = game.get("moves_uci")
        moves = _normalize_moves(raw_moves)
        if not moves:
            # Skip empty move lists
            continue
        processed_games.append({
            "type": "game",
            "moves": moves,
        })
    except Exception as e:
        print(f"Skipping game {i}: {e}")
        continue

with open("../data/games.json", "w") as f:
    json.dump(processed_games, f)

print("Saved games.json with", len(processed_games), "games")

# ---- Load chess puzzles dataset ----
print("\nLoading puzzles dataset using streaming mode...")
puzzles_ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)

# Take first M puzzles using streaming
processed_puzzles = []
print(f"Processing {puzzles_target} puzzles...")
for i, p in enumerate(tqdm(puzzles_ds, total=puzzles_target, desc="Puzzles")):
    if i >= puzzles_target:
        break
    try:
        fen = p.get("FEN")
        solution_raw = p.get("Moves")
        solution_moves = _normalize_moves(solution_raw)
        if not fen or not solution_moves:
            continue
        processed_puzzles.append({

            "type": "puzzle",
            "fen": fen,
            "solution": solution_moves,
        })
    except Exception as e:
        print(f"Skipping puzzle {i}: {e}")
        continue

with open("../data/puzzles.json", "w") as f:
    json.dump(processed_puzzles, f)

print("Saved puzzles.json with", len(processed_puzzles), "puzzles")
