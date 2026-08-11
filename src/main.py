from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import time
import chess
import torch
import sys
import os
import traceback
import random

# ==========================================
# 1. SETUP & IMPORTS
# ==========================================
# Define paths
TRAINING_PATH = r"C:\Users\Matthew\Desktop\chessbot\chessbot\src"
MODEL_PATH = os.path.join(TRAINING_PATH, "checkpoint.pth")

if TRAINING_PATH not in sys.path:
    sys.path.insert(0, TRAINING_PATH)

try:
    from config import Config
    from model import ChessNet, ParallelMCTS, decode_move, MCTSNode, load_checkpoint
    print(">> [DEBUG] Imports successful.")
except ImportError as e:
    print(f"!! [CRITICAL] Import Error: {e}")
    sys.exit()

# ==========================================
# 2. LOAD MODEL (ON GPU)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">> [SYSTEM] Inference Device: {device}")

engine_net = None

def load_brain():
    global engine_net
    if os.path.exists(MODEL_PATH):
        try:
            print(f">> [DEBUG] Loading Super Model...", flush=True)
            temp_net = ChessNet().to(device)

            state, meta = load_checkpoint(MODEL_PATH, device)
            temp_net.load_state_dict(state)
            temp_net.eval()

            engine_net = temp_net
            if meta:
                print(f">> [INFO] Checkpoint metadata: {meta}", flush=True)
            print(f">> [SUCCESS] Model loaded on {device}", flush=True)
            return True
        except Exception as e:
            print(f"!! [CRITICAL] Model Load Failed: {e}", flush=True)
            print("!! Is the training script currently writing to the file?")
            return False
    else:
        print(f"!! [WARNING] Checkpoint not found at {MODEL_PATH}", flush=True)
        return False

# Load immediately on startup
load_brain()

# ==========================================
# 3. AI LOGIC
# ==========================================
def get_alphazero_move(board: chess.Board, time_limit_ms=None):
    if engine_net is None:
        # Try reloading in case training just finished a chunk
        if not load_brain():
            return None, {}

    try:
        # Override Config for PLAYING (Higher quality than training)
        Config.device = device
        Config.num_simulations = 800 

        mcts = ParallelMCTS(engine_net)
        roots = mcts.search([board], time_limit_ms=time_limit_ms)
        root = roots[0]

        best_visits = -1
        best_prior = -1.0
        best_move_idx = -1
        move_probs = {}
        total_visits = sum(n.n for n in root.children.values())

        for idx, node in root.children.items():
            prob = float(node.n / total_visits) if total_visits > 0 else 0.0

            try:
                mv = decode_move(idx, board)
                move_probs[mv] = prob

                # Tie-break on prior probability, not iteration order. With
                # a very low time budget (clock running down - see above),
                # search() can return after only the initial root expansion
                # and no simulations at all, leaving every child at n=0;
                # without this, the "best" move would just be whichever
                # legal move happened to come first from board.legal_moves.
                if node.n > best_visits or (node.n == best_visits and node.p > best_prior):
                    best_visits = node.n
                    best_prior = node.p
                    best_move_idx = idx
            except:
                continue

        if best_move_idx == -1:
            return None, {}

        return decode_move(best_move_idx, board), move_probs

    except Exception as e:
        print(f"!! [CRITICAL] Inference Crash: {e}", flush=True)
        traceback.print_exc()
        return None, {}

# ==========================================
# 4. CHESS MANAGER INTERFACE
# ==========================================
from src.utils import chess_manager, GameContext

@chess_manager.entrypoint
def bot(ctx: GameContext):
    legal_moves = list(ctx.board.legal_moves)
    if not legal_moves:
        return None

    # Try AI
    try:
        # Spend a slice of remaining clock, not the whole thing - conservative
        # since we don't know the harness's increment or its own per-move
        # deadline. Without this, num_simulations=800 was a fixed cost
        # regardless of time left: a forfeit risk if the clock ever runs low,
        # and simulations left on the table whenever there's plenty of time.
        time_budget_ms = None
        if ctx.timeLeft is not None:
            # `if ctx.timeLeft:` would treat a legitimate 0 (clock run down)
            # as "no time info" and disable the budget entirely - the exact
            # forfeit risk this exists to prevent. And a fixed floor here
            # (e.g. "always >= 500ms") can itself exceed what's actually
            # left when the clock is low, so the floor is capped by half of
            # whatever time remains instead of being a flat constant.
            target = max(50, ctx.timeLeft * 0.04)
            time_budget_ms = min(20000, target, ctx.timeLeft * 0.5)

        print(f">> [BOT] Thinking... (Fen: {ctx.board.fen()}, "
              f"timeLeft={ctx.timeLeft}ms, budget={time_budget_ms}ms)", flush=True)
        best_move, probs = get_alphazero_move(ctx.board, time_limit_ms=time_budget_ms)

        if best_move and best_move in legal_moves:
            ctx.logProbabilities(probs)
            print(f">> [BOT] Playing: {best_move.uci()}", flush=True)
            return best_move
            
    except Exception as e:
        print(f"!! [BOT] Crash: {e}", flush=True)

    # Fallback
    print(">> [BOT] Random Fallback", flush=True)
    uniform_prob = 1.0 / len(legal_moves)
    fallback_probs = {m: float(uniform_prob) for m in legal_moves}
    ctx.logProbabilities(fallback_probs)
    return random.choice(legal_moves)

@chess_manager.reset
def reset(ctx: GameContext):
    print(">> [GAME] Reset. Reloading Brain...", flush=True)
    # Reload the model on every game reset to get the latest Training progress!
    load_brain()
