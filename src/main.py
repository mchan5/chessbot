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
MODEL_PATH = os.path.join(TRAINING_PATH, "most_recent.pth")

if TRAINING_PATH not in sys.path:
    sys.path.insert(0, TRAINING_PATH)

try:
    from config import Config
    from model import ChessNet, ParallelMCTS, decode_move, MCTSNode
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
            
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            temp_net.load_state_dict(checkpoint)
            temp_net.eval()
            
            engine_net = temp_net
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
        roots = mcts.search([board])
        root = roots[0]

        best_visits = -1
        best_move_idx = -1
        move_probs = {}
        total_visits = sum(n.n for n in root.children.values())

        for idx, node in root.children.items():
            prob = float(node.n / total_visits) if total_visits > 0 else 0.0
            
            try:
                mv = decode_move(idx)
                move_probs[mv] = prob
                
                if node.n > best_visits:
                    best_visits = node.n
                    best_move_idx = idx
            except:
                continue

        if best_move_idx == -1:
            return None, {}

        return decode_move(best_move_idx), move_probs

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
        print(f">> [BOT] Thinking... (Fen: {ctx.board.fen()})", flush=True)
        best_move, probs = get_alphazero_move(ctx.board)
        
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
