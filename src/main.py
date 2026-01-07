# main.py - ChessHacks bot using AlphaZero model
from .utils import chess_manager, GameContext
from chess import Move
import chess
import chess.pgn
import random
import io
import torch
import numpy as np
import os
import sys

# ----------------------------
# Import training code
# ----------------------------
# Add the training repo to path so we can import train module
# TRAINING_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "ash-hf", "src")
TRAINING_PATH = r"C:\Users\Happy\Desktop\ChessHacks\src"
sys.path.insert(0, TRAINING_PATH)

try:
    from train import ChessNet, MCTS, move_to_index
    print("Successfully imported training modules")
except Exception as e:
    print(f"Failed to import training modules: {e}")
    print(f"Looking in: {TRAINING_PATH}")
    ChessNet = None
    MCTS = None
    move_to_index = None

# ----------------------------
# Load AlphaZero model
# ----------------------------
MODEL_PATH = r"C:\Users\Happy\Desktop\ChessHacks\weights\best_model.pt"
sys.path.insert(1, MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
mcts = None

if os.path.exists(MODEL_PATH):
    try:
        print(f"Loading AlphaZero model from {MODEL_PATH}...")

        # Create model with optimized architecture
        model = ChessNet(
            input_channels=18,
            num_filters=64,
            num_residual_blocks=3
        ).to(device)

        # Load checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        state_dict = checkpoint['model_state_dict']

        # Handle torch.compile prefix if present
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                cleaned_state_dict[k[len('_orig_mod.'):]] = v
            else:
                cleaned_state_dict[k] = v

        model.load_state_dict(cleaned_state_dict)
        model.eval()

        # Create MCTS instance
        # Use fewer simulations for speed in online play
        mcts = MCTS(model, num_simulations=200, batch_size=8)

        iteration = checkpoint.get('iteration', 'unknown')
        print(f"Model loaded successfully (iteration {iteration})")
        print(f"Using device: {device}")
        print(f"MCTS simulations: 50")

    except Exception as e:
        print(f"Failed to load model: {e}")
        model = None
        mcts = None
else:
    if ChessNet is None:
        print("Training modules not found - using random moves")
    elif not os.path.exists(MODEL_PATH):
        print(f"Model checkpoint not found at {MODEL_PATH}")
        print("Download the model first:")
        print("  cd ash-hf/src")
        print("  modal run download_model.py --mode download")


# ----------------------------
# Get move from AlphaZero model
# ----------------------------
def get_move_from_model(board: chess.Board) -> tuple[Move, dict]:
    """
    Get best move using AlphaZero MCTS.

    Returns:
        tuple: (best_move, move_probabilities_dict)
    """
    if model is None or mcts is None or move_to_index is None:
        print("MODEL NOT LOADED")
        raise RuntimeError("Model not loaded")
        
    # Run MCTS search
    visit_counts = mcts.search(board)

    # Get legal moves and their visit counts
    legal_moves = list(board.legal_moves)
    move_visits = {}

    for move in legal_moves:
        idx = move_to_index(move)
        move_visits[move] = visit_counts[idx]

    # Normalize to probabilities
    total_visits = sum(move_visits.values())
    if total_visits > 0:
        move_probs = {m: v / total_visits for m, v in move_visits.items()}
    else:
        # If no visits (shouldn't happen), uniform distribution
        move_probs = {m: 1.0 / len(legal_moves) for m in legal_moves}

    # Pick move with highest visit count
    best_move = max(move_visits, key=move_visits.get)

    return best_move, move_probs


# ----------------------------
# Fallback: Random move
# ----------------------------
def get_random_move(board: chess.Board) -> tuple[Move, dict]:
    """Fallback random move selection"""
    legal_moves = list(board.legal_moves)
    move_probs = {m: 1.0 / len(legal_moves) for m in legal_moves}
    best_move = random.choice(legal_moves)
    return best_move, move_probs

# ----------------------------
# ChessHacks entrypoint
# ----------------------------
@chess_manager.entrypoint
def bot(ctx: GameContext):
    print("\n" + "="*50)
    print("ChessHacks Bot - AlphaZero Model")
    print("="*50)

    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")

    print(f"Position: {ctx.board.fen()}")
    print(f"Legal moves: {len(legal_moves)}")

    # Try to use model, fallback to random
    try:
        if model is not None:
            print("Using AlphaZero model (50 MCTS simulations)...")
            move_obj, move_probs = get_move_from_model(ctx.board)
            print(f"Model selected: {move_obj.uci()}")

            # Show top 3 moves
            sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)[:3]
            print("Top moves:")
            for m, prob in sorted_moves:
                print(f"  {m.uci()}: {prob:.2%}")

        else:
            print("Model not loaded, using random move")
            move_obj, move_probs = get_random_move(ctx.board)

    except Exception as e:
        print(f"Model error: {e}")
        print("Falling back to random move")
        move_obj, move_probs = get_random_move(ctx.board)

    # Verify move is legal
    if move_obj not in legal_moves:
        print(f"Model returned illegal move: {move_obj.uci()}")
        print("Using random legal move")
        move_obj, move_probs = get_random_move(ctx.board)

    # Log probabilities for devtools
    ctx.logProbabilities(move_probs)

    print(f"Playing: {move_obj.uci()}")
    print("="*50 + "\n")

    return move_obj


@chess_manager.reset
def reset(ctx: GameContext):
    """Called when a new game begins"""
    print("\n New game starting - resetting bot state")
    # MCTS is stateless, no reset needed
    pass

# # main.py
# import io
# import random
# import torch
# import chess
# import chess.pgn
# from huggingface_hub import hf_hub_download
# from .utils import chess_manager, GameContext

# # ----------------------------
# # Model Configuration
# # ----------------------------
# HF_REPO_ID = "lazy-guy12/chess-llama"  # Hugging Face repo
# MODEL_FILENAME = "best_move.pt"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ----------------------------
# # Download and load model
# # ----------------------------
# model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)
# model = torch.load(model_path, map_location=device)
# model.eval()
# model.to(device)

# # ----------------------------
# # Helper: Encode board for model
# # ----------------------------
# def encode_board(board: chess.Board) -> torch.Tensor:
#     """
#     Converts a chess.Board into a tensor suitable for your model.
#     Adjust this function to match your model's expected input format.
#     Example: 8x8x12 one-hot encoding flattened to a vector.
#     """
#     tensor = torch.zeros(8, 8, 12)  # placeholder
#     # TODO: Fill tensor with pieces from board
#     # for example, loop over board squares and set corresponding channels
#     return tensor.flatten().unsqueeze(0)  # batch dimension

# # ----------------------------
# # Helper: Convert model output to UCI move
# # ----------------------------
# def output_to_uci(output, board: chess.Board) -> str:
#     """
#     Converts model output into a legal UCI move string.
#     You need to implement this based on your model's output format.
#     """
#     # Placeholder: choose a random legal move if you can't parse output
#     legal_moves = list(board.legal_moves)
#     if not legal_moves:
#         return None
#     return random.choice(legal_moves).uci()

# # ----------------------------
# # Generate move using model
# # ----------------------------
# def get_move(board: chess.Board) -> str:
#     """
#     Predicts the next move given the current board.
#     Falls back to a random legal move if prediction is invalid.
#     """
#     input_tensor = encode_board(board).to(device)
#     with torch.no_grad():
#         output = model(input_tensor)
#     predicted_move = output_to_uci(output, board)

#     # Ensure move is legal
#     if predicted_move:
#         try:
#             move_obj = chess.Move.from_uci(predicted_move)
#             if move_obj in board.legal_moves:
#                 return move_obj.uci()
#         except ValueError:
#             pass

#     # Fallback
#     legal_moves = list(board.legal_moves)
#     return random.choice(legal_moves).uci() if legal_moves else None

# # ----------------------------
# # Convert PGN to Board
# # ----------------------------
# def pgn_to_board(pgn: str) -> chess.Board:
#     """
#     Converts PGN string to a chess.Board object.
#     """
#     game = chess.pgn.read_game(io.StringIO(pgn))
#     if game is None:
#         raise ValueError("Invalid PGN")
#     board = game.board()
#     for move in game.mainline_moves():
#         board.push(move)
#     return board

# # ----------------------------
# # Chess Manager Entrypoint
# # ----------------------------
# @chess_manager.entrypoint
# def bot(ctx: GameContext):
#     board = ctx.board

#     # Build PGN history (optional)
#     exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
#     game = chess.pgn.Game.from_board(board)
#     pgn_history = game.accept(exporter)

#     # Predict move
#     try:
#         move_uci = get_move(board)
#         move_obj = chess.Move.from_uci(move_uci)
#         if move_obj not in board.legal_moves:
#             raise ValueError("Illegal move generated by model")
#     except Exception:
#         move_obj = random.choice(list(board.legal_moves))

#     # Log uniform probabilities for devtools
#     legal_moves = list(board.legal_moves)
#     move_probs = {m: 1 / len(legal_moves) for m in legal_moves} if legal_moves else {}
#     ctx.logProbabilities(move_probs)

#     return move_obj

# # ----------------------------
# # Chess Manager Reset
# # ----------------------------
# @chess_manager.reset
# def reset(ctx: GameContext):
#     # Reset any model state if needed
#     pass

# # # main.py
# # from .utils import chess_manager, GameContext
# # from chess import Move
# # import chess
# # import chess.pgn
# # import random
# # import io
# # import torch
# # from transformers import AutoModelForCausalLM, AutoTokenizer

# # # ----------------------------
# # # Load Hugging Face model
# # # ----------------------------
# # MODEL_NAME = "lazy-guy12/chess-llama"  # your model
# # CACHE_DIR = "./.model_cache"

# # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
# # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # model.to(device)
# # model.eval()


# # # ----------------------------
# # # Convert PGN to board
# # # ----------------------------
# # def pgn_to_board(pgn: str) -> chess.Board:
# #     game = chess.pgn.read_game(io.StringIO(pgn))
# #     if game is None:
# #         raise ValueError("Invalid PGN")
# #     board = game.board()
# #     for move in game.mainline_moves():
# #         board.push(move)
# #     return board


# # # ----------------------------
# # # Get move from model
# # # ----------------------------
# # def get_move(pgn: str) -> str:
# #     board = pgn_to_board(pgn)

# #     # Convert move history to UCI string for model input
# #     move_history = " ".join([m.uci() for m in board.move_stack])
# #     inputs = tokenizer(move_history, return_tensors="pt")
# #     for k in inputs:
# #         inputs[k] = inputs[k].to(device)

# #     # Generate predicted move(s)
# #     with torch.no_grad():
# #         outputs = model.generate(**inputs, do_sample=True, top_k=50, num_return_sequences=1)

# #     # Decode generated text
# #     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# #     uci_moves = generated_text.strip().split()
# #     predicted_uci = uci_moves[-1] if uci_moves else None

# #     # Ensure move is legal
# #     legal_moves = list(board.legal_moves)
# #     if predicted_uci:
# #         try:
# #             move_obj = chess.Move.from_uci(predicted_uci)
# #             if move_obj in legal_moves:
# #                 return move_obj.uci()
# #         except ValueError:
# #             pass

# #     # Fallback to random legal move
# #     return random.choice(legal_moves).uci()


# # # ----------------------------
# # # ChessHacks entrypoint
# # # ----------------------------
# # @chess_manager.entrypoint
# # def bot(ctx: GameContext):
# #     print("Cooking move...")
# #     print(ctx.board.move_stack)

# #     legal_moves = list(ctx.board.generate_legal_moves())
# #     if not legal_moves:
# #         ctx.logProbabilities({})
# #         raise ValueError("No legal moves available")

# #     # Build PGN history
# #     exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
# #     game = chess.pgn.Game.from_board(ctx.board)
# #     pgn_history = game.accept(exporter)

# #     # Generate move (fallback to random if model fails)
# #     try:
# #         move_uci = get_move(pgn_history)
# #         move_obj = Move.from_uci(move_uci)
# #         if move_obj not in legal_moves:
# #             raise ValueError("Illegal move generated by model")
# #     except Exception:
# #         move_obj = random.choice(legal_moves)

# #     # Log uniform probabilities for devtools
# #     move_probs = {m: 1 / len(legal_moves) for m in legal_moves}
# #     ctx.logProbabilities(move_probs)

# #     return move_obj


# # @chess_manager.reset
# # def reset(ctx: GameContext):
# #     # Reset any model state if needed
# #     pass


# # # from .utils import chess_manager, GameContext
# # # from chess import Move
# # # import random
# # # import time

# # # # Write code here that runs once
# # # # Can do things like load models from huggingface, make connections to subprocesses, etcwenis

# # # from transformers import AutoModel
# # # import os

# # # # Load model from Hugging Face
# # # model = AutoModel.from_pretrained(
# # #     "lazy-guy12/chess-llama",
# # #     cache_dir="./.model_cache"  # Cache locally
# # # )

# # # print('Done loading')

# # # def get_move(pgn: str) -> str:
# # #     # Use your model for inference
# # #     # ...
# # #     pass

# # # @chess_manager.entrypoint
# # # def test_func(ctx: GameContext):
# # #     # This gets called every time the model needs to make a move
# # #     # Return a python-chess Move object that is a legal move for the current position

# # #     print("Cooking move...")
# # #     print(ctx.board.move_stack)
# # #     time.sleep(0.1)

# # #     legal_moves = list(ctx.board.generate_legal_moves())
# # #     if not legal_moves:
# # #         ctx.logProbabilities({})
# # #         raise ValueError("No legal moves available (i probably lost didn't i)")

# # #     move_weights = [random.random() for _ in legal_moves]
# # #     total_weight = sum(move_weights)
# # #     # Normalize so probabilities sum to 1
# # #     move_probs = {
# # #         move: weight / total_weight
# # #         for move, weight in zip(legal_moves, move_weights)
# # #     }
# # #     ctx.logProbabilities(move_probs)

# # #     return random.choices(legal_moves, weights=move_weights, k=1)[0]


# # # @chess_manager.reset
# # # def reset_func(ctx: GameContext):
# # #     # This gets called when a new game begins
# # #     # Should do things like clear caches, reset model state, etc.
# # #     pass
