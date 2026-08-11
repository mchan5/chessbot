"""
Supervised pretraining on expert games + puzzles (the "bootstrap" stage
before self-play, see src/training/training.py).

Trains the same architecture/encoding used everywhere else in the repo
(src/model.py's ChessNet / board_to_tensor) - a checkpoint produced here
loads directly into src/main.py or gauntlet.py, no conversion needed.
"""
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import chess
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../src
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import Config
from model import ChessNet, ParallelMCTS, encode_move, decode_move, board_to_tensor, save_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Determine paths (works both locally and on Modal)
if os.path.exists("/root/weights"):  # Modal environment
    WEIGHTS_DIR = "/root/weights"
    DATA_DIR = "/root/data"
else:  # Local environment
    WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "../weights")
    DATA_DIR = os.path.join(SCRIPT_DIR, "../data")

os.makedirs(WEIGHTS_DIR, exist_ok=True)


# ==================== DATASET ====================

def _outcome_from_winner(winner):
    """winner is "white", "black", or None (draw/unknown) - see preprocess.py."""
    if winner == "white":
        return 1.0
    if winner == "black":
        return -1.0
    return 0.0


class ChessDataset(Dataset):
    """Dataset for supervised learning from games and puzzles"""
    def __init__(self, games_path=None, puzzles_path=None):
        self.data = []

        if games_path is None:
            games_path = os.path.join(DATA_DIR, "games.json")
        if puzzles_path is None:
            puzzles_path = os.path.join(DATA_DIR, "puzzles.json")

        with open(games_path) as f:
            games = json.load(f)

        print(f"Processing {len(games)} games...")
        for game_data in tqdm(games[:1000]):  # Limit for faster training
            if game_data["type"] == "game":
                self._process_game(game_data["moves"], game_data.get("winner"))

        with open(puzzles_path) as f:
            puzzles = json.load(f)

        print(f"Processing {len(puzzles)} puzzles...")
        for puzzle_data in tqdm(puzzles[:500]):  # Limit for faster training
            if puzzle_data["type"] == "puzzle":
                self._process_puzzle(puzzle_data["fen"], puzzle_data["solution"])

        print(f"Total positions: {len(self.data)}")

    def _process_game(self, moves_uci, winner):
        """Extract positions and moves from a game, labeling each position's
        value from the actual result (signed to the side to move) instead of
        a flat 0.0 - the value head can't learn anything about who's winning
        if every non-puzzle example claims the position is a dead draw."""
        outcome = _outcome_from_winner(winner)
        board = chess.Board()

        for move_uci in moves_uci:
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    encoded_board = board_to_tensor(board)
                    move_idx = encode_move(move)
                    value = outcome if board.turn == chess.WHITE else -outcome

                    self.data.append((encoded_board, move_idx, value))
                    board.push(move)
                else:
                    break
            except Exception:
                break

    def _process_puzzle(self, fen, solution_moves):
        """Extract positions from puzzle. Puzzle solutions are the
        objectively-best/winning move in that position, so value=1.0 (from
        the mover's perspective) is a deliberate signal, not the bug this
        function's sibling above fixes."""
        try:
            board = chess.Board(fen)
            for move_uci in solution_moves:
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in board.legal_moves:
                        encoded_board = board_to_tensor(board)
                        move_idx = encode_move(move)
                        value = 1.0

                        self.data.append((encoded_board, move_idx, value))
                        board.push(move)
                    else:
                        break
                except Exception:
                    break
        except Exception:
            pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board, move_idx, value = self.data[idx]
        return (
            torch.FloatTensor(board),
            torch.LongTensor([move_idx]),
            torch.FloatTensor([value])
        )


# ==================== TRAINING ====================

def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch.

    ChessNet's policy head already applies log_softmax internally (see
    model.py - inference relies on that), so the policy loss here is
    NLLLoss against those log-probabilities directly, not CrossEntropyLoss
    (which would apply a second log_softmax on top and train against the
    wrong distribution).
    """
    model.train()
    total_loss = 0
    policy_loss_sum = 0
    value_loss_sum = 0

    criterion_policy = nn.NLLLoss()
    criterion_value = nn.MSELoss()

    for boards, target_moves, target_values in tqdm(dataloader, desc="Training"):
        boards = boards.to(device)
        target_moves = target_moves.to(device).squeeze(1)
        target_values = target_values.to(device).squeeze(1)

        optimizer.zero_grad()

        log_probs, values = model(boards)

        policy_loss = criterion_policy(log_probs, target_moves)
        value_loss = criterion_value(values.squeeze(), target_values)

        loss = policy_loss + value_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        policy_loss_sum += policy_loss.item()
        value_loss_sum += value_loss.item()

    n = len(dataloader)
    return total_loss / n, policy_loss_sum / n, value_loss_sum / n


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    criterion_policy = nn.NLLLoss()
    criterion_value = nn.MSELoss()

    with torch.no_grad():
        for boards, target_moves, target_values in dataloader:
            boards = boards.to(device)
            target_moves = target_moves.to(device).squeeze(1)
            target_values = target_values.to(device).squeeze(1)

            log_probs, values = model(boards)

            policy_loss = criterion_policy(log_probs, target_moves)
            value_loss = criterion_value(values.squeeze(), target_values)
            loss = policy_loss + value_loss

            total_loss += loss.item()

            predicted = torch.argmax(log_probs, dim=1)
            correct += (predicted == target_moves).sum().item()
            total += target_moves.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0

    return avg_loss, accuracy


# ==================== MAIN ====================

def main():
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10

    print("Loading dataset...")
    dataset = ChessDataset()

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    print("Initializing model...")
    model = ChessNet().to(device)
    print(f"Architecture: {Config.num_residual_blocks} blocks x {Config.num_filters} filters")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*60}")

        train_loss, policy_loss, value_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.4f} (Policy: {policy_loss:.4f}, Value: {value_loss:.4f})")

        val_loss, val_accuracy = evaluate(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        scheduler.step()
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(WEIGHTS_DIR, "best_model.pt")
            save_checkpoint(best_model_path, model,
                             run_notes=f"supervised epoch {epoch+1}, val_acc={val_accuracy:.4f}",
                             epoch=epoch + 1, val_loss=val_loss, val_accuracy=val_accuracy)
            print(f"Saved best model (val_loss: {val_loss:.4f})")

        if (epoch + 1) % 2 == 0:
            checkpoint_path = os.path.join(WEIGHTS_DIR, f"checkpoint_epoch_{epoch+1}.pt")
            save_checkpoint(checkpoint_path, model,
                             run_notes=f"supervised epoch {epoch+1}",
                             epoch=epoch + 1)

    final_model_path = os.path.join(WEIGHTS_DIR, "final_model.pt")
    save_checkpoint(final_model_path, model, run_notes="supervised training complete")
    print("\n" + "="*60)
    print(f"Training complete! Models saved to {WEIGHTS_DIR}")
    print("="*60)

    print("\nDemonstrating MCTS search on starting position...")
    model.eval()
    Config.device = device
    Config.num_simulations = 50
    mcts = ParallelMCTS(model)
    board = chess.Board()
    root = mcts.search([board])[0]

    print("\nTop moves by MCTS:")
    top = sorted(root.children.items(), key=lambda kv: kv[1].n, reverse=True)[:5]
    for idx, node in top:
        if node.n > 0:
            print(f"  {decode_move(idx, board).uci()}: {node.n} visits")


if __name__ == "__main__":
    main()
