"""
Continuous self-play training loop for chess AI.
Based on AlphaZero's reinforcement learning approach.
"""
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import chess
import math
from tqdm import tqdm
import time
from datetime import datetime

# Enable TensorFloat32 for better performance on Ampere GPUs (A10G, A100)
torch.set_float32_matmul_precision('high')

from train import ChessNet, MCTS, encode_board, move_to_index, device

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists("/root/data"):  # Modal environment
    WEIGHTS_DIR = "/root/data/weights"
    DATA_DIR = "/root/data"
    SELFPLAY_DIR = "/root/data/selfplay"
else:  # Local environment
    WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "../weights")
    DATA_DIR = os.path.join(SCRIPT_DIR, "../data")
    SELFPLAY_DIR = os.path.join(SCRIPT_DIR, "../selfplay_games")

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(SELFPLAY_DIR, exist_ok=True)


class SelfPlayDataset(Dataset):
    """Dataset from self-play games"""
    def __init__(self, game_records):
        self.data = []

        for game in game_records:
            for position in game['positions']:
                self.data.append(position)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board, policy, value = self.data[idx]
        return (
            torch.FloatTensor(board),
            torch.FloatTensor(policy),
            torch.FloatTensor([value])
        )


def play_self_play_game(model, num_simulations=50, temperature=1.0, debug=False):
    """
    Play one self-play game using MCTS.

    Args:
        model: Neural network
        num_simulations: MCTS simulations per move
        temperature: Exploration temperature (higher = more random)
        debug: Print timing information

    Returns:
        Game record with positions, policies, and outcome
    """
    import time as time_module
    game_start = time_module.time()

    board = chess.Board()
    mcts = MCTS(model, num_simulations=num_simulations)

    game_history = []
    move_count = 0
    max_moves = 200  # Prevent infinite games

    while not board.is_game_over() and move_count < max_moves:
        move_start = time_module.time()

        if debug and move_count == 0:
            print(f"  Starting first move (this may be slow due to CUDA warmup)...")
        # Get MCTS policy
        visit_counts = mcts.search(board)

        # Apply temperature
        if temperature > 0:
            visit_counts_temp = visit_counts ** (1.0 / temperature)
            policy = visit_counts_temp / visit_counts_temp.sum()
        else:
            # Greedy - pick best move
            policy = np.zeros_like(visit_counts)
            policy[np.argmax(visit_counts)] = 1.0

        # Store position and policy
        encoded_board = encode_board(board)
        game_history.append({
            'board': encoded_board,
            'policy': policy,
            'turn': board.turn
        })

        # Sample move from policy
        legal_moves = list(board.legal_moves)
        move_probs = []
        moves = []

        for move in legal_moves:
            idx = move_to_index(move)
            if idx < len(policy) and policy[idx] > 0:
                move_probs.append(policy[idx])
                moves.append(move)

        if not moves:
            break

        # Normalize probabilities
        move_probs = np.array(move_probs)
        move_probs = move_probs / move_probs.sum()

        # Sample move
        chosen_move = np.random.choice(moves, p=move_probs)
        board.push(chosen_move)
        move_count += 1

    # Determine game outcome
    result = board.result()
    if result == "1-0":
        outcome = 1.0  # White wins
    elif result == "0-1":
        outcome = -1.0  # Black wins
    else:
        outcome = 0.0  # Draw

    # Assign values from winner's perspective
    game_record = {
        'positions': [],
        'outcome': outcome,
        'move_count': move_count
    }

    for entry in game_history:
        # Value from current player's perspective
        if entry['turn'] == chess.WHITE:
            value = outcome
        else:
            value = -outcome

        game_record['positions'].append((
            entry['board'],
            entry['policy'],
            value
        ))

    if debug:
        game_time = time_module.time() - game_start
        print(f"  Game completed in {game_time:.1f}s ({move_count} moves, {game_time/max(move_count,1):.2f}s/move)")
        print(f"  Result: {result} (outcome value: {outcome})")
        if board.outcome():
            print(f"  Termination: {board.outcome().termination}")

    return game_record


def train_on_selfplay(model, optimizer, game_records, epochs=3, batch_size=64, use_amp=True):
    """Train model on self-play data with optional mixed precision"""
    dataset = SelfPlayDataset(game_records)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    total_loss = 0

    for epoch in range(epochs):
        epoch_loss = 0

        for boards, target_policies, target_values in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            boards = boards.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device).squeeze(1)

            optimizer.zero_grad()

            # Forward pass with automatic mixed precision
            if use_amp and scaler:
                with torch.cuda.amp.autocast():
                    policy_logits, values = model(boards)
                    target_moves = torch.argmax(target_policies, dim=1)
                    policy_loss = criterion_policy(policy_logits, target_moves)
                    value_loss = criterion_value(values.squeeze(), target_values)
                    loss = policy_loss + value_loss

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular precision
                policy_logits, values = model(boards)
                target_moves = torch.argmax(target_policies, dim=1)
                policy_loss = criterion_policy(policy_logits, target_moves)
                value_loss = criterion_value(values.squeeze(), target_values)
                loss = policy_loss + value_loss
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        total_loss += epoch_loss / len(dataloader)

    avg_loss = total_loss / epochs
    return avg_loss


def continuous_training(
    num_iterations=100,
    games_per_iteration=50,
    mcts_simulations=50,
    train_epochs_per_iteration=3,
    batch_size=64,
    learning_rate=0.001,
    checkpoint_every=5
):
    """
    Main continuous training loop.

    Args:
        num_iterations: Number of self-play + training iterations
        games_per_iteration: Self-play games to generate per iteration
        mcts_simulations: MCTS simulations per move
        train_epochs_per_iteration: Training epochs on new data
        batch_size: Training batch size
        learning_rate: Learning rate
        checkpoint_every: Save checkpoint every N iterations
    """
    print("=" * 60)
    print("CONTINUOUS SELF-PLAY TRAINING")
    print("=" * 60)
    print(f"Iterations: {num_iterations}")
    print(f"Games per iteration: {games_per_iteration}")
    print(f"MCTS simulations: {mcts_simulations}")
    print(f"Training epochs per iteration: {train_epochs_per_iteration}")
    print(f"Device: {device}")
    print("=" * 60)

    # Initialize or load model
    model = ChessNet(
        input_channels=18,
        num_filters=64,  # Reduced from 128
        num_residual_blocks=3  # Reduced from 5
    ).to(device)

    # Compile model for faster inference (PyTorch 2.0+)
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("Model compiled with torch.compile for faster inference")
    except:
        print("torch.compile not available, using uncompiled model")

    # Try to load existing model
    best_model_path = os.path.join(WEIGHTS_DIR, "best_model.pt")
    start_iteration = 0

    if os.path.exists(best_model_path):
        print(f"\nFound existing checkpoint at {best_model_path}")
        try:
            checkpoint = torch.load(best_model_path, map_location=device)
            state_dict = checkpoint['model_state_dict']

            # Remove _orig_mod. prefix if present (from torch.compile)
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('_orig_mod.'):
                    cleaned_state_dict[k[len('_orig_mod.'):]] = v
                else:
                    cleaned_state_dict[k] = v

            # Try loading - this will fail if architecture changed
            model.load_state_dict(cleaned_state_dict, strict=True)
            start_iteration = checkpoint.get('iteration', 0)
            print(f"Successfully loaded checkpoint, resuming from iteration {start_iteration}")

        except (RuntimeError, KeyError) as e:
            print(f"\n  Checkpoint architecture mismatch (old model had different size)")
            print(f"  This is expected if you changed network parameters")
            print(f"  Starting fresh with new optimized architecture")
            print(f"  Old checkpoint will be overwritten\n")
            start_iteration = 0
    else:
        print("\nNo existing checkpoint found, starting from scratch")
        start_iteration = 0

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Continuous loop
    for iteration in range(start_iteration, num_iterations):
        iteration_start = time.time()

        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")

        # === SELF-PLAY ===
        print(f"\nGenerating {games_per_iteration} self-play games...")
        game_records = []

        model.eval()
        for game_num in tqdm(range(games_per_iteration), desc="Self-play"):
            game_record = play_self_play_game(
                model,
                num_simulations=mcts_simulations,
                temperature=1.0 if iteration < 10 else 0.5,  # Reduce exploration over time
                debug=(game_num == 0)  # Debug first game of each iteration
            )
            game_records.append(game_record)

        # Statistics
        total_positions = sum(len(g['positions']) for g in game_records)
        avg_moves = np.mean([g['move_count'] for g in game_records])
        white_wins = sum(1 for g in game_records if g['outcome'] == 1.0)
        black_wins = sum(1 for g in game_records if g['outcome'] == -1.0)
        draws = sum(1 for g in game_records if g['outcome'] == 0.0)

        print(f"\nSelf-play stats:")
        print(f"  Total positions: {total_positions}")
        print(f"  Avg moves/game: {avg_moves:.1f}")
        print(f"  Results: W:{white_wins} B:{black_wins} D:{draws}")

        # === TRAINING ===
        print(f"\nTraining on {total_positions} positions...")
        train_loss = train_on_selfplay(
            model,
            optimizer,
            game_records,
            epochs=train_epochs_per_iteration,
            batch_size=batch_size
        )

        print(f"Training loss: {train_loss:.4f}")

        # === SAVE ===
        if (iteration + 1) % checkpoint_every == 0:
            checkpoint_path = os.path.join(WEIGHTS_DIR, f"iteration_{iteration+1}.pt")
            torch.save({
                'iteration': iteration + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, checkpoint_path)
            print(f"✓ Saved checkpoint: {checkpoint_path}")

        # Always update best model
        torch.save({
            'iteration': iteration + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
        }, best_model_path)

        # Save game records
        games_file = os.path.join(SELFPLAY_DIR, f"games_iter_{iteration+1}.json")
        with open(games_file, 'w') as f:
            # Save without numpy arrays (just metadata)
            metadata = [{
                'outcome': g['outcome'],
                'move_count': g['move_count'],
                'num_positions': len(g['positions'])
            } for g in game_records]
            json.dump(metadata, f)

        iteration_time = time.time() - iteration_start
        print(f"\nIteration time: {iteration_time/60:.1f} minutes")
        print(f"Estimated time remaining: {(num_iterations - iteration - 1) * iteration_time / 3600:.1f} hours")


if __name__ == "__main__":
    # Run continuous training
    continuous_training(
        num_iterations=1000,        # Run for 1000 iterations (effectively infinite)
        games_per_iteration=25,     # 25 games per iteration (faster)
        mcts_simulations=50,        # 50 MCTS sims per move
        train_epochs_per_iteration=2,  # 2 epochs per iteration
        batch_size=64,
        learning_rate=0.001,
        checkpoint_every=5          # Save every 5 iterations
    )
