"""
Continuous self-play training loop for chess AI.
Based on AlphaZero's reinforcement learning approach: self-play -> replay
buffer -> train -> gauntlet the candidate against the incumbent production
checkpoint -> promote only if it wins by a significant margin.

Trains the SAME architecture and board encoding used for live play
(src/model.py's ChessNet / ParallelMCTS / board_to_tensor) - a checkpoint
produced here can be loaded directly by src/main.py or gauntlet.py without
any conversion.
"""
import json
import os
import shutil
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import chess
from tqdm import tqdm

# Enable TensorFloat32 for better performance on Ampere GPUs (A10G, A100, 3090)
torch.set_float32_matmul_precision('high')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))        # .../src/training
SRC_DIR = os.path.dirname(SCRIPT_DIR)                            # .../src
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import Config
from model import (
    ChessNet, ParallelMCTS, decode_move, board_to_tensor,
    save_checkpoint, load_checkpoint,
)
from gauntlet import NeuralPlayer, run_match

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths. Self-play checkpoints are kept separate from the production
# checkpoint - promotion (see maybe_promote below) is a deliberate,
# gauntlet-gated copy, not something that happens as a side effect of
# training.
if os.path.exists("/root/data"):  # Modal environment
    WEIGHTS_DIR = "/root/data/weights"
    SELFPLAY_DIR = "/root/data/selfplay"
    # /root/src (where this repo's own src/checkpoint.pth lands via
    # add_local_dir in modal_training.py) is baked into the container image
    # at build time - it is NOT backed by the mounted volume, so anything
    # written there is silently discarded the moment the container exits.
    # Production must live under /root/data instead so a promotion actually
    # survives. Seed it from the image's checkpoint on the very first run
    # on this volume; every run after that reads/writes only this path.
    PRODUCTION_CHECKPOINT = "/root/data/checkpoint.pth"
    _image_checkpoint = os.path.join(SRC_DIR, "checkpoint.pth")
    if not os.path.exists(PRODUCTION_CHECKPOINT) and os.path.exists(_image_checkpoint):
        os.makedirs(os.path.dirname(PRODUCTION_CHECKPOINT), exist_ok=True)
        shutil.copy2(_image_checkpoint, PRODUCTION_CHECKPOINT)
        print(f"Seeded persistent production checkpoint from {_image_checkpoint}")
else:  # Local environment
    WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "selfplay_weights")
    SELFPLAY_DIR = os.path.join(SCRIPT_DIR, "selfplay_games")
    PRODUCTION_CHECKPOINT = os.path.join(SRC_DIR, "checkpoint.pth")

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(SELFPLAY_DIR, exist_ok=True)

SELFPLAY_BEST_PATH = os.path.join(WEIGHTS_DIR, "selfplay_best.pt")
SELFPLAY_CANDIDATE_PATH = os.path.join(WEIGHTS_DIR, "selfplay_candidate.pt")


class SelfPlayDataset(Dataset):
    """Dataset from self-play games. Each position is (board_tensor,
    4096-dim MCTS visit-count policy distribution, scalar value target)."""

    def __init__(self, game_records):
        self.data = []
        for game in game_records:
            self.data.extend(game['positions'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board_tensor, policy, value = self.data[idx]
        return (
            torch.FloatTensor(board_tensor),
            torch.FloatTensor(policy),
            torch.FloatTensor([value]),
        )


def generate_selfplay_games(model, num_games, num_simulations, num_parallel_games=None,
                             temperature=1.0, max_moves=200, debug=False, dirichlet_epsilon=0.25):
    """
    Play num_games self-play games with up to num_parallel_games running
    simultaneously, all driven through ONE ParallelMCTS - each call to
    mcts.search(boards) advances every active game by one ply and batches
    the NN evaluation across all of them (on top of the per-board virtual-
    loss batching ParallelMCTS already does internally). This is what
    Config.num_parallel_games was for; before this, self-play played
    exactly one game at a time and that setting was never read anywhere.

    As games finish they're replaced with fresh ones so the batch stays
    full and GPU utilization doesn't taper off near the end of a run.

    dirichlet_epsilon: root exploration noise (AlphaZero default 0.25),
    mixed into root priors at every move of every game here - this is the
    ONE place in the codebase that should ever pass a nonzero value, since
    main.py/gauntlet.py's evaluation calls must stay noise-free.
    """
    if num_parallel_games is None:
        num_parallel_games = Config.num_parallel_games
    num_parallel_games = max(1, min(num_parallel_games, num_games))

    Config.device = device
    Config.num_simulations = num_simulations
    mcts = ParallelMCTS(model)

    def new_game_state():
        return {'board': chess.Board(), 'history': [], 'move_count': 0}

    active = [new_game_state() for _ in range(num_parallel_games)]
    started = num_parallel_games
    completed_records = []
    start_time = time.time()

    while len(completed_records) < num_games and active:
        boards = [g['board'] for g in active]
        roots = mcts.search(boards, dirichlet_epsilon=dirichlet_epsilon)

        still_active = []
        for g, root in zip(active, roots):
            board = g['board']

            if root.children:
                move_indices = list(root.children.keys())
                counts = np.array([root.children[idx].n for idx in move_indices], dtype=np.float64)

                if temperature > 0 and counts.sum() > 0:
                    weights = counts ** (1.0 / temperature)
                    if weights.sum() <= 0:
                        weights = np.ones_like(weights)
                    probs = weights / weights.sum()
                    chosen = np.random.choice(len(move_indices), p=probs)
                else:
                    probs = np.zeros_like(counts)
                    probs[np.argmax(counts)] = 1.0
                    chosen = int(np.argmax(counts))

                # Full 4096-dim policy target: the MCTS-improved distribution
                # over legal moves (root.children is already restricted to
                # legal moves, so no separate legality filtering is needed).
                policy = np.zeros(4096, dtype=np.float32)
                for i, idx in enumerate(move_indices):
                    policy[idx] = probs[i]

                g['history'].append({
                    'board_tensor': board_to_tensor(board),
                    'policy': policy,
                    'turn': board.turn,
                })

                move = decode_move(move_indices[chosen], board)
                board.push(move)
                g['move_count'] += 1

            finished = board.is_game_over() or g['move_count'] >= max_moves or not root.children
            if finished:
                result = board.result() if board.is_game_over() else "1/2-1/2"
                outcome = 1.0 if result == "1-0" else (-1.0 if result == "0-1" else 0.0)
                positions = [
                    (e['board_tensor'], e['policy'], outcome if e['turn'] == chess.WHITE else -outcome)
                    for e in g['history']
                ]
                completed_records.append({
                    'positions': positions, 'outcome': outcome, 'move_count': g['move_count'],
                })
                if debug:
                    print(f"  Game {len(completed_records)}/{num_games} done: "
                          f"{result} in {g['move_count']} moves "
                          f"({len(completed_records)}/{num_games} complete, "
                          f"{time.time() - start_time:.1f}s elapsed)")
                if started < num_games:
                    still_active.append(new_game_state())
                    started += 1
            else:
                still_active.append(g)

        active = still_active

    return completed_records[:num_games]


def train_on_selfplay(model, optimizer, game_records, epochs=3, batch_size=64, use_amp=None):
    """Train model on self-play data.

    Policy target is a soft distribution (MCTS visit counts), not a single
    label - ChessNet.forward() already returns log-softmax policy output,
    so the correct loss is a soft cross-entropy against those log-probs
    (standard AlphaZero policy loss), not nn.CrossEntropyLoss (which wants
    integer class labels and would throw away most of the MCTS signal).
    """
    if use_amp is None:
        use_amp = device.type == "cuda"

    dataset = SelfPlayDataset(game_records)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    total_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for boards, target_policies, target_values in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            boards = boards.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=use_amp):
                log_probs, values = model(boards)
                policy_loss = -(target_policies * log_probs).sum(dim=1).mean()
                value_loss = torch.nn.functional.mse_loss(values, target_values)
                loss = policy_loss + value_loss

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        total_loss += epoch_loss / len(dataloader)

    return total_loss / epochs


def evaluate_candidate_vs_incumbent(candidate_path, incumbent_path, games, sims, verbose=True):
    """Gauntlet the candidate against the current incumbent using the exact
    same paired-opening match logic gauntlet.py's CLI uses (run_match) -
    one gauntlet, whether you invoke it by hand or from the training loop."""
    player_a = NeuralPlayer(candidate_path, device, sims)
    player_b = NeuralPlayer(incumbent_path, device, sims)
    try:
        return run_match(player_a, player_b, games=games, verbose=verbose)
    finally:
        player_a.close()
        player_b.close()


def maybe_promote(candidate_path, production_path, result, min_score_fraction=0.55, min_games=10):
    """Promote (copy candidate -> production) only if it cleared a minimum
    win-rate margin over a minimum number of games. This is a simple
    threshold, not a rigorous sequential test (SPRT) - good enough to stop
    noise-level "improvements" from silently overwriting production, which
    is the actual goal here, not statistical precision.
    """
    if result['games'] < min_games:
        print(f"Only {result['games']} games played (< min_games={min_games}) - not promoting.")
        return False

    score_fraction = result['a_score'] / result['games']
    print(f"Candidate score vs incumbent: {result['a_score']}/{result['games']} "
          f"({score_fraction:.1%}), Elo diff {result['elo_diff']:+.0f}")

    if score_fraction >= min_score_fraction:
        shutil.copy2(candidate_path, production_path)
        print(f"PROMOTED: {candidate_path} -> {production_path}")
        return True

    print(f"Not promoted (needed >= {min_score_fraction:.0%}, scored {score_fraction:.1%}).")
    return False


def continuous_training(
    num_iterations=100,
    games_per_iteration=50,
    num_parallel_games=None,
    mcts_simulations=100,
    train_epochs_per_iteration=3,
    batch_size=64,
    learning_rate=0.001,
    checkpoint_every=5,
    replay_buffer_iters=5,
    promote_every=1,
    promotion_games=20,
    promotion_sims=200,
    promotion_min_score_fraction=0.55,
    production_checkpoint_path=None,
):
    """
    Args:
        replay_buffer_iters: train on positions pooled from the last N
            iterations' self-play games, not just the freshest batch - more
            stable than training on a small, fully-fresh dataset every time.
        promote_every: run the gauntlet-gated promotion check every N
            iterations (0 disables promotion entirely - self-play still
            proceeds and improves its own trajectory, just never overwrites
            production).
        production_checkpoint_path: override for testing - defaults to the
            real src/checkpoint.pth.
    """
    if production_checkpoint_path is None:
        production_checkpoint_path = PRODUCTION_CHECKPOINT

    print("=" * 60)
    print("CONTINUOUS SELF-PLAY TRAINING (production architecture)")
    print("=" * 60)
    print(f"Iterations: {num_iterations}")
    print(f"Games per iteration: {games_per_iteration} (parallel: {num_parallel_games or Config.num_parallel_games})")
    print(f"MCTS simulations: {mcts_simulations}")
    print(f"Training epochs per iteration: {train_epochs_per_iteration}")
    print(f"Replay buffer: last {replay_buffer_iters} iterations")
    print(f"Architecture: {Config.num_residual_blocks} blocks x {Config.num_filters} filters")
    print(f"Promotion target: {production_checkpoint_path}")
    print(f"Device: {device}")
    print("=" * 60)

    model = ChessNet().to(device)

    # Resume self-play progress if it exists; otherwise bootstrap from the
    # current production checkpoint (much better starting point than random
    # init); otherwise start from scratch.
    start_iteration = 0
    if os.path.exists(SELFPLAY_BEST_PATH):
        source = SELFPLAY_BEST_PATH
    elif os.path.exists(production_checkpoint_path):
        source = production_checkpoint_path
    else:
        source = None

    if source:
        try:
            state, meta = load_checkpoint(source, device)
            model.load_state_dict(state, strict=True)
            start_iteration = meta.get("iteration", 0) if source == SELFPLAY_BEST_PATH else 0
            print(f"\nLoaded weights from {source}")
            if meta:
                print(f"  metadata: {meta}")
            print(f"  resuming from iteration {start_iteration}")
        except (RuntimeError, KeyError) as e:
            print(f"\n  Checkpoint at {source} didn't match the current architecture ({e})")
            print("  Starting fresh with random init\n")
            start_iteration = 0
    else:
        print("\nNo checkpoint found anywhere, starting from scratch")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    replay_buffer = deque(maxlen=replay_buffer_iters)

    for iteration in range(start_iteration, num_iterations):
        iteration_start = time.time()
        print(f"\n{'=' * 60}")
        print(f"ITERATION {iteration + 1}/{num_iterations}")
        print(f"{'=' * 60}")

        print(f"\nGenerating {games_per_iteration} self-play games...")
        model.eval()
        game_records = generate_selfplay_games(
            model, games_per_iteration, mcts_simulations,
            num_parallel_games=num_parallel_games,
            temperature=1.0 if iteration < 10 else 0.5,
            debug=True,
        )
        replay_buffer.append(game_records)
        buffered_games = [g for iter_games in replay_buffer for g in iter_games]

        total_positions = sum(len(g['positions']) for g in game_records)
        buffered_positions = sum(len(g['positions']) for g in buffered_games)
        avg_moves = np.mean([g['move_count'] for g in game_records])
        white_wins = sum(1 for g in game_records if g['outcome'] == 1.0)
        black_wins = sum(1 for g in game_records if g['outcome'] == -1.0)
        draws = sum(1 for g in game_records if g['outcome'] == 0.0)

        print(f"\nSelf-play stats (this iteration):")
        print(f"  Positions: {total_positions}  |  Avg moves/game: {avg_moves:.1f}")
        print(f"  Results: W:{white_wins} B:{black_wins} D:{draws}")
        print(f"  Replay buffer: {len(buffered_games)} games / {buffered_positions} positions "
              f"across {len(replay_buffer)} iteration(s)")

        print(f"\nTraining on {buffered_positions} positions...")
        train_loss = train_on_selfplay(
            model, optimizer, buffered_games,
            epochs=train_epochs_per_iteration, batch_size=batch_size,
        )
        print(f"Training loss: {train_loss:.4f}")

        run_notes = f"selfplay iter {iteration + 1}/{num_iterations}, avg_moves={avg_moves:.1f}"
        if (iteration + 1) % checkpoint_every == 0:
            checkpoint_path = os.path.join(WEIGHTS_DIR, f"selfplay_iteration_{iteration + 1}.pt")
            save_checkpoint(checkpoint_path, model, run_notes=run_notes,
                             iteration=iteration + 1, train_loss=train_loss)
            print(f"Saved checkpoint: {checkpoint_path}")

        save_checkpoint(SELFPLAY_BEST_PATH, model, run_notes=run_notes,
                         iteration=iteration + 1, train_loss=train_loss)

        games_file = os.path.join(SELFPLAY_DIR, f"games_iter_{iteration + 1}.json")
        with open(games_file, 'w') as f:
            json.dump([{
                'outcome': g['outcome'],
                'move_count': g['move_count'],
                'num_positions': len(g['positions']),
            } for g in game_records], f)

        if promote_every and (iteration + 1) % promote_every == 0 and os.path.exists(production_checkpoint_path):
            print(f"\nGauntlet: candidate vs incumbent ({promotion_games} games, {promotion_sims} sims/move)...")
            save_checkpoint(SELFPLAY_CANDIDATE_PATH, model, run_notes=f"candidate @ iter {iteration + 1}",
                             iteration=iteration + 1)
            result = evaluate_candidate_vs_incumbent(
                SELFPLAY_CANDIDATE_PATH, production_checkpoint_path,
                games=promotion_games, sims=promotion_sims,
            )
            maybe_promote(SELFPLAY_CANDIDATE_PATH, production_checkpoint_path, result,
                          min_score_fraction=promotion_min_score_fraction)

        iteration_time = time.time() - iteration_start
        print(f"\nIteration time: {iteration_time / 60:.1f} minutes")
        remaining = (num_iterations - iteration - 1) * iteration_time
        print(f"Estimated time remaining: {remaining / 3600:.1f} hours")


if __name__ == "__main__":
    continuous_training(
        num_iterations=1000,
        games_per_iteration=25,
        mcts_simulations=100,
        train_epochs_per_iteration=2,
        batch_size=64,
        learning_rate=0.001,
        checkpoint_every=5,
    )
