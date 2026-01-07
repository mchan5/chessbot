"""
Modal deployment for continuous self-play training.
This will run indefinitely, continuously improving the model.
"""
import modal

app = modal.App("chess-continuous-training")

# Persistent volume for weights
volume = modal.Volume.from_name("chess-weights", create_if_missing=True)

# Image with all dependencies
import os
script_dir = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "numpy",
        "tqdm",
        "python-chess",
    )
    .add_local_dir(script_dir, "/root/src")  # Copy entire src directory
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,  # 24 hour timeout (increase as needed)
    volumes={"/root/data": volume},  # Mount once, subdirectories for weights and selfplay
)
def continuous_train(
    num_iterations: int = 1000,
    games_per_iteration: int = 25,
    mcts_simulations: int = 50,
):
    """
    Run continuous self-play training.

    Args:
        num_iterations: Number of iterations (set high for continuous)
        games_per_iteration: Games to play per iteration
        mcts_simulations: MCTS simulations per move
    """
    import sys
    sys.path.insert(0, "/root/src")

    from train_continuous import continuous_training

    print("=" * 60)
    print("Starting continuous training on Modal")
    print("=" * 60)

    continuous_training(
        num_iterations=num_iterations,
        games_per_iteration=games_per_iteration,
        mcts_simulations=mcts_simulations,
        train_epochs_per_iteration=2,
        batch_size=64,
        learning_rate=0.001,
        checkpoint_every=5
    )

    # Commit volume to persist all changes
    volume.commit()

    print("\n" + "=" * 60)
    print("Continuous training completed!")
    print("=" * 60)


@app.local_entrypoint()
def main(
    iterations: int = 1000,
    games: int = 25,
    simulations: int = 50,
):
    """
    Launch continuous training on Modal.

    Usage:
        modal run train_continuous_modal.py
        modal run train_continuous_modal.py --iterations 10000 --games 50
    """
    print(f"Launching continuous training on Modal...")
    print(f"  Iterations: {iterations}")
    print(f"  Games per iteration: {games}")
    print(f"  MCTS simulations: {simulations}")
    print("\nThis will run for a long time. Press Ctrl+C to stop monitoring.")
    print("The job will continue running on Modal even if you disconnect.\n")

    continuous_train.remote(
        num_iterations=iterations,
        games_per_iteration=games,
        mcts_simulations=simulations,
    )
