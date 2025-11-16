import sys
import numpy as np

try:
    import chess
except Exception:
    print('Skipping MCTS noise smoke test: python-chess not installed in environment')
    sys.exit(0)

try:
    import torch
except Exception:
    print('Skipping MCTS noise smoke test: torch not installed in environment')
    sys.exit(0)

from src.train import MCTS, POLICY_SIZE


class DummyModel:
    def __call__(self, boards_tensor):
        batch = boards_tensor.shape[0]
        # Return uniform logits and zero values
        policy_logits = torch.zeros(batch, POLICY_SIZE)
        values = torch.zeros(batch, 1)
        return policy_logits, values


def run_once(dirichlet_eps):
    model = DummyModel()
    mcts = MCTS(model, num_simulations=8, batch_size=4, dirichlet_alpha=0.3, dirichlet_eps=dirichlet_eps, temperature=1.0)
    board = chess.Board()
    dist = mcts.search(board)
    return dist


if __name__ == '__main__':
    # With eps=0 there should be (likely) a deterministic result across runs
    a = run_once(0.0)
    b = run_once(0.0)
    print('eps=0 equal:', np.array_equal(a, b))

    # With eps>0 runs should differ due to Dirichlet noise
    c = run_once(0.5)
    d = run_once(0.5)
    print('eps>0 equal:', np.array_equal(c, d))

    if np.array_equal(c, d):
        raise SystemExit('Dirichlet noise did not change distributions as expected')
    else:
        print('Dirichlet noise appears to inject variability (smoke test passed)')
