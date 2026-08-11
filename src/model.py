import os
import subprocess
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np
from config import Config

# Bump this whenever board_to_tensor's channel layout changes, so old
# checkpoints can be told apart from ones matching the current encoding.
ENCODING_VERSION = 1
ARCH_ID = f"resnet-{Config.num_residual_blocks}x{Config.num_filters}"


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def save_checkpoint(path, model, run_notes="", **extra):
    """Save weights plus the metadata needed to know what they are later.

    Fixes the "which of checkpoint2.pth/checkpoint3.pth/supervised_64.pth
    came from where" problem: every future checkpoint records its arch,
    the board-encoding version it expects, the git commit that produced it,
    and a free-text note. Callers can pass arbitrary extra metadata (e.g.
    iteration=5, train_loss=0.42) as keyword args.
    """
    state = model.state_dict()
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    torch.save({
        "state_dict": state,
        "arch": ARCH_ID,
        "encoding_version": ENCODING_VERSION,
        "git_sha": _git_sha(),
        "run_notes": run_notes,
        **extra,
    }, path)


def load_checkpoint(path, device):
    """Load a checkpoint saved by save_checkpoint(), or any of the older
    ad-hoc formats already in this repo (a raw state_dict, or a dict with a
    "model_state_dict" key). Returns (state_dict, metadata); metadata is {}
    for legacy checkpoints that never recorded any.
    """
    raw = torch.load(path, map_location=device)
    metadata = {}
    if isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
        metadata = {k: v for k, v in raw.items() if k != "state_dict"}
    elif isinstance(raw, dict) and "model_state_dict" in raw:
        state = raw["model_state_dict"]
    else:
        state = raw
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    return state, metadata

# ==========================================
# 1. HELPERS: Move Encoding & Board Tensor
# ==========================================
def encode_move(move):
    # Simple hash: from_sq * 64 + to_sq (Max 4096)
    return move.from_square * 64 + move.to_square

def decode_move(idx, board=None):
    f, t = divmod(idx, 64)
    # Underpromotions (to N/B/R) collide with the queen-promotion index since
    # the action space only encodes from/to squares. We can't tell which
    # promotion piece was intended, so always resolve to queen promotion -
    # legal and correct in the vast majority of real positions. True
    # underpromotion support needs a larger action space + retrain.
    if board is not None:
        piece = board.piece_at(f)
        if piece and piece.piece_type == chess.PAWN and t // 8 in (0, 7):
            return chess.Move(f, t, promotion=chess.QUEEN)
    return chess.Move(f, t)

def board_to_tensor(board):
    # 18 Channels: 
    # 0-5: White Pieces, 6-11: Black Pieces
    # 12-13: Repetitions
    # 14: Color (All 1 if White, All 0 if Black)
    # 15: Total Move Count (scaled)
    # 16-17: Castling Rights
    pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    matrix = np.zeros((18, 8, 8), dtype=np.float32)
    
    for i, piece in enumerate(pieces):
        for sq in board.pieces(piece, chess.WHITE):
            row, col = divmod(sq, 8)
            matrix[i, 7-row, col] = 1
        for sq in board.pieces(piece, chess.BLACK):
            row, col = divmod(sq, 8)
            matrix[i+6, 7-row, col] = 1
            
    if board.turn == chess.WHITE: matrix[14, :, :] = 1
    matrix[15, :, :] = len(board.move_stack) / 100.0 # Normalized game length
    
    # Castling
    if board.has_kingside_castling_rights(chess.WHITE): matrix[16, :, 4:] = 1
    if board.has_queenside_castling_rights(chess.WHITE): matrix[16, :, :4] = 1
    if board.has_kingside_castling_rights(chess.BLACK): matrix[17, :, 4:] = 1
    if board.has_queenside_castling_rights(chess.BLACK): matrix[17, :, :4] = 1
    
    return matrix

# ==========================================
# 2. NEURAL NETWORK (12-Block Super Model)
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 18 channels -> Config.num_filters
        self.conv1 = nn.Conv2d(18, Config.num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(Config.num_filters)
        
        # The "Tower"
        self.res_tower = nn.ModuleList([
            ResidualBlock(Config.num_filters) for _ in range(Config.num_residual_blocks)
        ])
        
        # Policy Head
        self.p_conv = nn.Conv2d(Config.num_filters, 32, kernel_size=1)
        self.p_bn = nn.BatchNorm2d(32)
        self.p_fc = nn.Linear(32 * 8 * 8, 4096) 
        
        # Value Head
        self.v_conv = nn.Conv2d(Config.num_filters, 8, kernel_size=1)
        self.v_bn = nn.BatchNorm2d(8)
        self.v_fc1 = nn.Linear(8 * 8 * 8, 256)
        self.v_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.res_tower:
            x = block(x)
            
        p = F.relu(self.p_bn(self.p_conv(x)))
        p = p.view(-1, 32 * 8 * 8)
        p = self.p_fc(p)
        p = F.log_softmax(p, dim=1) 
        
        v = F.relu(self.v_bn(self.v_conv(x)))
        v = v.view(-1, 8 * 8 * 8)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v))
        
        return p, v

# ==========================================
# 3. MCTS LOGIC (The Thinking Engine)
# ==========================================
class MCTSNode:
    def __init__(self, prior):
        self.p = prior
        self.n = 0
        self.w = 0
        self.children = {}

class ParallelMCTS:
    """
    Batches across boards (as before) AND within a single board's tree via
    virtual loss: each outer round collects up to `batch_size` in-flight
    leaves per board before doing one NN forward pass, instead of one
    leaf-per-forward-pass. For a single live game (the common case - main.py
    and gauntlet.py both call search([board])), the old version had no
    batching at all and did Config.num_simulations sequential batch-size-1
    forward passes; this is what actually uses the GPU.

    Virtual loss mechanics: when a simulation's selection descent claims a
    node, that node's n is incremented immediately (a simulation really is
    in flight through it) and a temporary penalty is added to w so other
    in-flight selections in the same batch see it as worse and tend to
    explore elsewhere. Once the leaf's real value is known, backprop removes
    exactly that penalty and adds the real value - net effect per node is
    identical to running the simulations one at a time, just computed in
    fewer, larger NN calls.
    """
    def __init__(self, model, virtual_loss=1.0):
        self.model = model
        self.virtual_loss = virtual_loss

    def search(self, boards, batch_size=None, time_limit_ms=None,
               dirichlet_epsilon=0.0, dirichlet_alpha=None):
        """
        dirichlet_epsilon/dirichlet_alpha: root exploration noise, off by
        default. This is a self-play concern only - mixing noise into root
        priors makes the search sometimes explore a move the network
        wouldn't otherwise try, which is exactly what you want when
        generating training data (a purely greedy/on-policy self-play loop
        can't discover better lines it never considers) and exactly what you
        don't want during actual play or gauntlet evaluation, where it would
        just be forced random quality loss. Callers that want it (see
        generate_selfplay_games in training.py) pass it explicitly;
        main.py/gauntlet.py never do, so this defaults to a no-op for them.
        """
        if batch_size is None:
            batch_size = Config.mcts_batch_size
        # `if time_limit_ms` (truthy check) would treat a legitimate 0 -
        # "you have no time left" - the same as "no limit was given", and
        # run the full unbounded search. Has to be an explicit None check.
        deadline = time.time() + time_limit_ms / 1000 if time_limit_ms is not None else None

        # 1. Create Roots
        roots = [MCTSNode(0) for _ in boards]

        # 2. Expand Roots (Initial Guess)
        tensors = torch.tensor(np.array([board_to_tensor(b) for b in boards])).to(Config.device)
        with torch.no_grad():
            pi, _ = self.model(tensors)
        pi = pi.exp().cpu().numpy()

        for i, root in enumerate(roots):
            legal_moves = list(boards[i].legal_moves)
            indices = [encode_move(move) for move in legal_moves]
            priors = pi[i][indices]

            if dirichlet_epsilon > 0 and legal_moves:
                alpha = dirichlet_alpha if dirichlet_alpha is not None else Config.dirichlet_alpha
                noise = np.random.dirichlet([alpha] * len(legal_moves))
                priors = (1 - dirichlet_epsilon) * priors + dirichlet_epsilon * noise

            for idx, p in zip(indices, priors):
                root.children[idx] = MCTSNode(p)

        # 3. Run simulations in batches
        simulations_done = 0
        while simulations_done < Config.num_simulations:
            if deadline is not None and time.time() >= deadline:
                break
            cur_batch = min(batch_size, Config.num_simulations - simulations_done)

            # A. Selection - cur_batch independent descents per board
            leaves = []
            paths = []
            leaf_boards = []

            for i, root in enumerate(roots):
                for _ in range(cur_batch):
                    node = root
                    b_copy = boards[i].copy()
                    path = [node]
                    node.n += 1
                    node.w += self.virtual_loss

                    # UCB Selection
                    while node.children and not b_copy.is_game_over():
                        best_score = -99999
                        best_child = None
                        sqrt_n = np.sqrt(node.n)

                        for act, child in node.children.items():
                            # UCB Formula
                            score = (-child.w / child.n if child.n > 0 else 0) + \
                                    Config.c_puct * child.p * (sqrt_n / (1 + child.n))

                            if score > best_score:
                                best_score = score
                                best_child = child
                                action = act

                        if best_child:
                            node = best_child
                            b_copy.push(decode_move(action, b_copy))
                            path.append(node)
                            node.n += 1
                            node.w += self.virtual_loss
                        else:
                            break

                    leaves.append(node)
                    paths.append(path)
                    leaf_boards.append(b_copy)

            # B. Evaluation
            tensors_eval = []
            valid_indices = []
            values = [0] * len(leaf_boards)

            for i, b in enumerate(leaf_boards):
                if b.is_game_over():
                    res = b.outcome().winner
                    if res is None: values[i] = 0
                    else: values[i] = 1 if res == b.turn else -1
                else:
                    tensors_eval.append(board_to_tensor(b))
                    valid_indices.append(i)

            if tensors_eval:
                t_batch = torch.tensor(np.array(tensors_eval)).to(Config.device)
                with torch.no_grad():
                    p_batch, v_batch = self.model(t_batch)

                p_batch = p_batch.exp().cpu().numpy()
                v_batch = v_batch.cpu().numpy()

                for j, idx in enumerate(valid_indices):
                    values[idx] = v_batch[j].item()
                    node = leaves[idx]
                    board = leaf_boards[idx]

                    # Expand the leaf (guarded: the same node can be reached
                    # twice within one batch before either occurrence gets
                    # expanded - only build children once)
                    if not node.children:
                        for move in board.legal_moves:
                            node.children[encode_move(move)] = MCTSNode(p_batch[j][encode_move(move)])

            # C. Backpropagation - undo the virtual loss, apply the real value
            for i, path in enumerate(paths):
                v = values[i]
                for node in reversed(path):
                    node.w += (v - self.virtual_loss)
                    v = -v  # Flip perspective for opponent

            simulations_done += cur_batch

        return roots
