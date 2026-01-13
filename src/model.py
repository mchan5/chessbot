import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np
from config import Config

# ==========================================
# 1. HELPERS: Move Encoding & Board Tensor
# ==========================================
def encode_move(move):
    # Simple hash: from_sq * 64 + to_sq (Max 4096)
    return move.from_square * 64 + move.to_square

def decode_move(idx):
    f, t = divmod(idx, 64)
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
    def __init__(self, model):
        self.model = model
    
    def search(self, boards):
        # 1. Create Roots
        roots = [MCTSNode(0) for _ in boards]
        
        # 2. Expand Roots (Initial Guess)
        tensors = torch.tensor(np.array([board_to_tensor(b) for b in boards])).to(Config.device)
        with torch.no_grad():
            pi, _ = self.model(tensors)
        pi = pi.exp().cpu().numpy()
        
        for i, root in enumerate(roots):
            for move in boards[i].legal_moves:
                idx = encode_move(move)
                root.children[idx] = MCTSNode(pi[i][idx])
                
        # 3. Run Simulations (This was missing!)
        for _ in range(Config.num_simulations):
            # A. Selection
            leaves = []
            paths = []
            leaf_boards = []
            
            for i, root in enumerate(roots):
                node = root
                b_copy = boards[i].copy()
                path = [node]
                
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
                        b_copy.push(decode_move(action))
                        path.append(node)
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
                    
                    # Expand the leaf
                    for move in board.legal_moves:
                        node.children[encode_move(move)] = MCTSNode(p_batch[j][encode_move(move)])

            # C. Backpropagation
            for i, path in enumerate(paths):
                v = values[i]
                for node in reversed(path):
                    node.w += v
                    node.n += 1
                    v = -v # Flip perspective for opponent
                    
        return roots
    
    