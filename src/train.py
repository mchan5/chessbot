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

# Policy mapping constants
# Base policy maps (from_square * 64 + to_square) => 4096 entries
# Promotions are encoded by extending the policy with 4 slots per from-to
# index: index = POLICY_BASE + (from*64 + to) * 4 + promo_idx
# where promo_idx maps to [QUEEN, ROOK, BISHOP, KNIGHT]. This keeps
# a simple deterministic mapping and ensures promotions round-trip.
PROMOTION_PIECES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
POLICY_BASE = 64 * 64
POLICY_SIZE = POLICY_BASE * (1 + len(PROMOTION_PIECES))

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Determine paths (works both locally and on Modal)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists("/root/weights"):  # Modal environment
    WEIGHTS_DIR = "/root/weights"
    DATA_DIR = "/root/data"
else:  # Local environment
    WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "../weights")
    DATA_DIR = os.path.join(SCRIPT_DIR, "../data")

os.makedirs(WEIGHTS_DIR, exist_ok=True)


# ==================== NEURAL NETWORK ====================

class ChessNet(nn.Module):
    """
    Neural network for chess position evaluation.
    Takes board state as input and outputs:
    - Policy: probability distribution over legal moves
    - Value: position evaluation (-1 to 1)
    """
    def __init__(self, input_channels=18, num_filters=128, num_residual_blocks=5):
        super(ChessNet, self).__init__()

        # Initial convolutional block
        self.conv_initial = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_residual_blocks)
        ])

        # Policy head - predicts move probabilities
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, POLICY_SIZE)
        )

        # Value head - predicts position score
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv_initial(x)
        for block in self.residual_blocks:
            x = block(x)

        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value


class ResidualBlock(nn.Module):
    """Residual block for deeper network training"""
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


# ==================== BOARD ENCODING ====================

def encode_board(board):
    """
    Encode chess board to neural network input tensor.
    Uses 18 channels: 12 for pieces + 6 for meta info.
    Optimized for speed with minimal allocations.
    """
    planes = np.zeros((18, 8, 8), dtype=np.float32)

    # Piece planes (6 piece types * 2 colors) - optimized loop
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank, file = divmod(square, 8)
            channel = piece_map[piece.piece_type] + (0 if piece.color else 6)
            planes[channel, rank, file] = 1.0

    # Meta information planes - vectorized where possible
    if board.turn == chess.WHITE:
        planes[12] = 1.0  # Faster than [:, :] assignment

    # Castling rights - batch check
    castling = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK)
    ]
    for i, has_right in enumerate(castling):
        if has_right:
            planes[13 + i] = 1.0

    # En passant
    if board.ep_square:
        rank, file = divmod(board.ep_square, 8)
        planes[17, rank, file] = 1.0

    return planes


def move_to_index(move):
    """Convert move to policy index.

    Mapping:
    - Non-promotion moves: index = from_square * 64 + to_square
    - Promotion moves: index = POLICY_BASE + (from*64 + to) * N + promo_idx
      where N = len(PROMOTION_PIECES) and promo_idx is the index of the
      promotion piece in PROMOTION_PIECES.

    This deterministic mapping lets the policy uniquely represent promotion
    piece choices.
    """
    base_idx = move.from_square * 64 + move.to_square
    if move.promotion is None:
        return base_idx

    # Promotion: map to extended region
    try:
        promo_idx = PROMOTION_PIECES.index(move.promotion)
    except ValueError:
        # Unknown promotion type - fallback to base index
        return base_idx

    return POLICY_BASE + base_idx * len(PROMOTION_PIECES) + promo_idx


def index_to_move(index, board):
    """Convert policy index to move"""
    # Non-promotion region
    if index < POLICY_BASE:
        from_square = index // 64
        to_square = index % 64
        move = chess.Move(from_square, to_square)
        if move in board.legal_moves:
            return move
        # No valid non-promotion move found
        return None

    # Promotion region
    rem = index - POLICY_BASE
    n_promos = len(PROMOTION_PIECES)
    base_idx = rem // n_promos
    promo_idx = rem % n_promos

    from_square = base_idx // 64
    to_square = base_idx % 64
    promotion = PROMOTION_PIECES[promo_idx]

    promoted_move = chess.Move(from_square, to_square, promotion=promotion)
    if promoted_move in board.legal_moves:
        return promoted_move

    return None


# ==================== MCTS ====================

class MCTSNode:
    """Monte Carlo Tree Search Node"""
    def __init__(self, board, parent=None, move=None, prior=0):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.prior = prior

        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.is_expanded = False

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visit_count, c_puct=1.5):
        """Upper Confidence Bound for Trees"""
        prior_score = c_puct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)
        if self.visit_count > 0:
            value_score = self.value()
        else:
            value_score = 0
        return value_score + prior_score

    def select_child(self, c_puct=1.5):
        """Select child with highest UCB score"""
        return max(self.children.values(),
                   key=lambda node: node.ucb_score(self.visit_count, c_puct))

    def expand(self, policy_probs, dirichlet_alpha=0.0, dirichlet_eps=0.0):
        """Expand node by adding children for legal moves.

        If this node is the root (parent is None) and dirichlet_eps>0,
        mix the normalized policy probabilities with a Dirichlet noise
        vector: probs = (1 - eps) * probs + eps * dir_noise.
        """
        if self.is_expanded:
            return

        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            self.is_expanded = True
            return

        # Normalize probabilities over legal moves
        move_probs = {}
        total_prob = 0
        probs_list = []
        for move in legal_moves:
            idx = move_to_index(move)
            prob = float(policy_probs[idx]) if idx < len(policy_probs) else 1e-8
            prob = max(prob, 1e-8)
            move_probs[move] = prob
            probs_list.append(prob)
            total_prob += prob

        # Normalize into proper distribution
        if total_prob > 0:
            for move in legal_moves:
                move_probs[move] /= total_prob

        # If root node, optionally apply Dirichlet noise to encourage exploration
        if self.parent is None and dirichlet_eps and dirichlet_alpha > 0:
            alpha = dirichlet_alpha
            eps = dirichlet_eps
            # Dirichlet over legal move dimension
            noise = np.random.dirichlet([alpha] * len(legal_moves))
            for i, move in enumerate(legal_moves):
                move_probs[move] = (1 - eps) * move_probs[move] + eps * noise[i]

            # Renormalize after mixing
            s = sum(move_probs.values())
            if s > 0:
                for move in legal_moves:
                    move_probs[move] /= s

        # Create child nodes
        for move in legal_moves:
            new_board = self.board.copy()
            new_board.push(move)
            self.children[move] = MCTSNode(new_board, parent=self, move=move, prior=move_probs[move])

        self.is_expanded = True

    def backpropagate(self, value):
        """Backpropagate value up the tree"""
        self.visit_count += 1
        self.value_sum += value
        if self.parent:
            self.parent.backpropagate(-value)  # Negate for opponent


class MCTS:
    """Monte Carlo Tree Search with batched evaluation

    Configurable parameters:
    - c_puct: controls exploration-exploitation in UCB
    - dirichlet_alpha / dirichlet_eps: root noise mixture (AlphaZero style)
    - temperature: final distribution temperature (1.0 = standard, <1 more greedy)
    """
    def __init__(self, model, num_simulations=100, batch_size=8, c_puct=1.5,
                 dirichlet_alpha=0.3, dirichlet_eps=0.25, temperature=1.0):
        self.model = model
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.temperature = temperature

    @torch.no_grad()
    def search(self, board):
        """Run MCTS with batched neural network evaluation"""
        root = MCTSNode(board)

        # Process simulations in batches
        for batch_start in range(0, self.num_simulations, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.num_simulations)
            nodes_to_eval = []
            terminal_nodes = []

            # Collect batch of nodes to evaluate
            for _ in range(batch_end - batch_start):
                node = root

                # Selection - traverse to leaf
                while node.is_expanded and node.children:
                    if node.board.is_game_over():
                        break
                    node = node.select_child(self.c_puct)

                # Separate terminal vs non-terminal nodes
                if node.board.is_game_over():
                    result = node.board.result()
                    if result == "1-0":
                        value = 1.0 if node.board.turn == chess.BLACK else -1.0
                    elif result == "0-1":
                        value = 1.0 if node.board.turn == chess.WHITE else -1.0
                    else:
                        value = 0.0
                    terminal_nodes.append((node, value))
                else:
                    nodes_to_eval.append(node)

            # Batch evaluate non-terminal nodes
            if nodes_to_eval:
                # Encode all boards at once
                boards = np.array([encode_board(n.board) for n in nodes_to_eval])
                boards_tensor = torch.FloatTensor(boards).to(device)

                # Single batched forward pass
                policy_logits_batch, values_batch = self.model(boards_tensor)
                policy_probs_batch = torch.softmax(policy_logits_batch, dim=1).cpu().numpy()
                values_batch = values_batch.cpu().numpy().flatten()

                # Expand and backpropagate (pass dirichlet params for root expansion)
                for i, node in enumerate(nodes_to_eval):
                    node.expand(policy_probs_batch[i], dirichlet_alpha=self.dirichlet_alpha,
                                dirichlet_eps=self.dirichlet_eps)
                    node.backpropagate(float(values_batch[i]))

            # Backpropagate terminal nodes
            for node, value in terminal_nodes:
                node.backpropagate(value)

        # Return visit count distribution as policy (optionally apply temperature)
        visit_counts = np.zeros(POLICY_SIZE)
        for move, child in root.children.items():
            visit_counts[move_to_index(move)] = child.visit_count

        # Apply temperature to produce final distribution
        temp = self.temperature
        if temp is None or temp == 1.0:
            return visit_counts

        # Temperature <= 0 treated as greedy deterministic
        if temp <= 0.0:
            probs = np.zeros_like(visit_counts, dtype=float)
            if visit_counts.sum() > 0:
                probs[np.argmax(visit_counts)] = 1.0
            return probs

        # Soften/sharpen distribution: counts^(1/T)
        with np.errstate(divide='ignore', invalid='ignore'):
            adjusted = np.power(visit_counts, 1.0 / temp)
        s = adjusted.sum()
        if s > 0:
            adjusted = adjusted / s
        return adjusted


# ==================== DATASET ====================

class ChessDataset(Dataset):
    """Dataset for supervised learning from games and puzzles"""
    def __init__(self, games_path=None, puzzles_path=None):
        self.data = []

        # Use default paths if not provided
        if games_path is None:
            games_path = os.path.join(DATA_DIR, "games.json")
        if puzzles_path is None:
            puzzles_path = os.path.join(DATA_DIR, "puzzles.json")

        # Load games
        with open(games_path) as f:
            games = json.load(f)

        print(f"Processing {len(games)} games...")
        for game_data in tqdm(games[:1000]):  # Limit for faster training
            if game_data["type"] == "game":
                self._process_game(game_data["moves"])

        # Load puzzles
        with open(puzzles_path) as f:
            puzzles = json.load(f)

        print(f"Processing {len(puzzles)} puzzles...")
        for puzzle_data in tqdm(puzzles[:500]):  # Limit for faster training
            if puzzle_data["type"] == "puzzle":
                self._process_puzzle(puzzle_data["fen"], puzzle_data["solution"])

        print(f"Total positions: {len(self.data)}")

    def _process_game(self, moves_uci):
        """Extract positions and moves from a game"""
        board = chess.Board()

        for move_uci in moves_uci:
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    # Store position and target move
                    encoded_board = encode_board(board)
                    move_idx = move_to_index(move)

                    # Assign outcome value (simplified: 0 for now, would need game result)
                    value = 0.0

                    self.data.append((encoded_board, move_idx, value))
                    board.push(move)
                else:
                    break
            except:
                break

    def _process_puzzle(self, fen, solution_moves):
        """Extract positions from puzzle"""
        try:
            board = chess.Board(fen)
            for move_uci in solution_moves:
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in board.legal_moves:
                        encoded_board = encode_board(board)
                        move_idx = move_to_index(move)
                        value = 1.0  # Puzzle solutions are "winning"

                        self.data.append((encoded_board, move_idx, value))
                        board.push(move)
                    else:
                        break
                except:
                    break
        except:
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
    """Train for one epoch"""
    model.train()
    total_loss = 0
    policy_loss_sum = 0
    value_loss_sum = 0

    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()

    for boards, target_moves, target_values in tqdm(dataloader, desc="Training"):
        boards = boards.to(device)
        target_moves = target_moves.to(device).squeeze(1)
        target_values = target_values.to(device).squeeze(1)

        optimizer.zero_grad()

        # Forward pass
        policy_logits, values = model(boards)

        # Calculate losses
        policy_loss = criterion_policy(policy_logits, target_moves)
        value_loss = criterion_value(values.squeeze(), target_values)

        # Combined loss
        loss = policy_loss + value_loss

        # Backward pass
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

    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()

    with torch.no_grad():
        for boards, target_moves, target_values in dataloader:
            boards = boards.to(device)
            target_moves = target_moves.to(device).squeeze(1)
            target_values = target_values.to(device).squeeze(1)

            policy_logits, values = model(boards)

            policy_loss = criterion_policy(policy_logits, target_moves)
            value_loss = criterion_value(values.squeeze(), target_values)
            loss = policy_loss + value_loss

            total_loss += loss.item()

            # Accuracy
            predicted = torch.argmax(policy_logits, dim=1)
            correct += (predicted == target_moves).sum().item()
            total += target_moves.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0

    return avg_loss, accuracy


# ==================== MAIN ====================

def main():
    # Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    NUM_RESIDUAL_BLOCKS = 5  # Reduced for faster training
    NUM_FILTERS = 128  # Reduced for faster training

    # Load dataset
    print("Loading dataset...")
    dataset = ChessDataset()  # Uses default paths

    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Initialize model
    print("Initializing model...")
    model = ChessNet(
        input_channels=18,
        num_filters=NUM_FILTERS,
        num_residual_blocks=NUM_RESIDUAL_BLOCKS
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*60}")

        # Train
        train_loss, policy_loss, value_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.4f} (Policy: {policy_loss:.4f}, Value: {value_loss:.4f})")

        # Validate
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Step scheduler
        scheduler.step()
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(WEIGHTS_DIR, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
            }, best_model_path)
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")

        # Save checkpoint
        if (epoch + 1) % 2 == 0:
            checkpoint_path = os.path.join(WEIGHTS_DIR, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

    # Save final model
    final_model_path = os.path.join(WEIGHTS_DIR, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print("\n" + "="*60)
    print(f"Training complete! Models saved to {WEIGHTS_DIR}")
    print("="*60)

    # Demonstrate MCTS inference
    print("\nDemonstrating MCTS search on starting position...")
    model.eval()
    mcts = MCTS(model, num_simulations=50)
    board = chess.Board()
    visit_counts = mcts.search(board)

    # Get top 5 moves
    top_indices = np.argsort(visit_counts)[-5:][::-1]
    print("\nTop moves by MCTS:")
    for idx in top_indices:
        move = index_to_move(idx, board)
        if move and visit_counts[idx] > 0:
            print(f"  {move.uci()}: {visit_counts[idx]} visits")


if __name__ == "__main__":
    main()
