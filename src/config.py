import torch

class Config:
    # --- HARDWARE ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_parallel_games = 64   
    
    # 12 blocks, 128 filters (~10 Million params)
    num_residual_blocks = 12 
    num_filters = 128        
    
    # --- MCTS SETTINGS ---
    num_simulations = 100    # Simulations per move during training
    c_puct = 1.0             # Exploration constant
    mcts_batch_size = 16     # In-flight leaves collected per NN forward pass
                              # (virtual loss) - see ParallelMCTS in model.py
    dirichlet_alpha = 0.3     # Root noise concentration (AlphaZero's chess value).
                              # Self-play only - see ParallelMCTS.search's dirichlet_epsilon
    
    # --- TRAINING HYPERPARAMETERS ---
    batch_size = 256         # Reduced from 512 to fit bigger model in VRAM
    learning_rate = 0.002    # Slightly higher start, we will decay it
    weight_decay = 1e-4      # Regularization to prevent overfitting
    momentum = 0.9
    
    # --- DATA LIMITS ---
    max_positions = 5000000  # Stream more data for the bigger brain

