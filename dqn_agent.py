import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
import time
import logging
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional, Union, Any

# Configure logging
def setup_logger(log_dir=None, name="hangman_training", level=logging.INFO):
    """
    Set up a logger with console and file handlers.
    
    Args:
        log_dir: Directory to save log files. If None, only console logging is used.
        name: Logger name
        level: Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_dir is provided
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        file_handler.setLevel(level)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create a named tuple for storing transitions
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""
    def __init__(self, capacity=100000):
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, next_state, reward, done):
        """Store a transition."""
        self.memory.append(Transition(state, action, next_state, reward, done))
    
    def sample(self, batch_size):
        """Sample a batch of transitions."""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class TransformerBlock(nn.Module):
    """
    Transformer encoder block for processing sequential data.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attention_mask=None, key_padding_mask=None):
        # Pre-normalization architecture
        normalized_x = self.norm1(x)
        
        # Self-attention
        attn_output, _ = self.attn(
            normalized_x, normalized_x, normalized_x, 
            attn_mask=attention_mask,
            key_padding_mask=key_padding_mask
        )
        
        # Residual connection
        x = x + self.dropout(attn_output)
        
        # Pre-normalization for feed-forward
        normalized_x = self.norm2(x)
        
        # Feed-forward with residual connection
        x = x + self.ff(normalized_x)
        
        return x

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    """
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer to make it part of the module's state
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # Add positional encoding to the input embeddings
        return x + self.pe[:, :x.size(1)]

class HangmanDQNNetwork(nn.Module):
    """
    DQN network with Transformer architecture for Hangman.
    
    Uses transformers for processing the revealed word sequence and 
    combines with other inputs through MLPs.
    """
    def __init__(
        self,
        max_word_length: int,
        action_dim: int = 26,
        embedding_dim: int = 32,
        num_transformer_layers: int = 2,
        num_heads: int = 2,
        ff_dim: int = 128,
        mlp_hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super(HangmanDQNNetwork, self).__init__()
        
        self.max_word_length = max_word_length
        
        # Word embedding: -2 (padding), -1 (hidden), 0-25 (a-z) -> 0-27
        self.word_embedding = nn.Embedding(28, embedding_dim, padding_idx=0)
        
        # Positional encoding for the word sequence
        self.positional_encoding = PositionalEncoding(embedding_dim, max_word_length)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, ff_dim, dropout)
            for _ in range(num_transformer_layers)
        ])
        
        # Enable skip connections if more than one transformer layer
        self.use_skip_connections = num_transformer_layers > 1
        
        # Feature embeddings
        self.guessed_letters_embedding = nn.Linear(26, 32)
        self.letter_freq_embedding = nn.Linear(26, 32)
        self.letter_prob_embedding = nn.Linear(26, 32)
        self.last_action_embedding = nn.Linear(26, 32)
        
        # Game state embedding
        self.game_state_embedding = nn.Linear(1, 32)  # Only word length
        
        # Total dimension of all specialized features
        self.other_features_dim = 32 * 4 + 32  # 4 letter features + game state
        
        # Feature fusion network
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.other_features_dim, mlp_hidden_dim),
            nn.LayerNorm(mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.LayerNorm(mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Combined dimension
        self.combined_dim = embedding_dim + mlp_hidden_dim
        
        # Q-value head
        self.q_head = nn.Sequential(
            nn.Linear(self.combined_dim, mlp_hidden_dim * 2),
            nn.LayerNorm(mlp_hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim * 2, mlp_hidden_dim),
            nn.LayerNorm(mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, action_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights for better training stability."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, state, action_mask=None):
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor [batch_size, obs_dim]
            action_mask: Optional mask for actions [batch_size, action_dim]
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        # Handle both batched and unbatched inputs
        batch_size = state.size(0) if len(state.shape) > 1 else 1
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # Add batch dimension
        
        # Split the input state into components
        word_part = state[:, :self.max_word_length]  # Revealed word
        other_parts = state[:, self.max_word_length:]  # Other inputs
        
        # Further split other inputs
        idx = 0
        guessed_letters = other_parts[:, idx:idx+26]
        idx += 26
        word_length = other_parts[:, idx:idx+1]
        idx += 1
        letter_frequencies = other_parts[:, idx:idx+26]
        idx += 26
        last_action = other_parts[:, idx:idx+26]
        idx += 26
        letter_probabilities = other_parts[:, idx:idx+26]
        
        # Game state features (just word length now)
        game_state = word_length
        
        # Process word with transformer
        # Convert revealed word from (-2, -1, 0-25) to (0, 1, 2-27) for embedding
        word_indices = (word_part + 2).long().clamp(0, 27)
        
        # Create padding mask for transformer (1 for padding positions, 0 for valid positions)
        padding_mask = (word_indices == 0)
        
        # Embed the word
        word_embedded = self.word_embedding(word_indices)
        word_embedded = self.positional_encoding(word_embedded)
        
        # Apply transformer layers with skip connections
        transformer_outputs = [word_embedded]
        
        for transformer_layer in self.transformer_layers:
            word_embedded = transformer_layer(
                word_embedded,
                key_padding_mask=padding_mask
            )
            transformer_outputs.append(word_embedded)
        
        # If using skip connections, combine all transformer outputs
        if self.use_skip_connections and len(transformer_outputs) > 2:
            # Skip first element as it's the input
            word_embedded = torch.stack(transformer_outputs[1:], dim=0).mean(dim=0)
        
        # Pool word embedding
        word_encoding = word_embedded.mean(dim=1)
        
        # Process specialized features
        guessed_letters_enc = self.guessed_letters_embedding(guessed_letters)
        letter_freq_enc = self.letter_freq_embedding(letter_frequencies)
        letter_prob_enc = self.letter_prob_embedding(letter_probabilities)
        last_action_enc = self.last_action_embedding(last_action)
        game_state_enc = self.game_state_embedding(game_state)
        
        # Combine all specialized features
        other_features = torch.cat([
            guessed_letters_enc,
            letter_freq_enc,
            letter_prob_enc,
            last_action_enc,
            game_state_enc
        ], dim=1)
        
        # Process combined features
        other_encoding = self.feature_fusion(other_features)
        
        # Combine word and other encodings
        combined = torch.cat([word_encoding, other_encoding], dim=1)
        
        # Generate Q-values
        q_values = self.q_head(combined)
        
        # Apply action mask if provided
        if action_mask is not None:
            # Set invalid actions to large negative values
            q_values = q_values.masked_fill(action_mask == 0, float('-1e9'))
        
        return q_values


class HangmanDQNAgent:
    """
    Deep Q-Network Agent for the Hangman game.
    """
    def __init__(
        self,
        max_word_length: int,
        action_dim: int = 26,
        embedding_dim: int = 32,
        num_transformer_layers: int = 2,
        num_heads: int = 2,
        ff_dim: int = 128,
        mlp_hidden_dim: int = 128,
        dropout: float = 0.1,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: int = 50000,
        target_update_freq: int = 1000,
        batch_size: int = 64,
        buffer_size: int = 100000,
        device: str = 'cpu'
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.device = device
        
        # Initialize Q-networks
        self.policy_net = HangmanDQNNetwork(
            max_word_length=max_word_length,
            action_dim=action_dim,
            embedding_dim=embedding_dim,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout=dropout
        ).to(device)
        
        self.target_net = HangmanDQNNetwork(
            max_word_length=max_word_length,
            action_dim=action_dim,
            embedding_dim=embedding_dim,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout=dropout
        ).to(device)
        
        # Initialize target network with policy network weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Initialize step counter
        self.steps_done = 0
        
        # Metrics tracking
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'win_rate': [],
            'loss': [],
            'current_phase': 0
        }
    
    def select_action(self, state, training=True):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current environment state
            training: Whether the agent is in training mode
            
        Returns:
            int: Selected action
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # Create action mask to prevent guessing already guessed letters
        guessed_letters = state[self.policy_net.max_word_length:self.policy_net.max_word_length+26]
        action_mask = torch.FloatTensor(1 - guessed_letters).to(self.device)
        
        # Calculate current epsilon
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-self.steps_done / self.epsilon_decay)
        
        self.steps_done += 1
        
        if training and random.random() < epsilon:
            # Choose random valid action
            valid_actions = np.where(guessed_letters == 0)[0]
            if len(valid_actions) > 0:
                return random.choice(valid_actions)
            else:
                # Fallback if no valid actions (should not happen in properly designed environment)
                return random.randrange(self.action_dim)
        else:
            # Choose best action from Q-values
            with torch.no_grad():
                # Apply action mask
                q_values = self.policy_net(state_tensor, action_mask.unsqueeze(0))
                return q_values.argmax().item()
    
    def store_transition(self, state, action, next_state, reward, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(
            state,
            action,
            next_state,
            reward,
            done
        )
    
    def update(self):
        """Update the policy network using a batch of experiences."""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0  # Not enough samples yet
        
        # Sample batch from replay buffer
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Convert to tensors and move to device
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(np.array(batch.action)).to(self.device)
        reward_batch = torch.FloatTensor(np.array(batch.reward)).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(np.array(batch.done)).to(self.device)
        
        # Create action masks for Q-value calculations
        action_masks = []
        next_action_masks = []
        
        for i in range(self.batch_size):
            # Extract guessed letters from state
            guessed_letters = state_batch[i, self.policy_net.max_word_length:self.policy_net.max_word_length+26]
            action_mask = 1.0 - guessed_letters
            action_masks.append(action_mask)
            
            # Extract guessed letters from next state
            next_guessed_letters = next_state_batch[i, self.policy_net.max_word_length:self.policy_net.max_word_length+26]
            next_action_mask = 1.0 - next_guessed_letters
            next_action_masks.append(next_action_mask)
        
        action_masks = torch.stack(action_masks).to(self.device)
        next_action_masks = torch.stack(next_action_masks).to(self.device)
        
        # Compute Q(s_t, a) - the current Q-values for the actions taken
        q_values = self.policy_net(state_batch, action_masks)
        state_action_values = q_values.gather(1, action_batch.unsqueeze(1))
        
        # Compute V(s_{t+1}) for all next states using the target network
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch, next_action_masks)
            next_state_values = next_q_values.max(1)[0]
            
            # Correct for terminal states (V(s_t+1) = 0 if s_t+1 is terminal)
            next_state_values = next_state_values * (1 - done_batch)
            
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update the target network with the policy network's weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, path):
        """Save model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics,
            'steps_done': self.steps_done,
            'hyperparams': {
                'max_word_length': self.policy_net.max_word_length,
                'gamma': self.gamma,
                'epsilon_start': self.epsilon_start,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'target_update_freq': self.target_update_freq,
                'batch_size': self.batch_size,
            }
        }, path)
        logging.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.metrics = checkpoint.get('metrics', self.metrics)
        self.steps_done = checkpoint.get('steps_done', 0)
        logging.info(f"Model loaded from {path}")


def train_hangman_dqn_agent(
    env, 
    agent, 
    max_episodes=10000, 
    max_steps=100, 
    update_freq=4, 
    save_freq=500, 
    log_freq=100, 
    eval_freq=500, 
    eval_episodes=50, 
    save_path='./models',
    device='cpu',
    log_dir=None,
    eval_callback=None
):
    """
    Train a DQN agent on the Hangman environment.
    
    Args:
        env: Hangman environment
        agent: DQN agent
        max_episodes: Maximum number of episodes to train for
        max_steps: Maximum steps per episode
        update_freq: Frequency of network updates (in steps)
        save_freq: Frequency of model saving (in episodes)
        log_freq: Frequency of logging (in episodes)
        eval_freq: Frequency of evaluation (in episodes)
        eval_episodes: Number of episodes to evaluate on
        save_path: Directory to save models
        device: Device to train on (cuda/cpu)
        log_dir: Directory to save logs
        eval_callback: Function to call for evaluation (takes agent, episode, logger as args)
    """
    # Set up logger
    logger = setup_logger(log_dir, "hangman_dqn_training")
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    # Initialize counters and statistics
    episode_count = 0
    step_count = 0
    update_count = 0
    total_wins = 0
    total_games = 0
    
    # Start training
    training_start_time = time.time()
    
    while episode_count < max_episodes:
        # Reset environment
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        
        # Episode loop
        while not done and episode_steps < max_steps:
            # Select action
            action = agent.select_action(state)
            
            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, next_state, reward, done)
            
            # Update state and counters
            state = next_state
            episode_reward += reward
            episode_steps += 1
            step_count += 1
            
            # Update policy if enough steps have been collected
            if step_count % update_freq == 0:
                loss = agent.update()
                if loss > 0:  # Only log if update was performed
                    agent.metrics['loss'].append(loss)
                    update_count += 1
                    
                    if update_count % 100 == 0:
                        logger.info(f"Update {update_count}: Loss: {loss:.4f}, Epsilon: {agent.epsilon_end + (agent.epsilon_start - agent.epsilon_end) * np.exp(-agent.steps_done / agent.epsilon_decay):.4f}")
            
            # Update target network periodically
            if step_count % agent.target_update_freq == 0:
                agent.update_target_network()
                logger.info(f"Updated target network at step {step_count}")
        
        # End of episode
        episode_count += 1
        
        # Track win/loss
        if 'phase_advanced' in info:
            if terminated and episode_reward > 0:  # Winning condition
                total_wins += 1
            total_games += 1
        
        # Store episode metrics
        agent.metrics['episode_rewards'].append(episode_reward)
        agent.metrics['episode_lengths'].append(episode_steps)
        agent.metrics['current_phase'] = env.curriculum.current_phase
        
        # Calculate win rate over last 100 episodes
        if total_games > 0:
            win_rate = total_wins / total_games
            agent.metrics['win_rate'].append(win_rate)
        
        # Logging
        if episode_count % log_freq == 0:
            avg_reward = np.mean(agent.metrics['episode_rewards'][-log_freq:])
            avg_length = np.mean(agent.metrics['episode_lengths'][-log_freq:])
            curr_win_rate = np.mean(agent.metrics['win_rate'][-log_freq:]) if agent.metrics['win_rate'] else 0
            avg_loss = np.mean(agent.metrics['loss'][-100:]) if agent.metrics['loss'] else 0
            
            logger.info(f"Episode {episode_count}/{max_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.2f} | "
                  f"Win Rate: {curr_win_rate:.2f} | "
                  f"Avg Loss: {avg_loss:.4f} | "
                  f"Curriculum Phase: {env.curriculum.current_phase + 1}/{len(env.curriculum.phases)}")
        
        # Save model
        if episode_count % save_freq == 0:
            model_path = os.path.join(save_path, f"hangman_dqn_model_episode_{episode_count}.pt")
            agent.save_model(model_path)
        
        # Evaluation
        if episode_count % eval_freq == 0:
            if eval_callback:
                # Use the provided evaluation callback (which uses the eval environment)
                eval_callback(agent, episode_count, logger)
            else:
                # Fallback to evaluating on the training environment
                evaluate_dqn_agent(env, agent, eval_episodes, logger)
    
    # Final save
    final_model_path = os.path.join(save_path, "hangman_dqn_model_final.pt")
    agent.save_model(final_model_path)
    
    training_time = time.time() - training_start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Final model saved to {final_model_path}")
    
    return agent


def evaluate_dqn_agent(env, agent, num_episodes=100, logger=None):
    """
    Evaluate a trained DQN agent.
    
    Args:
        env: Hangman environment
        agent: Trained DQN agent
        num_episodes: Number of episodes to evaluate on
        logger: Logger instance
        
    Returns:
        tuple: (Average reward, Win rate)
    """
    if logger is None:
        logger = logging.getLogger("hangman_dqn_evaluation")
    
    total_rewards = 0
    wins = 0
    
    for i in range(num_episodes):
        episode_reward = 0
        done = False
        
        # Reset environment
        state, _ = env.reset()
        
        while not done:
            # Select action without exploration
            action = agent.select_action(state, training=False)
            
            # Execute action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            # Check for win
            if terminated and episode_reward > 0:  # Winning condition for Hangman
                wins += 1
        
        total_rewards += episode_reward
    
    avg_reward = total_rewards / num_episodes
    win_rate = wins / num_episodes
    
    logger.info(f"Evaluation Results | Avg Reward: {avg_reward:.2f} | Win Rate: {win_rate:.2f}")
    
    return avg_reward, win_rate