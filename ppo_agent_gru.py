import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os
import logging
import time

class MaskedCategorical(Categorical):
    """
    Custom categorical distribution that handles action masking properly.
    """
    def __init__(self, logits, mask=None):
        if mask is not None:
            # Apply mask by setting masked logits to -inf
            logits = logits.masked_fill(mask == 0, float('-inf'))
        super(MaskedCategorical, self).__init__(logits=logits)


class GruHangmanNetwork(nn.Module):
    """
    Simplified Actor-Critic network with GRU for Hangman.
    
    Uses GRU for processing the revealed word sequence and 
    combines with letter features through a simpler architecture.
    """
    def __init__(
        self,
        max_word_length: int,
        action_dim: int = 26,
        embedding_dim: int = 32,
        gru_layers: int = 2,
        gru_hidden_dim: int = 64,
        mlp_hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        super(GruHangmanNetwork, self).__init__()
        
        self.max_word_length = max_word_length
        
        # Simple word embedding
        self.word_embedding = nn.Embedding(28, embedding_dim, padding_idx=0)
        
        # GRU layer instead of Transformer
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_hidden_dim > 1 else 0
        )
        
        # Simple letter features processing
        self.letter_features_embedding = nn.Linear(26 * 3 + 1, mlp_hidden_dim)  # guessed, freq, prob + word_length
        
        # Combined dimension
        combined_dim = gru_hidden_dim + mlp_hidden_dim
        
        # Simpler policy head (Actor)
        self.policy_head = nn.Sequential(
            nn.Linear(combined_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, action_dim)
        )
        
        # Simpler value head (Critic)
        self.value_head = nn.Sequential(
            nn.Linear(combined_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, 1)
        )
        
        # For action masking
        self.action_mask = None
        
        # Initialize weights with standard method
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Standard weight initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def set_action_mask(self, mask):
        """Set action mask to prevent already guessed letters."""
        self.action_mask = mask
    
    def clear_action_mask(self):
        """Clear action mask."""
        self.action_mask = None
    
    def forward(self, state):
        """
        Forward pass through the network with simplified processing.
        
        Args:
            state: Input state tensor [batch_size, obs_dim]
            
        Returns:
            tuple: (Action distribution, Value estimate)
        """
        batch_size = state.size(0)
        
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
        # Skip lase action
        idx += 26
        letter_probabilities = other_parts[:, idx:idx+26]
        
        # Process word with GRU
        # Convert word indices for embedding
        word_indices = (word_part + 2).long().clamp(0, 27)
        
        # Create padding mask (1 for padding positions, 0 for valid positions)
        padding_mask = (word_indices == 0)
        
        # Embed the word
        word_embedded = self.word_embedding(word_indices)
        
        # Use GRU to process the sequence
        # Pack the sequence to handle variable lengths
        lengths = (~padding_mask).sum(dim=1).cpu()
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            word_embedded, 
            lengths, 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # Process with GRU
        gru_output, hidden = self.gru(packed_embedded)

        # Use the final hidden state as the word encoding
        word_encoding = hidden[-1]
        
        # Combine letter features
        letter_features = torch.cat([
            guessed_letters,
            letter_frequencies,
            letter_probabilities,
            word_length,
        ], dim=1)
        
        letter_encoding = F.relu(self.letter_features_embedding(letter_features))
        
        # Combine word and letter encodings
        combined = torch.cat([word_encoding, letter_encoding], dim=1)
        
        # Generate policy logits
        policy_logits = self.policy_head(combined)
        
        # Use action masking with custom distribution
        if self.action_mask is not None:
            action_dist = MaskedCategorical(policy_logits, self.action_mask)
        else:
            action_dist = Categorical(logits=policy_logits)
        
        # Generate value estimate
        value = self.value_head(combined).squeeze(-1)
        
        return action_dist, value


class PPOMemory:
    """Simplified memory buffer for storing trajectories."""
    def __init__(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def store(self, state, action, prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def get_batches(self, batch_size):
        """Generate mini-batches for PPO updates."""
        n_states = len(self.states)
        batch_start_idx = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+batch_size] for i in batch_start_idx]
        
        return batches


class GruHangmanPPOAgent:
    """
    Simplified PPO Agent for Hangman with GRU-based policy.
    """
    def __init__(
        self,
        max_word_length: int,
        action_dim: int = 26,
        embedding_dim: int = 32,
        gru_layers: int = 2,
        gru_hidden_dim: int = 64,
        mlp_hidden_dim: int = 64,
        dropout: float = 0.1,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        batch_size: int = 64,
        device: str = 'cpu'
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.device = device
        
        # Initialize the actor-critic network
        self.policy = GruHangmanNetwork(
            max_word_length=max_word_length,
            action_dim=action_dim,
            embedding_dim=embedding_dim,
            gru_layers=gru_layers,
            gru_hidden_dim=gru_hidden_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout=dropout
        ).to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Initialize memory
        self.memory = PPOMemory()
        
        # Metrics tracking
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'win_rate': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': [],
            'current_phase': 0
        }
    
    def select_action(self, state, training=True):
        """
        Select an action from the policy, with action masking for already guessed letters.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Create action mask to prevent guessing already guessed letters
        guessed_letters = state[self.policy.max_word_length:self.policy.max_word_length+26]
        action_mask = torch.FloatTensor(1 - guessed_letters).unsqueeze(0).to(self.device)
        
        # Set action mask
        self.policy.set_action_mask(action_mask)
        
        # Get action distribution and value
        with torch.no_grad():
            action_dist, value = self.policy(state_tensor)
        
        # Sample action or get the best action
        if training:
            action = action_dist.sample()
        else:
            # During evaluation, pick the most probable action
            probs = action_dist.probs
            action = torch.argmax(probs, dim=1)
        
        # Get log probability of the action
        log_prob = action_dist.log_prob(action)
        
        # Clear action mask
        self.policy.clear_action_mask()
        
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, state, action, prob, reward, value, done):
        """Store transition in memory."""
        self.memory.store(
            torch.FloatTensor(state).to(self.device),
            action,
            prob,
            reward,
            value,
            done
        )
    
    def compute_advantages(self, rewards, values, dones, next_value):
        """Compute advantages using Generalized Advantage Estimation (GAE)."""
        advantages = []
        advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
                
            # Convert boolean done flag to float (0.0 or 1.0)
            if isinstance(dones[t], bool):
                next_non_terminal = 1.0 - float(dones[t])
            else:
                # If it's already a tensor, ensure it's float type before subtraction
                next_non_terminal = 1.0 - dones[t].float()
            
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            
            advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * advantage
            advantages.insert(0, advantage)
        
        return torch.tensor(advantages, dtype=torch.float).to(self.device)
    
    def update(self, next_value=0):
        """Update policy using the collected rollouts."""
        # Convert lists to tensors
        states = torch.stack(self.memory.states).to(self.device)
        actions = torch.tensor(self.memory.actions, dtype=torch.long).to(self.device)
        old_probs = torch.tensor(self.memory.probs, dtype=torch.float).to(self.device)
        rewards = torch.tensor(self.memory.rewards, dtype=torch.float).to(self.device)
        values = torch.tensor(self.memory.values, dtype=torch.float).to(self.device)
        dones = torch.tensor(self.memory.dones, dtype=torch.bool).to(self.device)
        
        # Compute returns and advantages
        advantages = self.compute_advantages(rewards, values, dones, next_value)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update loop
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_loss = 0
        
        for _ in range(self.update_epochs):
            # Generate random indices for batches
            batch_indices = np.arange(len(states))
            np.random.shuffle(batch_indices)
            
            for start in range(0, len(batch_indices), self.batch_size):
                end = start + self.batch_size
                batch_idx = batch_indices[start:end]
                
                # Get batch data
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_probs = old_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                # Create action masks for already guessed letters in each state
                guessed_letters_indices = slice(
                    self.policy.max_word_length,
                    self.policy.max_word_length + 26
                )
                batch_guessed_letters = batch_states[:, guessed_letters_indices]
                batch_action_masks = 1.0 - batch_guessed_letters
                
                # Set batch action mask
                self.policy.set_action_mask(batch_action_masks)
                
                # Forward pass
                action_dist, values_pred = self.policy(batch_states)
                
                # Get new log probs
                new_log_probs = action_dist.log_prob(batch_actions)
                entropy = action_dist.entropy().mean()
                
                # Compute ratio
                ratio = torch.exp(new_log_probs - batch_old_probs)
                
                # Compute surrogate objectives
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                
                # Compute policy loss
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = F.mse_loss(values_pred, batch_returns)
                
                # Compute total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Update policy
                self.optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Clear action mask
                self.policy.clear_action_mask()
                
                # Accumulate losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_loss += loss.item()
        
        # Calculate average losses
        n_batches = (len(states) + self.batch_size - 1) // self.batch_size * self.update_epochs
        avg_policy_loss = total_policy_loss / n_batches
        avg_value_loss = total_value_loss / n_batches
        avg_entropy = total_entropy / n_batches
        avg_total_loss = total_loss / n_batches
        
        # Update metrics
        self.metrics['policy_loss'].append(avg_policy_loss)
        self.metrics['value_loss'].append(avg_value_loss)
        self.metrics['entropy'].append(avg_entropy)
        self.metrics['total_loss'].append(avg_total_loss)
        
        # Clear memory
        self.memory.clear()
        
        return avg_policy_loss, avg_value_loss, avg_entropy, avg_total_loss
    
    def save_model(self, path):
        """Save model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics,
            'hyperparams': {
                'max_word_length': self.policy.max_word_length,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_epsilon': self.clip_epsilon,
                'value_coef': self.value_coef,
                'entropy_coef': self.entropy_coef,
                'max_grad_norm': self.max_grad_norm,
                'update_epochs': self.update_epochs,
                'batch_size': self.batch_size,
            }
        }, path)
        logging.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            logging.warning(f"Failed to load optimizer state: {e}")
        
        self.metrics = checkpoint.get('metrics', self.metrics)
        logging.info(f"Model loaded from {path}")


def setup_logger(log_dir=None, name="hangman_training", level=logging.INFO):
    """Set up a logger with console and file handlers."""
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


def train_hangman_agent(
    env, 
    agent, 
    max_episodes=10000, 
    max_steps=100, 
    update_freq=2048, 
    save_freq=500, 
    log_freq=100, 
    eval_freq=500, 
    eval_episodes=50,
    save_path='./models',
    device='cpu',
    log_dir=None
):
    """Train a PPO agent on the Hangman environment."""
    # Set up logger
    logger = setup_logger(log_dir, "hangman_training")
    
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
            action, log_prob, value = agent.select_action(state)
            
            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, log_prob, reward, value, done)
            
            # Update state and counters
            state = next_state
            episode_reward += reward
            episode_steps += 1
            step_count += 1
            
            # Update policy if enough steps have been collected
            if step_count % update_freq == 0:
                # Get value estimate of the next state if not done
                if done:
                    next_value = 0
                else:
                    _, _, next_value = agent.select_action(state)
                
                # Update policy
                policy_loss, value_loss, entropy, total_loss = agent.update(next_value)
                
                update_count += 1
                logger.info(f"Update {update_count}: Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, "
                      f"Entropy: {entropy:.4f}, Total Loss: {total_loss:.4f}")
        
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
        if hasattr(env, 'curriculum') and hasattr(env.curriculum, 'current_phase'):
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
            
            phase_info = ""
            if hasattr(env, 'curriculum') and hasattr(env.curriculum, 'current_phase'):
                phase_info = f" | Curriculum Phase: {env.curriculum.current_phase + 1}/{len(env.curriculum.phases)}"
                
            logger.info(f"Episode {episode_count}/{max_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.2f} | "
                  f"Win Rate: {curr_win_rate:.2f}{phase_info}")
        
        # Save model
        if episode_count % save_freq == 0:
            model_path = os.path.join(save_path, f"hangman_ppo_model_episode_{episode_count}.pt")
            agent.save_model(model_path)
        
        # Evaluation
        if episode_count % eval_freq == 0:
            evaluate_agent(env, agent, eval_episodes, logger)
    
    # Final save
    final_model_path = os.path.join(save_path, "hangman_ppo_model_final.pt")
    agent.save_model(final_model_path)
    
    training_time = time.time() - training_start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Final model saved to {final_model_path}")
    
    return agent


def evaluate_agent(env, agent, num_episodes=100, logger=None):
    """Evaluate a trained agent."""
    if logger is None:
        logger = logging.getLogger("hangman_evaluation")
    
    total_rewards = 0
    wins = 0
    
    for i in range(num_episodes):
        episode_reward = 0
        done = False
        
        # Reset environment
        state, _ = env.reset()
        
        while not done:
            # Select action without exploration
            action, _, _ = agent.select_action(state, training=False)
            
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