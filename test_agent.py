import numpy
import argparse
import torch.serialization
import os
from tqdm import tqdm

# Add safe globals for numpy serialization
torch.serialization.add_safe_globals([
    numpy.core.multiarray.scalar,
    numpy.dtype,
    numpy.dtypes.Float64DType,
])

from ppo_agent import HangmanPPOAgent
from ppo_agent_gru import GruHangmanPPOAgent
from env import HangmanEnv

def load_agent_params_from_file(filepath):
    params = {}
    try:
        with open(filepath, 'r') as file:
            for line in file:
                if line.strip() == '' or line.strip().startswith('#'):
                    continue
                    
                # Parser key : value
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    try:
                        if value.isdigit():
                            params[key] = int(value)
                        elif value.replace('.', '', 1).isdigit():
                            params[key] = float(value)
                        else:
                            params[key] = value
                    except ValueError:
                        params[key] = value

    except Exception as e:
        print(f"Error loading agent parameters: {e}")
        return {}
    
    return params


def play_agent(env, agent, num_episodes=100, verbose=False, progress_bar=True):
    """
    Play the game with the agent and track wins.
    
    Args:
        env: The Hangman environment
        agent: The RL agent
        num_episodes: Number of episodes to play
        verbose: Whether to print detailed game progress
        progress_bar: Whether to show a progress bar
    
    Returns:
        tuple: (total_wins, win_rate, total_rewards, avg_reward)
    """
    total_wins = 0
    total_rewards = 0
    word_lengths = []
    win_by_length = {}
    
    # Setup progress bar if requested
    episode_range = tqdm(range(num_episodes), desc="Playing games") if progress_bar else range(num_episodes)
    
    for episode in episode_range:
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        # Track word length
        word_length = len(env.target_word)
        word_lengths.append(word_length)
        if word_length not in win_by_length:
            win_by_length[word_length] = {"wins": 0, "total": 0}
        win_by_length[word_length]["total"] += 1
        
        if verbose:
            print(f"\n===== Episode {episode + 1} =====")
            print(f"Word length: {word_length}")
            print(env.render())
        
        while not done:
            # Select action
            action, _, _ = agent.select_action(obs, training=False)
            
            # Skip if the letter has already been guessed
            if obs[env.max_word_length + action] > 0:
                # If in verbose mode, show skipped action
                if verbose:
                    print(f"Skipping already guessed letter: {env.idx_to_letter[action]}")
                
                # Try to find a non-guessed letter
                guessed_letters = obs[env.max_word_length:env.max_word_length+26]
                unguessed_indices = [i for i, guessed in enumerate(guessed_letters) if guessed == 0]
                
                if unguessed_indices:
                    # Take the first unguessed letter as fallback
                    action = unguessed_indices[0]
                else:
                    # No more letters to guess, just continue with the original action
                    pass
            
            # Take the action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            if verbose:
                print(f"\nAction: {env.idx_to_letter[action]}")
                print(f"Reward: {reward:.2f}")
                print(env.render())
        
        # Check if won (remaining attempts > 0 at termination indicates win)
        if terminated and env.remaining_attempts > 0:
            total_wins += 1
            win_by_length[word_length]["wins"] += 1
        
        total_rewards += total_reward
        
        if verbose:
            print(f"Episode {episode + 1} finished with total reward: {total_reward:.2f}")
            if "phase_advanced" in info and info["phase_advanced"]:
                print("Curriculum phase advanced!")
            print(f"Word was: {env.target_word}")
    
    # Calculate statistics
    win_rate = total_wins / num_episodes
    avg_reward = total_rewards / num_episodes
    
    # Print summary
    print(f"\n===== Results after {num_episodes} episodes =====")
    print(f"Wins: {total_wins}/{num_episodes} (Win Rate: {win_rate:.2f})")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Current Curriculum Phase: {env.curriculum.current_phase + 1}/{len(env.curriculum.phases)}")
    
    # Print win rate by word length
    print("\nWin rate by word length:")
    for length, stats in sorted(win_by_length.items()):
        if stats["total"] > 0:
            length_win_rate = stats["wins"] / stats["total"]
            print(f"  Length {length}: {stats['wins']}/{stats['total']} ({length_win_rate:.2f})")
    
    return total_wins, win_rate, total_rewards, avg_reward


def get_available_models(models_dir="./models"):
    """Get a list of available model directories."""
    if not os.path.exists(models_dir):
        return []
    
    # List directories in the models folder
    model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    
    return model_dirs


def parse_arguments():
    """Parse command line arguments for the script."""
    parser = argparse.ArgumentParser(description='Test Hangman RL Agent')
    
    # Model selection arguments
    parser.add_argument('--model_type', type=str, choices=['ppo', 'ppo_gru'], default='ppo_gru',help='Type of model to use (ppo or ppo_gru)')
    parser.add_argument('--model_run', type=str, default='ppo_gru_hangman_run_20250427_004317',help='Specific model run to use (e.g., ppo_gru_hangman_run_20250427_004317)')
    parser.add_argument('--checkpoint', type=str, default='hangman_ppo_model_episode_1578000.pt',help='Specific checkpoint to load (default: latest)')
    parser.add_argument('--config', type=str, default=None,help='Path to config file (default: use the run\'s config.txt)')
    
    # Environment arguments
    parser.add_argument('--word_list', type=str, default="eval_words.txt",help='Path to the word list file')
    parser.add_argument('--render_mode', type=str, choices=['human', 'ansi', None], default="human",help='Render mode for the environment')
    
    # Testing arguments
    parser.add_argument('--episodes', type=int, default=1000,help='Number of episodes to run')
    parser.add_argument('--verbose', action='store_true',help='Print detailed information during testing')
    parser.add_argument('--progress', action='store_true',help='Show progress bar during testing')
    parser.add_argument('--seed', type=int, default=None,help='Random seed for reproducibility')
    
    # Device selection
    parser.add_argument('--device', type=str, default=None,help='Device to run on (cpu, cuda, or mps). Overrides config file.')
    
    return parser.parse_args()


def main():
    """Main function to run the testing script."""
    args = parse_arguments()
    
    # Set random seed if specified
    if args.seed is not None:
        numpy.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # If model_run not specified, let user select interactively
    model_run = args.model_run

    # Determine paths
    model_dir = os.path.join("./models", model_run)
    config_path = args.config if args.config else os.path.join("./logs", model_run, "config.txt")
    
    # Load agent parameters
    agent_params = load_agent_params_from_file(config_path)
    if not agent_params:
        print(f"Failed to load agent parameters from {config_path}. Exiting.")
        return
    
    # Override device if specified
    if args.device:
        agent_params['device'] = args.device
    
    # Initialize the agent based on model type
    if args.model_type == 'ppo_gru':
        agent = GruHangmanPPOAgent(
            max_word_length=20,
            embedding_dim=agent_params.get('embedding_dim'),
            gru_layers=agent_params.get('num_gru_layers'),
            gru_hidden_dim=agent_params.get('gru_hidden_dim'),
            mlp_hidden_dim=agent_params.get('mlp_hidden_dim'),
            dropout=agent_params.get('dropout'),
            gamma=agent_params.get('gamma'),
            gae_lambda=agent_params.get('gae_lambda'),
            clip_epsilon=agent_params.get('clip_epsilon'),
            value_coef=agent_params.get('value_coef'),
            entropy_coef=agent_params.get('entropy_coef'),
            max_grad_norm=agent_params.get('max_grad_norm'),
            update_epochs=agent_params.get('update_epochs'),
            batch_size=agent_params.get('batch_size'),
            device=agent_params.get('device')
        )

    else:  # ppo
        agent = HangmanPPOAgent(
            max_word_length=20,
            embedding_dim=agent_params.get('embedding_dim'),
            transformer_dim=agent_params.get('transformer_dim'),
            num_transformer_layers=agent_params.get('num_transformer_layers'),
            num_heads=agent_params.get('num_heads'),
            ff_dim=agent_params.get('ff_dim'),
            mlp_hidden_dim=agent_params.get('mlp_hidden_dim'),
            dropout=agent_params.get('dropout'),
            gamma=agent_params.get('gamma'),
            gae_lambda=agent_params.get('gae_lambda'),
            clip_epsilon=agent_params.get('clip_epsilon'),
            value_coef=agent_params.get('value_coef'),
            entropy_coef=agent_params.get('entropy_coef'),
            max_grad_norm=agent_params.get('max_grad_norm'),
            update_epochs=agent_params.get('update_epochs'),
            batch_size=agent_params.get('batch_size'),
            device=agent_params.get('device')
        )
    
    # If checkpoint not specified, let user select interactively
    checkpoint = args.checkpoint
    
    # Load the model
    checkpoint_path = os.path.join(model_dir, checkpoint)
    print(f"Loading model from {checkpoint_path}")
    agent.load_model(checkpoint_path)
    
    # Create the environment
    env = HangmanEnv(word_list_path=args.word_list, render_mode=args.render_mode)
    
    # Play the game and track performance
    print(f"Testing agent over {args.episodes} episodes...")
    total_wins, win_rate, total_rewards, avg_reward = play_agent(
        env, 
        agent, 
        num_episodes=args.episodes, 
        verbose=args.verbose,
        progress_bar=args.progress
    )
    
    # Return success
    return True


if __name__ == "__main__":
    main()