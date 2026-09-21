import os
import argparse
import torch
import numpy as np
import logging
from datetime import datetime

from env import HangmanEnv
from ppo_agent import HangmanPPOAgent, train_hangman_agent, evaluate_agent, setup_logger
from ppo_agent_gru import GruHangmanPPOAgent  # Import the new GRU agent
from dqn_agent import HangmanDQNAgent, train_hangman_dqn_agent, evaluate_dqn_agent


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Train an agent for Hangman')
    
    # Environment settings
    parser.add_argument('--train_word_list', type=str, default="train_words.txt", help='Path to the training word list file')
    parser.add_argument('--eval_word_list', type=str, default="eval_words.txt", help='Path to the evaluation word list file')
    parser.add_argument('--render_mode', type=str, default=None, choices=[None, 'human', 'ansi'], help='Render mode for environment')
    
    # Algorithm selection
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'ppo_gru', 'dqn'], 
                        help='RL algorithm to use (ppo, ppo_gru, or dqn)')
    
    # Model settings
    parser.add_argument('--embedding_dim', type=int, default=32, help='Dimension of word embeddings')
    parser.add_argument('--transformer_dim', type=int, default=64, help='Dimension of transformer')
    parser.add_argument('--gru_hidden_dim', type=int, default=64, help='Dimension of GRU hidden state')
    parser.add_argument('--num_transformer_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--num_gru_layers', type=int, default=2, help='Number of GRU layers')
    parser.add_argument('--num_heads', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--ff_dim', type=int, default=64, help='Feed-forward dimension')
    parser.add_argument('--mlp_hidden_dim', type=int, default=64, help='MLP hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # Common training settings
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--max_episodes', type=int, default=5000000, help='Maximum number of episodes')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum steps per episode')
    parser.add_argument('--save_freq', type=int, default=1000, help='Frequency of model saving (in episodes)')
    parser.add_argument('--log_freq', type=int, default=1000, help='Frequency of logging (in episodes)')
    parser.add_argument('--eval_freq', type=int, default=1000, help='Frequency of evaluation (in episodes)')
    parser.add_argument('--eval_episodes', type=int, default=100, help='Number of episodes to evaluate on')
    
    # PPO-specific settings
    parser.add_argument('--gae_lambda', type=float, default=0.99, help='GAE lambda parameter (PPO only)')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='PPO clip epsilon (PPO only)')
    parser.add_argument('--value_coef', type=float, default=0.01, help='Value loss coefficient (PPO only)')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy coefficient (PPO only)')
    parser.add_argument('--update_epochs', type=int, default=20, help='Number of PPO update epochs (PPO only)')
    parser.add_argument('--max_grad_norm', type=float, default=2.0, help='Maximum gradient norm (PPO only)')
    parser.add_argument('--update_freq', type=int, default=5000, help='Frequency of policy updates in steps (PPO only)')
    
    # DQN-specific settings
    parser.add_argument('--epsilon_start', type=float, default=1.0, help='Starting epsilon for exploration (DQN only)')
    parser.add_argument('--epsilon_end', type=float, default=0.1, help='Final epsilon for exploration (DQN only)')
    parser.add_argument('--epsilon_decay', type=int, default=50000, help='Epsilon decay steps (DQN only)')
    parser.add_argument('--target_update_freq', type=int, default=1000, help='Target network update frequency (DQN only)')
    parser.add_argument('--buffer_size', type=int, default=10000, help='Replay buffer size (DQN only)')
    parser.add_argument('--dqn_update_freq', type=int, default=4, help='DQN network update frequency (DQN only)')
    
    # Paths
    parser.add_argument('--save_dir', type=str, default='./models', help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save logs')
    parser.add_argument('--load_model', type=str, default=None, help='Path to load a pretrained model')
    
    # Misc
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to train on (cuda/cpu)')
    parser.add_argument('--log_level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Set device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"{args.algorithm}_hangman_run_{timestamp}"
    save_path = os.path.join(args.save_dir, base_dir)
    log_dir = os.path.join(args.log_dir, base_dir)
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logger(log_dir=log_dir, name="hangman_training")
    
    # Save configuration
    config_path = os.path.join(log_dir, "config.txt")
    with open(config_path, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    logger.info(f"Configuration saved to {config_path}")
    
    # Create training environment
    train_env = HangmanEnv(word_list_path=args.train_word_list, render_mode=args.render_mode)
    
    # Create evaluation environment
    eval_env = HangmanEnv(word_list_path=args.eval_word_list, render_mode=None)
    
    # Create agent based on selected algorithm
    if args.algorithm == 'ppo':
        # Create PPO agent with Transformer
        agent = HangmanPPOAgent(
            max_word_length=train_env.max_word_length,
            action_dim=26,  # English alphabet
            embedding_dim=args.embedding_dim,
            transformer_dim=args.transformer_dim,
            num_transformer_layers=args.num_transformer_layers,
            num_heads=args.num_heads,
            ff_dim=args.ff_dim,
            mlp_hidden_dim=args.mlp_hidden_dim,
            dropout=args.dropout,
            lr=args.lr,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_epsilon=args.clip_epsilon,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            update_epochs=args.update_epochs,
            batch_size=args.batch_size,
            device=args.device
        )
        
        # Print training information
        logger.info(f"Starting Hangman PPO training with transformer architecture")
        logger.info(f"Training word list: {args.train_word_list}")
        logger.info(f"Evaluation word list: {args.eval_word_list}")
        logger.info(f"Device: {args.device}")
        logger.info(f"Curriculum phases: {len(train_env.curriculum.phases)}")
        logger.info(f"Max word length: {train_env.max_word_length}")
        logger.info(f"Models will be saved to: {save_path}")
        logger.info(f"Starting training for {args.max_episodes} episodes...")
        
        # Load pretrained model if specified
        if args.load_model and os.path.exists(args.load_model):
            agent.load_model(args.load_model)
            logger.info(f"Loaded pretrained PPO model from {args.load_model}")
        
        # Define evaluation function that uses the evaluation environment
        def eval_callback(agent, episode, logger):
            return evaluate_agent(eval_env, agent, num_episodes=args.eval_episodes, logger=logger)
        
        # Train with PPO
        train_hangman_agent(
            env=train_env,
            agent=agent,
            max_episodes=args.max_episodes,
            max_steps=args.max_steps,
            update_freq=args.update_freq,
            save_freq=args.save_freq,
            log_freq=args.log_freq,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            save_path=save_path,
            device=args.device,
            log_dir=log_dir,
            eval_callback=eval_callback
        )
        
        # Final evaluation
        logger.info("Training completed. Running final evaluation...")
        final_reward, final_win_rate = evaluate_agent(eval_env, agent, num_episodes=100, logger=logger)

    elif args.algorithm == 'ppo_gru':
        # Create simplified PPO agent with GRU
        agent = GruHangmanPPOAgent(
            max_word_length=train_env.max_word_length,
            action_dim=26,  # English alphabet
            embedding_dim=args.embedding_dim,
            gru_layers=args.num_gru_layers,
            gru_hidden_dim=args.gru_hidden_dim,
            mlp_hidden_dim=args.mlp_hidden_dim,
            dropout=args.dropout,
            lr=args.lr,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_epsilon=args.clip_epsilon,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            update_epochs=args.update_epochs,
            batch_size=args.batch_size,
            device=args.device
        )
        
        # Print training information
        logger.info(f"Starting Hangman PPO training with GRU architecture")
        logger.info(f"Training word list: {args.train_word_list}")
        logger.info(f"Evaluation word list: {args.eval_word_list}")
        logger.info(f"Device: {args.device}")
        logger.info(f"Curriculum phases: {len(train_env.curriculum.phases)}")
        logger.info(f"Max word length: {train_env.max_word_length}")
        logger.info(f"Models will be saved to: {save_path}")
        logger.info(f"Starting training for {args.max_episodes} episodes...")
        
        # Load pretrained model if specified
        if args.load_model and os.path.exists(args.load_model):
            agent.load_model(args.load_model)
            logger.info(f"Loaded pretrained PPO GRU model from {args.load_model}")
        
        # Define evaluation function that uses the evaluation environment
        def eval_callback(agent, episode, logger):
            return evaluate_agent(eval_env, agent, num_episodes=args.eval_episodes, logger=logger)
        
        # Train with PPO (using the same training function)
        train_hangman_agent(
            env=train_env,
            agent=agent,
            max_episodes=args.max_episodes,
            max_steps=args.max_steps,
            update_freq=args.update_freq,
            save_freq=args.save_freq,
            log_freq=args.log_freq,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            save_path=save_path,
            device=args.device,
            log_dir=log_dir,
            eval_callback=eval_callback
        )
        
        # Final evaluation
        logger.info("Training completed. Running final evaluation...")
        final_reward, final_win_rate = evaluate_agent(eval_env, agent, num_episodes=100, logger=logger)
        
    elif args.algorithm == 'dqn':
        # Create DQN agent
        agent = HangmanDQNAgent(
            max_word_length=train_env.max_word_length,
            action_dim=26,  # English alphabet
            embedding_dim=args.embedding_dim,
            num_transformer_layers=args.num_transformer_layers,
            num_heads=args.num_heads,
            ff_dim=args.ff_dim,
            mlp_hidden_dim=args.mlp_hidden_dim,
            dropout=args.dropout,
            lr=args.lr,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            target_update_freq=args.target_update_freq,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            device=args.device
        )
        
        # Print training information
        logger.info(f"Starting Hangman DQN training with transformer architecture")
        logger.info(f"Training word list: {args.train_word_list}")
        logger.info(f"Evaluation word list: {args.eval_word_list}")
        logger.info(f"Device: {args.device}")
        logger.info(f"Curriculum phases: {len(train_env.curriculum.phases)}")
        logger.info(f"Max word length: {train_env.max_word_length}")
        logger.info(f"Models will be saved to: {save_path}")
        logger.info(f"Starting training for {args.max_episodes} episodes...")
        
        # Load pretrained model if specified
        if args.load_model and os.path.exists(args.load_model):
            agent.load_model(args.load_model)
            logger.info(f"Loaded pretrained DQN model from {args.load_model}")
        
        # Define evaluation function that uses the evaluation environment
        def eval_callback(agent, episode, logger):
            return evaluate_dqn_agent(eval_env, agent, num_episodes=args.eval_episodes, logger=logger)
        
        # Train with DQN
        train_hangman_dqn_agent(
            env=train_env,
            agent=agent,
            max_episodes=args.max_episodes,
            max_steps=args.max_steps,
            save_freq=args.save_freq,
            log_freq=args.log_freq,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            save_path=save_path,
            device=args.device,
            log_dir=log_dir,
            eval_callback=eval_callback
        )
        
        # Final evaluation
        logger.info("Training completed. Running final evaluation...")
        final_reward, final_win_rate = evaluate_dqn_agent(eval_env, agent, num_episodes=100, logger=logger)
    
    else:
        logger.error(f"Unknown algorithm: {args.algorithm}")
        return
    
    logger.info(f"Final evaluation results | Avg Reward: {final_reward:.2f} | Win Rate: {final_win_rate:.2f}")
    logger.info(f"Training completed. Models saved to {save_path}")


if __name__ == "__main__":
    main()