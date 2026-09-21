import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import re
import random
import time
import argparse
from collections import Counter
from tqdm import tqdm

# Import the Hangman environment
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train import MultiLabelCatBoostClassifier, MultiLabelXGBoostClassifier
from env import HangmanEnv

class HangmanMLAgent:
    """
    An enhanced agent that uses a pre-trained multilabel model combined with
    pattern-based analysis to play Hangman more effectively.
    """
    def __init__(self, model_path, dictionary_path):
        """
        Initialize the agent with the pre-trained model and dictionary.
        
        Args:
            model_path: Path to the pre-trained model file (.pkl)
            dictionary_path: Path to the word list file
            exploration_rate: Probability of taking a random action for exploration
        """
        self.model = self.load_model(model_path)
        self.words = self.build_dictionary(dictionary_path)
        
        # Setup letter mappings
        self.alpha = "abcdefghijklmnopqrstuvwxyz"
        self.letter_to_idx = {letter: i for i, letter in enumerate(self.alpha)}
        self.idx_to_letter = {i: letter for i, letter in enumerate(self.alpha)}
        
        # Value mappings for prediction
        self.value = {"a":1,"b":2,"c":3,"d":4,"e":5,"f":6,"g":7,
                      "h":8,"i":9,"j":10,"k":11,"l":12,"m":13,"n":14,
                      "o":15,"p":16,"q":17,"r":18,"s":19,"t":20,"u":21,
                      "v":22,"w":23,"x":24,"y":25,"z":26,"_":0}
        self.value_rev = {v: k for k, v in self.value.items()}
        
        print(f"Loaded model from {model_path}")
        print(f"Loaded {len(self.words)} words from dictionary")
    
    def load_model(self, model_path):
        """Load the pre-trained model from a pickle file."""
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def build_dictionary(self, dictionary_file_location):
        """Load the word list from a file."""
        text_file = open(dictionary_file_location, "r")
        full_dictionary = text_file.read().splitlines()
        text_file.close()
        return full_dictionary
    
    def prediction(self, word):
        """
        Get letter predictions using the trained model.
        
        Args:
            tc: Current state of the word with _ for unknown letters
            
        Returns:
            tuple: (ordered letters by probability, their probabilities)
        """
        # Prepare input for model
        input = np.full(80, -1)
        for i in range(len(word)):
            input[i] = self.value.get(word[i], 0)
            input[80-len(word)+i] = self.value.get(word[i], 0)

        # Get model predictions
        predictions = self.model.predict(np.array([input]))[0]

        # Sort letters by probability
        perm = ""
        prob = []
        for i in range(26):
            perm += self.alpha[np.argmax(predictions)]
            prob.append(predictions[np.argmax(predictions)])
            predictions[np.argmax(predictions)] = -1
            
        return perm, prob
    
    def create_substrings(self, word, guessed=[], n=6, threshold=0.1, multiple=False):
        """
        Create substrings for pattern analysis to find potential letters.
        
        Args:
            word: Current state of the word
            guessed: Already guessed letters
            n: Size of substring window
            threshold: Minimum probability threshold
            multiple: Whether to consider patterns with multiple unknowns
            
        Returns:
            tuple: (best letter, its probability) or (None, None) if no good match
        """
        substring = []
        for i in range(len(word)):
            if word[i] == '_' and i >= 0 and i <= len(word) - 1:
                if multiple:
                    # Replace all '_' with '.'
                    substring.append((word[max(i-n+1,0):i] + '.' + word[i+1:min(i+n,len(word))]).replace("_","*"))
                
                if not multiple and (i==0 or word[i-1] != '_') and (i==len(word)-1 or word[i+1] != '_'): 
                    substring.append(word[max(i-n+1,0):i] + '.' + word[i+1:min(i+n,len(word))])
        
        # Process substrings to ensure they're the right length and format
        substrings = []
        for word in substring:
            for i in range(0, len(word) - n + 1):
                substrings.append(word[i:i+n])
        
        # Vaild substring
        if multiple:
            substrings = [x for x in substrings if x.count('*') < 2]
        else:
            substrings = [x for x in substrings if x.count('_') == 0]
        
        substrings = [x for x in substrings if len(x) == n]

        if not substrings:
            return None, None
        
        ans = []
        
        # Find potential letters from dictionary words matching patterns
        for x in substrings:
            ind = x.index('.')
            letters = []

            if multiple:
                x = x.replace("*", ".")
            
            pattern = re.compile(x)
            for word in self.words:
                match = pattern.search(word)
                if match:
                    l = word[match.start() + ind]
                    if l not in guessed:
                        letters.append(l)

            if not letters:
                continue
            
            # Calculate letter probabilities
            letter_probs = [[letter[0], letter[1]/len(letters), letter[1]] for letter in Counter(letters).most_common()]
            ans.append(letter_probs)
        
        if not ans:
            return None, None
        
        # Combine probabilities across all patterns
        probs = np.zeros(26)
        for x in ans:
            for y in x:
                probs[self.value[y[0]] - 1] += y[1]

        # Normalize
        if probs.sum() > 0:
            probs = probs / probs.sum()
        
        # Check if best probability meets threshold
        if probs[np.argmax(probs)] < threshold:
            return None, None
            
        return self.value_rev[np.argmax(probs) + 1], probs[np.argmax(probs)]
    
    def choose_action(self, observation, remaining_attempts):
        """
        Choose the next letter to guess based on current observation.
        
        Args:
            observation: Current observation from environment
            env: Hangman environment instance
            
        Returns:
            int: Action to take (letter index)
        """
        # Extract relevant information from observation
        revealed_word = observation[:20]
        guessed_letters = observation[20:46]

        # Compute revealed words length
        padding_mask = (revealed_word == -2)
        length = (~padding_mask).sum().item()
        revealed_word_mask = (revealed_word != -1) & (~padding_mask)
        revealed_word_length = revealed_word_mask.sum().item()
        
        # Convert to word format with _ for unknown letters
        word_state = ""
        for i in range(length):  # Use target word length
            if i < len(revealed_word) and revealed_word[i] >= 0:
                idx = int(revealed_word[i])
                word_state += self.idx_to_letter[idx]
            else:
                word_state += "_"
        
        # Get guessed letters list
        guessed = []
        for i, is_guessed in enumerate(guessed_letters):
            if is_guessed > 0:
                guessed.append(self.idx_to_letter[i])

        # Dynamic threshold for pattern analysis
        base_threshold = 0.1
        remaining_pct = remaining_attempts / 6
        dynamic_threshold = base_threshold * (0.5 + 0.5 * remaining_pct)  

        # Prediction based on pattern analysis
        pattern_candidates = {}
    
        # Adaptive window size based on word length
        word_length = len(word_state)
        # window_sizes = self.get_adaptive_windows(word_state, guessed, remaining_attempts)
        window_sizes = [3, 4, 5, 6, 7, 8, 9]
        
        # Get pattern predictions
        for window_size in window_sizes:
            if window_size < len(word_state):
                letter, prob = self.create_substrings(word_state, guessed, window_size, threshold=dynamic_threshold, multiple=True)
                if letter is not None:
                    pattern_candidates[letter] = max(pattern_candidates.get(letter, 0), prob)
        
        # Specific case: use small window without multiple unknowns
        if len(pattern_candidates) == 0 or max(pattern_candidates.values(), default=0) < dynamic_threshold:
            letter, prob = self.create_substrings(word_state, guessed, 3, threshold=dynamic_threshold*0.8)
            if letter is not None:
                pattern_candidates[letter] = max(pattern_candidates.get(letter, 0), prob)
    
        # Get model's letter prediction
        perm, probs = self.prediction(word_state)
        
        # Find first unguessed letter from model predictions
        model_candidates = {}
        for i in range(len(perm)):
            if perm[i] not in guessed:
                model_candidates[perm[i]] = probs[i]
        
        # Combined predictions based on pattern and model
        combined_candidates = {}
        
        # Give higher weight to pattern predictions
        for letter, prob in pattern_candidates.items():
            if revealed_word_length <= 2 and word_length <= 6:
                combined_candidates[letter] = prob * 1.0
            else:
                combined_candidates[letter] = prob * 1.0
        
        # Add model predictions
        for letter, prob in model_candidates.items():
            if letter in combined_candidates:
                combined_candidates[letter] = max(combined_candidates[letter], prob)
            else:
                combined_candidates[letter] = prob
        
        # Choose best candidate
        if combined_candidates:
            best_letter = max(combined_candidates.items(), key=lambda x: x[1])[0]
            return self.letter_to_idx[best_letter]
        
        # Fallback to random unguessed letter
        unguessed = [letter for letter in self.alpha if letter not in guessed]
        if unguessed:
            return self.letter_to_idx[random.choice(unguessed)]
        else:
            return random.randint(0, 25)  # Should rarely happen

    def get_adaptive_windows(self, word_state, guessed_letters, remaining_attempts):

        # Define letter frequency and position sensitivity
        letter_properties = {
            # High frequency letters (common)
            'high_freq': 'etaoinshr',
            # Medium frequency letters
            'mid_freq': 'dlcumwfgy', 
            # Low frequency letters (rare)
            'low_freq': 'pbjvkqxz',
            # Position sensitive letters
            'start_letters': 'stcpabwm',  # Common at word start
            'end_letters': 'syedtnrg'     # Common at word end
        }
        word_length = len(word_state)
        
        if word_length <= 5:
            base_windows = [2, 3]
        elif word_length <= 8:
            base_windows = [3, 4, 5, 6, 7]
        else:
            base_windows = [3, 4, 5, 6, 7, 8, 9]
        
        # Known and unknown positions
        known_positions = [i for i, c in enumerate(word_state) if c != '_']
        unknown_positions = [i for i, c in enumerate(word_state) if c == '_']
        
        # If known positions are more than 60%, use smaller windows
        if len(known_positions) > word_length * 0.6:
            smaller_windows = [w for w in range(2, min(base_windows) + 1)]
            base_windows = smaller_windows + base_windows
        
        # Adjust windows based on known positions
        has_start_known = 0 in known_positions
        has_end_known = (word_length - 1) in known_positions
        
        # Unguessed letters
        unguessed = [l for l in 'abcdefghijklmnopqrstuvwxyz' if l not in guessed_letters]
        
        # Check for position sensitive letters
        has_start_letters = any(l in letter_properties['start_letters'] for l in unguessed)
        has_end_letters = any(l in letter_properties['end_letters'] for l in unguessed)
        
        # Adjust windows
        final_windows = base_windows.copy()
        
        # If start unknown and has start sensitive letters, add small window
        if not has_start_known and has_start_letters and 0 in unknown_positions:
            if 2 not in final_windows:
                final_windows.insert(0, 2)
        
        # If end unknown and has end sensitive letters, add small window
        if not has_end_known and has_end_letters and (word_length - 1) in unknown_positions:
            if 2 not in final_windows:
                final_windows.insert(0, 2)
        
        # Ensure windows are not larger than word length
        final_windows = [w for w in final_windows if w < word_length]
        
        # If no windows left, add a small window
        if not final_windows:
            final_windows = [min(2, word_length - 1)]
        
        return final_windows


def evaluate_agent(agent, env, num_episodes=1000, render_every=100):
    """
    Evaluate the agent's performance over multiple episodes.
    
    Args:
        agent: The agent to evaluate
        env: The Hangman environment
        num_episodes: Number of episodes to run
        render_every: How often to render an episode
        
    Returns:
        dict: Statistics about the agent's performance
    """
    wins = 0
    total_rewards = []
    episode_lengths = []
    remaining_attempts_when_winning = []
    curriculum_phase_history = [0]  # Start with phase 0
    word_lengths = []
    win_by_length = {}
    
    print(f"Evaluating agent over {num_episodes} episodes...")
    
    for episode in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        done = False
        cumulative_reward = 0
        steps = 0
        
        # Record word length
        word_legth = len(env.target_word)
        word_lengths.append(word_legth)
        
        # Track if episode should be rendered
        should_render = (episode % render_every == 0)

        if word_legth not in win_by_length:
            win_by_length[word_legth] = {"wins": 0, "total": 0}
        win_by_length[word_legth]["total"] += 1
        
        while not done:
            action = agent.choose_action(obs, env.remaining_attempts)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Optionally render
            if should_render:
                env.render()
                time.sleep(0.5)  # Pause to make rendering visible
            
            cumulative_reward += reward
            steps += 1
            done = terminated or truncated
            obs = next_obs
            
            # Check if phase advanced
            if done and info.get("phase_advanced", False):
                curriculum_phase_history.append(env.curriculum.current_phase)
        
        # Record episode stats
        total_rewards.append(cumulative_reward)
        episode_lengths.append(steps)
        
        # Record win/loss
        if terminated and env.remaining_attempts > 0:  # Win condition
            wins += 1
            remaining_attempts_when_winning.append(env.remaining_attempts)
            win_by_length[word_legth]["wins"] += 1
    
    # Calculate statistics
    win_rate = wins / num_episodes
    avg_reward = np.mean(total_rewards)
    avg_episode_length = np.mean(episode_lengths)
    avg_remaining_attempts = np.mean(remaining_attempts_when_winning) if remaining_attempts_when_winning else 0
    avg_word_length = np.mean(word_lengths)
    
    stats = {
        "win_rate": win_rate,
        "avg_reward": avg_reward,
        "avg_episode_length": avg_episode_length,
        "avg_remaining_attempts": avg_remaining_attempts,
        "final_curriculum_phase": env.curriculum.current_phase,
        "avg_word_length": avg_word_length
    }
    
    print(f"Evaluation complete:")
    print(f"  Win rate: {win_rate:.2f}")
    print(f"  Average reward: {avg_reward:.2f}")
    print(f"  Average episode length: {avg_episode_length:.2f}")
    print(f"  Average word length: {avg_word_length:.2f}")
    print(f"  Average remaining attempts when winning: {avg_remaining_attempts:.2f}")
    print(f"  Final curriculum phase: {env.curriculum.current_phase + 1}/{len(env.curriculum.phases)}")

    # Print win rate by word length
    print("\nWin rate by word length:")
    for length, stats in sorted(win_by_length.items()):
        if stats["total"] > 0:
            length_win_rate = stats["wins"] / stats["total"]
            print(f"  Length {length}: {stats['wins']}/{stats['total']} ({length_win_rate:.2f})")
    
    # Plot performance
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 3, 1)
    plt.plot(total_rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Plot curriculum phase
    plt.subplot(2, 3, 2)
    plt.step(range(len(curriculum_phase_history)), curriculum_phase_history)
    plt.title('Curriculum Phase Progression')
    plt.xlabel('Advancement Event')
    plt.ylabel('Phase')
    
    # Plot episode lengths
    plt.subplot(2, 3, 3)
    plt.plot(episode_lengths)
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    # Plot win rate over time (moving average)
    plt.subplot(2, 3, 4)
    window_size = min(100, num_episodes // 10)
    wins_binary = [1 if r > 0 and l <= 6 else 0 for r, l in zip(total_rewards, episode_lengths)]
    win_rate_moving_avg = np.convolve(wins_binary, np.ones(window_size)/window_size, mode='valid')
    plt.plot(win_rate_moving_avg)
    plt.title(f'Win Rate (Moving Avg, Window={window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    
    # Plot word length distribution
    plt.subplot(2, 3, 5)
    plt.hist(word_lengths, bins=range(3, 21), alpha=0.7)
    plt.title('Word Length Distribution')
    plt.xlabel('Word Length')
    plt.ylabel('Count')
    
    # Plot remaining attempts when winning
    plt.subplot(2, 3, 6)
    if remaining_attempts_when_winning:
        plt.hist(remaining_attempts_when_winning, bins=range(7), alpha=0.7)
        plt.title('Remaining Attempts When Winning')
        plt.xlabel('Attempts Left')
        plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('enhanced_agent_performance.png')
    plt.close()
    
    return stats


def play_interactive(agent, env):
    """
    Let the agent play the game, with human-readable output.
    
    Args:
        agent: The agent to use
        env: The Hangman environment
        
    Returns:
        tuple: (total_reward, steps)
    """
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    print("=== New Hangman Game ===")
    print(f"Target word length: {len(env.target_word)}")
    print(f"Curriculum phase: {env.curriculum.current_phase + 1}/{len(env.curriculum.phases)}")
    
    while not done:
        # Render current state
        env.render()
        remaining_attempts = env.remaining_attempts
        
        # Get action from agent
        action = agent.choose_action(obs, remaining_attempts)
        letter = agent.idx_to_letter[action]
        
        print(f"\nAgent guesses: {letter}")
        
        # Take step in environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Display result
        if letter in env.target_word:
            print(f"Correct! '{letter}' is in the word. Reward: {reward:.2f}")
        else:
            print(f"Wrong! '{letter}' is not in the word. Reward: {reward:.2f}")
        
        total_reward += reward
        steps += 1
        done = terminated or truncated
        obs = next_obs
        
        # Slow down for readability
        time.sleep(1)
    
    # Show final state
    env.render()
    
    if terminated and env.remaining_attempts > 0:
        print("\nGame won!")
    else:
        print(f"\nGame lost! The word was: {env.target_word}")
    
    print(f"Total reward: {total_reward:.2f}")
    print(f"Total steps: {steps}")
    
    return total_reward, steps


def main():
    parser = argparse.ArgumentParser(description="Enhanced Hangman Agent")
    parser.add_argument('--model_path', type=str, default='models_trained_2/final_model_0.8458.pkl',help='Path to the trained model file')
    parser.add_argument('--word_list', type=str, default='words_250000_train.txt',help='Path to the word list file')
    parser.add_argument('--mode', type=str, choices=['evaluate', 'play'], default='evaluate',help='Mode: evaluate (multiple episodes) or play (single interactive game)')
    parser.add_argument('--episodes', type=int, default=100,help='Number of episodes for evaluation mode')
    
    args = parser.parse_args()
    
    # Initialize environment
    env = HangmanEnv(word_list_path='words_250000_train.txt', render_mode="human")
    
    # Initialize agent
    agent = HangmanMLAgent(
        model_path=args.model_path, 
        dictionary_path=args.word_list
    )
    
    # Run in selected mode
    if args.mode == 'evaluate':
        evaluate_agent(agent, env, num_episodes=args.episodes)
    else:  # play mode
        num_games = 5
        print(f"Playing {num_games} games...")
        
        total_rewards = []
        for game in range(num_games):
            print(f"\n=== Game {game+1}/{num_games} ===")
            reward, steps = play_interactive(agent, env)
            total_rewards.append(reward)
            
            if game < num_games - 1:
                input("\nPress Enter for next game...")
        
        print(f"\nAverage reward over {num_games} games: {np.mean(total_rewards):.2f}")


if __name__ == "__main__":
    main()