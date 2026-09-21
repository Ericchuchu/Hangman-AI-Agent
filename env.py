import gymnasium as gym
import numpy as np
import re
import random
from gymnasium import spaces
from typing import Optional, List, Dict, Any, Tuple


class Curriculum:
    """
    Manages curriculum learning for the Hangman environment.
    Progressively increases difficulty based on agent performance.
    """
    def __init__(self, phases=None):
        """
        Initialize the curriculum with predefined phases.
        
        Args:
            phases: List of dictionaries containing phase configurations.
                   Each phase defines word length range and max attempts.
        """
        # Default curriculum phases if none provided
        if phases is None:
            self.phases = [
                {"min_word_length": 3, "max_word_length": 7, "max_attempts": 6, "win_threshold": 0.6},
                {"min_word_length": 7, "max_word_length": 9, "max_attempts": 6, "win_threshold": 0.6},
                {"min_word_length": 9, "max_word_length": 11, "max_attempts": 6, "win_threshold": 0.5},
                {"min_word_length": 11, "max_word_length": 15, "max_attempts": 6, "win_threshold": 0.5},
                {"min_word_length": 14, "max_word_length": 20, "max_attempts": 6, "win_threshold": 0.45},
            ]
        else:
            self.phases = phases

        self.current_phase = 0
        self.phase_stats = {"games": 0, "wins": 0}
    
    def get_config(self):
        """Get the configuration for the current phase."""
        return self.phases[self.current_phase]
    
    def record_game_result(self, win: bool):
        """
        Record the result of a game and check if criteria are met to advance phase.
        
        Args:
            win: Whether the agent won the game.
        
        Returns:
            bool: Whether the phase was advanced.
        """
        self.phase_stats["games"] += 1
        if win:
            self.phase_stats["wins"] += 1
        
        # Check if we should advance to the next phase
        if (self.phase_stats["games"] >= 200 and 
            self.phase_stats["wins"] / self.phase_stats["games"] >= self.phases[self.current_phase]["win_threshold"]):
            return self.advance_phase()
        return False
    
    def advance_phase(self):
        """
        Advance to the next curriculum phase if available.
        
        Returns:
            bool: Whether the phase was advanced.
        """
        if self.current_phase < len(self.phases) - 1:
            self.current_phase += 1
            # Reset statistics for the new phase
            self.phase_stats = {"games": 0, "wins": 0}
            return True
        return False


class HangmanEnv(gym.Env):
    """
    A Gymnasium environment for the Hangman game.
    Optimized for reinforcement learning with curriculum learning and efficient state representation.
    """
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(self, word_list_path: str, render_mode=None):
        """
        Initialize the Hangman environment.
        
        Args:
            word_list_path: Path to a text file containing words, one word per line.
            render_mode: The render mode to use.
        """
        super(HangmanEnv, self).__init__()
        
        # Load the word list
        with open(word_list_path, 'r') as f:
            self.word_list = [word.strip().lower() for word in f.readlines() if word.strip()]
        
        # Cache words by length for faster access
        self.words_by_length = self._cache_words_by_length()
        
        # Generate letter frequency statistics from the word list (precomputation)
        self.letter_stats = self._compute_letter_statistics()
        
        # Initialize curriculum
        self.curriculum = Curriculum([{"min_word_length": 3, "max_word_length": 6, "max_attempts": 6, "win_threshold": 1.0}])
        
        # Action space: 26 letters (a-z)
        self.action_space = spaces.Discrete(26)
        
        # Calculate observation space dimension
        self.max_word_length = max(phase["max_word_length"] for phase in self.curriculum.phases)
        self.max_word_length = 20
        
        # Observation space components:
        # 1. Revealed word: max_word_length (each position is -2 for hidden, 0-25 for revealed)
        # 2. Guessed letters: 26 (one-hot encoding of guessed letters)
        # 3. Word length: 1 (normalized)
        # 4. Letter frequencies: 26 (global letter frequencies)
        # 5. Last action: 26 (one-hot encoding of last action)
        # 6. Letter probabilities: 26 (probabilities for each letter)
        obs_dim = (
            self.max_word_length +  # Revealed word
            26 +                    # Guessed letters
            1 +                     # Word length
            26 +                    # Letter frequencies
            26 +                    # Last action
            26                      # Letter probabilities
        )
        
        # Observation space: Box with values in [0, 1]
        self.observation_space = spaces.Box(
            low=-2.0, high=25.0, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize alphabet and letter mappings
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"
        self.letter_to_idx = {letter: i for i, letter in enumerate(self.alphabet)}
        self.idx_to_letter = {i: letter for i, letter in enumerate(self.alphabet)}
        
        # Set render mode
        self.render_mode = render_mode
        
        # Initialize state variables (will be set in reset)
        self.target_word = None
        self.revealed_word = None
        self.guessed_letters = None
        self.max_attempts = None
        self.remaining_attempts = None
        self.last_action = None
        self.letter_probabilities = None
        
        # Cache for regex patterns
        self._regex_cache = {}
        
        # Reset the environment
        self.reset()
    
    def _cache_words_by_length(self):
        """
        Cache words by their length for faster filtering during reset.
        
        Returns:
            dict: Dictionary mapping word lengths to lists of words.
        """
        words_by_length = {}
        for word in self.word_list:
            length = len(word)
            if length not in words_by_length:
                words_by_length[length] = []
            words_by_length[length].append(word)
        return words_by_length
    
    def _compute_letter_statistics(self):
        """
        Compute letter frequency statistics from the word list.
        
        Returns:
            dict: Dictionary containing letter frequency information.
        """
        # Count total occurrences of each letter
        letter_counts = {letter: 0 for letter in 'abcdefghijklmnopqrstuvwxyz'}
        total_letters = 0
        
        # Count words containing each letter
        words_containing = {letter: 0 for letter in 'abcdefghijklmnopqrstuvwxyz'}
        total_words = len(self.word_list)
        
        for word in self.word_list:
            seen_letters = set()
            for letter in word:
                if letter in letter_counts:
                    letter_counts[letter] += 1
                    total_letters += 1
                    
                    if letter not in seen_letters:
                        words_containing[letter] += 1
                        seen_letters.add(letter)
        
        # Calculate normalized frequencies
        letter_frequencies = {
            letter: count / total_letters if total_letters > 0 else 0
            for letter, count in letter_counts.items()
        }
        
        # Calculate word appearance frequencies
        word_frequencies = {
            letter: count / total_words if total_words > 0 else 0
            for letter, count in words_containing.items()
        }
        
        return {
            "letter_frequencies": letter_frequencies,
            "word_frequencies": word_frequencies
        }
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Random seed.
            options: Additional options.
        
        Returns:
            tuple: Initial observation and info dict.
        """
        super().reset(seed=seed)
        
        # Get the current curriculum phase configuration
        phase_config = self.curriculum.get_config()
        
        # Select a word based on the current phase
        min_len = phase_config["min_word_length"] + random.choice([-1, 0, 1])
        max_len = phase_config["max_word_length"] + random.choice([-1, 0, 1])
        min_len = max(min_len, 3)  # prevent too small
        max_len = min(max_len, self.max_word_length)
        
        # Get all eligible words
        eligible_words = []
        for length in range(min_len, max_len + 1):
            if length in self.words_by_length:
                eligible_words.extend(self.words_by_length[length])
        
        if not eligible_words:
            # Fallback if no eligible words are found
            eligible_words = [word for word in self.word_list if min_len <= len(word) <= max_len]

        if random.random() < 0.2:
            eligible_words = self.word_list # no filtering
        
        self.target_word = random.choice(eligible_words)
        
        # Initialize the revealed word (-2 for hidden letters)
        self.revealed_word = np.full(self.max_word_length, -2, dtype=np.float32)
        
        # Set actual word letters (0-25 for a-z)
        for i, letter in enumerate(self.target_word):
            if i < self.max_word_length:  # Ensure we don't exceed max_word_length
                letter_idx = self.letter_to_idx.get(letter, -1)
                # Mark as unrevealed (-1) initially
                self.revealed_word[i] = -1
        
        # Reset guessed letters
        self.guessed_letters = np.zeros(26, dtype=np.float32)
        
        # Set attempts based on current phase
        self.max_attempts = phase_config["max_attempts"]
        self.remaining_attempts = self.max_attempts
        
        # Reset last action (no action taken yet)
        self.last_action = np.zeros(26, dtype=np.float32)
        
        # Update the rest of the state
        self._update_state()
        
        # Compute initial letter probabilities
        self.letter_probabilities = self._compute_letter_probabilities()
        
        # Return the initial observation and an empty info dict
        return self._get_observation(), {}
    
    def _update_state(self):
        """Update state variables based on the current game state."""
        # Recompute letter probabilities
        self.letter_probabilities = self._compute_letter_probabilities()
    
    def _get_observation(self):
        """
        Construct the observation vector.
        
        Returns:
            numpy.ndarray: The observation vector.
        """
        # 1. Revealed word
        word_obs = np.copy(self.revealed_word)
        
        # 2. Guessed letters
        guessed_letters_obs = np.copy(self.guessed_letters)
        
        # 3. Word length (normalized)
        word_length_obs = np.array([len(self.target_word) / self.max_word_length], dtype=np.float32)
        
        # 4. Letter frequencies
        letter_freqs = np.array(
            [self.letter_stats["letter_frequencies"][letter] for letter in self.alphabet],
            dtype=np.float32
        )
        
        # 5. Last action
        last_action_obs = np.copy(self.last_action)
        
        # 6. Letter probabilities
        letter_probs_obs = np.copy(self.letter_probabilities)
        
        # Concatenate all components
        obs = np.concatenate([
            word_obs,
            guessed_letters_obs,
            word_length_obs,
            letter_freqs,
            last_action_obs,
            letter_probs_obs
        ])
        
        return obs
    
    def _compute_letter_probabilities(self) -> np.ndarray:
        """
        Compute probabilities for each letter using regex-based filtering.
        
        Returns:
            numpy.ndarray: Array of probabilities for each letter.
        """
        # Create a pattern based on the current revealed word
        pattern = ""
        for i in range(len(self.target_word)):
            if i < len(self.revealed_word) and self.revealed_word[i] >= 0:
                # Letter is revealed
                letter = self.idx_to_letter[int(self.revealed_word[i])]
                pattern += letter
            else:
                # Letter is not revealed
                # Exclude already guessed letters
                excluded = ""
                for j, guessed in enumerate(self.guessed_letters):
                    if guessed > 0:
                        excluded += self.idx_to_letter[j]
                
                if excluded:
                    pattern += f"[^{excluded}]"
                else:
                    pattern += "."
        
        # Check if pattern is in cache
        if pattern in self._regex_cache:
            matching_words = self._regex_cache[pattern]
        else:
            try:
                # Compile the regex pattern
                regex = re.compile(f"^{pattern}$")
                
                # Filter the word list to find matching words
                # Only consider words of the same length as target word
                word_length = len(self.target_word)
                candidate_words = self.words_by_length.get(word_length, [])
                
                matching_words = [
                    word for word in candidate_words if regex.match(word)
                ]
                
                # Cache the result
                self._regex_cache[pattern] = matching_words
                
            except re.error:
                # Fallback if regex pattern is invalid
                return np.zeros(26, dtype=np.float32)
        
        # Count letter occurrences in matching words
        letter_counts = {letter: 0 for letter in self.alphabet}
        
        # Vectorized counting of letters
        if matching_words:
            # Join all matching words and count letter frequencies
            all_letters = ''.join(matching_words)
            for letter in self.alphabet:
                letter_counts[letter] = all_letters.count(letter)
        
        # Calculate probabilities
        total_letters = sum(letter_counts.values())
        probabilities = np.zeros(26, dtype=np.float32)
        
        if total_letters > 0:
            for letter, count in letter_counts.items():
                letter_idx = self.letter_to_idx[letter]
                probabilities[letter_idx] = count / total_letters
        
        # Zero out probabilities for already guessed letters
        probabilities *= (1 - self.guessed_letters)
                
        return probabilities
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take an action in the environment.
        
        Args:
            action: Index of the letter to guess (0-25 for a-z).
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # Convert action index to letter
        guessed_letter = self.idx_to_letter[action]
        
        # Check if letter was already guessed
        if self.guessed_letters[action] > 0:
            # Letter already guessed - penalize
            reward = -5.0
            
            # Update last action
            self.last_action = np.zeros(26, dtype=np.float32)
            self.last_action[action] = 1.0
            
            return self._get_observation(), reward, False, False, {}
        
        # Mark letter as guessed
        self.guessed_letters[action] = 1.0
        
        # Update last action
        self.last_action = np.zeros(26, dtype=np.float32)
        self.last_action[action] = 1.0
        
        # Check if the guessed letter is in the target word
        is_correct = guessed_letter in self.target_word
        
        # Get probability of the guessed letter
        letter_prob = self.letter_probabilities[action]
        
        if is_correct:
            # Letter found in target word
            # Count occurrences and reveal them
            target_word_array = np.array(list(self.target_word))
            mask = (target_word_array == guessed_letter)
            newly_revealed = np.sum(mask)
            
            # Update revealed word
            for i, is_match in enumerate(mask):
                if is_match:
                    self.revealed_word[i] = action
            
            # Reward for correct guess based on frequency
            base_reward = 5.0 * newly_revealed
            
            # Extra reward based on letter probability (strategic guessing)
            heuristic_reward = 10.0 * letter_prob if letter_prob > 0 else 2.0
                
            reward = base_reward + heuristic_reward
        else:
            # Incorrect guess - penalize
            self.remaining_attempts -= 1
            
            # Base penalty for incorrect guess
            base_penalty = -4.0
            
            # Additional penalty for strategic mistakes (guessing improbable letters)
            if letter_prob < 0.1:
                # Low probability letter was incorrect - higher penalty
                heuristic_penalty = -2.0 * (1.0 - letter_prob)
            else:
                # Higher probability letter was incorrect - lower penalty
                heuristic_penalty = -2.0 * (1.0 - letter_prob)
                
            reward = base_penalty + heuristic_penalty
        
        # Update state
        self._update_state()
        
        # Check if game is won (all letters revealed) - more efficient check
        # Convert target word to letter indices
        target_indices = np.array([self.letter_to_idx[letter] for letter in self.target_word])
        # Compare with revealed word (only up to the length of target word)
        revealed_indices = self.revealed_word[:len(self.target_word)]
        # Check if all indices match
        all_revealed = np.array_equal(target_indices, revealed_indices)
        
        terminated = False
        info = {}
        
        if all_revealed:
            # Game won
            terminated = True
            
            # Win reward + efficiency bonus
            efficiency_factor = self.remaining_attempts / self.max_attempts
            reward += 50.0 + (10.0 * efficiency_factor)
            
            # Record win for curriculum advancement
            phase_advanced = self.curriculum.record_game_result(win=True)
            info["phase_advanced"] = phase_advanced
            
        elif self.remaining_attempts <= 0:
            # Game lost
            terminated = True
            reward -= 20.0
            
            # Record loss for curriculum tracking
            phase_advanced = self.curriculum.record_game_result(win=False)
            info["phase_advanced"] = phase_advanced
        
        return self._get_observation(), reward, terminated, False, info
    
    def render(self):
        """
        Render the current state of the environment.
        
        Returns:
            str: A string representation of the current state.
        """
        if self.render_mode is None:
            return
        
        # Construct a string representation of the revealed word
        revealed = ""
        for i in range(len(self.target_word)):
            if i < len(self.revealed_word) and self.revealed_word[i] >= 0:
                letter_idx = int(self.revealed_word[i])
                letter = self.idx_to_letter[letter_idx]
                revealed += letter
            else:
                revealed += "_"
        
        # Construct a string of guessed letters
        guessed = ""
        for i, is_guessed in enumerate(self.guessed_letters):
            if is_guessed > 0:
                guessed += self.idx_to_letter[i] + " "
        
        # Construct the render output
        output = (
            f"Word: {revealed}\n"
            f"Guessed letters: {guessed}\n"
            f"Remaining attempts: {self.remaining_attempts}/{self.max_attempts}\n"
            f"Curriculum phase: {self.curriculum.current_phase + 1}/{len(self.curriculum.phases)}"
        )
        
        if self.render_mode == "human":
            print(output)
        
        return output