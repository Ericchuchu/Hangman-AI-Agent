"""Final ensemble solver: routes every guess to a PPO or gradient-boosting agent.

| word length | remaining attempts | agent               |
|-------------|--------------------|---------------------|
| <= 4        | any                | PPO-GRU             |
| 5 to 6      | any                | short-word XGBoost  |
| 7 to 10     | >= 5               | CatBoost            |
| 7 to 10     | < 5                | XGBoost             |
| > 10        | any                | XGBoost             |

`guess` and the helper methods are taken verbatim from the evaluation notebook; only the
dictionary path became a constructor argument. The trained models and run logs they load
are not part of the repository (see .gitignore), so train them first or adjust the paths.
"""
import collections
import re

import numpy as np

from ml_agents import HangmanMLAgent


def load_dictionary(path):
    with open(path, "r") as handle:
        return handle.read().splitlines()


class EnsembleHangmanSolver:
    def __init__(self, dictionary_path="cleaned_words.txt"):
        self.dictionary_path = dictionary_path
        self.guessed_letters = []
        self.full_dictionary = load_dictionary(dictionary_path)
        self.full_dictionary_common_letter_sorted = collections.Counter("".join(self.full_dictionary)).most_common()
        self.current_dictionary = []
        self.remaining_attempts = 6

    def new_game(self, attempts=6):
        """Reset the per-game state (the game loop calls this before the first guess)."""
        self.guessed_letters = []
        self.current_dictionary = []
        self.remaining_attempts = attempts

    def guess(self, word): # word input example: "_ p p _ e "
        if not self.current_dictionary:
            self.current_dictionary = self.full_dictionary
        
        # Initialize agent
        if not hasattr(self, 'agent_initialized'):
            import numpy
            import torch.serialization
            torch.serialization.add_safe_globals([
                numpy.core.multiarray.scalar,
                numpy.dtype,
                numpy.dtypes.Float64DType,
            ])
            from ppo_agent_gru import GruHangmanPPOAgent
            from ppo_agent import HangmanPPOAgent
            
            # Load ppo agent
            # Load config.txt
            agent_params = self._load_agent_params_from_file("./logs/ppo_gru_hangman_run_20250427_004317/config.txt")
            
            self.agent_ppo_gru = GruHangmanPPOAgent(
                max_word_length = 20,
                embedding_dim = agent_params.get('embedding_dim'),
                gru_layers =  agent_params.get('num_gru_layers'),
                gru_hidden_dim =  agent_params.get('gru_hidden_dim'),
                mlp_hidden_dim=  agent_params.get('mlp_hidden_dim'),
                dropout=  agent_params.get('dropout'),
                lr=  agent_params.get('lr'),
                gamma=  agent_params.get('gamma'),
                gae_lambda=  agent_params.get('gae_lambda'),
                clip_epsilon=  agent_params.get('clip_epsilon'),
                value_coef=  agent_params.get('value_coef'),
                entropy_coef=  agent_params.get('entropy_coef'),
                max_grad_norm=  agent_params.get('max_grad_norm'),
                update_epochs=  agent_params.get('update_epochs'),
                batch_size=  agent_params.get('batch_size'),
                device=  agent_params.get('device')
            )
            
            # Load model
            self.agent_ppo_gru.load_model("./models/ppo_gru_hangman_run_20250427_004317/hangman_ppo_model_episode_1578000.pt")

            # Load PPO Transformer agent
            agent_params = self._load_agent_params_from_file("./logs/ppo_hangman_run_20250427_004325/config.txt")

            self.agent_ppo_transformer = HangmanPPOAgent(
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

            self.agent_ppo_transformer.load_model("./models/ppo_hangman_run_20250427_004325/hangman_ppo_model_episode_1262000.pt")

            # Load ML agent
            self.agent_ml_xgb = HangmanMLAgent(
                model_path='ml_process/models_trained_2/final_model_0.8458.pkl', 
                dictionary_path=self.dictionary_path
            )

            self.agent_ml_cat = HangmanMLAgent(
                model_path='ml_process/models_trained/final_model_0.7011.pkl', 
                dictionary_path=self.dictionary_path
            )

            self.agent_ml_short = HangmanMLAgent(
                model_path='ml_process/models_trained_short/final_model_0.5858.pkl', 
                dictionary_path=self.dictionary_path
            )

            self.agent_initialized = True
            self.current_state = None

        # Word convert to environment state
        clean_word = word[::2]  

        state = self._convert_to_state(clean_word)

        guessed_word = []
        for i, char in enumerate(clean_word):
            if char != '_':
                guessed_word.append(char)

        # Select action
        if len(clean_word) <= 4:
            action,_,_ = self.agent_ppo_gru.select_action(state, training=False)

        elif 5 <= len(clean_word) < 7:
            action = self.agent_ml_short.choose_action(state, self.remaining_attempts)

        elif 7 <= len(clean_word) <= 10:
            if self.remaining_attempts >= 5:
                action = self.agent_ml_cat.choose_action(state, self.remaining_attempts)
            else:
                action = self.agent_ml_xgb.choose_action(state, self.remaining_attempts)
        
        else:
            action = self.agent_ml_xgb.choose_action(state, self.remaining_attempts)
        
        # Convert to alphabet
        guess_letter = chr(97 + action)  # a=97, b=98, ...
        
        # Record alphabet
        self.guessed_letters.append(guess_letter)
        
        return guess_letter

    def _load_agent_params_from_file(self, filepath):
        # Load agent params
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
                            else:
                                params[key] = float(value)
                        except ValueError:
                            params[key] = value

        except Exception as e:
            print(f"Error loading agent parameters: {e}")
            return {}
        
        return params

    def _convert_to_state(self, clean_word):
        max_word_length = 20  
        
        # Initialize revealed word (-2 : padding, -1 : hidden, 0-25 : revealed words)
        revealed_word = np.full(max_word_length, -2, dtype=np.float32)
        for i, char in enumerate(clean_word):
            if i < max_word_length:
                if char == '_':
                    revealed_word[i] = -1  # hidden words
                else:
                    revealed_word[i] = ord(char) - 97  # revealed words
        
        # Guessed letter
        guessed_letters = np.zeros(26, dtype=np.float32)
        for letter in self.guessed_letters:
            idx = ord(letter) - 97
            if 0 <= idx < 26:
                guessed_letters[idx] = 1.0
        
        # Word length
        word_length = np.array([len(clean_word) / max_word_length], dtype=np.float32)
        
        # Letter frequencies
        letter_freqs = self._get_letter_frequencies()
        
        # Last action
        last_action = np.zeros(26, dtype=np.float32)
        if self.guessed_letters:
            last_idx = ord(self.guessed_letters[-1]) - 97
            if 0 <= last_idx < 26:
                last_action[last_idx] = 1.0
        
        
        # Letter probabilities
        letter_probs = self._compute_letter_probabilities(clean_word)
        
        # Combine all components
        state = np.concatenate([
            revealed_word,
            guessed_letters,
            word_length,
            letter_freqs,
            last_action,
            letter_probs
        ])
        
        return state

    def _get_letter_frequencies(self):
        letter_freqs = np.zeros(26, dtype=np.float32)
        total_count = sum(count for _, count in self.full_dictionary_common_letter_sorted)
        
        if total_count > 0:
            for letter, count in self.full_dictionary_common_letter_sorted:
                if 'a' <= letter <= 'z':
                    idx = ord(letter) - 97
                    letter_freqs[idx] = count / total_count
                    
        return letter_freqs

    def _compute_letter_probabilities(self, clean_word):
        pattern = ""
        for char in clean_word:
            if char == '_':
                pattern += "."
            else:
                pattern += char
        
        # Find matching words
        regex = re.compile(f"^{pattern}$")
        matching_words = [w for w in self.full_dictionary if len(w) == len(clean_word) and regex.match(w)]
        
        # Calculate letter probabilities
        letter_counts = {chr(97+i): 0 for i in range(26)}
        if matching_words:
            all_letters = "".join(matching_words)
            for letter in letter_counts:
                letter_counts[letter] = all_letters.count(letter)
        
        # Convert to probabilities
        total_letters = sum(letter_counts.values())
        probabilities = np.zeros(26, dtype=np.float32)
        
        if total_letters > 0:
            for letter, count in letter_counts.items():
                idx = ord(letter) - 97
                probabilities[idx] = count / total_letters
        
        # Set probabilities of guessed letters to 0
        for letter in self.guessed_letters:
            idx = ord(letter) - 97
            if 0 <= idx < 26:
                probabilities[idx] = 0.0
        
        return probabilities
