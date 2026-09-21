"""Gradient-boosting agents used by the final ensemble solver.

Verbatim from the evaluation notebook: multi-label XGBoost and CatBoost classifiers
(one binary classifier per letter) and the agent that combines their predictions with
dictionary pattern matching. Slightly newer than the copies in ml_process/.
"""
import os
import re
import pickle
import random
import collections
from collections import Counter

import numpy as np
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

class MultiLabelXGBoostClassifier:
    def __init__(self, num_classes=26, iterations=1500, learning_rate=0.1, depth=6, verbose=100):
        # Initialize XGBoost parameters
        self.params = {
            'n_estimators': iterations,        
            'learning_rate': learning_rate,
            'max_depth': depth,            
            'objective': 'binary:logistic',    
            'eval_metric': 'auc',              
            'random_state': 0,
            'early_stopping_rounds': 50,
            'verbosity': 1 if verbose > 0 else 0,
            'n_jobs': -1                        
        }
        
        # Initialize 26 XGBoost classifiers, one for each letter
        self.classifiers = [XGBClassifier(**self.params) for _ in range(num_classes)]
        self.verbose = verbose
    
    def fit(self, X, y):
        """
        Train all 26 classifiers, one for each letter.
        
        Parameters:
        -----------
        X : DataFrame
            Feature data
        y : DataFrame
            Binary target variables (one column per letter)
        """
        for i, letter in enumerate("abcdefghijklmnopqrstuvwxyz"):
            if self.verbose:
                print(f"\nTraining classifier for letter '{letter}'")
            
            try:
                self.classifiers[i].fit(
                    X, 
                    y[letter],
                    eval_set=[(X, y[letter])], 
                    verbose=self.verbose
                )
                
                if self.verbose > 0:
                    importances = self.classifiers[i].feature_importances_
                    indices = np.argsort(importances)[-5:]  # Top 5 features
                    print(f"  Top 5 features for '{letter}': {X.columns[indices]}")
                    
            except Exception as e:
                print(f"Error training classifier for letter '{letter}': {e}")
                self.classifiers[i] = XGBClassifier(**self.params)
    
    def predict(self, X):
        """
        Predict probabilities for each letter.
        
        Parameters:
        -----------
        X : DataFrame or numpy array
            Feature data
            
        Returns:
        --------
        np.array
            Predicted probabilities for each letter (shape: n_samples x 26)
        """
        predictions = np.zeros((len(X), len(self.classifiers)))
        for i, clf in enumerate(self.classifiers):
            try:
                predictions[:, i] = clf.predict_proba(X)[:, 1] 
            except Exception as e:
                print(f"Error predicting with classifier {i}: {e}")
                predictions[:, i] = 0.5
        return predictions
    
    def save(self, filename):
        """Save the model to a file."""
        print(f"Saving model to {filename}...")
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        print("Model saved successfully.")
    
    @classmethod
    def load(cls, filename):
        """Load a model from a file."""
        print(f"Loading model from {filename}...")
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        print("Model loaded successfully.")
        return model


class MultiLabelCatBoostClassifier:
    def __init__(self, num_classes=26, iterations=1500, learning_rate=0.1, depth=6, verbose=100):
        # Initialize CatBoost parameters
        self.params = {
            'iterations': iterations,
            'learning_rate': learning_rate,
            'depth': depth,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'random_seed': 0,
            'early_stopping_rounds': 50,
        }
        
        # Initialize 26 CatBoost classifiers, one for each letter
        self.classifiers = [CatBoostClassifier(**self.params) for _ in range(num_classes)]
        self.verbose = verbose
    
    def predict(self, X):
        """
        Predict probabilities for each letter.
        
        Parameters:
        -----------
        X : DataFrame
            Feature data
            
        Returns:
        --------
        np.array
            Predicted probabilities for each letter (shape: n_samples x 26)
        """
        # Predict probabilities for each letter
        predictions = np.zeros((len(X), len(self.classifiers)))
        for i, clf in enumerate(self.classifiers):
            try:
                predictions[:, i] = clf.predict_proba(X)[:, 1]  # Probability of class '1'
            except Exception as e:
                print(f"Error predicting with classifier {i}: {e}")
                # Use 0.5 as default probability if prediction fails
                predictions[:, i] = 0.5
        return predictions
    
    def save(self, filename):
        """Save the model to a file."""
        print(f"Saving model to {filename}...")
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        print("Model saved successfully.")
    
    @classmethod
    def load(cls, filename):
        """Load a model from a file."""
        print(f"Loading model from {filename}...")
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        print("Model loaded successfully.")
        return model

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
        if word_length <= 7:
            window_sizes = [2, 3, 4, 5, 6]
        else :
            window_sizes = [3 ,4, 5, 6, 7, 8, 9]
        
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
            if word_length <= 7:
                if revealed_word_length <= 2:
                    combined_candidates[letter] = prob * 1.1
                elif revealed_word_length >= len(word_state) - 2 and remaining_attempts <= 2:
                    combined_candidates[letter] = prob * 1.1
                else:
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
            base_windows = [3]
        elif word_length <= 8:
            base_windows = [4, 5, 6]
        else:
            base_windows = [4, 5, 6, 7, 8]
        
        # Known and unknown positions
        known_positions = [i for i, c in enumerate(word_state) if c != '_']
        unknown_positions = [i for i, c in enumerate(word_state) if c == '_']
        
        # If known positions are more than 60%, use smaller windows
        if len(known_positions) > word_length * 0.6:
            smaller_windows = [w for w in range(3, min(base_windows) + 1)]
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
            if 3 not in final_windows:
                final_windows.insert(0, 3)
        
        # If end unknown and has end sensitive letters, add small window
        if not has_end_known and has_end_letters and (word_length - 1) in unknown_positions:
            if 3 not in final_windows:
                final_windows.insert(0, 3)
        
        # Ensure windows are not larger than word length
        final_windows = [w for w in final_windows if w < word_length]
        
        # If no windows left, add a small window
        if not final_windows:
            final_windows = [min(3, word_length - 1)]
        
        return final_windows
