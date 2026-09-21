# Hangman AI Agent

Agents that play Hangman (six wrong guesses allowed) against words they have not seen: a reinforcement-learning approach (PPO with GRU or Transformer encoders, DQN, curriculum learning) and a supervised approach (one XGBoost or CatBoost classifier per letter combined with dictionary pattern matching). The final solver routes each guess to the agent that works best for the word length and the number of attempts left.

| Path | Purpose |
|---|---|
| `env.py`, `text_preprocess.py` | Gymnasium environment with curriculum phases; word-list cleaning and train/evaluation split |
| `ppo_agent.py`, `ppo_agent_gru.py`, `dqn_agent.py` | PPO (Transformer and GRU) and DQN agents |
| `train_agent.py`, `test_agent.py`, `test_hangman.py` | Training, evaluation and interactive play |
| `ml_process/` | Feature generation, training and testing of the per-letter gradient-boosting classifiers |
| `ml_agents.py`, `ensemble_solver.py` | The model classes and the routing logic of the final solver |
| `word_analysis.txt` | Letter-frequency and word-length statistics of the cleaned word list |

## Word lists

No word list is included. Use any newline-separated English word list (for example `words_alpha.txt` from the public-domain [dwyl/english-words](https://github.com/dwyl/english-words) project), then build the cleaned, training and evaluation lists:

```bash
pip install -r requirements.txt
python text_preprocess.py --input_file words_alpha.txt   # writes cleaned_words.txt, train_words.txt, eval_words.txt
python train_agent.py                                     # PPO / DQN training
awk 'length($0)>=3 && length($0)<=7' cleaned_words.txt > short_words.txt   # input of the short-word model
python ml_process/data_processing.py && python ml_process/train.py   # gradient-boosting agents
```

`ml_process/data_processing.py` reads `short_words.txt` (the words of 3 to 7 letters). `ml_process/test.py` opens the word list under the file name used in the original runs, `words_250000_train.txt`, regardless of `--word_list`; rename your list or edit that line.

Trained models and run logs are not stored in the repository; `ensemble_solver.py` loads them from the paths of the original runs, so adjust those paths after training.

## RL approach

### Text Preprocessing (`text_preprocess.py`)
- Parallel Processing: Efficient word list cleaning using multiprocessing
- Filtering Criteria: Removes invalid words based on length, unique letter count, and repetitive patterns
- Data Splitting: Automatically divides word list into training and evaluation sets
- Statistical Analysis: Generates comprehensive letter frequency and word length distributions

### Environment (`env.py`)
- Action Space: 26 discrete actions (one for each English letter a-z)
- Observation Space: 185 features in total. A high-dimensional vector composed of:
  - Revealed Word (max word length, -2 for sequence padding, -1 for hidden, 0-25 for revealed letters)
  - Guessed Letters (26-dimensional one-hot)
  - Normalized Word Length (1-dimensional)
  - Global Letter Frequencies (26-dimensional)
  - Last Action (26-dimensional one-hot)
  - Estimated Letter Probabilities (26-dimensional)

- Reward Design:
  - Positive reward for correct guesses (scaled by occurrences)
  - Penalty for wrong guesses (severity based on letter probability)
  - Bonus for guessing the full word
  - Heavy penalty for repeated guesses

- Termination Conditions:
  - Win: all letters correctly revealed
  - Loss: all attempts used up

- Curriculum Learning:
  - The environment automatically increases difficulty based on the agent's win rate.
  - Early phases feature short, easy words; later phases introduce longer, harder words.
  - Phase transitions occur when performance thresholds (e.g., win rate ≥ 60%) are met.

### PPO Agent Implementation (`ppo_agent.py` & `ppo_agent_gru.py`)
- Proximal Policy Optimization (PPO) algorithm for reinforcement learning
- GRU-based residual neural network architecture for sequence processing in ppo_agent_gru.py
- Transformer-based residual neural network architecture for sequence processing in ppo_agent.py
- MLP process embedding features for policy and value networks
- Action masking to prevent already-guessed letters
- Generalized Advantage Estimation (GAE) for more stable training

### DQN Agent Implementation (`dqn_agent.py`)
- Deep Q-Network (DQN) algorithm for reinforcement learning
- Action masking to prevent already-guessed letters
- Transformer-based residual neural network architecture for sequence processing in dqn_agent.py
- MLP process embedding features for policy and value networks

### Training and Evaluation (`train_agent.py` & `test_agent.py`)
- Comprehensive training pipeline with logging and checkpoints
- Performance evaluation metrics including win rate and reward tracking
- Interactive play mode for demonstration

### Used RL Techniques
- Reward Shaping: Designed reward function that considers letter frequency and strategic guessing
- Experience Replay: Used in DQN implementation to improve sample efficiency
- Curriculum Learning: Progressive difficulty increase based on agent performance metrics
- Residual Connections: Implemented in neural network architectures to improve gradient flow
- Attention Mechanisms: Used in transformer-based models to focus on relevant parts of the word state
- Exploration Strategies: Epsilon-greedy with annealing schedule for DQN and entropy-based exploration for PPO

### Results (`test_agent.py`)
- Our reinforcement learning approach showed excellent performance (approximately 90% accuracy) in the custom Hangman environment but dropped significantly to around 10% accuracy when applied to actual practice scenarios.
The primary issue appears to be overfitting to the letter frequency distributions and state representations in our training environment. The RL agent learned patterns specific to our training data rather than generalizable strategies for playing Hangman.
To address this limitation, we've shifted to an RL approach to ML approach and deliberately excludes letter frequency features from the state representation. This modification forces the agent to learn more generalizable strategies based on word patterns and game dynamics rather than memorizing statistical distributions from the training set.

## ML approach (`ml_process`)

### Data Processing (`data_processing.py`)
- Feature Engineering: Transforms words into numerical feature vectors with 80 positional features and 26 letter mask features
- Combinatorial Processing: Generates training examples from letter combinations of each word
- Parallel Processing: Uses multiprocessing to handle large dictionaries efficiently
- Memory Management: Implements chunk-based processing to avoid memory overflow with large datasets
- Data Analysis: Provides statistical insights into word length and unique letter distributions

### Model Training (`train.py`)
- Multi-label Classification: Uses CatBoost classifiers for predicting letter probabilities 
- Multi-label Classification: Uses XGBoost classifiers for predicting letter probabilities 
- Multi-label Classification: Uses XGBoost classifiers for predicting letter probabilities (special case for short letters)
- Using different model dealing for the long and short words. This is bracuse I found that the performance for original xgboost on short words (about < 7 letters) is not good. So using specific short-word xgboost for short words (5 and 6 letters) and the catboost will act as a regularizer for the long words (about > 7 letters).Then ensemble the three models to get the final action. More specific approach will mentioned in the final prediction. 
- Ensemble Approach: For all Multi-label Classification, Trains 26 separate classifiers, one for each letter of the alphabet
- Batched Training: Implements memory-efficient batch training for handling large datasets
- Confusion Matrix Analysis: Generates detailed performance metrics for each letter classifier, and balanced accuracy

### Model Testing (`test.py`)
- Pattern Analysis: Implements substring-based pattern matching for unknown letters
- Probability-based Guessing: Uses trained model to predict letter probabilities in partially revealed words
- Adaptive Strategy: Combines model predictions with game-specific heuristics

## Finalized guess Function and logic of ML agent (ML-Based Hangman Letter Predictor)

### ML agent logic : Pattern-Based Substring Matching (Dictionary Logic)

To enhance guesses in early or ambiguous stages, the agent employs a regex-like substring extraction and frequency estimation approach:

#### Substring Extraction Logic

- From the revealed word, sliding windows of length *n* (e.g., 3–8) are formed with a wildcard (`.`) at unknown positions.
  - Example: `'a _ _ l e'` → `a..le`, `.pp.e`
- Substrings are filtered to ensure they:
  - Include exactly one unknown (`.`)
  - Do not contain consecutive wildcards (optional strict mode)
  - Match known letter positions

#### Dictionary-Based Letter Probability

1. Each regex substring is matched against a preloaded dictionary of English words (about 250,000 in the original runs).
2. For matches, the character at the wildcard position is extracted.
3. Letter frequencies are computed from all matched words.
4. The result is a probability distribution over possible next letters.

#### Integration with ML Prediction

- The agent combines the pattern-derived probabilities with ML model predictions using dynamic weighting:
  - Heavier weight is given to pattern logic when few letters are revealed.
  - ML predictions dominate when more of the word is known or it's near the endgame.


### Model Selection Strategy

The agent dynamically selects a specialized model based on the length of the partially revealed word and how many incorrect attempts remain:

| Word Length      | Remaining Attempts | Model Used           |
|------------------|--------------------|-----------------------|
| ≤ 4              | Any                | PPO-GRU agent         |
| 5 to 6           | Any                | Short-word ML model   |
| 7 to 10          | ≥ 5                | CatBoost model        |
| 7 to 10          | < 5                | XGBoost model         |
| > 10             | Any                | XGBoost model         |

### Final guess function (`ensemble_solver.py`)

1. **Convert Word to State Vector**  
   A length-80 numeric vector is constructed based on forward/backward fill of the revealed letters.

2. **Model Selection**  
   The agent selects one of the following: PPO-GRU, XGBoost (short/full), or CatBoost.

3. **Model-Based Letter Ranking**  
   The selected model returns a probability vector over 26 letters.

4. **Pattern-Based Letter Ranking**  
   Substrings are extracted and used to build a separate distribution from dictionary matches.

5. **Combined Scoring & Selection**  
   Both scores are merged. The letter with the highest combined score that has not yet been guessed is selected.

6. **Fallback Logic**  
   If no good letter is found (e.g., all probabilities low), the agent picks a random unguessed letter.

## Limitations

- The reinforcement-learning agents reached about 90% wins in the training environment but about 10% on words from outside it; they overfitted to the letter-frequency features of the training list. That is why the final solver relies mainly on the gradient-boosting agents.
- The win rate of the final ensemble solver is not reported in this repository.
- Trained models and word lists are not included, and `ensemble_solver.py` loads models from the paths of the original runs.
