import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import gc
from tqdm import tqdm
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
from xgboost import XGBClassifier

# Define the alphabet
alpha = "abcdefghijklmnopqrstuvwxyz"

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
            'n_jobs': -1 ,  
            # 'tree_method': 'hist',
            # 'device':'cuda'                  
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


def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data and print metrics."""
    print("Evaluating model on test data...")
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracies = []
    print("\nConfusion matrices and balanced accuracies for each letter:")
    
    plt.figure(figsize=(20, 15))
    for i, letter in enumerate(alpha):
        # Convert probabilities to binary predictions
        binary_preds = np.round(y_pred[:, i])
        
        # Calculate metrics
        cm = confusion_matrix(y_test[letter], binary_preds)
        balanced_acc = balanced_accuracy_score(y_test[letter], binary_preds)
        accuracies.append(balanced_acc)
        
        # Print metrics
        print(f"\nLetter '{letter}':")
        print(f"Confusion Matrix:\n{cm}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        
        # Plot confusion matrix
        plt.subplot(5, 6, i+1)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"Letter '{letter}': {balanced_acc:.2f}")
        plt.colorbar()
        plt.xticks([0, 1], ['Neg', 'Pos'])
        plt.yticks([0, 1], ['Neg', 'Pos'])
        
        # Add values to the plot
        thresh = cm.max() / 2
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                plt.text(col, row, format(cm[row, col], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[row, col] > thresh else "black")
        
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300)
    plt.close()
    
    # Overall accuracy
    mean_acc = np.mean(accuracies)
    print(f"\nOverall mean balanced accuracy: {mean_acc:.4f}")
    
    # Plot accuracy distribution
    plt.figure(figsize=(12, 6))
    plt.bar(alpha, accuracies)
    plt.axhline(y=mean_acc, color='r', linestyle='-', label=f'Mean: {mean_acc:.4f}')
    plt.title('Balanced Accuracy by Letter')
    plt.xlabel('Letter')
    plt.ylabel('Balanced Accuracy')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('accuracy_by_letter.png', dpi=300)
    plt.close()
    
    return mean_acc, accuracies


def load_and_process_data(data_dir="hangman_data", output_file="combined_hangman_data.parquet", max_chunks=None):
    """
    Load and combine all parquet files from data directory.
    """
    # Check if combined file already exists
    if os.path.exists(output_file):
        print(f"Combined file {output_file} already exists. Loading...")
        return pd.read_parquet(output_file)
    
    # Get all parquet files
    files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    if max_chunks:
        files = files[:max_chunks]
    
    print(f"Found {len(files)} data chunks to process")
    
    # Read and combine all files
    all_data = []
    total_rows = 0
    
    # Process in smaller batches to manage memory
    batch_size = 10  # Process 10 files at a time
    num_batches = (len(files) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(files))
        batch_files = files[start_idx:end_idx]
        
        print(f"Processing batch {batch_idx+1}/{num_batches} ({len(batch_files)} files)")
        
        # Load batch files
        batch_dfs = []
        for file in tqdm(batch_files, desc=f"Loading batch {batch_idx+1}"):
            file_path = os.path.join(data_dir, file)
            df = pd.read_parquet(file_path)
            batch_dfs.append(df)
            total_rows += len(df)
        
        # Combine batch and save to a temporary file
        if batch_dfs:
            combined_batch = pd.concat(batch_dfs, ignore_index=True)
            temp_file = f"temp_batch_{batch_idx}.parquet"
            combined_batch.to_parquet(temp_file)
            
            # Add temp file to the list of all_data
            all_data.append(temp_file)
            
            # Clear memory
            del combined_batch, batch_dfs
            gc.collect()
    
    print(f"Processed {total_rows} total rows across {len(files)} files")
    
    # Now combine all temporary batch files
    print("Combining all batches into final dataset...")
    final_dfs = []
    
    for temp_file in all_data:
        df = pd.read_parquet(temp_file)
        final_dfs.append(df)
    
    combined_df = pd.concat(final_dfs, ignore_index=True)
    
    # Save the final combined dataset
    combined_df.to_parquet(output_file)
    print(f"Saved combined data with {len(combined_df)} rows to {output_file}")
    
    # Clean up temporary files
    for temp_file in all_data:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    return combined_df


def train_on_sample(data_dir="hangman_data", max_files=5):
    """Train on a small sample of data for testing purposes."""
    print(f"Training on a small sample ({max_files} files) for testing...")
    
    # Load sample data
    files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')][:max_files]
    dfs = []
    
    for file in files:
        file_path = os.path.join(data_dir, file)
        df = pd.read_parquet(file_path)
        dfs.append(df)
    
    dataset = pd.concat(dfs, ignore_index=True)
    print(f"Sample dataset shape: {dataset.shape}")
    
    # Split data
    feature_cols = [str(x) for x in range(80)]
    target_cols = [x for x in alpha]
    
    X_train, X_test, y_train, y_test = train_test_split(
        dataset[feature_cols], 
        dataset[target_cols], 
        test_size=0.2, 
        random_state=42
    )
    
    model = MultiLabelCatBoostClassifier(iterations=100, verbose=10)
    
    for i, letter in enumerate(alpha):
        print(f"\nTraining sample classifier for letter '{letter}'")
        
        model.classifiers[i].fit(
            X_train, 
            y_train[letter],
            eval_set=[(X_test, y_test[letter])],
            verbose=10
        )
    
    mean_acc, _ = evaluate_model(model, X_test, y_test)
    
    model.save('models_trained/test_model.pkl')
    print(f"Test training completed with accuracy: {mean_acc:.4f}")
    
    return mean_acc


def train_batched_model(data_dir, combined_data_file, model_dir):
    """Batch training for computation efficiency"""
    alpha_batches = [
        "abcd",
        "efgh",
        "ijkl",
        "mnop",
        "qrst",
        "uvwx",
        "yz"
    ]
    
    dataset = load_and_process_data(data_dir, combined_data_file)
    
    feature_cols = [str(x) for x in range(80)]
    target_cols = [x for x in alpha]
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        dataset[feature_cols], 
        dataset[target_cols], 
        test_size=0.2, 
        random_state=42
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, 
        y_temp, 
        test_size=0.5, 
        random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    del dataset, X_temp, y_temp
    gc.collect()
    
    existing_models = {}
    for letter in alpha:
        letter_model_path = os.path.join(model_dir, f'letter_{letter}_model.pkl')
        if os.path.exists(letter_model_path):
            print(f"Found existing model for letter '{letter}'")
            existing_models[letter] = letter_model_path
    
    # Create main model
    main_model = MultiLabelXGBoostClassifier(
        iterations=2000,  
        learning_rate=0.1,
        depth=10,
        verbose=50
    )
    
    # Load existing classifiers
    for letter, model_path in existing_models.items():
        letter_idx = alpha.index(letter)
        try:
            with open(model_path, 'rb') as f:
                letter_classifier = pickle.load(f)
                main_model.classifiers[letter_idx] = letter_classifier
                print(f"Loaded existing classifier for letter '{letter}'")
        except Exception as e:
            print(f"Error loading classifier for letter '{letter}': {e}")
    
    # Train for each batch of letters
    for batch_idx, batch_letters in enumerate(alpha_batches):
        print(f"Training batch {batch_idx+1}/{len(alpha_batches)}: {batch_letters}")
        
        # Train model for this batch of letters
        for letter in batch_letters:
            letter_idx = alpha.index(letter)
            letter_model_path = os.path.join(model_dir, f'letter_{letter}_model.pkl')
            
            # Skip if this letter already has a trained model
            if letter in existing_models:
                print(f"Skipping letter '{letter}' as it already has a trained model")
                continue
                
            print(f"\nTraining classifier for letter '{letter}'")
            
            # Handle severe class imbalance
            pos_count = y_train[letter].sum()
            neg_count = len(y_train) - pos_count
            
            if pos_count / neg_count < 0.1:  # High imbalance
                pos_indices = y_train[y_train[letter] == 1].index
                neg_indices = y_train[y_train[letter] == 0].index
                
                # Sample from negative samples to match 10 times the positive samples
                sample_size = min(10 * pos_count, neg_count)
                sampled_neg_indices = np.random.choice(neg_indices, size=sample_size, replace=False)
                training_indices = np.concatenate([pos_indices, sampled_neg_indices])
                
                X_subset = X_train.loc[training_indices]
                y_subset = y_train[letter].loc[training_indices]
                
                print(f"  Balanced training data for '{letter}': {len(pos_indices)} positive, {len(sampled_neg_indices)} negative samples")
            else:
                X_subset = X_train
                y_subset = y_train[letter]
                print(f"  Using all data for '{letter}': {pos_count} positive, {neg_count} negative samples")
            
            try:
                # Train single letter classifier
                main_model.classifiers[letter_idx].fit(
                    X_subset, 
                    y_subset,
                    eval_set=[(X_val, y_val[letter])],
                    verbose=50
                )
                
                # Save this letter's classifier
                with open(letter_model_path, 'wb') as f:
                    pickle.dump(main_model.classifiers[letter_idx], f)
                print(f"Saved classifier for letter '{letter}' to {letter_model_path}")
                
                # Print feature importance
                feat_importance = main_model.classifiers[letter_idx].get_feature_importance()
                top_features = np.argsort(feat_importance)[-5:]  # Top 5 features
                print(f"  Top 5 features for '{letter}': {X_train.columns[top_features]}")
                
            except Exception as e:
                print(f"Error training letter '{letter}': {e}")
                
                # If normal training fails, try with stricter memory limits
                try:
                    print(f"Retrying with stricter memory limits for letter '{letter}'...")
                    
                    # Create a new classifier with stricter memory limits
                    strict_params = main_model.params.copy()
                    strict_params['used_ram_limit'] = '8gb'  # Lower memory limit
                    strict_params['thread_count'] = 2  # Reduce thread count
                    
                    temp_classifier = CatBoostClassifier(**strict_params)
                    
                    # Train with stricter settings
                    temp_classifier.fit(
                        X_subset, 
                        y_subset,
                        eval_set=[(X_val, y_val[letter])],
                        verbose=50
                    )
                    
                    # If successful, replace the classifier in the main model
                    main_model.classifiers[letter_idx] = temp_classifier
                    
                    # Save this letter's classifier
                    with open(letter_model_path, 'wb') as f:
                        pickle.dump(temp_classifier, f)
                    print(f"Successfully trained and saved classifier for letter '{letter}' with strict memory limits")
                    
                except Exception as e2:
                    print(f"Retry also failed for letter '{letter}': {e2}")
            
            # Release memory
            gc.collect()
        
        # Save progress after batch
        batch_model_path = os.path.join(model_dir, f'batch_{batch_idx+1}_model.pkl')
        main_model.save(batch_model_path)
        print(f"Saved progress after batch {batch_idx+1} to {batch_model_path}")
        
        # If enough letters have been trained, try evaluating the model
        trained_letters = [l for l in alpha if os.path.exists(os.path.join(model_dir, f'letter_{l}_model.pkl'))]
        if len(trained_letters) >= 5:  # At least 5 letters trained
            print(f"Evaluating model with {len(trained_letters)} letters trained...")
            try:
                # Use validation set for evaluation
                mean_acc, _ = evaluate_model(main_model, X_val, y_val)
                
                # Save progress model with accuracy
                progress_model_path = os.path.join(model_dir, f'progress_model_{mean_acc:.4f}.pkl')
                main_model.save(progress_model_path)
            except Exception as e:
                print(f"Error evaluating model: {e}")
    
    # All letters trained, perform final evaluation
    print("All letters trained. Performing final evaluation...")
    try:
        # Use test set for final evaluation
        mean_acc, _ = evaluate_model(main_model, X_test, y_test)
        
        # Save final model
        final_model_path = os.path.join(model_dir, f'final_model_{mean_acc:.4f}.pkl')
        main_model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")
    except Exception as e:
        print(f"Error in final evaluation: {e}")
        
        # Still save final model, even if evaluation fails
        final_model_path = os.path.join(model_dir, 'final_model.pkl')
        main_model.save(final_model_path)
        print(f"Final model saved to {final_model_path} (without accuracy)")
    
    return main_model


def main():
    """Main function to run the training process."""
    # Configuration
    data_dir = "hangman_data"  # Directory with processed data chunks
    combined_data_file = "combined_hangman_data.parquet"  # Combined data file
    model_dir = "models_trained_2"  # Directory for saving models
    sample_test_data = False  # Use a smaller dataset for testing
    
    # Create model directory if needed
    os.makedirs(model_dir, exist_ok=True)
    
    # Quick test on sample data first
    if sample_test_data:
        print("Running a quick test on sample data first...")
        sample_acc = train_on_sample(data_dir, max_files=5)
        print(f"Sample training complete with accuracy: {sample_acc:.4f}")
        
        # Ask if user wants to continue with full training
        user_input = input("Continue with full training? (y/n): ")
        if user_input.lower() != 'y':
            print("Exiting without full training.")
            return
    
    # Start batched training
    print("Starting batched training...")
    trained_model = train_batched_model(data_dir, combined_data_file, model_dir)
    
    print("Training process completed.")


if __name__ == "__main__":
    main()