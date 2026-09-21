import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from collections import Counter
import multiprocessing as mp
from tqdm import tqdm
import gc
import os

# Mapping of letters to values
value = {"a":1,"b":2,"c":3,"d":4,"e":5,"f":6,"g":7,
         "h":8,"i":9,"j":10,"k":11,"l":12,"m":13,"n":14,
         "o":15,"p":16,"q":17,"r":18,"s":19,"t":20,"u":21,
         "v":22,"w":23,"x":24,"y":25,"z":26}

def build_dictionary(dictionary_file_location):
    """Read dictionary from file"""
    text_file = open(dictionary_file_location, "r")
    full_dictionary = text_file.read().splitlines()
    text_file.close()
    return full_dictionary

def get_combinations(letters, max_comb_size=None):
    """Generate letter combinations, optionally limit combination size to reduce computation"""
    combinations = []
    max_size = max_comb_size if max_comb_size else len(letters)
    for i in range(1, min(len(letters)+1, max_size+1)):
        combinations += list(itertools.combinations(letters, i))
    return combinations

def process_word_chunk(args):
    """Process a chunk of words, creating feature vectors for each letter combination of each word
    Returns a DataFrame instead of yielding it to avoid pickling issues with generators"""
    word_chunk, max_comb_size, chunk_id = args
    all_rows = []
    
    # Track progress with print statements (tqdm won't work well with multiprocessing)
    print(f"Processing chunk {chunk_id} with {len(word_chunk)} words")
    
    for word_idx, word in enumerate(word_chunk):
        word_set = set(word)
        # Limit combination size if the letter set is too large
        effective_max_size = min(max_comb_size, len(word_set)) if max_comb_size else len(word_set)
        
        for comb_size in range(1, effective_max_size + 1):
            for comb in itertools.combinations(word_set, comb_size):
                # Initialize feature vector
                features = np.full(80, -1, dtype=np.int8)
                letter_mask = np.zeros(26, dtype=np.int8)
                
                # Fill feature vector
                k = len(word)
                for i in range(k):
                    if word[i] in comb:
                        features[i] = value[word[i]]
                        features[80-k+i] = value[word[i]]
                    else:
                        features[i] = 0
                        features[80-k+i] = 0
                        letter_idx = value[word[i]] - 1
                        letter_mask[letter_idx] = 1
                
                # Add feature vector to the rows list
                all_rows.append(np.concatenate([features, letter_mask]).tolist())
    
    # Create a single DataFrame with all rows
    df = pd.DataFrame(all_rows, dtype=np.int8)
    if not df.empty:
        df.columns = [str(x) for x in range(80)] + list("abcdefghijklmnopqrstuvwxyz")
    
    print(f"Chunk {chunk_id} completed with {len(all_rows)} rows")
    return df

def save_dataframe(df, output_dir, chunk_number):
    """Save DataFrame to a compressed file"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"hangman_data_chunk_{chunk_number:05d}.parquet")
    df.to_parquet(filepath, compression='gzip')
    return filepath

def process_dictionary(dictionary, output_dir, num_processes=None, max_comb_size=3, chunk_size=500):
    """Process the entire dictionary and save as multiple smaller chunks"""
    # Set number of processes
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare dictionary chunks - make chunks smaller to avoid memory issues
    total_words = len(dictionary)
    print(f"Processing {total_words} words using {num_processes} processes")
    
    # Split dictionary into smaller chunks for multiprocessing
    word_chunks = []
    for i in range(0, total_words, chunk_size):
        chunk_words = dictionary[i:min(i+chunk_size, total_words)]
        word_chunks.append((chunk_words, max_comb_size, len(word_chunks)))
    
    # Track data chunk number
    chunk_number = 0
    
    # Process dictionary chunks using a process pool
    with mp.Pool(processes=num_processes) as pool:
        results = []
        
        # Process chunks in batches to control memory usage
        for batch_idx in range(0, len(word_chunks), num_processes*2):
            batch_chunks = word_chunks[batch_idx:batch_idx + num_processes*2]
            print(f"Processing batch {batch_idx//num_processes + 1} of {(len(word_chunks) + num_processes*2 - 1)//(num_processes*2)}")
            
            # Process batch of chunks
            batch_results = pool.map(process_word_chunk, batch_chunks)
            
            # Save results and clear memory
            for df in batch_results:
                if not df.empty:
                    filepath = save_dataframe(df, output_dir, chunk_number)
                    print(f"Saved data chunk {chunk_number} to {filepath}, shape: {df.shape}")
                    chunk_number += 1
                
                # Release memory
                del df
            
            # Force garbage collection
            gc.collect()
            
            print(f"Batch {batch_idx//num_processes + 1} completed and saved")

def analyze_dictionary(dictionary):
    """Analyze word length distribution in the dictionary"""
    # Count number of letters in each word
    letter_counts = [len(set(word)) for word in dictionary]
    word_lengths = [len(word) for word in dictionary]
    
    # Plot letter count distribution
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(letter_counts, bins=25, color='skyblue', edgecolor='black')
    plt.title('Distribution of Unique Letters per Word')
    plt.xlabel('Number of Unique Letters')
    plt.ylabel('Number of Words')
    
    # Plot word length distribution
    plt.subplot(1, 2, 2)
    plt.hist(word_lengths, bins=25, color='lightgreen', edgecolor='black')
    plt.title('Word Length Distribution')
    plt.xlabel('Word Length')
    plt.ylabel('Number of Words')
    
    plt.tight_layout()
    plt.savefig('dictionary_analysis.png', dpi=300)
    plt.close()
    
    print("Dictionary analysis complete, results saved to dictionary_analysis.png")
    
    # Return word length statistics
    length_counter = Counter(word_lengths)
    letter_counter = Counter(letter_counts)
    
    return {
        'word_length_stats': length_counter,
        'unique_letters_stats': letter_counter,
        'avg_word_length': np.mean(word_lengths),
        'avg_unique_letters': np.mean(letter_counts)
    }

def main():
    """Main function"""
    dictionary_file = "short_words.txt"
    output_dir = "hangman_data_short"
    
    # Use smaller combination size to reduce memory usage
    max_comb_size = 4  # Limit combination size to reduce computation
    
    # Read dictionary
    print(f"Reading dictionary {dictionary_file}...")
    dictionary = build_dictionary(dictionary_file)
    
    # Analyze dictionary
    print("Analyzing dictionary...")
    stats = analyze_dictionary(dictionary)
    print(f"Average word length: {stats['avg_word_length']:.2f}")
    print(f"Average unique letters per word: {stats['avg_unique_letters']:.2f}")
    
    # Process dictionary with smaller chunk size to manage memory better
    print(f"Processing dictionary and saving data to {output_dir}...")
    process_dictionary(dictionary, output_dir, max_comb_size=max_comb_size, chunk_size=1000)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()