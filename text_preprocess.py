import os
import re
import string
import argparse
from collections import Counter
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import random

def load_word_list(file_path):
    """
    Efficiently load word list using generators to avoid loading large files at once
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            word = line.strip().lower()
            if word:  # Skip empty lines
                yield word

def is_valid_word(word, min_length=3, max_length=15, min_unique_letters=2, 
                  allow_proper_nouns=False, valid_words_set=None):
    """
    Check if a word meets all validity criteria
    """
    # Basic length check
    if not (min_length <= len(word) <= max_length):
        return False
    
    # Alphabetic characters only
    if not word.isalpha():
        return False
    
    # Proper noun check (capitalized words)
    if not allow_proper_nouns and word[0].isupper():
        return False
    
    # Unique letter count check
    if len(set(word)) < min_unique_letters:
        return False
    
    # Check against valid words set if provided
    if valid_words_set is not None and word not in valid_words_set:
        return False
    
    # Check for repetitive patterns
    # Examples: 'aaaa', 'abababab', etc.
    for i in range(1, len(word)//2 + 1):
        pattern = word[:i]
        if pattern * (len(word)//i) + pattern[:len(word)%i] == word and i < len(word)//2:
            return False
    
    return True

def process_chunk(chunk, criteria):
    """
    Process a chunk of the word list
    """
    valid_words = []
    for word in chunk:
        if is_valid_word(word, **criteria):
            valid_words.append(word)
    return valid_words

def clean_word_list_parallel(file_path, output_path, min_length=3, max_length=15, 
                            min_unique_letters=2, allow_proper_nouns=False, 
                            valid_words_path=None, n_processes=None, chunk_size=10000):
    """
    Clean word list in parallel for improved performance
    """
    # Load valid words set if provided
    valid_words_set = None
    if valid_words_path:
        print(f"Loading valid words from {valid_words_path}...")
        valid_words_set = set()
        with open(valid_words_path, 'r', encoding='utf-8') as f:
            for line in f:
                valid_words_set.add(line.strip().lower())
        print(f"Loaded {len(valid_words_set)} valid words.")
    
    # Set filtering criteria
    criteria = {
        'min_length': min_length,
        'max_length': max_length,
        'min_unique_letters': min_unique_letters,
        'allow_proper_nouns': allow_proper_nouns,
        'valid_words_set': valid_words_set
    }
    
    # Calculate file size and line count (for progress display)
    total_lines = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
    
    # Prepare for parallel processing
    n_processes = n_processes or max(1, mp.cpu_count() - 1)
    print(f"Using {n_processes} processes for processing.")
    
    # Read and process words in chunks
    all_valid_words = []
    chunks = []
    current_chunk = []
    
    print(f"Reading and processing {file_path}...")
    with tqdm(total=total_lines) as pbar:
        for word in load_word_list(file_path):
            current_chunk.append(word)
            pbar.update(1)
            
            if len(current_chunk) >= chunk_size:
                chunks.append(current_chunk)
                current_chunk = []
                
                # Process chunks in parallel when enough have accumulated
                if len(chunks) >= n_processes * 2:  # Keep process pool busy
                    with mp.Pool(n_processes) as pool:
                        results = pool.map(partial(process_chunk, criteria=criteria), chunks)
                        for result in results:
                            all_valid_words.extend(result)
                    chunks = []
        
        # Process any remaining chunks
        if current_chunk:
            chunks.append(current_chunk)
        
        if chunks:
            with mp.Pool(n_processes) as pool:
                results = pool.map(partial(process_chunk, criteria=criteria), chunks)
                for result in results:
                    all_valid_words.extend(result)
    
    # Save results
    print(f"Writing {len(all_valid_words)} cleaned words to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_valid_words))
    
    return all_valid_words

def analyze_word_list(words):
    """
    Analyze word list and generate statistics
    """
    analysis = {
        'total_words': len(words),
        'avg_length': sum(len(word) for word in words) / len(words) if words else 0,
        'length_distribution': Counter(len(word) for word in words),
        'letter_frequency': Counter(''.join(words)),
        'unique_letters_distribution': Counter(len(set(word)) for word in words)
    }
    
    # Find most common and least common letters
    analysis['most_common_letters'] = analysis['letter_frequency'].most_common(5)
    analysis['least_common_letters'] = analysis['letter_frequency'].most_common()[:-6:-1]
    
    return analysis

def print_analysis(analysis):
    """
    Print word list analysis results
    """
    print("\n===== Word List Analysis =====")
    print(f"Total words: {analysis['total_words']}")
    print(f"Average length: {analysis['avg_length']:.2f}")
    
    print("\nLength distribution:")
    for length, count in sorted(analysis['length_distribution'].items()):
        percentage = (count / analysis['total_words']) * 100
        print(f"  {length} letters: {count} ({percentage:.1f}%)")
    
    print("\nMost common letters:")
    for letter, count in analysis['most_common_letters']:
        percentage = (count / sum(analysis['letter_frequency'].values())) * 100
        print(f"  '{letter}': {count} ({percentage:.1f}%)")
    
    print("\nLeast common letters:")
    for letter, count in analysis['least_common_letters']:
        percentage = (count / sum(analysis['letter_frequency'].values())) * 100
        print(f"  '{letter}': {count} ({percentage:.1f}%)")
    
    print("\nUnique letters distribution:")
    for unique_count, word_count in sorted(analysis['unique_letters_distribution'].items()):
        percentage = (word_count / analysis['total_words']) * 100
        print(f"  {unique_count} unique letters: {word_count} ({percentage:.1f}%)")

def save_analysis(analysis, output_path):
    """
    Save analysis results to file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("===== Word List Analysis =====\n")
        f.write(f"Total words: {analysis['total_words']}\n")
        f.write(f"Average length: {analysis['avg_length']:.2f}\n\n")
        
        f.write("Length distribution:\n")
        for length, count in sorted(analysis['length_distribution'].items()):
            percentage = (count / analysis['total_words']) * 100
            f.write(f"  {length} letters: {count} ({percentage:.1f}%)\n")
        
        f.write("\nLetter frequency:\n")
        for letter, count in sorted(analysis['letter_frequency'].items()):
            percentage = (count / sum(analysis['letter_frequency'].values())) * 100
            f.write(f"  '{letter}': {count} ({percentage:.1f}%)\n")
        
        f.write("\nUnique letters distribution:\n")
        for unique_count, word_count in sorted(analysis['unique_letters_distribution'].items()):
            percentage = (word_count / analysis['total_words']) * 100
            f.write(f"  {unique_count} unique letters: {word_count} ({percentage:.1f}%)\n")
    
def split_word_list(full_word_list, train_ratio=0.8, seed=42):
    """
    Split word list into training and evaluation sets.
    """
    random.seed(seed)
    shuffled = full_word_list.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    train_words = shuffled[:split_idx]
    eval_words = shuffled[split_idx:]
    
    return train_words, eval_words

def main():
    parser = argparse.ArgumentParser(description='Optimized Word List Cleaner and Analyzer')
    parser.add_argument('--input_file',default='words_250000_train.txt', help='Input word list file path')
    parser.add_argument('--output', '-o', default='cleaned_words.txt', help='Output file path for cleaned word list')
    parser.add_argument('--min-length', type=int, default=3, help='Minimum word length')
    parser.add_argument('--max-length', type=int, default=20, help='Maximum word length')
    parser.add_argument('--min-unique', type=int, default=1, help='Minimum number of unique letters')
    parser.add_argument('--valid-words', help='Path to valid words list file (optional)')
    parser.add_argument('--allow-proper', action='store_true', help='Allow proper nouns (capitalized words)')
    parser.add_argument('--processes', type=int, help='Number of processes to use for parallel processing')
    parser.add_argument('--chunk-size', type=int, default=10000, help='Number of words per processing chunk')
    parser.add_argument('--analyze', action='store_true', help='Analyze the cleaned word list')
    parser.add_argument('--analysis-output', default='word_analysis.txt', help='Output file path for analysis results')
    parser.add_argument('--train-output', default='train_words.txt', help='Output file path for training set')
    parser.add_argument('--eval-output', default='eval_words.txt', help='Output file path for evaluation set')
    
    args = parser.parse_args()
    
    # Ensure input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        return
    
    # Clean word list
    print(f"Starting word list cleaning: {args.input_file}")
    cleaned_words = clean_word_list_parallel(
        args.input_file, 
        args.output,
        min_length=args.min_length,
        max_length=args.max_length,
        min_unique_letters=args.min_unique,
        allow_proper_nouns=args.allow_proper,
        valid_words_path=args.valid_words,
        n_processes=args.processes,
        chunk_size=args.chunk_size
    )
    
    print(f"Original word count: {sum(1 for _ in load_word_list(args.input_file))}")
    print(f"Cleaned word count: {len(cleaned_words)}")
    print(f"Cleaned word list saved to: {args.output}")

    # Split word list into training and evaluation sets
    train_words, eval_words = split_word_list(cleaned_words)
    
    # Save training and evaluation sets
    with open(args.train_output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_words))
    
    with open(args.eval_output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(eval_words))
    
    print(f"Training set saved to: {args.train_output}")
    print(f"Evaluation set saved to: {args.eval_output}")
    
    # Analyze word list if requested
    if args.analyze:
        print("Analyzing word list...")
        analysis = analyze_word_list(cleaned_words)
        print_analysis(analysis)
        
        # Save analysis results
        save_analysis(analysis, args.analysis_output)
        print(f"Analysis results saved to: {args.analysis_output}")

if __name__ == "__main__":
    main()