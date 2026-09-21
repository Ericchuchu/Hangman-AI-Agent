#!/usr/bin/env python
"""
Test script for the Hangman environment.
This script demonstrates how to interact with the HangmanEnv class.
"""

import numpy as np
import os
from env import HangmanEnv

# Create a simple word list file for testing
def create_test_word_list():
    word_list_path = "test_words.txt"
    words = [
        "apple", "banana", "cherry", "date", "elderberry",
        "fig", "grape", "honeydew", "kiwi", "lemon",
        "mango", "nectarine", "orange", "papaya", "quince",
        "raspberry", "strawberry", "tangerine", "watermelon",
        "cat", "dog", "elephant", "fox", "giraffe",
        "horse", "iguana", "jaguar", "koala", "lion",
        "monkey", "newt", "octopus", "penguin", "quail",
        "rabbit", "snake", "tiger", "vulture", "wolf"
    ]
    
    with open(word_list_path, 'w') as f:
        for word in words:
            f.write(word + '\n')
    
    return word_list_path

def play_random_agent(env, num_episodes=5):
    """Play the game with a random agent."""
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        print(f"\n===== Episode {episode + 1} =====")
        print(env.render())
        
        while not done:
            # Random action (choose a letter randomly)
            action = env.action_space.sample()
            
            # Skip if the letter has already been guessed
            if obs[env.max_word_length + action] > 0:
                continue
            
            # Take the action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            # Print the current state
            print(f"\nAction: {env.idx_to_letter[action]}")
            print(f"Reward: {reward:.2f}")
            print(env.render())
        
        print(f"Episode {episode + 1} finished with total reward: {total_reward:.2f}")
        if "phase_advanced" in info and info["phase_advanced"]:
            print("Curriculum phase advanced!")

def play_interactive(env, max_episodes=5):
    """Play the game interactively with user input."""
    for episode in range(max_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        print(f"\n===== Episode {episode + 1} =====")
        print(env.render())
        
        while not done:
            # Get user input
            letter = input("\nEnter a letter (a-z) or 'q' to quit: ").lower()
            
            if letter == 'q':
                print("Quitting the game...")
                return
            
            if len(letter) != 1 or not letter.isalpha():
                print("Please enter a single letter.")
                continue
            
            # Convert letter to action
            action = env.letter_to_idx.get(letter, -1)
            
            if action == -1:
                print("Invalid letter. Try again.")
                continue
            
            # Check if already guessed
            if obs[env.max_word_length + action] > 0:
                print(f"You've already guessed '{letter}'. Try another letter.")
                continue
            
            # Take the action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            # Print the current state
            print(f"Reward: {reward:.2f}")
            print(env.render())
        
        print(f"Episode {episode + 1} finished with total reward: {total_reward:.2f}")
        if "phase_advanced" in info and info["phase_advanced"]:
            print("Curriculum phase advanced!")
        
        # Ask if the user wants to continue
        if episode < max_episodes - 1:
            continue_game = input("\nPlay another round? (y/n): ").lower()
            if continue_game != 'y':
                break

def main():
    # Create a test word list
    word_list_path = create_test_word_list()
    
    # Create the environment
    env = HangmanEnv(word_list_path=word_list_path, render_mode="human")
    
    print("Welcome to the Hangman Environment Test!")
    print("1. Watch a random agent play")
    print("2. Play interactively")
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == '1':
        play_random_agent(env)
    elif choice == '2':
        play_interactive(env)
    else:
        print("Invalid choice. Exiting.")
    
    # Clean up
    if os.path.exists(word_list_path):
        os.remove(word_list_path)

if __name__ == "__main__":
    main()
