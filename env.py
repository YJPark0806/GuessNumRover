import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class NumberRecognitionEnv(gym.Env):
    """
    Custom Environment for Number Recognition using Reinforcement Learning.
    This environment uses preprocessed MNIST data loaded from 'mnist_data.npy'.
    """

    def __init__(self):
        super(NumberRecognitionEnv, self).__init__()

        # Load the preprocessed MNIST data
        self.mnist_data = np.load("mnist_data.npy", allow_pickle=True).item()

        # Define action space
        # Actions: 0 - Up, 1 - Down, 2 - Left, 3 - Right
        self.action_space = spaces.Discrete(4)

        # Define observation space as a single Box space by flattening map_data and probabilities
        # map_data is 50x50 and probabilities is 10, so total dimension is 50*50 + 10
        self.observation_space = spaces.Box(low=-1, high=1, shape=(50 * 50 + 10,), dtype=np.float32)

        # Initialize other variables
        self.episode_number = 0  # Track episode number
        self.max_steps = 700  # Maximum steps per episode
        self.reset()

    def reset(self):
        # Increment episode number
        if hasattr(self, 'episode_number'):
            self.episode_number += 1

        # Choose a random target number between 0 and 9
        self.target_number = np.random.randint(0, 10)

        # Reset the robot's current position to (25, 25)
        self.current_position = [25, 25]

        # Initialize the explored map as a 50x50 matrix filled with -1 (unexplored)
        self.explored_map = np.full((50, 50), -1, dtype=int)

        # Initialize the probabilities vector as zeros
        self.probabilities = np.zeros(10, dtype=float)

        # Initialize the cumulative reward
        self.cumulative_reward = 0

        # Initialize step counter
        self.current_step = 0

        # Update the input matrix for the initial position
        self.update_explored_map()

        # Calculate initial probabilities based on the input matrix
        self.update_probabilities()

        # Return the initial observation
        return self.get_observation()

    def step(self, action):
        # Increment the step counter
        self.current_step += 1
        print(f"step {self.current_step}")

        # Handle movement actions
        if action == 0:  # Move up
            self.current_position[0] = max(0, self.current_position[0] - 1)
        elif action == 1:  # Move down
            self.current_position[0] = min(49, self.current_position[0] + 1)
        elif action == 2:  # Move left
            self.current_position[1] = max(0, self.current_position[1] - 1)
        elif action == 3:  # Move right
            self.current_position[1] = min(49, self.current_position[1] + 1)

        # Update the explored map and probabilities
        self.update_explored_map()
        self.update_probabilities()

        # Print the probabilities vector after each step
        print("Probabilities vector:", self.probabilities)
        print('==================================================================')

        # Calculate reward and check if episode is done
        reward = self.calculate_reward()
        self.cumulative_reward += reward

        # Check termination conditions
        done = False
        if np.sum(self.probabilities > 0.95) == 1:  # End if one number is highly likely
            done = True
            print(f"Episode {self.episode_number} finished.")
            guessed_number = np.argmax(self.probabilities)
            print("Probabilities vector at episode end:", self.probabilities)
            print(f"Answer: {self.target_number}, Guessed: {guessed_number}, Return: {self.cumulative_reward}")

        # Terminate if maximum steps reached
        elif self.current_step >= self.max_steps:
            done = True
            print(f"Episode {self.episode_number} terminated due to reaching max steps of {self.max_steps}.")
            guessed_number = np.argmax(self.probabilities)
            print("Probabilities vector at max steps:", self.probabilities)
            print(f"Answer: {self.target_number}, Guessed: {guessed_number}, Return: {self.cumulative_reward}")

        return self.get_observation(), reward, done, {}

    def get_observation(self):
        # Flatten the explored_map and concatenate with probabilities to create a single observation vector
        map_data_flattened = self.explored_map.flatten()
        observation = np.concatenate([map_data_flattened, self.probabilities]).astype(np.float32)
        return observation

    def update_explored_map(self):
        # Mark the current position as explored and set its value in explored_map
        x, y = self.current_position

        # Assume `actual_map` is the true underlying map of the target digit (0 and 1 matrix for the target number)
        actual_map = self.mnist_data[self.target_number][0]  # Example usage, using the first sample of the target number

        # Update explored_map with the actual value at (x, y)
        self.explored_map[x, y] = actual_map[x, y]  # Set to 1 if the actual map has a 1, otherwise set to 0

    def update_probabilities(self):
        """
        Update the probabilities vector by calculating similarity scores with MNIST data.
        """
        self.probabilities = self.calculate_probabilities(self.explored_map)

    def calculate_probabilities(self, explored_map):
        """
        Calculate the probability vector for each digit (0-9) based on the explored map.
        The probabilities are normalized to form a probability distribution.

        Args:
            explored_map (np.array): 50x50 matrix representing the explored map (-1 for unexplored, 0 or 1 for explored).

        Returns:
            np.array: Probability vector of length 10, with values representing the normalized likelihood of each digit.
        """
        probabilities = np.zeros(10, dtype=float)

        # Calculate similarity score for each digit
        for number in range(10):
            score = 0
            for data_matrix in self.mnist_data[number]:
                # Only compare positions where explored_map is not -1 (i.e., where it has been explored)
                match_matrix = (explored_map != -1) * data_matrix  # Only consider explored parts
                score += np.sum((explored_map == 1) * match_matrix)  # Match only where explored_map == 1 and data_matrix == 1
            probabilities[number] = score

        # Normalize to make it a probability distribution
        total_score = np.sum(probabilities)
        if total_score > 0:
            probabilities = probabilities / total_score

        return probabilities

    def calculate_reward(self):
        # Reward based on probabilities and exploration efficiency
        reward = 0

        # Give reward for reducing candidate numbers based on probability threshold
        for i in range(10):
            if self.probabilities[i] < 0.05 * np.max(self.probabilities):
                reward += 1

        # Penalty for each step to encourage efficient exploration
        reward -= 0.05

        # Additional reward if a candidate's probability significantly decreases
        for i in range(10):
            if self.probabilities[i] < 0.3 * np.max(self.probabilities):
                reward += 3

        return reward

    def render(self, mode="human"):
        # Display the current explored map
        plt.figure(figsize=(5, 5))
        plt.imshow(self.explored_map, cmap="gray")
        plt.title("Explored Map")
        plt.axis("off")
        plt.show()