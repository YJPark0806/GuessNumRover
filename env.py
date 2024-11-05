import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import time

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
        self.max_steps = 1000  # Maximum steps per episode
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

        # Initialize the visited map to track visited cells
        self.visited_map = np.zeros((50, 50), dtype=bool)  # False means unvisited, True means visited
        self.visited_map[25, 25] = True  # Mark initial position as visited

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

        # Save the previous position
        x, y = self.current_position

        # Handle movement actions
        if action == 0:  # Move up
            self.current_position[0] = max(0, self.current_position[0] - 1)
        elif action == 1:  # Move down
            self.current_position[0] = min(49, self.current_position[0] + 1)
        elif action == 2:  # Move left
            self.current_position[1] = max(0, self.current_position[1] - 1)
        elif action == 3:  # Move right
            self.current_position[1] = min(49, self.current_position[1] + 1)

        # Check if the new position has been visited before
        reward = 0
        x_new, y_new = self.current_position
        if self.visited_map[x_new, y_new]:
            # Apply penalty if revisiting a previously visited position
            reward = -5
        else:
            # Mark the position as visited if it was not visited before
            self.visited_map[x_new, y_new] = True

        # Update the explored map and probabilities
        self.update_explored_map()
        self.update_probabilities()

        # Calculate the additional reward (or penalty) from exploration and probabilities
        reward += self.calculate_reward()
        self.cumulative_reward += reward

        # Print the current step information
        '''
        print(
            f"Step {self.current_step} | Current Position {self.current_position} | Probability Score {self.probabilities}")
        '''

        # Check termination conditions
        done = False

        # Check if the largest probability value constitutes 95% or more of the total
        if np.max(self.probabilities) >= 0.95 * np.sum(self.probabilities) and self.current_step >= 100:
            done = True
            guessed_number = np.argmax(self.probabilities)
            if guessed_number == self.target_number:
                reward += 100  # Additional reward for correct guess
                print("Correct guess! +100 reward.")
            print(f"Episode {self.episode_number} finished due to high confidence in one candidate.")
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
            #print(f"Score for: {number}, is: {score}")
        '''
        # Normalize to make it a probability distribution
        total_score = np.sum(probabilities)
        if total_score > 0:
            probabilities = probabilities / total_score
        '''

        return probabilities

    def calculate_reward(self):
        # Reward based on probabilities and exploration efficiency
        reward = 0

        # Sort probabilities in descending order
        sorted_probabilities = np.sort(self.probabilities)[::-1]
        # ex) [3, 8, 5] -> [8, 5, 3]

        # Check if top 3 probabilities make up more than 90% of the total
        if np.sum(sorted_probabilities[:3]) >= 0.9 * np.sum(self.probabilities):
            reward += 3
        elif np.sum(sorted_probabilities[:3]) >= 0.7 * np.sum(self.probabilities):
            reward += 2
        elif np.sum(sorted_probabilities[:3]) >= 0.5 * np.sum(self.probabilities):
            reward += 1

        # Check if top 2 probabilities make up more than 90% of the total
        if np.sum(sorted_probabilities[:2]) >= 0.9 * np.sum(self.probabilities):
            reward += 5
        elif np.sum(sorted_probabilities[:2]) >= 0.8 * np.sum(self.probabilities):
            reward += 4
        elif np.sum(sorted_probabilities[:2]) >= 0.7 * np.sum(self.probabilities):
            reward += 3

        # Penalty for each step to encourage efficient exploration
        reward -= 0.05

        return reward

    def render(self, mode="human"):
        # Enable interactive mode for live updates in a single window
        plt.ion()

        # Display the current explored map with the robot's current position in red
        plt.figure(1)  # Use a single figure ID to keep the same window
        plt.clf()  # Clear the current content to update

        # Copy the explored map and highlight the current position
        explored_map_with_robot = np.copy(self.explored_map)

        # Mark the robot's current position with a distinct value, e.g., 2
        x, y = self.current_position
        explored_map_with_robot[x, y] = 2

        # Create a custom colormap to display the robot's position in red
        cmap = plt.cm.gray
        cmap.set_over('red')  # Set 'over' color to red for the robot's position

        plt.imshow(explored_map_with_robot, cmap=cmap, vmax=1.5)  # Use vmax=1.5 to show red for the robot position
        plt.title(f"Explored Map - Step {self.current_step}")
        plt.axis("off")

        plt.draw()  # Draw the updated figure
        plt.pause(0.5)  # Pause briefly to allow for the update
