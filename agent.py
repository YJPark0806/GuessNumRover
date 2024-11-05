import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from env import NumberRecognitionEnv  # Import the NumberRecognitionEnv class from env.py


def train_agent():
    """
    Train a PPO agent on the NumberRecognitionEnv environment and log to TensorBoard.
    """
    # Initialize the custom environment
    env = NumberRecognitionEnv()

    # Initialize the PPO agent with TensorBoard logging
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./number_recognition_tensorboard/")

    # Train the agent
    print("Starting training...")
    model.learn(total_timesteps=200000, log_interval=10)  # Set the number of timesteps for training
    print("Training completed.")

    # Save the trained model
    model.save("number_recognition_agent")
    print("Model saved as 'number_recognition_agent'.")

    # Evaluate the trained model
    print("Evaluating the trained model...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")


def test_agent():
    """
    Test the trained PPO agent on the NumberRecognitionEnv environment.
    """
    # Load the custom environment
    env = NumberRecognitionEnv()

    # Load the trained model
    model = PPO.load("number_recognition_agent")
    print("Loaded trained model 'number_recognition_agent'.")

    # Run a test episode
    obs = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    while not done:
        # Predict the next action
        action, _states = model.predict(obs, deterministic=True)

        # Take the action in the environment
        obs, reward, done, info = env.step(action)

        # Accumulate reward and increase step count
        total_reward += reward
        step_count += 1

        # Render the environment (optional, for visualization)
        env.render()

    print(f"Test episode completed in {step_count} steps with total reward: {total_reward}")


if __name__ == "__main__":
    # Train the agent
    train_agent()

    # Test the trained agent
    #test_agent()

    # To visualize training in TensorBoard, run this command in your terminal:
    # tensorboard --logdir=./number_recognition_tensorboard/
