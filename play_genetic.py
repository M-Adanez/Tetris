
import os

import ale_py
import numpy as np
import gymnasium as gym


# Configuration
ENV_NAME = "ALE/Tetris-v5"
MODEL_PATH = "out/best_tetris_genome.npy"

MAX_STEPS = 4_000
SEED = 321

INPUT_DIM = 128
HIDDEN_DIM = 32
OUTPUT_DIM = 5


def load_model(path: str):
    """Load the saved genome and unpack its neural network weights."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

    genome = np.load(path).astype(np.float32)

    expected_size = (
        INPUT_DIM * HIDDEN_DIM
        + HIDDEN_DIM
        + HIDDEN_DIM * OUTPUT_DIM
        + OUTPUT_DIM
    )

    if genome.ndim != 1 or genome.size != expected_size:
        raise ValueError(
            f"Invalid genome shape: {genome.shape}. "
            f"Expected ({expected_size},)."
        )

    idx = 0

    w1_size = INPUT_DIM * HIDDEN_DIM
    W1 = genome[idx:idx + w1_size].reshape(INPUT_DIM, HIDDEN_DIM)
    idx += w1_size

    b1 = genome[idx:idx + HIDDEN_DIM]
    idx += HIDDEN_DIM

    w2_size = HIDDEN_DIM * OUTPUT_DIM
    W2 = genome[idx:idx + w2_size].reshape(HIDDEN_DIM, OUTPUT_DIM)
    idx += w2_size

    b2 = genome[idx:idx + OUTPUT_DIM]

    return W1, b1, W2, b2


def select_action(obs: np.ndarray, weights) -> int:
    """Forward pass through the trained neural network."""
    W1, b1, W2, b2 = weights

    x = obs.astype(np.float32) / 255.0

    h = np.tanh(x @ W1 + b1)
    logits = h @ W2 + b2

    return int(np.argmax(logits))


def play_game():
    """Load the trained model and play one Tetris game."""

    gym.register_envs(ale_py)

    weights = load_model(MODEL_PATH)
    print(f"Loaded model: {MODEL_PATH}")

    env = gym.make(
        ENV_NAME,
        obs_type="ram",
        render_mode="human",
        repeat_action_probability=0.0,
    )

    obs, info = env.reset(seed=SEED)

    total_reward = 0.0
    steps = 0

    print("Starting Tetris...")

    try:
        for step in range(MAX_STEPS):
            action = select_action(obs, weights)

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            steps += 1

            if terminated or truncated:
                break

    except KeyboardInterrupt:
        print("\nGame interrupted by user.")

    finally:
        env.close()

    print("\nGame finished!")
    print(f"Steps:        {steps}")
    print(f"Total reward: {total_reward:.2f}")


if __name__ == "__main__":
    play_game()
