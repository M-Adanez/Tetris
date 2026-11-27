import os
from collections import deque

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# ---------------------------------------------------------------------------
# 1. Custom Observation Wrapper
# ---------------------------------------------------------------------------
class TetrisFeatureWrapper(gym.ObservationWrapper):
    """
    Extracts a 2x22x10 binary board (Channel 0: locked stack, Channel 1: active piece)
    and 12 normalized heuristic features computed strictly on the locked stack.
    """

    def __init__(
        self,
        env: gym.Env,
        y_bounds: tuple[int, int] = (27, 203),  # 176 px -> 22 rows * 8 px
        x_bounds: tuple[int, int] = (22, 62),   # 40 px  -> 10 cols * 4 px
        bg_color: tuple[int, int, int] = (111, 111, 111),
        color_tol: int = 10,
    ):
        super().__init__(env)
        self.rows = 22
        self.cols = 10
        self.y0, self.y1 = y_bounds
        self.x0, self.x1 = x_bounds
        self.bg_color = np.array(bg_color, dtype=np.int16)
        self.color_tol = color_tol

        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(
                    low=0.0, high=1.0, shape=(2, self.rows, self.cols), dtype=np.float32
                ),
                # [10 column heights, holes, bumpiness] normalized to [0, 1]
                "features": spaces.Box(
                    low=0.0, high=1.0, shape=(self.cols + 2,), dtype=np.float32
                ),
            }
        )

    def observation(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        full_board = self._extract_board(obs)
        locked, active = self._split_locked_and_active(full_board)
        features = self._compute_features(locked)

        board_2ch = np.stack([locked, active], axis=0).astype(np.float32)
        return {
            "board": board_2ch,
            "features": features,
        }

    def _extract_board(self, frame: np.ndarray) -> np.ndarray:
        crop = frame[self.y0 : self.y1, self.x0 : self.x1]
        cell_h = crop.shape[0] // self.rows
        cell_w = crop.shape[1] // self.cols

        trimmed = crop[: self.rows * cell_h, : self.cols * cell_w]
        cells = trimmed.reshape(self.rows, cell_h, self.cols, cell_w, 3)

        center_pixels = cells[:, cell_h // 2, :, cell_w // 2, :].astype(np.int16)
        diff = np.abs(center_pixels - self.bg_color).max(axis=-1)
        return (diff > self.color_tol).astype(np.uint8)

    def _split_locked_and_active(
        self, full_board: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        locked = np.zeros_like(full_board)
        queue = deque()

        bottom_row = self.rows - 1
        for c in range(self.cols):
            if full_board[bottom_row, c]:
                locked[bottom_row, c] = 1
                queue.append((bottom_row, c))

        while queue:
            r, c = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < self.rows
                    and 0 <= nc < self.cols
                    and full_board[nr, nc]
                    and not locked[nr, nc]
                ):
                    locked[nr, nc] = 1
                    queue.append((nr, nc))

        active = full_board - locked
        return locked, active

    def _compute_features(self, locked_board: np.ndarray) -> np.ndarray:
        has_block = locked_board.any(axis=0)
        first_occupied = locked_board.argmax(axis=0)
        heights = np.where(has_block, self.rows - first_occupied, 0)

        block_above = np.maximum.accumulate(locked_board, axis=0)
        holes = int(np.sum((locked_board == 0) & (block_above == 1)))

        bumpiness = int(np.sum(np.abs(np.diff(heights))))

        norm_heights = heights / float(self.rows)
        norm_holes = holes / float(self.rows * self.cols)
        norm_bumpiness = bumpiness / float((self.cols - 1) * self.rows)

        return np.concatenate(
            [norm_heights, [norm_holes, norm_bumpiness]]
        ).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Custom CNN + MLP Feature Extractor
# ---------------------------------------------------------------------------
class TetrisCNNMLPExtractor(BaseFeaturesExtractor):
    """
    Small-kernel CNN for the 2x22x10 board + MLP for the 12 scalar features.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 128,
        mlp_output_dim: int = 32,
    ):
        super().__init__(
            observation_space, features_dim=cnn_output_dim + mlp_output_dim
        )

        board_shape = observation_space["board"].shape      # (2, 22, 10)
        features_dim = observation_space["features"].shape[0]  # 12

        self.cnn = nn.Sequential(
            nn.Conv2d(board_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, *board_shape)).shape[1]

        self.cnn_linear = nn.Sequential(
            nn.Linear(n_flatten, cnn_output_dim),
            nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, mlp_output_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        board_emb = self.cnn_linear(self.cnn(observations["board"]))
        feat_emb = self.mlp(observations["features"])
        return torch.cat([board_emb, feat_emb], dim=1)


# ---------------------------------------------------------------------------
# 3. Environment Factory & Training Loop
# ---------------------------------------------------------------------------
def make_tetris_env() -> gym.Env:
    env = gym.make("ALE/Tetris-v5")
    return TetrisFeatureWrapper(env)


def main():
    gym.register_envs(ale_py)

    # Directories for logs and saved weights
    log_dir = "./logs/tetris_ppo"
    ckpt_dir = "./checkpoints/tetris_ppo"
    best_model_dir = "./models/best_tetris_ppo"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    # Vectorized training & evaluation environments
    n_envs = 8
    train_env = make_vec_env(make_tetris_env, n_envs=n_envs, seed=42)
    eval_env = make_vec_env(make_tetris_env, n_envs=1, seed=100)

    # Custom policy configuration
    policy_kwargs = dict(
        features_extractor_class=TetrisCNNMLPExtractor,
        features_extractor_kwargs=dict(cnn_output_dim=128, mlp_output_dim=32),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
        normalize_images=False,  # Keep [0, 1] binary board unscaled
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=512,          # 512 * 8 envs = 4,096 steps per rollout
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,        # Encourages exploration early on
        tensorboard_log=log_dir,
        verbose=1,
        device="auto",
    )

    # Callbacks: periodic checkpoints + save best model by eval reward
    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1),
        save_path=ckpt_dir,
        name_prefix="ppo_tetris",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=log_dir,
        eval_freq=max(25_000 // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
    )

    # Train
    total_timesteps = 1_000_000
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList([checkpoint_cb, eval_cb]),
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model...")
    finally:
        model.save("./models/tetris_ppo_final")
        train_env.close()
        eval_env.close()
        print("Saved final model to ./models/tetris_ppo_final.zip")


if __name__ == "__main__":
    main()