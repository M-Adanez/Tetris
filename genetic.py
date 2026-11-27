
import os
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

import ale_py
import numpy as np
import gymnasium as gym

gym.register_envs(ale_py)


SEED = 123
ENV_NAME = "ALE/Tetris-v5"

POP_SIZE = 200
GENERATIONS = 300
ELITE_COUNT = 10
EPS_PER_EVAL = 4
MAX_STEPS = 4_000

K = 4
CROSS_OVER_RATE = 0.25
MUTATION_RATE = 0.02
MUTATION_STD = 0.10

INPUT_DIM = 128
HIDDEN_DIM = 32
OUTPUT_DIM = 5
GENOME_SIZE = (INPUT_DIM * HIDDEN_DIM) + HIDDEN_DIM + (HIDDEN_DIM * OUTPUT_DIM) + OUTPUT_DIM

OUTPUT_PATH = "out/best_tetris_genome.npy"


def init_population(pop_size: int, genome_size: int, seed: int = SEED) -> np.ndarray:
    """Create the initial population matrix of random 1D genomes."""
    rng = np.random.default_rng(seed=seed)
    mat_dim = (pop_size, genome_size)
    mean = 0.0
    std_dev = 0.25

    rand_mat = rng.normal(
        loc=mean,
        scale=std_dev,
        size=mat_dim
    ).astype(np.float32)

    return rand_mat


def unpack_weights(
        genome: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reshape a flat 1D genome vector into (W1, b1, W2, b2) network parameters."""

    expected_size = (
        INPUT_DIM * HIDDEN_DIM
        + HIDDEN_DIM
        + HIDDEN_DIM * OUTPUT_DIM
        + OUTPUT_DIM
    )

    assert genome.ndim == 1
    assert genome.size == expected_size

    idx = 0
    w1_size = INPUT_DIM * HIDDEN_DIM
    W1 = genome[idx : idx + w1_size].copy().reshape(INPUT_DIM, HIDDEN_DIM)

    idx += w1_size
    b1 = genome[idx : idx + HIDDEN_DIM].copy()
    
    idx += HIDDEN_DIM
    w2_size = HIDDEN_DIM * OUTPUT_DIM
    W2 = genome[idx : idx + w2_size].copy().reshape(HIDDEN_DIM, OUTPUT_DIM)

    idx += w2_size
    b2 = genome[idx : idx + OUTPUT_DIM].copy()
    idx += OUTPUT_DIM

    assert idx == genome.size

    return W1, b1, W2, b2


def select_action(
        obs: np.ndarray,
        weights: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> int:
    """Run a forward pass on a 128-byte RAM observation and return an action ID (0-4)."""
    W1, b1, W2, b2 = weights
    x = obs.astype(np.float32) / 255.

    z1 = (x @ W1) + b1
    h = np.tanh(z1)

    logits = (h @ W2) + b2
    best_action = int(np.argmax(logits))

    return best_action



def evaluate_genome(args: tuple[np.ndarray, np.ndarray, int]) -> float:
    """Run one genome across multiple seeded episodes and return its mean fitness."""
    genome, seeds, max_steps = args
    weights = unpack_weights(genome)

    env = gym.make(ENV_NAME, obs_type="ram", repeat_action_probability=0.)
    total_fitness = 0.

    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        ep_score = 0.
        steps = 0

        for _ in range(max_steps):
            action = select_action(obs, weights)
            obs, reward, terminated, truncated, _ = env.step(action)

            ep_score += float(reward)
            steps += 1

            if terminated or truncated:
                break

        ep_fitness = (ep_score * 500.) + (steps * 0.01)
        total_fitness += ep_fitness

    env.close()
    return total_fitness / len(seeds)


def select_parent(
        population: np.ndarray, fitness: np.ndarray, k: int = 4
    ) -> np.ndarray:
    """Select one parent genome using k-way tournament selection."""
    pop_size = population.shape[0]

    candidate_indices = np.random.choice(pop_size, size=k, replace=False)
    candidate_fitness = fitness[candidate_indices]

    best_local_idx = np.argmax(candidate_fitness)
    winner_idx = candidate_indices[best_local_idx]

    return population[winner_idx]


def crossover(
    parent_a: np.ndarray, parent_b: np.ndarray, crossover_rate: float = 0.7
    ) -> np.ndarray:
    """Combine two parent genomes into one child chromosome."""

    if np.random.uniform() >= crossover_rate:
        return parent_a.copy()

    mask = np.random.random(parent_a.shape) < 0.5
    child = np.where(mask, parent_a, parent_b)

    return child


def mutate(
        genome: np.ndarray, mutation_rate: float = 0.05, mutation_std: float = 0.15
    ) -> np.ndarray:
    """Apply random Gaussian perturbations to a subset of genes."""
    mask = np.random.random(genome.shape) < mutation_rate
    num_mutations = np.sum(mask)

    noise = np.random.normal(loc=0., scale=mutation_std, size=num_mutations).astype(np.float32)
    genome[mask] += noise

    return genome


def train(
    pop_size: int = POP_SIZE,
    generations: int = GENERATIONS,
    elite_count: int = ELITE_COUNT,
    episodes_per_eval: int = EPS_PER_EVAL,
    max_steps: int = MAX_STEPS,
) -> None:
    """Run the main evolutionary loop across generations and save the best genome."""
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    rng = np.random.default_rng(SEED)
    population = init_population(pop_size=pop_size, genome_size=GENOME_SIZE)
    best_ever_fitness = float("-inf")
    workers = min(os.cpu_count() or 4, pop_size)
    chunksize = max(1, pop_size // (workers * 4))

    with ProcessPoolExecutor(max_workers=workers) as pool:
        gen_bar = tqdm(range(generations), desc="Evolution", unit="gen")

        for gen in gen_bar:
            gen_seeds = rng.integers(0, 2**32, size=episodes_per_eval)
            tasks = [(genome, gen_seeds, max_steps) for genome in population]

            fitness = np.array(
                list(
                    tqdm(
                        pool.map(evaluate_genome, tasks, chunksize=chunksize),
                        total=pop_size,
                        desc=f"Gen {gen}",
                        unit="ind",
                        leave=False,
                    )
                ),
                dtype=np.float32,
            )

            ranked_idx = np.argsort(fitness)[::-1]
            population = population[ranked_idx]
            fitness = fitness[ranked_idx]

            if fitness[0] > best_ever_fitness:
                best_ever_fitness = float(fitness[0])
                np.save(OUTPUT_PATH, population[0])

            gen_bar.set_postfix(
                best=f"{fitness[0]:.2f}",
                mean=f"{fitness.mean():.2f}",
                record=f"{best_ever_fitness:.2f}",
            )

            next_pop = [ind.copy() for ind in population[:elite_count]]

            while len(next_pop) < pop_size:
                parent_a = select_parent(population=population, fitness=fitness, k=K)
                parent_b = select_parent(population=population, fitness=fitness, k=K)

                child = crossover(
                    parent_a=parent_a, parent_b=parent_b, crossover_rate=CROSS_OVER_RATE
                )
                progress = gen / generations
                current_std = MUTATION_STD * (1.0 - progress) + 0.03 * progress
                child = mutate(
                    child, mutation_rate=MUTATION_RATE, mutation_std=current_std
                )
                next_pop.append(child)

            population = np.array(next_pop, dtype=np.float32)



if __name__ == "__main__":
    train()