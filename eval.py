"""VLA-0: Evaluation on LIBERO environments."""

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from train import VLA0

# LIBERO imports (will be available after installing libero)
try:
    import libero
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    LIBERO_AVAILABLE = True
except ImportError as e:
    LIBERO_AVAILABLE = False
    print(f"Warning: LIBERO import failed: {e}")
    import traceback
    traceback.print_exc()


# ============================================================================
# Action Ensemble
# ============================================================================

class ActionEnsemble:
    """Temporal ensemble of action predictions."""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def add(self, actions: np.ndarray):
        """Add a new action sequence (10, 7)."""
        self.buffer.append(actions)

    def get_current_action(self) -> np.ndarray:
        """Average all predictions for current timestep."""
        if len(self.buffer) == 0:
            return np.zeros(7, dtype=np.float32)

        # Each prediction in buffer is (10, 7)
        # We want prediction[i][0] for i-th prediction (its current timestep)
        current_actions = []
        for i, pred in enumerate(self.buffer):
            # Use the (len(buffer) - 1 - i)th timestep from this prediction
            timestep = len(self.buffer) - 1 - i
            if timestep < len(pred):
                current_actions.append(pred[timestep])

        if len(current_actions) == 0:
            return np.zeros(7, dtype=np.float32)

        # Average
        return np.mean(current_actions, axis=0).astype(np.float32)

    def reset(self):
        """Clear buffer."""
        self.buffer.clear()


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_libero(model_path: str, suite: str = 'libero_spatial', n_episodes: int = 50):
    """Evaluate VLA-0 on LIBERO benchmark."""

    if not LIBERO_AVAILABLE:
        print("LIBERO not installed. Cannot evaluate.")
        return

    # Load model
    print(f"Loading model from {model_path}")
    vla = VLA0()

    checkpoint = torch.load(model_path, map_location='cuda', weights_only=False)
    vla.model.load_state_dict(checkpoint['model_state_dict'])
    vla.model.eval()

    # Load benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite]()

    print(f"\nEvaluating on {suite}")
    print(f"Tasks: {task_suite.n_tasks}")
    print(f"Episodes per task: {n_episodes}")

    # Evaluate each task
    all_results = {}

    for task_id in range(task_suite.n_tasks):
        task_name = task_suite.get_task(task_id).name
        task_desc = task_suite.get_task_demonstration(task_id)[0]  # Get instruction

        print(f"\n{'='*60}")
        print(f"Task {task_id}: {task_name}")
        print(f"Description: {task_desc}")
        print(f"{'='*60}")

        successes = []

        for episode in tqdm(range(n_episodes), desc=f"Task {task_id}"):
            # Create environment
            env_args = {
                "bddl_file_name": task_suite.get_task_bddl_file_path(task_id),
                "camera_heights": 128,
                "camera_widths": 128,
            }
            env = OffScreenRenderEnv(**env_args)
            env.seed(episode)

            # Reset
            obs = env.reset()
            ensemble = ActionEnsemble(window_size=10)

            done = False
            steps = 0
            max_steps = 300

            while not done and steps < max_steps:
                # Get images
                agentview = obs['agentview_image']  # (128, 128, 3)
                eyeinhand = obs['robot0_eye_in_hand_image']  # (128, 128, 3)

                img1 = Image.fromarray(agentview)
                img2 = Image.fromarray(eyeinhand)

                # Predict actions
                action_sequences, _ = vla.predict([img1], [img2], [task_desc])
                action_seq = action_sequences[0]  # (10, 7)

                # Add to ensemble
                ensemble.add(action_seq)
                action = ensemble.get_current_action()

                # Step environment
                obs, reward, done, info = env.step(action)
                steps += 1

            # Check success
            success = done and info.get('success', False)
            successes.append(1.0 if success else 0.0)

            env.close()

        # Compute success rate
        success_rate = np.mean(successes) * 100
        all_results[task_name] = success_rate

        print(f"\n{task_name}: {success_rate:.1f}% ({int(np.sum(successes))}/{n_episodes})")

    # Overall results
    print(f"\n{'='*60}")
    print(f"Overall Results - {suite}")
    print(f"{'='*60}")

    for task_name, success_rate in all_results.items():
        print(f"{task_name:40s}: {success_rate:5.1f}%")

    overall_success = np.mean(list(all_results.values()))
    print(f"\n{'Mean Success Rate':40s}: {overall_success:5.1f}%")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--suite', type=str, default='libero_spatial',
                        choices=['libero_spatial', 'libero_object', 'libero_goal',
                                'libero_10', 'libero_90'],
                        help='LIBERO benchmark suite')
    parser.add_argument('--n-episodes', type=int, default=50,
                        help='Episodes per task')

    args = parser.parse_args()

    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return

    evaluate_libero(args.checkpoint, args.suite, args.n_episodes)


if __name__ == '__main__':
    main()
