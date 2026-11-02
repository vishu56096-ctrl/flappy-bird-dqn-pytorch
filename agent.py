# Importing this package registers FlappyBird-v0 with Gymnasium.
import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import os
import argparse
import random
import time


if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(BASE_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)


class Agent:
    def __init__(self, param_set):
        self.param_set = param_set
        parameter_file = os.path.join(BASE_DIR, "parameters.yaml")
        with open(parameter_file, "r", encoding="utf-8") as file:
            all_param_set = yaml.safe_load(file)
            params = all_param_set[param_set]

        self.env_id = params["env_id"]
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]
        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]
        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]
        self.network_sync_rate = params["network_sync_rate"]
        self.reward_threshold = params["reward_threshold"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        self.log_file = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.episode_log_file = os.path.join(
            RUNS_DIR, f"{self.param_set}_episodes.csv"
        )
        self.model_file = os.path.join(RUNS_DIR, f"{self.param_set}.pt")
        self.latest_model_file = os.path.join(
            RUNS_DIR, f"{self.param_set}_latest.pt"
        )

    def run(
        self, is_training=True, render=False, episodes=None, duration_minutes=None
    ):
        env = gym.make(self.env_id, render_mode="human" if render else None)
        try:
            num_states = env.observation_space.shape[0]
            num_actions = env.action_space.n
            policy_dqn = DQN(num_states, num_actions).to(device)
            episode_offset = 0

            if is_training:
                if os.path.exists(self.latest_model_file):
                    policy_dqn.load_state_dict(
                        torch.load(self.latest_model_file, map_location=device)
                    )
                    print(f"Resuming from {self.latest_model_file}")
                elif os.path.exists(self.model_file):
                    policy_dqn.load_state_dict(
                        torch.load(self.model_file, map_location=device)
                    )
                    print(f"Resuming from best model {self.model_file}")

                memory = ReplayMemory(self.replay_memory_size)
                epsilon = self.epsilon_init
                target_dqn = DQN(num_states, num_actions).to(device)
                target_dqn.load_state_dict(policy_dqn.state_dict())
                steps = 0
                self.optimizer = optim.Adam(
                    policy_dqn.parameters(), lr=self.alpha
                )
                best_reward = float("-inf")
                if os.path.exists(self.log_file):
                    with open(self.log_file, "r", encoding="utf-8") as file:
                        previous_rewards = [
                            float(
                                line.split(" = ", 1)[1].split(
                                    " for episode=", 1
                                )[0]
                            )
                            for line in file
                            if line.startswith("best reward = ")
                        ]
                    if previous_rewards:
                        best_reward = max(previous_rewards)
                if os.path.exists(self.episode_log_file):
                    with open(
                        self.episode_log_file, "r", encoding="utf-8", newline=""
                    ) as file:
                        rows = csv.DictReader(file)
                        for row in rows:
                            episode_offset = int(row["episode"])
            else:
                policy_dqn.load_state_dict(
                    torch.load(self.model_file, map_location=device)
                )
                policy_dqn.eval()

            episode_numbers = (
                range(episodes) if episodes is not None else itertools.count()
            )
            deadline = (
                time.monotonic() + duration_minutes * 60
                if duration_minutes is not None
                else None
            )
            for episode in episode_numbers:
                episode_number = episode_offset + episode + 1
                if deadline is not None and time.monotonic() >= deadline:
                    break

                state, _ = env.reset()
                state = torch.tensor(state, dtype=torch.float, device=device)
                episode_reward = 0
                done = False

                while (
                    not done
                    and episode_reward < self.reward_threshold
                    and (deadline is None or time.monotonic() < deadline)
                ):
                    if is_training and random.random() < epsilon:
                        action = torch.tensor(
                            env.action_space.sample(),
                            dtype=torch.long,
                            device=device,
                        )
                    else:
                        with torch.no_grad():
                            action = (
                                policy_dqn(state.unsqueeze(dim=0))
                                .squeeze(dim=0)
                                .argmax()
                            )

                    next_state, reward, terminated, truncated, _ = env.step(
                        action.item()
                    )
                    done = terminated or truncated
                    episode_reward += reward

                    reward_tensor = torch.tensor(
                        reward, dtype=torch.float, device=device
                    )
                    next_state = torch.tensor(
                        next_state, dtype=torch.float, device=device
                    )
                    if is_training:
                        memory.append(
                            (state, action, next_state, reward_tensor, terminated)
                        )
                        steps += 1
                        if len(memory) >= self.mini_batch_size:
                            mini_batch = memory.sample(self.mini_batch_size)
                            self.optimize(mini_batch, policy_dqn, target_dqn)

                        if steps >= self.network_sync_rate:
                            target_dqn.load_state_dict(policy_dqn.state_dict())
                            steps = 0
                    state = next_state

                if is_training:
                    epsilon = max(
                        epsilon * self.epsilon_decay, self.epsilon_min
                    )
                    is_best = episode_reward > best_reward

                    if is_best:
                        log_msg = (
                            f"best reward = {episode_reward} "
                            f"for episode={episode_number}"
                        )
                        with open(self.log_file, "a", encoding="utf-8") as file:
                            file.write(log_msg + "\n")
                        torch.save(policy_dqn.state_dict(), self.model_file)
                        best_reward = episode_reward

                    torch.save(policy_dqn.state_dict(), self.latest_model_file)
                    write_header = not os.path.exists(
                        self.episode_log_file
                    ) or os.path.getsize(self.episode_log_file) == 0
                    with open(
                        self.episode_log_file, "a", encoding="utf-8", newline=""
                    ) as file:
                        writer = csv.writer(file)
                        if write_header:
                            writer.writerow(
                                [
                                    "episode",
                                    "reward",
                                    "epsilon",
                                    "best_reward_so_far",
                                ]
                            )
                        writer.writerow(
                            [
                                episode_number,
                                episode_reward,
                                epsilon,
                                max(best_reward, episode_reward),
                            ]
                        )

                print(
                    f"episode = {episode_number} "
                    f"reward = {episode_reward} & "
                    f"epsilon={epsilon if is_training else 'n/a'} "
                    f"best={max(best_reward, episode_reward) if is_training else 'n/a'}"
                )

                if is_training and episode_reward >= self.reward_threshold:
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    print(f"Training time limit reached after episode {episode_number}.")
                    break
        except KeyboardInterrupt:
            if is_training:
                torch.save(policy_dqn.state_dict(), self.latest_model_file)
                print(
                    "\nTraining interrupted. Latest model weights saved to "
                    f"{self.latest_model_file}"
                )
            else:
                print("\nEvaluation interrupted.")
        finally:
            env.close()

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, next_states, rewards, terminations = zip(*mini_batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(
            terminations, dtype=torch.float, device=device
        )

        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.gamma * target_dqn(
                next_states
            ).max(dim=1)[0]

        current_q = policy_dqn(states).gather(
            dim=1, index=actions.unsqueeze(dim=1)
        ).squeeze(dim=1)
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and render for five minutes, or use --play to evaluate."
    )
    parser.add_argument("hyperparameters", help="Name of the parameter set.")
    parser.add_argument(
        "--train",
        help="Train and render the agent (this is also the default).",
        action="store_true",
    )
    parser.add_argument(
        "--play",
        help="Evaluate the best saved model without training.",
        action="store_true",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        help="Maximum number of episodes (default: 10000 for training, 5 for evaluation).",
    )
    parser.add_argument(
        "--minutes",
        type=float,
        help="Training time limit in minutes (default: 5).",
    )
    parser.add_argument(
        "--render",
        help="Show the game window (already enabled by default).",
        action="store_true",
    )
    parser.add_argument(
        "--no-render",
        help="Train without opening the game window.",
        action="store_true",
    )
    args = parser.parse_args()

    if args.train and args.play:
        parser.error("--train and --play cannot be used together")
    if args.render and args.no_render:
        parser.error("--render and --no-render cannot be used together")
    is_training = not args.play

    episodes = args.episodes
    if episodes is None:
        episodes = 10000 if is_training else 5
    duration_minutes = args.minutes
    if is_training and duration_minutes is None and args.episodes is None:
        duration_minutes = 5
    if episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.minutes is not None and args.minutes <= 0:
        parser.error("--minutes must be greater than 0")
    if args.minutes is not None and not is_training:
        parser.error("--minutes can only be used while training")

    agent = Agent(param_set=args.hyperparameters)

    if is_training:
        agent.run(
            is_training=True,
            render=not args.no_render,
            episodes=episodes,
            duration_minutes=duration_minutes,
        )
    else:
        agent.run(
            is_training=False,
            render=not args.no_render,
            episodes=episodes,
        )