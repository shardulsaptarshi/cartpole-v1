import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# ============================================================
# Hyperparameters
# ============================================================
LEARNING_RATE = 1e-3
GAMMA = 0.99              # discount factor
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995     # multiplied each episode
BUFFER_SIZE = 10000
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 10   # update target net every N episodes
NUM_EPISODES = 1500

LEARNING_RATE = 5e-4       # was 1e-3 — slower learning, less overshooting
TARGET_UPDATE_FREQ = 20    # was 10 — more stable target
BUFFER_SIZE = 50000        # was 10000 — keeps more diverse experience
# ============================================================
# Q-Network: maps state -> Q-values for each action
# ============================================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Replay Buffer: stores past transitions for off-policy training
# ============================================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
# DQN Agent
# ============================================================
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.epsilon = EPSILON_START

    def select_action(self, state):
        # Epsilon-greedy: explore with probability epsilon, else act greedily
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def train_step(self):
        # Don't train until we have enough samples
        if len(self.buffer) < BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        # Q(s, a) for actions actually taken
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Bellman target: r + gamma * max_a' Q_target(s', a'), zeroed if done
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]
            target = rewards + GAMMA * next_q * (1 - dones)

        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


# ============================================================
# Training Loop
# ============================================================
def train():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)

    episode_rewards = []
    best_avg = -float("inf")  # track best moving average

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.buffer.push(state, action, reward, next_state, float(done))
            agent.train_step()
            state = next_state
            total_reward += reward

        agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)

        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target()

        episode_rewards.append(total_reward)

        # Save the BEST model, not the final one
        if len(episode_rewards) >= 50:
            current_avg = np.mean(episode_rewards[-50:])
            if current_avg > best_avg:
                best_avg = current_avg
                torch.save(agent.q_net.state_dict(), "cartpole_dqn.pth")

        if episode % 10 == 0:
            avg = np.mean(episode_rewards[-50:]) if episode_rewards else 0
            print(f"Episode {episode:4d} | Reward: {total_reward:6.1f} | "
                  f"Avg(50): {avg:6.1f} | Best: {best_avg:6.1f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

        # Early stopping when solved
        if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) >= 475:
            print(f"\nSolved at episode {episode}! Stopping early.")
            break

    env.close()

    print(f"\nBest 50-ep average: {best_avg:.1f}")
    print("Best model saved to cartpole_dqn.pth")

    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, alpha=0.5, label="Episode reward")
    if len(episode_rewards) >= 50:
        moving_avg = np.convolve(episode_rewards, np.ones(50) / 50, mode="valid")
        plt.plot(range(49, len(episode_rewards)), moving_avg, label="50-ep moving avg")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN on CartPole-v1")
    plt.legend()
    plt.savefig("training_curve.png")
    plt.show()

if __name__ == "__main__":
    train()