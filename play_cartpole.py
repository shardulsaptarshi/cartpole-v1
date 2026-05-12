import gymnasium as gym
import torch
from dqn_cartpole import QNetwork

env = gym.make("CartPole-v1", render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q_net = QNetwork(state_dim, action_dim)
q_net.load_state_dict(torch.load("cartpole_dqn.pth"))
q_net.eval()

for episode in range(5):
    state, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        with torch.no_grad():
            action = int(q_net(torch.FloatTensor(state).unsqueeze(0)).argmax().item())
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
    print(f"Episode {episode}: reward = {total_reward}")

env.close()