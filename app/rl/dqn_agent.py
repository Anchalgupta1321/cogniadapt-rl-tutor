import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Any

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class QNetwork(nn.Module):
    """Deep Q-Network for Student Difficulty Policy Optimization."""
    def __init__(self, state_dim: int = 8, hidden_dim: int = 64, action_dim: int = 3):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ReplayBuffer:
    """Experience Replay Buffer for storing and sampling transition tuples."""
    def __init__(self, capacity: int = 2000):
        self.capacity = capacity
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.position = 0

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    Deep Q-Network Agent with Target Network, Experience Replay, 
    and Bellman Optimization for Student Trajectory Learning.
    """
    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 3,
        lr: float = 0.001,
        gamma: float = 0.95,
        epsilon_start: float = 0.9,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        model_dir: str = "data/dqn_model"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.model_dir = model_dir
        
        # Policy & Target Networks
        self.policy_net = QNetwork(state_dim, 64, action_dim)
        self.target_net = QNetwork(state_dim, 64, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss() # Huber loss
        self.memory = ReplayBuffer(capacity=5000)
        
        self.total_steps = 0
        self.recent_losses: List[float] = []
        self.episode_rewards: List[float] = []

        os.makedirs(self.model_dir, exist_ok=True)
        self.load_model()

    def get_state_vector(
        self,
        current_diff_idx: int,
        recent_acc: float,
        streak: int,
        attempts_count: int,
        topic_accuracy: float = 0.5,
        avg_response_time: float = 5.0
    ) -> np.ndarray:
        """Constructs an 8-dimensional continuous state vector."""
        vec = np.array([
            float(current_diff_idx) / 2.0,                  # Normalized current difficulty [0, 1]
            float(recent_acc),                               # Recent overall accuracy [0, 1]
            np.clip(float(streak) / 5.0, -1.0, 1.0),        # Normalized streak [-1, 1]
            min(float(attempts_count) / 20.0, 1.0),          # Normalized attempt count [0, 1]
            float(topic_accuracy),                           # Topic-specific mastery score [0, 1]
            min(float(avg_response_time) / 30.0, 1.0),       # Response latency indicator [0, 1]
            1.0 if streak > 0 else 0.0,                     # Hot streak boolean flag
            1.0 if streak < 0 else 0.0                      # Cold streak boolean flag
        ], dtype=np.float32)
        return vec

    def select_action(self, state_vec: np.ndarray, evaluate: bool = False) -> int:
        """Selects action using Epsilon-Greedy or greedy policy during evaluation."""
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state_vec).unsqueeze(0)
            q_values = self.policy_net(state_t)
            return int(torch.argmax(q_values).item())

    def update_policy(self, batch_size: int = 32) -> float:
        """Performs one step of PyTorch gradient descent on the policy network."""
        if len(self.memory) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        # Compute Q(s, a)
        q_values = self.policy_net(states)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute max Q(s', a') with Target Network
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q = next_q_values.max(1)[0]
            expected_state_action_values = rewards + (self.gamma * max_next_q * (1 - dones))

        # Compute Huber Loss
        loss = self.criterion(state_action_values, expected_state_action_values)

        # Optimize policy
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.total_steps += 1

        # Soft update target network every 10 steps
        if self.total_steps % 10 == 0:
            self.soft_update_target()

        loss_val = float(loss.item())
        self.recent_losses.append(loss_val)
        if len(self.recent_losses) > 100:
            self.recent_losses.pop(0)

        if self.total_steps % 50 == 0:
            self.save_model()

        return loss_val

    def soft_update_target(self, tau: float = 0.05):
        """Soft update target network weights: theta_target = tau*theta_local + (1-tau)*theta_target."""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def save_model(self):
        """Saves weights and training state."""
        try:
            model_path = os.path.join(self.model_dir, "dqn_policy.pt")
            meta_path = os.path.join(self.model_dir, "dqn_meta.json")
            torch.save(self.policy_net.state_dict(), model_path)
            with open(meta_path, "w") as f:
                json.dump({
                    "total_steps": self.total_steps,
                    "epsilon": self.epsilon,
                    "replay_buffer_size": len(self.memory),
                    "recent_loss_avg": float(np.mean(self.recent_losses)) if self.recent_losses else 0.0
                }, f, indent=2)
        except Exception as e:
            print(f"[DQN] Failed to save model: {e}")

    def load_model(self):
        """Loads model weights if available."""
        model_path = os.path.join(self.model_dir, "dqn_policy.pt")
        meta_path = os.path.join(self.model_dir, "dqn_meta.json")
        if os.path.exists(model_path):
            try:
                self.policy_net.load_state_dict(torch.load(model_path))
                self.target_net.load_state_dict(self.policy_net.state_dict())
                if os.path.exists(meta_path):
                    with open(meta_path, "r") as f:
                        data = json.load(f)
                        self.total_steps = data.get("total_steps", 0)
                        self.epsilon = data.get("epsilon", self.epsilon)
                print(f"[DQN] Successfully loaded PyTorch model from {model_path}")
            except Exception as e:
                print(f"[DQN] Model load error: {e}")

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns deep RL diagnostics metrics."""
        sample_states = torch.eye(8)[:3] # Sample standard state vectors
        with torch.no_grad():
            q_preds = self.policy_net(sample_states).tolist()

        return {
            "algorithm": "Deep Q-Network (DQN) with PyTorch",
            "state_dimension": self.state_dim,
            "action_dimension": self.action_dim,
            "total_steps": self.total_steps,
            "epsilon": round(self.epsilon, 4),
            "replay_buffer_occupancy": f"{len(self.memory)}/{self.memory.capacity}",
            "average_huber_loss": round(float(np.mean(self.recent_losses)), 6) if self.recent_losses else 0.0,
            "sample_q_values": q_preds
        }

# Global DQN Singleton
dqn_agent = DQNAgent()
