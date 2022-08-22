import sys
IN_COLAB = "google.colab" in sys.modules

import os
import random
from typing import Dict, List, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
import matplotlib.pyplot as plt

if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed = 777
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(
        self, 
        state_size: int, 
        size: int, 
        batch_size: int = 32, 
    ):
        """Initialize."""
        self.obs_buf = np.zeros([size, state_size], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, state_size], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size = 0, 0

    def store(
        self, 
        obs: np.ndarray, 
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ):
        """Store the transition in buffer."""
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """Randomly sample a batch of experiences from memory."""
        indices = np.random.choice(self.size, size=self.batch_size, replace=False)

        return dict(
            obs=self.obs_buf[indices],
            next_obs=self.next_obs_buf[indices],
            acts=self.acts_buf[indices],
            rews=self.rews_buf[indices],
            done=self.done_buf[indices],
        )

    def __len__(self) -> int:
        return self.size

class Network(nn.Module):
    def __init__(self, state_size: int, action_size: int, 
    ):
        """Initialization."""
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_size, 128), 
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU(), 
            nn.Linear(128, action_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)

class DQNAgent:
    """DQN Agent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        epsilon (float): parameter for epsilon greedy policy
        epsilon_decay (float): step size to decrease epsilon
        max_epsilon (float): max value of epsilon
        min_epsilon (float): min value of epsilon
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including 
                           state, action, reward, next_state, done
    """

    def __init__(
        self, 
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        epsilon_decay: float,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.99,
    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            epsilon_decay (float): step size to decrease epsilon
            lr (float): learning rate
            max_epsilon (float): max value of epsilon
            min_epsilon (float): min value of epsilon
            gamma (float): discount factor
        """
        self.env = env
        
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma
        
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        
        # networks: dqn, dqn_target
        self.dqn = Network(self.state_size, self.action_size
                          ).to(self.device)
        self.dqn_target = Network(self.state_size, self.action_size
                          ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        self.memory = ReplayBuffer(
            self.state_size, memory_size, batch_size
        )
        
        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())
        
        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        self.is_test = False

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # epsilon greedy policy
        if self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.dqn(
                torch.FloatTensor(state).to(self.device)
            ).argmax()
            selected_action = selected_action.detach().cpu().numpy()
        
        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], 
    ) -> torch.Tensor:
        """Return dqn loss."""
        device     = self.device  # for shortening the following lines
        state      = torch.FloatTensor(samples["obs"]).to(device)
        action     = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward     = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        done       = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        
        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_Qs = self.dqn(state).gather(1, action)
        next_Q_targs = self.dqn_target(
            next_state
        ).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target_value = (reward + self.gamma * next_Q_targs * mask).to(self.device)

        # calculate dqn loss
        loss = F.smooth_l1_loss(curr_Qs, target_value)

        return loss

        
    def train_step(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        samples = self.memory.sample_batch()
        
        loss = self._compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
        return loss.item()
        
    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

# environment
env_name = "CartPole-v0"
env = gym.make(env_name)

# parameters
memory_size = 2000
target_update = 100
epsilon_decay = 1 / 2000
initial_random_steps = 5000

max_episodes = 300
batch_size = 32

# train
agent = DQNAgent(
    env, 
    memory_size, 
    batch_size, 
    target_update, 
    epsilon_decay,
)

if __name__ == "__main__":
    
    """Train the agent."""
    agent.is_test = False
    
    update_cnt    = 0
    epsilons      = []
    losses        = []
    scores        = []
    
    for episode in range(max_episodes):
        state = agent.env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state)
            # next_state, reward, done = agent.step(action)
            """Take an action and return the response of the env."""
            next_state, reward, done, _ = agent.env.step(action)
            agent.transition += [reward, next_state, done]
            agent.memory.store(*agent.transition)
            
            state = next_state
            episode_reward += reward

            # if episode ends
            if done:
                scores.append(episode_reward)
                print("Episode " + str(episode+1) + ": " + str(episode_reward))
                
            # if training is ready
            if (len(agent.memory) >= agent.batch_size):
                loss = agent.train_step()
                losses.append(loss)
                update_cnt += 1

                # linearly decrease epsilon
                agent.epsilon = max(
                    agent.min_epsilon, agent.epsilon - (
                        agent.max_epsilon - agent.min_epsilon
                    ) * agent.epsilon_decay
                )
                epsilons.append(agent.epsilon)

                # if hard update is needed
                if update_cnt % agent.target_update == 0:
                    agent._target_hard_update()
    
