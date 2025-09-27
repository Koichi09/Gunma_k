import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from minigrid.wrappers import ImgObsWrapper, ActionBonus, OneHotPartialObsWrapper

import random
import math
import numpy as np
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ===================== 追加: デバイス統一 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

################################################################################################

#--- 経験を保存するためのデータ構造 ---
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminated'))

# --- リプレイバッファ ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.stage_memories = {}  # 各ステージの経験を保持
        self.current_stage = 0

class QNetwork(nn.Module):
    def __init__(self, obs_space_shape, action_space_n):
        super(QNetwork, self).__init__()
        h, w, c = obs_space_shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            cnn_out_size = self.cnn(torch.zeros(1, c, h, w)).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_size, 256), nn.ReLU(),
            nn.Linear(256, action_space_n)
        )
    def forward(self, x):
        x = x.to(DEVICE).float() / 255.0
        x = x.permute(0, 3, 1, 2)
        return self.fc(self.cnn(x))

# --- DQNエージェント ---
class DQNAgent:
    def __init__(self, obs_space_shape, action_space_n):
        self.action_space_n = action_space_n
        self.policy_net = QNetwork(obs_space_shape, action_space_n).to(DEVICE)
        self.target_net = QNetwork(obs_space_shape, action_space_n).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.update_steps = 0  # ターゲット同期のステップカウンタ（学習ステップ基準）
        self.current_stage = 0
        self.episode_count = 0

    def select_action(self, state):
        state = self._ensure_tensor(state)

        # greedy
        with torch.no_grad():
            q_values = self.policy_net(state)
            return int(q_values.argmax(dim=-1).item())         # ★ int を返す

    def _ensure_tensor(self, state):
        """NumPy/torch どちらでも受けて [B,H,W,C] float32, 0-1 に揃える"""
        if isinstance(state, np.ndarray):
            t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE)
        else:
            t = state.to(DEVICE).float()
        if t.ndim == 3:                 # [H,W,C]
            t = t.unsqueeze(0)          # [1,H,W,C]
        # 値域が 0..255 なら正規化（簡易判定）
        if t.max() > 1.5:
            t = t / 255.0
        return t

    def load_model(self, filepath):
        """モデルとエージェントの状態を読み込み（torch.load使用）"""
        save_data = torch.load(filepath, map_location=DEVICE, weights_only=False)
        
        # モデルの状態を復元
        self.policy_net.load_state_dict(save_data['policy_net_state_dict'])
        self.target_net.load_state_dict(save_data['target_net_state_dict'])
        self.optimizer.load_state_dict(save_data['optimizer_state_dict'])
        
        # エージェントの状態を復元
        self.update_steps = save_data['update_steps']
        self.current_stage = save_data['current_stage']
        self.episode_count = save_data['episode_count']
        
        # リプレイバッファを復元
        self.replay_buffer.memory = deque(save_data['replay_buffer_memory'], maxlen=self.replay_buffer.memory.maxlen)
        self.replay_buffer.stage_memories = {k: deque(v, maxlen=self.replay_buffer.memory.maxlen//3) 
                                           for k, v in save_data['replay_buffer_stage_memories'].items()}
        self.replay_buffer.current_stage = save_data['replay_buffer_current_stage']
        
        print(f"Model loaded from {filepath}")
        print(f"Loaded: current_stage={self.current_stage + 1}, episode_count={self.episode_count}")
    
    @classmethod
    def create_from_saved(cls, filepath, obs_space_shape):
        """保存されたモデルから新しいエージェントを作成"""
        save_data = torch.load(filepath, map_location=DEVICE, weights_only=False)
        
        # 新しいエージェントを作成
        agent = cls(obs_space_shape, save_data['action_space_n'])
        
        # 状態を復元
        agent.load_model(filepath)
        
        return agent

################################################################################################

# DQNエージェントの設定
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 1250
REPLAY_BUFFER_SIZE = 25000
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
TARGET_UPDATE_FREQ = 50

MODEL_SAVE_PATH = "models/DQN_without_PER_stage2_final.pt"
ACTION_BONUS = False
ONE_HOT_ENCODE = True

# ① 可視化用のフル視点環境（rgb_array）
vis_env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")

# ② 推論用の観測（エージェント視点）環境
agent_env = gym.make("MiniGrid-Empty-16x16-v0")
if ONE_HOT_ENCODE:
    agent_env = OneHotPartialObsWrapper(agent_env)
if ACTION_BONUS:
    agent_env = ActionBonus(agent_env)
agent_env = ImgObsWrapper(agent_env)

# 録画ラッパ（フレームは vis_env から取る）
vis_env = RecordVideo(vis_env, "videos", name_prefix="side_by_side")

obs_shape = agent_env.observation_space.shape
action_n = 3  # env.action_space.n でもOK

print(f"Loading existing model from {MODEL_SAVE_PATH}")
agent = DQNAgent.create_from_saved(MODEL_SAVE_PATH, obs_shape)

obs_agent, _ = agent_env.reset(seed=0)
obs_vis, _   = vis_env.reset(seed=0)

done = False
while not done:
    # --- 観測をTensor化（0-1正規化も内部で対応） ---
    action = agent.select_action(obs_agent)  # ★ int を返す

    # 同じ action を両方に適用
    obs_agent, r, term1, trunc1, _ = agent_env.step(action)
    obs_vis,   r2, term2, trunc2, _ = vis_env.step(action)
    done = (term1 or trunc1)

vis_env.close()
agent_env.close()
plt.show()