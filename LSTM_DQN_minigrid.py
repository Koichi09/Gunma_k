import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from minigrid.wrappers import ImgObsWrapper

import random
import math
import numpy as np
from collections import deque, namedtuple
import matplotlib.pyplot as plt

# --- ハイパーパラメータ設定 ---
# カリキュラム学習の設定
MAZE_SIZES = [5, 8, 16]
EPISODES_PER_STAGE = 1000

# DQNエージェントの設定
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 30000
REPLAY_BUFFER_SIZE = 50000
BATCH_SIZE = 128
LEARNING_RATE = 3e-5 
TARGET_UPDATE_FREQ = 100

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#--- 経験を保存するためのデータ構造 ---
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward', 'terminated'))

# LSTM用のシーケンス経験
SequenceTransition = namedtuple('SequenceTransition',
                               ('state_sequence', 'action_sequence', 'reward_sequence', 'terminated_sequence'))

# --- リプレイバッファ ---
class ReplayBuffer:
    def __init__(self, capacity, sequence_length=10):
        self.memory = deque([], maxlen=capacity)
        self.sequence_length = sequence_length
        self.episode_buffer = []  # 現在のエピソードの経験を一時保存

    def push(self, *args):
        """単一の遷移を保存"""
        self.episode_buffer.append(Transition(*args))

    def end_episode(self):
        """エピソード終了時にシーケンスを保存"""
        if len(self.episode_buffer) >= self.sequence_length:
            # シーケンスを抽出
            for i in range(len(self.episode_buffer) - self.sequence_length + 1):
                sequence = self.episode_buffer[i:i + self.sequence_length]
                
                # 各シーケンスの形状:
                # t.state: [1, h, w, c] -> squeeze(0) -> [h, w, c]
                # t.action: [1, 1] -> squeeze(0) -> [1] 
                # t.reward: [1] -> [1]
                # t.terminated: [1] -> [1]
                state_seq = torch.stack([t.state.squeeze(0) for t in sequence])  # [seq_len, h, w, c]
                action_seq = torch.stack([t.action.squeeze(0) for t in sequence])  # [seq_len, 1]
                reward_seq = torch.stack([t.reward for t in sequence])  # [seq_len, 1]
                terminated_seq = torch.stack([t.terminated for t in sequence])  # [seq_len, 1]
                
                self.memory.append(SequenceTransition(
                    state_seq, action_seq, reward_seq, terminated_seq
                ))
        
        self.episode_buffer = []  # エピソードバッファをクリア

    def sample(self, batch_size):
        """シーケンスをサンプリング"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# --- CNN+LSTMベースのQネットワーク ---
class QNetwork(nn.Module):
    def __init__(self, obs_space_shape, action_space_n, hidden_dim=256):
        super(QNetwork, self).__init__()
        h, w, c = obs_space_shape
        self.hidden_dim = hidden_dim
        
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, stride=1, padding=1), # 3*7*7 -> 16*7*7
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # 16*7*7 -> 32*7*7
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 32*7*7 -> 64*7*7
            nn.ReLU(),
            nn.Flatten(), # 7*7*64 -> 3136
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w) 
            cnn_out_size = self.cnn(dummy_input).shape[1] # 3136

        # LSTMで時系列情報を処理
        self.lstm = nn.LSTM(cnn_out_size, hidden_dim, batch_first=True)
        
        # 行動価値の出力層
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_space_n)
        )

    def forward(self, x, hidden=None):
        """
        Args:
            x: 状態テンソル [batch_size, height, width, channels] または [batch_size, seq_len, height, width, channels]
            hidden: LSTMの隠れ状態 (h, c)
        """
        x = x.to(device).float() / 255.0 #正規化の必要あるか？
        
        # バッチ次元を確認
        if len(x.shape) == 4:  # 単一状態
            x = x.unsqueeze(1)  # [batch_size, 1, height, width, channels]
            single_state = True
        else:  # シーケンス状態
            single_state = False
            
        batch_size, seq_len = x.shape[0], x.shape[1]
        x = x.view(batch_size * seq_len, *x.shape[2:])  # [batch_size*seq_len, height, width, channels]
        x = x.permute(0, 3, 1, 2)  # [batch_size*seq_len, channels, height, width]
        
        # CNNで特徴抽出
        cnn_out = self.cnn(x)  # [batch_size*seq_len, cnn_out_size]
        cnn_out = cnn_out.view(batch_size, seq_len, -1)  # [batch_size, seq_len, cnn_out_size]
        
        # LSTMで時系列処理
        if hidden is None:
            hidden = self.init_hidden(batch_size)  # (h, c) 各々 [1, batch_size, hidden_dim]
            
        lstm_out, new_hidden = self.lstm(cnn_out, hidden)  # [batch_size, seq_len, hidden_dim]
        
        # 最後のタイムステップの出力を使用
        if single_state:
            lstm_out = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
        else:
            lstm_out = lstm_out  # [batch_size, seq_len, hidden_dim]
        
        # Q値を計算
        q_values = self.fc(lstm_out)  # [batch_size, action_space_n] または [batch_size, seq_len, action_space_n]
        
        return q_values, new_hidden
    
    def init_hidden(self, batch_size=1):
        h = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return (h, c)

# --- DQNエージェント ---
class DQNAgent:
    def __init__(self, obs_space_shape, action_space_n):
        self.action_space_n = action_space_n
        self.policy_net = QNetwork(obs_space_shape, action_space_n).to(device)
        self.target_net = QNetwork(obs_space_shape, action_space_n).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.steps_done = 0
        
        # LSTMの隠れ状態を管理
        self.hidden_state = None
        self.target_hidden_state = None

    def select_action(self, state):
        eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                        math.exp(-1. * self.steps_done / EPSILON_DECAY)
        self.steps_done += 1
        
        if random.random() > eps_threshold:
            with torch.no_grad():
                q_values, self.hidden_state = self.policy_net(state, self.hidden_state)
                return q_values.max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_space_n)]], device=device, dtype=torch.long)
    
    def reset_hidden_state(self):
        """エピソード開始時に隠れ状態をリセット"""
        self.hidden_state = None
        self.target_hidden_state = None

    def update_model(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        sequences = self.replay_buffer.sample(BATCH_SIZE)
        
        # シーケンスをバッチに変換
        # 各シーケンスの形状:
        # seq.state_sequence: [seq_len, h, w, c]
        # seq.action_sequence: [seq_len, 1]
        # seq.reward_sequence: [seq_len, 1] 
        # seq.terminated_sequence: [seq_len, 1]
        state_sequences = torch.stack([seq.state_sequence for seq in sequences])  # [batch_size, seq_len, h, w, c]
        action_sequences = torch.stack([seq.action_sequence for seq in sequences])  # [batch_size, seq_len, 1]
        reward_sequences = torch.stack([seq.reward_sequence for seq in sequences])  # [batch_size, seq_len, 1]
        terminated_sequences = torch.stack([seq.terminated_sequence for seq in sequences])  # [batch_size, seq_len, 1]

        # 現在のQ値を計算（LSTMの隠れ状態を考慮）
        current_q_values, _ = self.policy_net(state_sequences)  # [batch_size, seq_len, action_space_n]
        current_q_values = current_q_values.gather(2, action_sequences)  # [batch_size, seq_len, 1]

        # ターゲットQ値を計算
        with torch.no_grad():
            next_q_values, _ = self.target_net(state_sequences)  # [batch_size, seq_len, action_space_n]
            next_q_values = next_q_values.max(2)[0]  # [batch_size, seq_len]
            
            # ベルマン方程式でターゲットを計算
            # reward_sequences.squeeze(-1): [batch_size, seq_len]
            # next_q_values: [batch_size, seq_len]
            # terminated_sequences.squeeze(-1): [batch_size, seq_len]
            target_q_values = reward_sequences.squeeze(-1) + (GAMMA * next_q_values * (~ terminated_sequences.squeeze(-1)))  # [batch_size, seq_len]

        # 損失計算（シーケンス全体で平均）
        # current_q_values.squeeze(-1): [batch_size, seq_len]
        # target_q_values: [batch_size, seq_len]
        loss = F.mse_loss(current_q_values.squeeze(-1), target_q_values)  # スカラー

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# --- メインの学習ループ ---
if __name__ == "__main__":
    agent = None
    all_stage_rewards = {}

    for stage, size in enumerate(MAZE_SIZES):
        print(f"--- Curriculum Stage {stage + 1}: {size}x{size} Maze ---")
        env = gym.make(f'MiniGrid-Empty-{size}x{size}-v0')
        env = ImgObsWrapper(env)
        
        if agent is None:
            obs_shape = env.observation_space.shape
            action_n = env.action_space.n
            agent = DQNAgent(obs_shape, action_n)

        stage_rewards = []
        for episode in range(EPISODES_PER_STAGE):
            # エピソード開始時にLSTMの隠れ状態をリセット
            agent.reset_hidden_state()
            
            obs, info = env.reset()
            state = torch.from_numpy(obs).unsqueeze(0).to(device).float()  # [1, h, w, c]
            
            terminated, truncated = False, False
            episode_reward = 0
            while not terminated and not truncated:
                action = agent.select_action(state)  # [1, 1]
                obs, reward, terminated, truncated, info = env.step(action.item())
                episode_reward += reward

                next_state = torch.tensor(obs, device=device).unsqueeze(0).float() if not terminated else None  # [1, h, w, c] または None
                
                # リプレイバッファに保存するデータの形状:
                # state: [1, h, w, c]
                # action: [1, 1] 
                # next_state: [1, h, w, c] または None
                # reward: [1]
                # terminated: [1]
                agent.replay_buffer.push(state, action, next_state, 
                                         torch.tensor([reward], device=device), 
                                         torch.tensor(terminated, device=device))
                state = next_state
                agent.update_model()

            # エピソード終了時にシーケンスを保存
            agent.replay_buffer.end_episode()
            stage_rewards.append(episode_reward)

            if (episode + 1) % TARGET_UPDATE_FREQ == 0:
                agent.sync_target_network()
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(stage_rewards[-100:])
                print(f"Stage {stage+1}, Episode {episode+1}, Avg Reward (last 100): {avg_reward:.2f}")
        
        all_stage_rewards[f"{size}x{size}"] = stage_rewards
        print(f"--- Stage {stage + 1} Complete ---")

    env.close()

    # --- 結果のプロット ---
    plt.figure(figsize=(12, 7))
    total_episodes = 0
    for size, rewards in all_stage_rewards.items():
        moving_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
        episodes = np.arange(total_episodes, total_episodes + len(moving_avg))
        plt.plot(episodes, moving_avg, label=f'Stage: {size}')
        total_episodes += len(rewards)
    
    plt.title("DQN with Curriculum Learning Performance")
    plt.xlabel("Total Episodes")
    plt.ylabel("Average Reward (Moving Avg over 100 episodes)")
    plt.legend()
    plt.grid(True)
    plt.savefig("dqn_curriculum_performance.png")
    plt.show()