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
MAZE_SIZES = [5] #[5, 8, 16]
EPISODES_PER_STAGE = 100000 #10000

# DQNエージェントの設定
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 100000
REPLAY_BUFFER_SIZE = 50000
BATCH_SIZE = 64
LEARNING_RATE = 1e-4 
TARGET_UPDATE_FREQ = 1000

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#--- 経験を保存するためのデータ構造 ---
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward', 'terminated'))

# LSTM用のシーケンス経験（有効マスクを追加）
SequenceTransition = namedtuple('SequenceTransition',
                               ('state_sequence', 'action_sequence', 'reward_sequence', 'terminated_sequence', 'valid_sequence_mask'))

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
        ep_len = len(self.episode_buffer)
        if ep_len >= self.sequence_length:
            # スライディングウィンドウで固定長シーケンスを抽出
            for i in range(ep_len - self.sequence_length + 1):
                sequence = self.episode_buffer[i:i + self.sequence_length]
                state_seq = torch.stack([t.state.squeeze(0) for t in sequence])
                action_seq = torch.stack([t.action.squeeze(0) for t in sequence])
                reward_seq = torch.stack([t.reward for t in sequence])
                terminated_seq = torch.stack([t.terminated for t in sequence])
                valid_mask = torch.ones(self.sequence_length, dtype=torch.bool, device=state_seq.device)
                self.memory.append(SequenceTransition(state_seq, action_seq, reward_seq, terminated_seq, valid_mask))
        elif ep_len > 0:
            # 短いエピソードは末尾の遷移を繰り返してパディングして固定長にする
            last_transition = self.episode_buffer[-1]
            padded = list(self.episode_buffer)
            while len(padded) < self.sequence_length:
                # 末尾遷移を複製（報酬0、terminated=True を維持）
                pad_state = last_transition.state
                pad_action = last_transition.action
                pad_reward = torch.zeros_like(last_transition.reward)
                pad_terminated = torch.tensor(True, device=pad_state.device)
                padded.append(Transition(pad_state, pad_action, last_transition.next_state, pad_reward, pad_terminated))
            state_seq = torch.stack([t.state.squeeze(0) for t in padded])
            action_seq = torch.stack([t.action.squeeze(0) for t in padded])
            reward_seq = torch.stack([t.reward for t in padded])
            terminated_seq = torch.stack([t.terminated for t in padded])
            valid_mask = torch.zeros(self.sequence_length, dtype=torch.bool, device=state_seq.device)
            valid_mask[:ep_len] = True
            self.memory.append(SequenceTransition(state_seq, action_seq, reward_seq, terminated_seq, valid_mask))

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
        self.update_steps = 0  # ターゲット同期のステップカウンタ（学習ステップ基準）
        
        # LSTMの隠れ状態を管理
        self.hidden_state = None
        self.target_hidden_state = None

    def select_action(self, state):
        eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                        math.exp(-1.0 * self.steps_done / EPSILON_DECAY)
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
        # seq.state_sequence: [seq_len, h, w, c]
        # seq.action_sequence: [seq_len, 1]
        # seq.reward_sequence: [seq_len, 1]
        # seq.terminated_sequence: [seq_len] or [seq_len, 1]
        state_sequences = torch.stack([seq.state_sequence for seq in sequences]).to(device)  # [B, L, H, W, C]
        action_sequences = torch.stack([seq.action_sequence for seq in sequences]).to(device)  # [B, L, 1]
        reward_sequences = torch.stack([seq.reward_sequence for seq in sequences]).to(device)  # [B, L, 1]
        terminated_sequences = torch.stack([seq.terminated_sequence for seq in sequences]).to(device)  # [B, L] or [B, L, 1]
        valid_masks = torch.stack([seq.valid_sequence_mask for seq in sequences]).to(device)  # [B, L]

        # 1ステップずらしてTD(0)ターゲットを作るために、tとt+1で切り出し
        # 対象タイムステップは 0..L-2（最後は次状態がないため除外）
        state_t = state_sequences[:, :-1, ...]            # [B, L-1, H, W, C]
        action_t = action_sequences[:, :-1, :]            # [B, L-1, 1]
        reward_t = reward_sequences[:, :-1, :]            # [B, L-1, 1]
        # 終了フラグはtのものを使用
        term_t = terminated_sequences[:, :-1]
        if term_t.dim() == 3 and term_t.size(-1) == 1:
            term_t = term_t.squeeze(-1)                  # [B, L-1]
        state_tp1 = state_sequences[:, 1:, ...]          # [B, L-1, H, W, C]
        valid_t = valid_masks[:, :-1]                    # [B, L-1]

        # 現在のQ(s_t, a_t)
        current_q_all, _ = self.policy_net(state_t)      # [B, L-1, A]
        current_q_values = current_q_all.gather(2, action_t)  # [B, L-1, 1]

        # 目標 max_a' Q_target(s_{t+1}, a')
        with torch.no_grad():
            next_q_all, _ = self.target_net(state_tp1)   # [B, L-1, A]
            next_q_max = next_q_all.max(2)[0]            # [B, L-1]
            target_q_values = reward_t.squeeze(-1) + (GAMMA * next_q_max * (~term_t))  # [B, L-1]

        # 損失
        td_error = current_q_values.squeeze(-1) - target_q_values  # [B, L-1]
        # 無効部分を除外（Falseのところは0にし、分母は有効数）
        td_error = td_error * valid_t
        denom = valid_t.sum().clamp(min=1).float()
        loss = (td_error.pow(2).sum() / denom)

        self.optimizer.zero_grad()
        loss.backward()
        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # ターゲットネットを学習ステップ基準で同期
        self.update_steps += 1
        if self.update_steps % TARGET_UPDATE_FREQ == 0:
            self.sync_target_network()

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

                done_flag = terminated or truncated
                next_state = torch.tensor(obs, device=device).unsqueeze(0).float() if not done_flag else None  # [1, h, w, c] または None
                
                # リプレイバッファに保存するデータの形状:
                # state: [1, h, w, c]
                # action: [1, 1] 
                # next_state: [1, h, w, c] または None
                # reward: [1]
                # terminated: [1]
                agent.replay_buffer.push(state, action, next_state, 
                                         torch.tensor([reward], device=device), 
                                         torch.tensor(done_flag, device=device))
                state = next_state
                agent.update_model()

            # エピソード終了時にシーケンスを保存
            agent.replay_buffer.end_episode()
            stage_rewards.append(episode_reward)

            # 学習ステップ基準で同期するため、エピソード末の同期は削除
            
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