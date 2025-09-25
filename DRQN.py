# DQN

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
import os
from datetime import datetime

# wandbのインポート（オプション）
import wandb

# --- ハイパーパラメータ設定 ---
# カリキュラム学習の設定
MAZE_SIZES = [5, 8, 16]
EPISODES_PER_STAGE = 5000

# DQNエージェントの設定
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 5000
REPLAY_BUFFER_SIZE = 25000
BATCH_SIZE = 64
LEARNING_RATE = 1e-4 
TARGET_UPDATE_FREQ = 50

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
    def __init__(self, capacity, sequence_length=15):
        self.memory = deque([], maxlen=capacity)
        self.sequence_length = sequence_length
        self.episode_buffer = []  # 現在のエピソードの経験を一時保存
        self.stage_memories = {}  # 各ステージの経験を保持
        self.current_stage = 0

    def push(self, *args):
        """単一の遷移を保存"""
        self.episode_buffer.append(Transition(*args))

    def end_episode(self, stage=0):
        """エピソード終了時にシーケンスを保存"""
        self.current_stage = stage
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
                seq_transition = SequenceTransition(state_seq, action_seq, reward_seq, terminated_seq, valid_mask)
                self.memory.append(seq_transition)
                # ステージ別にも保存
                if stage not in self.stage_memories:
                    self.stage_memories[stage] = deque([], maxlen=self.memory.maxlen//3)  # 各ステージの1/3を保持
                self.stage_memories[stage].append(seq_transition)
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
            seq_transition = SequenceTransition(state_seq, action_seq, reward_seq, terminated_seq, valid_mask)
            self.memory.append(seq_transition)
            # ステージ別にも保存
            if stage not in self.stage_memories:
                self.stage_memories[stage] = deque([], maxlen=self.memory.maxlen//3)
            self.stage_memories[stage].append(seq_transition)

        self.episode_buffer = []  # エピソードバッファをクリア

    def sample(self, batch_size, stage=0):
        """シーケンスをサンプリング（古いステージの経験も含める）"""
        # ステージごとの重み付きサンプリング
        stage_samples = []
        stage_weights = []
        
        # 現在のステージ（重み: 0.4）
        if len(self.memory) > 0:
            current_samples = list(self.memory)
            stage_samples.append(current_samples)
            stage_weights.append(0.4)
        
        # 過去のステージ（重み: 0.6を均等分配）
        past_weight = 0.6 / max(1, stage) if stage > 0 else 0
        for past_stage in range(stage):
            if past_stage in self.stage_memories and len(self.stage_memories[past_stage]) > 0:
                past_samples = list(self.stage_memories[past_stage])
                stage_samples.append(past_samples)
                stage_weights.append(past_weight)
        
        # 重みに基づいて各ステージからサンプリング
        selected_samples = []
        for i, (samples, weight) in enumerate(zip(stage_samples, stage_weights)):
            if weight > 0 and len(samples) > 0:
                # このステージから取得するサンプル数
                num_from_stage = max(1, int(batch_size * weight))
                num_from_stage = min(num_from_stage, len(samples))
                
                # ランダムサンプリング
                stage_selected = random.sample(samples, num_from_stage)
                selected_samples.extend(stage_selected)
        
        # 不足分を現在のステージから補完
        if len(selected_samples) < batch_size and len(self.memory) > 0:
            additional_needed = batch_size - len(selected_samples)
            additional_samples = random.sample(list(self.memory), 
                                             min(additional_needed, len(self.memory)))
            selected_samples.extend(additional_samples)
        
        # まだ不足している場合は重複を許可
        while len(selected_samples) < batch_size:
            selected_samples.append(random.choice(selected_samples))
        
        # 最終的にランダムにシャッフルして順序の偏りを排除
        random.shuffle(selected_samples)
        return selected_samples[:batch_size]

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
        x = x.to(device).float()  #正規化の必要あるか？
        
        # バッチ次元を確認
        if len(x.shape) == 4:  # 単一状態
            x = x.unsqueeze(1)  # [batch_size, 1, height, width, channels]
            single_state = True
        else:  # シーケンス状態
            single_state = False
            
        batch_size, seq_len = x.shape[0], x.shape[1]
        x = x.reshape(batch_size * seq_len, *x.shape[2:])  # [batch_size*seq_len, height, width, channels]
        x = x.permute(0, 3, 1, 2)  # [batch_size*seq_len, channels, height, width]
        
        # CNNで特徴抽出
        cnn_out = self.cnn(x)  # [batch_size*seq_len, cnn_out_size]
        cnn_out = cnn_out.reshape(batch_size, seq_len, -1)  # [batch_size, seq_len, cnn_out_size]
        
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
        self.update_steps = 0  # ターゲット同期のステップカウンタ（学習ステップ基準）
        self.current_stage = 0
        self.episode_count = 0
        
        # LSTMの隠れ状態を管理
        self.hidden_state = None
        self.target_hidden_state = None

    def select_action(self, state):
        # ステージをまたいでepsilonが減衰
        eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                        math.exp(-1.0 * self.episode_count / EPSILON_DECAY)
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

        sequences = self.replay_buffer.sample(BATCH_SIZE, self.current_stage)
        
        # シーケンスをバッチに変換
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
    
    def set_stage(self, stage):
        """ステージを設定し、学習率を調整"""
        self.current_stage = stage
    
    def increment_episode(self):
        """エピソードカウントを増加"""
        self.episode_count += 1

    def sync_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, filepath):
        """モデルとエージェントの状態を保存（torch.save使用）"""
        save_data = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_steps': self.update_steps,
            'current_stage': self.current_stage,
            'episode_count': self.episode_count,
            'replay_buffer_memory': list(self.replay_buffer.memory),
            'replay_buffer_stage_memories': {k: list(v) for k, v in self.replay_buffer.stage_memories.items()},
            'replay_buffer_current_stage': self.replay_buffer.current_stage,
            'action_space_n': self.action_space_n,
            'model_config': {
                'hidden_dim': self.policy_net.hidden_dim,
                'sequence_length': self.replay_buffer.sequence_length
            }
        }
        
        # ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save(save_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """モデルとエージェントの状態を読み込み（torch.load使用）"""
        save_data = torch.load(filepath, map_location=device)
        
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
        print(f"Loaded: current_stage={self.current_stage}, episode_count={self.episode_count}")
    
    @classmethod
    def create_from_saved(cls, filepath, obs_space_shape):
        """保存されたモデルから新しいエージェントを作成"""
        save_data = torch.load(filepath, map_location=device)
        
        # 新しいエージェントを作成
        agent = cls(obs_space_shape, save_data['action_space_n'])
        
        # 状態を復元
        agent.load_model(filepath)
        
        return agent

# --- メインの学習ループ ---
if __name__ == "__main__":
    # 保存・読み込み設定
    SAVE_MODEL = True
    LOAD_MODEL = False  # 既存のモデルを読み込むかどうか
    MODEL_SAVE_PATH = "models/DRQN_model.pt"
    SAVE_FREQUENCY = 1000  # 何エピソードごとに保存するか
    
    # wandb設定
    USE_WANDB = True
    WANDB_PROJECT = "minigrid"
    WANDB_ENTITY = None  # あなたのwandbユーザー名（Noneの場合はデフォルト）
    
    # wandb初期化
    if USE_WANDB:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            config={
                "mode": "DRQN",
                "maze_sizes": MAZE_SIZES,
                "episodes_per_stage": EPISODES_PER_STAGE,
                "gamma": GAMMA,
                "epsilon_start": EPSILON_START,
                "epsilon_end": EPSILON_END,
                "epsilon_decay": EPSILON_DECAY,
                "replay_buffer_size": REPLAY_BUFFER_SIZE,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "target_update_freq": TARGET_UPDATE_FREQ,
                "sequence_length": 15,
                "device": str(device)
            },
            name=f"DRQN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # 使用例:
    # 1. 新しい学習を開始: LOAD_MODEL = False
    # 2. 既存のモデルから継続: LOAD_MODEL = True
    # 3. 手動でモデルを保存: agent.save_model("path/to/model.pt")
    # 4. 手動でモデルを読み込み: agent.load_model("path/to/model.pt")
    # 5. 保存されたモデルから新しいエージェント作成: 
    #    agent = DQNAgent.create_from_saved("path/to/model.pt", obs_shape)
    
    agent = None
    all_stage_rewards = {}

    for stage, size in enumerate(MAZE_SIZES):
        print(f"--- Curriculum Stage {stage + 1}: {size}x{size} Maze ---")
        env = gym.make(f'MiniGrid-Empty-{size}x{size}-v0')
        env = ImgObsWrapper(env)
        
        if agent is None:
            obs_shape = env.observation_space.shape
            action_n = 3 #env.action_space.n
            
            # 既存のモデルを読み込むかどうか
            if LOAD_MODEL and os.path.exists(MODEL_SAVE_PATH):
                print(f"Loading existing model from {MODEL_SAVE_PATH}")
                agent = DQNAgent.create_from_saved(MODEL_SAVE_PATH, obs_shape)
            else:
                agent = DQNAgent(obs_shape, action_n)
        
        # ステージを設定
        agent.set_stage(stage)

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
            agent.replay_buffer.end_episode(stage)
            agent.increment_episode()
            stage_rewards.append(episode_reward)

            # 学習ステップ基準で同期するため、エピソード末の同期は削除
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(stage_rewards[-100:])
                print(f"Stage {stage+1}, Episode {episode+1}, Avg Reward (last 100): {avg_reward:.2f}")
                
                # wandbにログ記録
                if USE_WANDB:
                    wandb.log({
                        "stage": stage + 1,
                        "episode": episode + 1,
                        "avg_reward_100": avg_reward,
                        "episode_reward": episode_reward,
                        "epsilon": EPSILON_END + (EPSILON_START - EPSILON_END) * 
                                 math.exp(-1.0 * agent.episode_count / EPSILON_DECAY),
                        "replay_buffer_size": len(agent.replay_buffer),
                        "learning_rate": agent.optimizer.param_groups[0]['lr']
                    })
            
            # 定期的にモデルを保存
            if SAVE_MODEL and (episode + 1) % SAVE_FREQUENCY == 0:
                agent.save_model(f"models/dqn_lstm_stage{stage+1}_ep{episode+1}.pt")
        
        # ステージ終了時にモデルを保存
        if SAVE_MODEL:
            agent.save_model(f"models/dqn_lstm_stage{stage+1}_final.pt")
            agent.save_model(MODEL_SAVE_PATH)  # 最新のモデルを保存
        
        # ステージ完了をwandbに記録
        if USE_WANDB:
            stage_avg_reward = np.mean(stage_rewards)
            stage_max_reward = np.max(stage_rewards)
            stage_std_reward = np.std(stage_rewards)
            
            wandb.log({
                "stage_completed": stage + 1,
                "stage_avg_reward": stage_avg_reward,
                "stage_max_reward": stage_max_reward,
                "stage_std_reward": stage_std_reward,
                "stage_episodes": len(stage_rewards)
            })
        
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
    
    # wandbにプロットをアップロード
    if USE_WANDB:
        wandb.log({"performance_plot": wandb.Image("dqn_curriculum_performance.png")})
    
    plt.show()
    
    # wandb終了
    if USE_WANDB:
        wandb.finish()