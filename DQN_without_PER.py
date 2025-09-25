# DQN_without_PER

import gymnasium as gym
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
BATCH_SIZE = 64 #もとは64だからそのままにしているがDQNの方はsequenceではないので学習回数が少なくなってしまう。これを解消するためにはどうすればいいか?
LEARNING_RATE = 1e-4 
TARGET_UPDATE_FREQ = 50

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#--- 経験を保存するためのデータ構造 ---
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward', 'terminated'))

# --- リプレイバッファ ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.stage_memories = {}  # 各ステージの経験を保持
        self.current_stage = 0

    def push(self, *args):
        """単一の遷移を保存"""
        self.current_stage = stage # ステージ別にも保存
        if stage not in self.stage_memories:
            self.stage_memories[stage] = deque([], maxlen=self.memory.maxlen//3)
        self.memory.append(Transition(*args))
        self.stage_memories[stage].append(Transition(*args))

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
        x = x.to(device).float() / 255.0
        x = x.permute(0, 3, 1, 2)
        return self.fc(self.cnn(x))

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

    def select_action(self, state):
        # ステージをまたいでepsilonが減衰
        eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                        math.exp(-1.0 * self.episode_count / EPSILON_DECAY)
        if random.random() > eps_threshold:
            with torch.no_grad():
                q_values = self.policy_net(state)
                return q_values.max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_space_n)]], device=device, dtype=torch.long)

    def update_model(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        transitions = self.replay_buffer.sample(BATCH_SIZE, self.current_stage)
        
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        with torch.no_grad():
            next_q_values = torch.zeros(BATCH_SIZE, device=device)
            next_q_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
            target_q_values = reward_batch + (GAMMA * next_q_values)
        loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))

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
    MODEL_SAVE_PATH = "models/DQN_without_PER_model.pt"
    SAVE_FREQUENCY = 1000  # 何エピソードごとに保存するか
    ACTION_BONUS = False
    ONE_HOT_ENCODE = True
    
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
                "mode": "DQN_without_PER",
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
                "sequence_length": 1,
                "device": str(device)
            },
            name=f"DQN_without_PER_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        if ACTION_BONUS == True:
            env = ActionBonus(env)
        if ONE_HOT_ENCODE == True:
            env = OneHotPartialObsWrapper(env)
        
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
            agent.increment_episode()
            stage_rewards.append(episode_reward)
            
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
                agent.save_model(f"models/DQN_without_PER_stage{stage+1}_ep{episode+1}.pt")
        
        # ステージ終了時にモデルを保存
        if SAVE_MODEL:
            agent.save_model(f"models/DQN_without_PER_stage{stage+1}_final.pt")
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
    
    plt.title("DQN without PER")
    plt.xlabel("Total Episodes")
    plt.ylabel("Average Reward (Moving Avg over 100 episodes)")
    plt.legend()
    plt.grid(True)
    plt.savefig("DQN without PER.png")
    
    # wandbにプロットをアップロード
    if USE_WANDB:
        wandb.log({"performance_plot": wandb.Image("DQN without PER.png")})
    
    plt.show()
    
    # wandb終了
    if USE_WANDB:
        wandb.finish()