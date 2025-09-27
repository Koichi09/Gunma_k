# visualize_minigrid_DRQN_annotated_fixed.py
# - DRQN (CNN+LSTM) の学習済みモデルを読み込み
# - 毎フレームの Q 値を右パネルにバーで表示した注釈付き動画を出力
# 依存: pip install pillow imageio imageio-ffmpeg

import os
from collections import deque, namedtuple

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from minigrid.wrappers import ImgObsWrapper, ActionBonus, OneHotPartialObsWrapper

from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

# ===================== 設定 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# モデルと出力
MODEL_SAVE_PATH = "models/DRQN_model.pt"
ANNOTATED_OUT   = "videos/drqn_q_annotated.mp4"

# 環境設定
ENV_ID = "MiniGrid-Empty-16x16-v0"
ONE_HOT_ENCODE = True
ACTION_BONUS   = False

# アクション（3アクション想定）
ACTION_LABELS = ["left", "right", "forward"]

# （ロード互換用の定数）
REPLAY_BUFFER_SIZE = 25000

# ===================== 可視化ユーティリティ =====================
def draw_q_overlay(frame_np, q_vals, action_idx, labels=None, panel_w=240):
    """右側にQバーを描いたフレームを返す。"""
    import numpy as np
    h, w, _ = frame_np.shape
    img = Image.fromarray(frame_np)
    canvas = Image.new("RGB", (w + panel_w, h), (30, 30, 30))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    pad, bar_h, gap = 10, 24, 8
    qmin, qmax = float(np.min(q_vals)), float(np.max(q_vals))
    if qmax == qmin:
        qmax = qmin + 1.0
    scale = (panel_w - 2 * pad) / (qmax - qmin)

    # 0 ライン
    if qmin <= 0.0 <= qmax:
        zero_x = int(w + pad + (-qmin) * scale)
        draw.line([(zero_x, pad), (zero_x, h - pad)], fill=(200, 200, 200), width=1)

    for i, qi in enumerate(q_vals):
        y  = pad + i * (bar_h + gap)
        x0 = w + pad
        x1 = x0 + int((qi - qmin) * scale)
        xa, xb = sorted([x0, x1])
        color = (80, 160, 255) if i == action_idx else (120, 120, 120)
        draw.rectangle([xa, y, xb, y + bar_h], fill=color)
        label = labels[i] if labels and i < len(labels) else f"a{i}"
        draw.text((w + pad, y - 12), f"{label}: {qi:.3f}", font=font, fill=(240, 240, 240))

    return np.array(canvas)

# ===================== モデル定義（学習時と合わせる） =====================
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'terminated'))
# ★ 学習時と同じフィールド名で定義（これが無いと torch.load で見つからずに失敗）
SequenceTransition = namedtuple(
    'SequenceTransition',
    ('state_sequence', 'action_sequence', 'reward_sequence', 'terminated_sequence', 'valid_sequence_mask')
)

# ★ PyTorch 2.6+ の安全ロードで許可リストに追加
torch.serialization.add_safe_globals([Transition, SequenceTransition])

class ReplayBuffer:
    """ロード互換のための最小実装（push/sampleは不要）"""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.stage_memories = {}
        self.current_stage = 0
    def __len__(self):
        return len(self.memory)

class QNetwork(nn.Module):
    def __init__(self, obs_space_shape, action_space_n, hidden_dim=256):
        super(QNetwork, self).__init__()
        h, w, c = obs_space_shape
        self.hidden_dim = hidden_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            cnn_out_size = self.cnn(dummy_input).shape[1]

        self.lstm = nn.LSTM(cnn_out_size, hidden_dim, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_space_n)
        )

    def forward(self, x, hidden=None):
        """
        x: [B,H,W,C] or [B,T,H,W,C] （学習時に合わせてここでは正規化しない）
        hidden: (h, c)
        """
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
        else:
            x = x.to(DEVICE).float()

        single_state = False
        if x.ndim == 4:
            x = x.unsqueeze(1)  # [B,1,H,W,C]
            single_state = True

        B, T = x.shape[0], x.shape[1]
        x = x.reshape(B * T, *x.shape[2:])     # [B*T,H,W,C]
        x = x.permute(0, 3, 1, 2)              # [B*T,C,H,W]

        cnn_out = self.cnn(x)                  # [B*T,F]
        cnn_out = cnn_out.reshape(B, T, -1)    # [B,T,F]

        if hidden is None:
            hidden = self.init_hidden(B)

        lstm_out, new_hidden = self.lstm(cnn_out, hidden)  # [B,T,H]

        if single_state:
            lstm_out = lstm_out[:, -1, :]     # [B,H]（単一時刻）

        q_values = self.fc(lstm_out)          # [B,A] or [B,T,A]
        return q_values, new_hidden

    def init_hidden(self, batch_size=1):
        h = torch.zeros(1, batch_size, self.hidden_dim, device=DEVICE)
        c = torch.zeros(1, batch_size, self.hidden_dim, device=DEVICE)
        return (h, c)

class DRQNAgent:
    def __init__(self, obs_space_shape, action_space_n):
        self.action_space_n = action_space_n
        self.policy_net = QNetwork(obs_space_shape, action_space_n).to(DEVICE)
        self.target_net = QNetwork(obs_space_shape, action_space_n).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.update_steps = 0
        self.current_stage = 0
        self.episode_count = 0

        self.hidden_state = None  # 推論用のLSTM隠れ状態

    def _ensure_tensor(self, state):
        """NumPy/torchどちらでも受けて [B,H,W,C] float32 に整形（正規化はしない）"""
        if isinstance(state, np.ndarray):
            t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE)
        else:
            t = state.to(DEVICE).float()
        if t.ndim == 3:  # [H,W,C]
            t = t.unsqueeze(0)
        return t

    def select_action_and_q(self, state):
        """hidden を前進させつつ、(greedy action, Q(s,·) as np.ndarray) を返す"""
        s = self._ensure_tensor(state)
        if self.hidden_state is None:
            self.hidden_state = self.policy_net.init_hidden(batch_size=s.size(0))
        with torch.no_grad():
            q, self.hidden_state = self.policy_net(s, self.hidden_state)  # [1,A]
            q_np = q.squeeze(0).detach().cpu().numpy()
            a = int(np.argmax(q_np))
        return a, q_np

    def reset_hidden_state(self):
        self.hidden_state = None

    def _torch_load_ckpt(self, filepath):
        """PyTorch 2.6+ の安全ロード優先。必要ならフォールバック。"""
        try:
            return torch.load(filepath, map_location=DEVICE, weights_only=True)
        except Exception:
            # weights_only=True が失敗する場合（環境差等）は非安全ロードにフォールバック
            return torch.load(filepath, map_location=DEVICE, weights_only=False)

    def load_model(self, filepath):
        save_data = self._torch_load_ckpt(filepath)

        self.policy_net.load_state_dict(save_data['policy_net_state_dict'])
        self.target_net.load_state_dict(save_data['target_net_state_dict'])
        self.optimizer.load_state_dict(save_data['optimizer_state_dict'])

        self.update_steps = save_data.get('update_steps', 0)
        self.current_stage = save_data.get('current_stage', 0)
        self.episode_count = save_data.get('episode_count', 0)

        # 互換のための復元（可視化自体には不要）
        try:
            self.replay_buffer.memory = deque(save_data['replay_buffer_memory'],
                                              maxlen=self.replay_buffer.memory.maxlen)
            self.replay_buffer.stage_memories = {
                k: deque(v, maxlen=self.replay_buffer.memory.maxlen // 3)
                for k, v in save_data['replay_buffer_stage_memories'].items()
            }
            self.replay_buffer.current_stage = save_data['replay_buffer_current_stage']
        except Exception:
            pass

        self.policy_net.to(DEVICE)
        self.target_net.to(DEVICE)

        print(f"Model loaded from {filepath}")
        print(f"Loaded: current_stage={self.current_stage}, episode_count={self.episode_count}")

    @classmethod
    def create_from_saved(cls, filepath, obs_space_shape):
        # 先に読み込んで action_space_n を取得
        tmp = torch.load(filepath, map_location=DEVICE, weights_only=True)
        agent = cls(obs_space_shape, tmp['action_space_n'])
        agent.load_model(filepath)
        return agent

# ===================== メイン（可視化ループ） =====================
def main():
    # 可視用（rgb_array）と推論用（部分観測）を用意
    vis_env = gym.make(ENV_ID, render_mode="rgb_array")
    agent_env = gym.make(ENV_ID)
    if ONE_HOT_ENCODE:
        agent_env = OneHotPartialObsWrapper(agent_env)
    if ACTION_BONUS:
        agent_env = ActionBonus(agent_env)
    agent_env = ImgObsWrapper(agent_env)

    os.makedirs("videos", exist_ok=True)
    vis_env = RecordVideo(vis_env, "videos", name_prefix="side_by_side")

    # モデル読み込み
    obs_shape = agent_env.observation_space.shape  # [H,W,C]
    print(f"Loading existing model from {MODEL_SAVE_PATH}")
    agent = DRQNAgent.create_from_saved(MODEL_SAVE_PATH, obs_shape)

    # reset → render の順序を厳守
    obs_agent, _ = agent_env.reset(seed=0)
    obs_vis, _   = vis_env.reset(seed=0)
    agent.reset_hidden_state()

    writer = imageio.get_writer(ANNOTATED_OUT, fps=15)
    print(f"[annot] writing annotated video to: {ANNOTATED_OUT}")

    done = False
    while not done:
        # 現在のフレーム（可視用）
        frame = vis_env.render()

        # Q と行動（pre-step）。LSTM hidden を前進させながら取得
        action, q = agent.select_action_and_q(obs_agent)

        # 注釈を重ねて保存
        annotated = draw_q_overlay(frame, q, action, ACTION_LABELS)
        writer.append_data(annotated)

        # 同じアクションで2つの環境を進める
        obs_agent, r, term1, trunc1, _ = agent_env.step(action)
        obs_vis,   r2, term2, trunc2, _ = vis_env.step(action)
        done = (term1 or trunc1)

    writer.close()
    print(f"[annot] saved: {ANNOTATED_OUT}")

    vis_env.close()
    agent_env.close()

if __name__ == "__main__":
    main()
