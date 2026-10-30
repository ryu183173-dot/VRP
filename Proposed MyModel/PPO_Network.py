import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorNetwork(nn.Module):
    """顧客を選択する確率分布を出力するActorネットワーク。"""

    def __init__(self, input_dims, n_actions, lr=0.0003):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dims, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

        # ネットワークをデバイスへ移動してからOptimizerを作成
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        return self.network(state)


class CriticNetwork(nn.Module):
    """状態価値 V(s) を推定するCriticネットワーク。"""

    def __init__(self, input_dims, lr=0.0003):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dims, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        return self.network(state).squeeze(-1)


class PPOAgent:
    """
    離散行動用 PPO エージェント。

    ・Actor-Critic構造
    ・行動マスク
    ・Clipped Surrogate Objective
    ・Generalized Advantage Estimation (GAE)
    ・車両ごとの軌跡管理

    trajectory_id は (depot, vehicle) の組を想定する。
    """

    def __init__(
        self,
        input_dims,
        n_actions,
        lr=0.0003,
        gamma=0.99,
        gae_lambda=0.95,
        batch_size=64,
        update_epochs=4,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.update_epochs = update_epochs
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.n_actions = n_actions

        self.actor = ActorNetwork(input_dims, n_actions, lr)
        self.critic = CriticNetwork(input_dims, lr)
        self.device = self.actor.device

        self.clear_memory()

    def clear_memory(self):
        """保存した経験をすべて消去する。"""

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.old_log_probs = []
        self.old_values = []
        self.action_masks = []

        # 各経験の所属車両を記録
        self.trajectory_ids = []

        # GAEの再帰計算を区切るフラグ
        self.trajectory_ends = []

    def choose_action(self, observation, valid_action_indices):
        """有効な行動候補から確率的に行動を選択する。"""


        if len(valid_action_indices) == 0:
            raise ValueError("選択可能な行動がありません。")

        if min(valid_action_indices) < 0 or max(valid_action_indices) >= self.n_actions:
            raise ValueError("valid_action_indices に範囲外の行動があります。")

        state = torch.as_tensor(
            np.asarray(observation, dtype=np.float32),
            dtype=torch.float32,
            device=self.device
        ).reshape(1, -1)


        action_mask = np.zeros(self.n_actions, dtype=np.bool_)
        action_mask[valid_action_indices] = True

        mask_tensor = torch.as_tensor(
            action_mask,
            dtype=torch.bool,
            device=self.device
        )


        with torch.no_grad():
            logits = self.actor(state).squeeze(0)


            logits = logits.masked_fill(~mask_tensor, -1e9)


            distribution = Categorical(logits=logits)

            action = distribution.sample()

            log_prob = distribution.log_prob(action)
            state_value = self.critic(state).squeeze(0)


        return (
            int(action.item()),
            float(log_prob.item()),
            float(state_value.item()),
            action_mask
        )

    def store_transition(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        log_prob,
        state_value,
        action_mask,
        trajectory_id,
        trajectory_end=False
    ):
        """
        経験を保存する。

        done:
            本当のエピソード終端かどうか。

        trajectory_end:
            車両の帰還などによるGAE計算上の軌跡の区切り。
            doneとは異なり、次状態価値のブートストラップを残す。
        """

        state = np.asarray(state, dtype=np.float32).reshape(-1)
        next_state = np.asarray(next_state, dtype=np.float32).reshape(-1)
        action_mask = np.asarray(action_mask, dtype=np.bool_)

        if action_mask.shape != (self.n_actions,):
            raise ValueError("action_mask のサイズが不正です。")

        if not action_mask[int(action)]:
            raise ValueError("保存された行動が action_mask で無効になっています。")

        self.states.append(state.copy())
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.next_states.append(next_state.copy())
        self.dones.append(bool(done))
        self.old_log_probs.append(float(log_prob))
        self.old_values.append(float(state_value))
        self.action_masks.append(action_mask.copy())

        self.trajectory_ids.append(trajectory_id)
        self.trajectory_ends.append(bool(trajectory_end))

    def mark_trajectory_end(self, trajectory_id, bootstrap_state=None):
        """
        指定した車両の直近の経験を軌跡の区切りにする。

        デポ帰還など、実際のエピソード終端ではない場合に使用する。
        bootstrap_stateが指定された場合は、次状態をその状態に更新する。
        """

        for i in range(len(self.trajectory_ids) - 1, -1, -1):

            if self.trajectory_ids[i] == trajectory_id:

                self.trajectory_ends[i] = True

                if bootstrap_state is not None:
                    bootstrap_state = np.asarray(
                        bootstrap_state,
                        dtype=np.float32
                    ).reshape(-1)

                    self.next_states[i] = bootstrap_state.copy()

                return True

        return False


    def mark_trajectory_terminal(self, trajectory_id):
        """
        指定した車両の直近の経験を本当の終端状態にする。

        全顧客への配送が完了した場合に使用する。
        done=Trueにすることで、次状態価値のブートストラップを停止する。
        """

        for i in range(len(self.trajectory_ids) - 1, -1, -1):

            if self.trajectory_ids[i] == trajectory_id:

                self.dones[i] = True
                self.trajectory_ends[i] = True

                return True

        return False

    def calculate_gae(self, next_values):
        """
        車両ごとに独立してGAEを計算する。

        delta_t = r_t + gamma * V(s_{t+1}) * (1-done) - V(s_t)

        A_t = delta_t
              + gamma * lambda * (1-done)
                * (1-trajectory_end) * A_{t+1}
        """

        n_samples = len(self.states)

        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.old_values, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        trajectory_ends = np.asarray(
            self.trajectory_ends,
            dtype=np.float32
        )

        next_values = np.asarray(next_values, dtype=np.float32)

        deltas = (
            rewards
            + self.gamma * next_values * (1.0 - dones)
            - values
        )

        advantages = np.zeros(n_samples, dtype=np.float32)

        # 車両ごとの経験インデックスをまとめる
        trajectory_groups = {}

        for i, trajectory_id in enumerate(self.trajectory_ids):
            if trajectory_id not in trajectory_groups:
                trajectory_groups[trajectory_id] = []

            trajectory_groups[trajectory_id].append(i)

        # 各車両の軌跡を独立して逆順に計算
        for indices in trajectory_groups.values():
            gae = 0.0

            for i in reversed(indices):
                continuation = (
                    (1.0 - dones[i])
                    * (1.0 - trajectory_ends[i])
                )

                gae = (
                    deltas[i]
                    + self.gamma
                    * self.gae_lambda
                    * continuation
                    * gae
                )

                advantages[i] = gae

        # Criticの教師信号はGAE + V(s)
        returns = advantages + values

        return advantages, returns

    def learn(self, force=False):
        """
        保存された経験を用いてPPOを更新する。

        force=True の場合、batch_size未満でも更新する。
        """

        n_samples = len(self.states)

        if n_samples == 0:
            return None

        if n_samples < self.batch_size and not force:
            return None

        states = torch.as_tensor(
            np.asarray(self.states),
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.as_tensor(
            self.actions,
            dtype=torch.long,
            device=self.device
        )

        next_states = torch.as_tensor(
            np.asarray(self.next_states),
            dtype=torch.float32,
            device=self.device
        )

        old_log_probs = torch.as_tensor(
            self.old_log_probs,
            dtype=torch.float32,
            device=self.device
        )

        old_values = torch.as_tensor(
            self.old_values,
            dtype=torch.float32,
            device=self.device
        )

        action_masks = torch.as_tensor(
            np.asarray(self.action_masks),
            dtype=torch.bool,
            device=self.device
        )

        # GAE計算に使う次状態価値を更新前のCriticで評価
        with torch.no_grad():
            next_values = self.critic(next_states).cpu().numpy()

        advantages_np, returns_np = self.calculate_gae(next_values)

        advantages = torch.as_tensor(
            advantages_np,
            dtype=torch.float32,
            device=self.device
        )

        returns = torch.as_tensor(
            returns_np,
            dtype=torch.float32,
            device=self.device
        )

        # Advantageの正規化
        if n_samples > 1:
            advantages = (
                advantages - advantages.mean()
            ) / (advantages.std(unbiased=False) + 1e-8)

        indices = np.arange(n_samples)
        last_loss = None

        for _ in range(self.update_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.batch_size):
                batch_indices = indices[
                    start:start + self.batch_size
                ]

                idx = torch.as_tensor(
                    batch_indices,
                    dtype=torch.long,
                    device=self.device
                )

                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]
                batch_masks = action_masks[idx]

                # Actor
                logits = self.actor(batch_states)
                logits = logits.masked_fill(~batch_masks, -1e9)

                distribution = Categorical(logits=logits)

                new_log_probs = distribution.log_prob(batch_actions)
                entropy = distribution.entropy().mean()

                # PPO probability ratio
                ratios = torch.exp(
                    new_log_probs - batch_old_log_probs
                )

                unclipped = ratios * batch_advantages

                clipped = torch.clamp(
                    ratios,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon
                ) * batch_advantages

                actor_loss = -torch.min(
                    unclipped,
                    clipped
                ).mean()

                # Critic
                values = self.critic(batch_states)

                critic_loss = nn.functional.mse_loss(
                    values,
                    batch_returns
                )

                # Combined loss
                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.entropy_coef * entropy
                )

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()

                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    max_norm=0.5
                )

                nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    max_norm=0.5
                )

                self.actor.optimizer.step()
                self.critic.optimizer.step()

                last_loss = float(loss.item())

        self.clear_memory()

        return last_loss
