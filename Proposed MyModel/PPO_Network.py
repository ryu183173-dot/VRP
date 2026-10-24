import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorNetwork(nn.Module):
    """顧客を選択する確率分布を出力するネットワーク。"""

    def __init__(self, input_dims, n_actions, lr=0.0003):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dims, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.network(state)


class CriticNetwork(nn.Module):
    """状態の価値 V(s) を推定するネットワーク。"""

    def __init__(self, input_dims, lr=0.0003):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dims, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.network(state).squeeze(-1)


class PPOAgent:
    """
    離散行動用 PPO エージェント。

    行動候補マスクを利用し、選択可能な顧客だけから行動を選ぶ。
    """

    def __init__(
        self,
        input_dims,
        n_actions,
        lr=0.0003,
        gamma=0.99,
        batch_size=64,
        update_epochs=4,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5
    ):
        self.gamma = gamma
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
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.old_log_probs = []
        self.old_values = []
        self.action_masks = []

    def choose_action(self, observation, valid_action_indices):

        if len(valid_action_indices) == 0:
            raise ValueError("選択可能な行動がありません。")

        state = torch.as_tensor(
            np.asarray(observation, dtype=np.float32),
            dtype=torch.float32,
            device=self.device
        ).reshape(1, -1)

        action_mask = np.zeros(self.n_actions, dtype=np.bool_)
        action_mask[valid_action_indices] = True

        mask_tensor = torch.as_tensor(
            action_mask, dtype=torch.bool, device=self.device
        )

        with torch.no_grad():
            logits = self.actor(state).squeeze(0)

            logits = logits.masked_fill(~mask_tensor, -1e9)

            distribution = Categorical(logits=logits)

            action = distribution.sample()

            log_prob = distribution.log_prob(action)
            state_value = self.critic(state).squeeze(0)

        result = (
            int(action.item()),
            float(log_prob.item()),
            float(state_value.item()),
            action_mask
        )
        
        return result

    def store_transition(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        log_prob,
        state_value,
        action_mask
    ):
        self.states.append(np.asarray(state, dtype=np.float32).copy())
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.next_states.append(
            np.asarray(next_state, dtype=np.float32).copy()
        )
        self.dones.append(bool(done))
        self.old_log_probs.append(float(log_prob))
        self.old_values.append(float(state_value))
        self.action_masks.append(
            np.asarray(action_mask, dtype=np.bool_).copy()
        )

    def learn(self, force=False):
        """
        保存された経験を使って PPO の更新を行う。
        force=True なら batch_size 未満の経験でも更新する。
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
            self.actions, dtype=torch.long, device=self.device
        )
        rewards = torch.as_tensor(
            self.rewards, dtype=torch.float32, device=self.device
        )
        next_states = torch.as_tensor(
            np.asarray(self.next_states),
            dtype=torch.float32,
            device=self.device
        )
        dones = torch.as_tensor(
            self.dones, dtype=torch.float32, device=self.device
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

        # 1ステップTD誤差をAdvantageとして使用する。
        with torch.no_grad():
            next_values = self.critic(next_states)
            targets = rewards + self.gamma * next_values * (1.0 - dones)
            advantages = targets - old_values

            if len(advantages) > 1:
                advantages = (
                    advantages - advantages.mean()
                ) / (advantages.std(unbiased=False) + 1e-8)


        indices = np.arange(n_samples)
        last_loss = None

        for _ in range(self.update_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.batch_size):
                batch_indices = indices[start:start + self.batch_size]

                idx = torch.as_tensor(
                    batch_indices, dtype=torch.long, device=self.device
                )

                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_targets = targets[idx]
                batch_masks = action_masks[idx]

                logits = self.actor(batch_states)
                logits = logits.masked_fill(~batch_masks, -1e9)

                distribution = Categorical(logits=logits)
                new_log_probs = distribution.log_prob(batch_actions)
                entropy = distribution.entropy().mean()

                # PPOの確率比とクリッピング目的関数
                ratios = torch.exp(
                    new_log_probs - batch_old_log_probs
                )

                unclipped = ratios * batch_advantages
                clipped = torch.clamp(
                    ratios,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon
                ) * batch_advantages

                actor_loss = -torch.min(unclipped, clipped).mean()

                values = self.critic(batch_states)
                critic_loss = nn.functional.mse_loss(
                    values, batch_targets
                )

                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.entropy_coef * entropy
                )

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), max_norm=0.5
                )
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), max_norm=0.5
                )

                self.actor.optimizer.step()
                self.critic.optimizer.step()

                last_loss = float(loss.item())

        self.clear_memory()
        return last_loss
