import os
import copy
import random
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DeepQNetwork(nn.Module):
    """
    Candidate-state scoring network.

    各候補顧客・デポの状態を入力し、
    その候補を選択した場合のQ値を1つ出力する。
    """

    def __init__(
        self,
        input_dims,
        n_actions=None,
        fc1_dims=256,
        fc2_dims=256,
        lr=0.003
    ):
        super(DeepQNetwork, self).__init__()

        self.input_dims = input_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, 1)

        self.device = T.device(
            "cuda:0" if T.cuda.is_available() else "cpu"
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # 探索率はこのネットワーク側で一元管理する
        self.exploration_proba = 1.0
        self.exploration_min = 0.01
        self.exploration_decay = 0.005

        self.to(self.device)

    def forward(self, state):
        state = state.to(self.device, dtype=T.float32)

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)

        return q_value

    def update_exploration_probability(self):
        """
        エピソードごとに探索率を指数的に減少させる。
        """
        self.exploration_proba = max(
            self.exploration_min,
            self.exploration_proba
            * np.exp(-self.exploration_decay)
        )


class DoubleDQNAgent:
    """
    Double DQN + Prioritized Experience Replay.

    経験は以下の形式で保存する。

    state:
        選択した候補の状態ベクトル

    reward:
        その行動によって得られた報酬

    next_candidate_states:
        次の行動で選択可能な候補状態の配列

    done:
        エピソード終了フラグ
    """

    def __init__(
        self,
        gamma,
        epsilon,
        lr,
        input_dims,
        batch_size,
        n_actions=None,
        max_mem_size=100000,
        eps_end=0.01,
        eps_dec=0.005,
        alpha=0.6,
        beta=0.4,
        beta_increment=0.0001,
        priority_epsilon=1e-6,
        fc1_dims=256,
        fc2_dims=256
    ):

        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec

        self.batch_size = batch_size
        self.input_dims = input_dims
        self.n_actions = n_actions

        self.mem_size = max_mem_size
        self.mem_cntr = 0

        # PER parameters
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.priority_epsilon = priority_epsilon

        self.device = T.device(
            "cuda:0" if T.cuda.is_available() else "cpu"
        )

        self.Q_eval = DeepQNetwork(
            input_dims=input_dims,
            n_actions=n_actions,
            fc1_dims=fc1_dims,
            fc2_dims=fc2_dims,
            lr=lr
        )

        self.target_Q = copy.deepcopy(self.Q_eval)
        self.target_Q.eval()

        # Experience memory
        self.state_memory = np.zeros(
            (self.mem_size, self.input_dims),
            dtype=np.float32
        )

        self.reward_memory = np.zeros(
            self.mem_size,
            dtype=np.float32
        )

        self.terminal_memory = np.zeros(
            self.mem_size,
            dtype=np.float32
        )

        # 次状態の候補数は可変なのでリストで保持する
        self.next_candidate_memory = [
            np.empty((0, self.input_dims), dtype=np.float32)
            for _ in range(self.mem_size)
        ]

        # Priorities
        self.priority_memory = np.zeros(
            self.mem_size,
            dtype=np.float32
        )

    def _prepare_candidates(self, candidate_states):
        """
        候補状態を (候補数, state_dim) のfloat32配列に変換する。
        """

        candidates = np.asarray(
            candidate_states,
            dtype=np.float32
        )

        if candidates.size == 0:
            return np.empty(
                (0, self.input_dims),
                dtype=np.float32
            )

        if candidates.ndim == 1:
            candidates = candidates.reshape(1, -1)

        if candidates.ndim != 2:
            raise ValueError(
                "candidate_states must be a 2D array."
            )

        if candidates.shape[1] != self.input_dims:
            raise ValueError(
                f"State dimension mismatch: "
                f"expected {self.input_dims}, "
                f"got {candidates.shape[1]}"
            )

        if not np.all(np.isfinite(candidates)):
            raise ValueError(
                "candidate_states contains NaN or Inf."
            )

        return candidates

    def choose_action(self, candidate_states):
        """
        候補状態の中から行動を選択する。

        戻り値:
            候補配列内のインデックス
        """

        candidates = self._prepare_candidates(candidate_states)

        if len(candidates) == 0:
            raise ValueError(
                "No feasible candidate is available."
            )

        # epsilon-greedy
        if random.random() < self.Q_eval.exploration_proba:
            return random.randrange(len(candidates))

        candidate_tensor = T.as_tensor(
            candidates,
            dtype=T.float32,
            device=self.device
        )

        self.Q_eval.eval()

        with T.no_grad():
            q_values = self.Q_eval(candidate_tensor).squeeze(-1)

        self.Q_eval.train()

        return int(T.argmax(q_values).item())

    def store_transition(
        self,
        state,
        reward,
        next_candidate_states,
        done
    ):
        """
        経験をリプレイメモリに保存する。
        """

        state = np.asarray(state, dtype=np.float32).reshape(-1)

        if state.shape[0] != self.input_dims:
            raise ValueError(
                f"State dimension mismatch: "
                f"expected {self.input_dims}, "
                f"got {state.shape[0]}"
            )

        if not np.all(np.isfinite(state)):
            raise ValueError("state contains NaN or Inf.")

        next_candidates = self._prepare_candidates(
            next_candidate_states
        )

        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.reward_memory[index] = float(reward)
        self.terminal_memory[index] = float(done)

        self.next_candidate_memory[index] = next_candidates

        current_size = min(self.mem_cntr, self.mem_size)

        if current_size > 0:
            max_priority = np.max(
                self.priority_memory[:current_size]
            )
        else:
            max_priority = 1.0

        if max_priority <= 0:
            max_priority = 1.0

        self.priority_memory[index] = max_priority

        self.mem_cntr += 1

    def get_priorities(self):
        """
        現在保存されている経験のサンプリング確率を返す。
        """

        current_size = min(self.mem_cntr, self.mem_size)

        if current_size == 0:
            return np.array([], dtype=np.float64)

        priorities = self.priority_memory[:current_size]

        scaled_priorities = (
            priorities + self.priority_epsilon
        ) ** self.alpha

        total_priority = np.sum(scaled_priorities)

        if total_priority <= 0 or not np.isfinite(total_priority):
            return np.ones(
                current_size,
                dtype=np.float64
            ) / current_size

        return scaled_priorities / total_priority

    def set_priorities(self, indices, td_errors):
        """
        TD errorに基づいて優先度を更新する。
        """

        td_errors = np.asarray(
            td_errors,
            dtype=np.float32
        ).reshape(-1)

        for index, td_error in zip(indices, td_errors):
            priority = (
                abs(float(td_error))
                + self.priority_epsilon
            )

            self.priority_memory[index] = priority

    def update_target_network(self):
        """
        評価ネットワークの重みをターゲットネットワークへコピーする。
        """

        self.target_Q.load_state_dict(
            self.Q_eval.state_dict()
        )

        self.target_Q.eval()

    def learn(self):
        """
        Double DQN + PERによる学習処理。

        Double DQN:
            1. Q_evalで次の候補のうち最大Q値の候補を選ぶ
            2. target_Qでその候補のQ値を評価する

        PER:
            TD errorに応じて経験を優先的にサンプリングする
        """

        current_size = min(self.mem_cntr, self.mem_size)

        if current_size < self.batch_size:
            return None

        sampling_probabilities = self.get_priorities()

        if len(sampling_probabilities) != current_size:
            return None

        # Prioritized sampling
        indices = np.random.choice(
            current_size,
            self.batch_size,
            replace=True,
            p=sampling_probabilities
        )

        selected_probabilities = sampling_probabilities[indices]

        # Importance sampling weights
        weights = (
            current_size * selected_probabilities
        ) ** (-self.beta)

        weights = weights / np.max(weights)

        weights = T.as_tensor(
            weights,
            dtype=T.float32,
            device=self.device
        )

        # Beta annealing
        self.beta = min(
            1.0,
            self.beta + self.beta_increment
        )

        states = T.as_tensor(
            self.state_memory[indices],
            dtype=T.float32,
            device=self.device
        )

        rewards = T.as_tensor(
            self.reward_memory[indices],
            dtype=T.float32,
            device=self.device
        )

        dones = T.as_tensor(
            self.terminal_memory[indices],
            dtype=T.float32,
            device=self.device
        )

        # Current Q values
        self.Q_eval.train()

        current_q_values = self.Q_eval(states).squeeze(-1)

        target_q_values = rewards.clone()

        td_errors = np.zeros(
            self.batch_size,
            dtype=np.float32
        )

        # Next-state Double DQN calculation
        with T.no_grad():

            for batch_index, memory_index in enumerate(indices):

                if dones[batch_index].item() >= 1.0:
                    continue

                next_candidates = (
                    self.next_candidate_memory[memory_index]
                )

                if len(next_candidates) == 0:
                    continue

                next_candidates_tensor = T.as_tensor(
                    next_candidates,
                    dtype=T.float32,
                    device=self.device
                )

                # Online network selects the best next candidate
                online_next_q = self.Q_eval(
                    next_candidates_tensor
                ).squeeze(-1)

                best_next_index = T.argmax(
                    online_next_q
                )

                # Target network evaluates the selected candidate
                target_next_q = self.target_Q(
                    next_candidates_tensor
                ).squeeze(-1)

                best_next_value = target_next_q[
                    best_next_index
                ]

                target_q_values[batch_index] = (
                    rewards[batch_index]
                    + self.gamma * best_next_value
                )

        # TD error
        td_error_tensor = (
            target_q_values - current_q_values
        )

        # Weighted squared TD error
        loss = T.mean(
            weights * td_error_tensor.pow(2)
        )

        self.Q_eval.optimizer.zero_grad()
        loss.backward()

        T.nn.utils.clip_grad_norm_(
            self.Q_eval.parameters(),
            max_norm=10.0
        )

        self.Q_eval.optimizer.step()

        # Update PER priorities
        td_errors = (
            td_error_tensor.detach()
            .abs()
            .cpu()
            .numpy()
        )

        self.set_priorities(indices, td_errors)

        return float(loss.item())

    def save_models(self, path="ddqn_checkpoint"):
        """
        評価ネットワークとターゲットネットワークを保存する。
        """

        os.makedirs(path, exist_ok=True)

        T.save(
            self.Q_eval.state_dict(),
            os.path.join(path, "Q_eval.pth")
        )

        T.save(
            self.target_Q.state_dict(),
            os.path.join(path, "target_Q.pth")
        )

    def load_models(self, path="ddqn_checkpoint"):
        """
        保存済みネットワークを読み込む。
        """

        eval_path = os.path.join(path, "Q_eval.pth")
        target_path = os.path.join(path, "target_Q.pth")

        self.Q_eval.load_state_dict(
            T.load(
                eval_path,
                map_location=self.device
            )
        )

        self.target_Q.load_state_dict(
            T.load(
                target_path,
                map_location=self.device
            )
        )

        self.Q_eval.train()
        self.target_Q.eval()