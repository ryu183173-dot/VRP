import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy


class DeepQNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims, fc2_dims, lr):
        super(DeepQNetwork, self).__init__()

        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.lr = lr

        self.exploration_proba = 1.0
        self.exploration_proba_decay = 0.005

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)

        # 1つの候補状態に対するQ値を出力
        self.fc3 = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss(reduction='none')

        self.device = T.device(
            'cuda:0' if T.cuda.is_available() else 'cpu'
        )
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

    def update_exploration_probability(self):
        self.exploration_proba *= np.exp(
            -self.exploration_proba_decay
        )


class DoubleDQNAgent:

    def __init__(
        self,
        gamma,
        epsilon,
        lr,
        input_dims,
        batch_size,
        n_actions,
        max_mem_size=10000,
        eps_end=0.01,
        eps_dec=5e-4
    ):

        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr

        self.n_actions = n_actions
        self.input_dims = input_dims

        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.Q_eval = DeepQNetwork(
            input_dims,
            n_actions,
            fc1_dims=256,
            fc2_dims=256,
            lr=lr
        )

        self.target_Q = copy.deepcopy(self.Q_eval)
        self.target_Q.eval()

        self.state_memory = np.zeros(
            (self.mem_size, input_dims),
            dtype=np.float32
        )

        self.reward_memory = np.zeros(
            self.mem_size,
            dtype=np.float32
        )

        self.terminal_memory = np.zeros(
            self.mem_size,
            dtype=bool
        )

        # 次状態の候補数は遷移ごとに異なるためリストで保存
        self.next_candidate_memory = [
            np.empty((0, input_dims), dtype=np.float32)
            for _ in range(self.mem_size)
        ]

        self.priorities = np.zeros(
            self.mem_size,
            dtype=np.float32
        )

    # --------------------------------------------------
    # Candidate state validation
    # --------------------------------------------------

    def _prepare_candidates(self, candidate_states):

        candidates = np.asarray(
            candidate_states,
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
                f"Expected {self.input_dims} state features, "
                f"but received {candidates.shape[1]}."
            )

        return candidates

    # --------------------------------------------------
    # Action selection
    # --------------------------------------------------

    def choose_action(self, candidate_states):

        candidates = self._prepare_candidates(candidate_states)

        if len(candidates) == 0:
            raise ValueError(
                "No candidate states were provided."
            )

        # Epsilon-greedy exploration
        if np.random.uniform() < self.Q_eval.exploration_proba:
            selected_index = np.random.randint(len(candidates))
            return selected_index

        states = T.as_tensor(
            candidates,
            dtype=T.float32,
            device=self.Q_eval.device
        )

        self.Q_eval.eval()

        with T.no_grad():
            q_values = self.Q_eval(states).squeeze(-1)

        self.Q_eval.train()

        selected_index = T.argmax(q_values).item()

        return selected_index

    # --------------------------------------------------
    # Experience replay buffer
    # --------------------------------------------------

    def store_transition(
        self,
        state,
        reward,
        next_candidate_states,
        done
    ):

        index = self.mem_cntr % self.mem_size

        state = np.asarray(
            state,
            dtype=np.float32
        ).reshape(-1)

        if len(state) != self.input_dims:
            raise ValueError(
                f"Expected state dimension {self.input_dims}, "
                f"but received {len(state)}."
            )

        next_candidates = self._prepare_candidates(
            next_candidate_states
        )

        self.state_memory[index] = state

        self.reward_memory[index] = reward

        self.next_candidate_memory[index] = (
            next_candidates.copy()
        )

        self.terminal_memory[index] = done

        current_max_priority = np.max(self.priorities)

        if current_max_priority == 0:
            current_max_priority = 1.0

        self.priorities[index] = current_max_priority

        self.mem_cntr += 1

    # --------------------------------------------------
    # Prioritized Experience Replay
    # --------------------------------------------------

    def get_priorities(self, priority_scale=0.6):

        max_mem = min(self.mem_cntr, self.mem_size)

        priorities = self.priorities[:max_mem]

        scaled_priorities = priorities ** priority_scale

        total = np.sum(scaled_priorities)

        if total <= 0:
            return np.ones(max_mem) / max_mem

        return scaled_priorities / total

    def set_priorities(
        self,
        indices,
        errors,
        offset=0.1
    ):

        for index, error in zip(indices, errors):
            self.priorities[index] = abs(float(error)) + offset

    # --------------------------------------------------
    # Learning: Double DQN + PER
    # --------------------------------------------------

    def learn(
        self,
        priority_scale=0.6,
        beta=0.4
    ):

        max_mem = min(self.mem_cntr, self.mem_size)

        if max_mem < self.batch_size:
            return None

        sample_probs = self.get_priorities(
            priority_scale
        )

        batch = np.random.choice(
            max_mem,
            self.batch_size,
            replace=True,
            p=sample_probs
        )

        # Importance sampling weights
        importance = (
            max_mem * sample_probs[batch]
        ) ** (-beta)

        importance /= np.max(importance)

        importance = T.as_tensor(
            importance,
            dtype=T.float32,
            device=self.Q_eval.device
        )

        state_batch = T.as_tensor(
            self.state_memory[batch],
            dtype=T.float32,
            device=self.Q_eval.device
        )

        reward_batch = T.as_tensor(
            self.reward_memory[batch],
            dtype=T.float32,
            device=self.Q_eval.device
        )

        terminal_batch = T.as_tensor(
            self.terminal_memory[batch],
            dtype=T.bool,
            device=self.Q_eval.device
        )

        # Q(s, selected candidate)
        q_eval = self.Q_eval(
            state_batch
        ).squeeze(-1)

        q_targets = []

        # Double DQN:
        # Online network selects the best next candidate.
        # Target network evaluates that candidate.
        with T.no_grad():

            for i, memory_index in enumerate(batch):

                reward = reward_batch[i]

                if terminal_batch[i]:
                    q_targets.append(reward)
                    continue

                next_candidates = (
                    self.next_candidate_memory[memory_index]
                )

                if len(next_candidates) == 0:
                    q_targets.append(reward)
                    continue

                next_states = T.as_tensor(
                    next_candidates,
                    dtype=T.float32,
                    device=self.Q_eval.device
                )

                online_q = self.Q_eval(
                    next_states
                ).squeeze(-1)

                best_next_index = T.argmax(
                    online_q
                ).item()

                target_q = self.target_Q(
                    next_states[best_next_index]
                ).squeeze()

                q_targets.append(
                    reward + self.gamma * target_q
                )

        q_target = T.stack(q_targets)

        td_errors = q_target - q_eval

        # Update priorities using actual sampled indices
        self.set_priorities(
            batch,
            td_errors.detach().cpu().numpy()
        )

        # PER-weighted squared TD error
        individual_losses = (
            td_errors.pow(2)
        )

        loss = T.mean(
            importance * individual_losses
        )

        self.Q_eval.optimizer.zero_grad()

        loss.backward()

        self.Q_eval.optimizer.step()

        self.epsilon = max(
            self.eps_min,
            self.epsilon - self.eps_dec
        )

        return loss.item()

    # --------------------------------------------------
    # Target network update
    # --------------------------------------------------

    def update_target_network(self):

        self.target_Q.load_state_dict(
            self.Q_eval.state_dict()
        )

        self.target_Q.eval()

    # --------------------------------------------------
    # Model saving
    # --------------------------------------------------

    def save_models(self):

        T.save(
            self.Q_eval.state_dict(),
            "./Q_network_DQN.pth"
        )

        T.save(
            self.target_Q.state_dict(),
            "./Target_network_DQN.pth"
        )