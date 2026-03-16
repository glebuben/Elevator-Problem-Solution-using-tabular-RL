import numpy as np
from collections import deque
import time

# ─────────────────────────────────────────────────────
# 1. SIMULATOR
# ─────────────────────────────────────────────────────

class Passenger:
    __slots__ = ['origin', 'destination', 'arrival_time', 'board_time']
    def __init__(self, origin, destination, arrival_time):
        self.origin = origin
        self.destination = destination
        self.arrival_time = arrival_time
        self.board_time = None

MOVE_UP = 0
MOVE_DOWN = 1
STOP = 2
WAIT = 3
ACTION_NAMES = ['MOVE_UP', 'MOVE_DOWN', 'STOP', 'WAIT']

DIR_DOWN = 0
DIR_IDLE = 1
DIR_UP = 2
DIR_NAMES = ['DOWN', 'IDLE', 'UP']


class ElevatorEnv:
    def __init__(self, num_floors=5, capacity=5, lam=0.05,
                 traffic='uniform', seed=None):
        self.num_floors = num_floors
        self.capacity = capacity
        self.lam = lam
        self.traffic = traffic
        self.rng = np.random.RandomState(seed)
        self.up_queues = [deque() for _ in range(num_floors)]
        self.down_queues = [deque() for _ in range(num_floors)]
        self.car_passengers = []
        self.car_floor = 0
        self.car_direction = DIR_IDLE
        self.time_step = 0
        self.total_wait_time = 0
        self.total_system_time = 0
        self.total_served = 0
        self.total_boarded = 0
        self.up_button_since = [-1] * num_floors
        self.down_button_since = [-1] * num_floors
        self._setup_traffic()

    def _setup_traffic(self):
        nf = self.num_floors
        if self.traffic == 'uniform':
            self.floor_arrival_probs = np.ones(nf) / nf
            self.dest_probs = {}
            for f in range(nf):
                p = np.ones(nf); p[f] = 0.0; p /= p.sum()
                self.dest_probs[f] = p
        elif self.traffic == 'up_peak':
            self.floor_arrival_probs = np.array([0.6, 0.1, 0.1, 0.1, 0.1])
            self.dest_probs = {}
            for f in range(nf):
                if f == 0:
                    p = np.array([0.0, 0.15, 0.25, 0.30, 0.30])
                else:
                    p = np.ones(nf); p[f] = 0.0; p /= p.sum()
                self.dest_probs[f] = p
        elif self.traffic == 'down_peak':
            self.floor_arrival_probs = np.array([0.05, 0.15, 0.20, 0.30, 0.30])
            self.dest_probs = {}
            for f in range(nf):
                if f == 0:
                    p = np.ones(nf); p[0] = 0.0; p /= p.sum()
                else:
                    p = np.zeros(nf); p[0] = 0.7
                    for j in range(nf):
                        if j != f and j != 0:
                            p[j] = 0.3 / max(nf - 2, 1)
                    p[f] = 0.0
                    s = p.sum()
                    if s > 0: p /= s
                self.dest_probs[f] = p
        elif self.traffic == 'mixed':
            self.floor_arrival_probs = np.array([0.35, 0.15, 0.15, 0.15, 0.20])
            self.dest_probs = {}
            for f in range(nf):
                p = np.ones(nf); p[f] = 0.0
                if f != 0: p[0] = 3.0
                p /= p.sum()
                self.dest_probs[f] = p
        else:
            raise ValueError(f"Unknown traffic: {self.traffic}")

    def reset(self):
        self.up_queues = [deque() for _ in range(self.num_floors)]
        self.down_queues = [deque() for _ in range(self.num_floors)]
        self.car_passengers = []
        self.car_floor = 0
        self.car_direction = DIR_IDLE
        self.time_step = 0
        self.total_wait_time = 0
        self.total_system_time = 0
        self.total_served = 0
        self.total_boarded = 0
        self.up_button_since = [-1] * self.num_floors
        self.down_button_since = [-1] * self.num_floors
        return self.get_state()

    def get_up_hall_calls(self):
        mask = 0
        for i in range(4):
            if len(self.up_queues[i]) > 0:
                mask |= (1 << i)
        return mask

    def get_down_hall_calls(self):
        mask = 0
        for i in range(4):
            if len(self.down_queues[i + 1]) > 0:
                mask |= (1 << i)
        return mask

    def get_car_calls(self):
        mask = 0
        for p in self.car_passengers:
            mask |= (1 << p.destination)
        return mask

    def get_num_passengers(self):
        return len(self.car_passengers)

    def get_state(self):
        return (self.car_floor, self.car_direction,
                self.get_up_hall_calls(), self.get_down_hall_calls(),
                self.get_car_calls(), self.get_num_passengers())

    def get_valid_actions(self):
        actions = [WAIT]
        if self.car_floor < self.num_floors - 1:
            actions.append(MOVE_UP)
        if self.car_floor > 0:
            actions.append(MOVE_DOWN)
        floor = self.car_floor
        has_exit = any(p.destination == floor for p in self.car_passengers)
        has_up = len(self.up_queues[floor]) > 0
        has_down = len(self.down_queues[floor]) > 0
        if has_exit or has_up or has_down:
            actions.append(STOP)
        return actions

    def _total_waiting(self):
        total = 0
        for f in range(self.num_floors):
            total += len(self.up_queues[f]) + len(self.down_queues[f])
        return total

    def _compute_observable_reward(self):
        reward = 0
        for f in range(self.num_floors):
            if self.up_button_since[f] >= 0:
                reward -= (self.time_step - self.up_button_since[f])
            if self.down_button_since[f] >= 0:
                reward -= (self.time_step - self.down_button_since[f])
        cc = self.get_car_calls()
        for f in range(5):
            if (cc >> f) & 1:
                reward -= 1
        return reward

    def _update_button_timestamps(self):
        for f in range(4):
            if len(self.up_queues[f]) > 0:
                if self.up_button_since[f] < 0:
                    self.up_button_since[f] = self.time_step
            else:
                self.up_button_since[f] = -1
        for f in range(1, 5):
            if len(self.down_queues[f]) > 0:
                if self.down_button_since[f] < 0:
                    self.down_button_since[f] = self.time_step
            else:
                self.down_button_since[f] = -1

    def step(self, action):
        valid = self.get_valid_actions()
        if action not in valid:
            action = WAIT

        if action == MOVE_UP:
            self.car_floor += 1
            self.car_direction = DIR_UP
        elif action == MOVE_DOWN:
            self.car_floor -= 1
            self.car_direction = DIR_DOWN
        elif action == WAIT:
            self.car_direction = DIR_IDLE
        elif action == STOP:
            remaining = []
            for p in self.car_passengers:
                if p.destination == self.car_floor:
                    self.total_system_time += (self.time_step - p.arrival_time)
                    self.total_served += 1
                else:
                    remaining.append(p)
            self.car_passengers = remaining

            direction = self.car_direction
            if direction == DIR_UP:
                queues_to_serve = [self.up_queues[self.car_floor]]
            elif direction == DIR_DOWN:
                queues_to_serve = [self.down_queues[self.car_floor]]
            else:
                queues_to_serve = [self.up_queues[self.car_floor],
                                   self.down_queues[self.car_floor]]
            for queue in queues_to_serve:
                while queue and len(self.car_passengers) < self.capacity:
                    pax = queue.popleft()
                    self.total_wait_time += (self.time_step - pax.arrival_time)
                    self.total_boarded += 1
                    pax.board_time = self.time_step
                    self.car_passengers.append(pax)

        self._generate_arrivals()
        self._update_button_timestamps()
        reward = self._compute_observable_reward()
        self.time_step += 1
        return self.get_state(), reward

    def _generate_arrivals(self):
        for f in range(self.num_floors):
            rate = self.lam * self.floor_arrival_probs[f] * self.num_floors
            n = self.rng.poisson(rate)
            for _ in range(n):
                dest = self.rng.choice(self.num_floors, p=self.dest_probs[f])
                if dest == f:
                    continue
                p = Passenger(f, dest, self.time_step)
                if dest > f:
                    self.up_queues[f].append(p)
                else:
                    self.down_queues[f].append(p)

    def get_pending_metrics(self):
        pw_time = pw_n = ps_time = ps_n = 0
        for f in range(self.num_floors):
            for p in self.up_queues[f]:
                pw_time += self.time_step - p.arrival_time
                pw_n += 1
            for p in self.down_queues[f]:
                pw_time += self.time_step - p.arrival_time
                pw_n += 1
        for p in self.car_passengers:
            ps_time += self.time_step - p.arrival_time
            ps_n += 1
        return pw_time, pw_n, ps_time, ps_n


# ─────────────────────────────────────────────────────
# 2. STATE ENCODING
# ─────────────────────────────────────────────────────

def _build_cc_np_lookup():
    pairs = []
    for cc in range(32):
        pop = bin(cc).count('1')
        for np_ in range(pop, 6):
            pairs.append((cc, np_))
    return pairs, {p: i for i, p in enumerate(pairs)}

VALID_CC_NP_PAIRS, CC_NP_LOOKUP = _build_cc_np_lookup()
NUM_CC_NP = len(VALID_CC_NP_PAIRS)

def state_to_index(state):
    cf, cd, uc, dc, cc, np_ = state
    idx = cf * 3 + cd
    idx = idx * 16 + uc
    idx = idx * 16 + dc
    idx = idx * NUM_CC_NP + CC_NP_LOOKUP[(cc, np_)]
    return idx

TOTAL_STATES = 5 * 3 * 16 * 16 * NUM_CC_NP
NUM_ACTIONS = 4


# ─────────────────────────────────────────────────────
# 3. HELPERS
# ─────────────────────────────────────────────────────

def has_up_call(uc, f):
    return f <= 3 and bool((uc >> f) & 1)

def has_down_call(dc, f):
    return f >= 1 and bool((dc >> (f - 1)) & 1)

def has_car_call(cc, f):
    return bool((cc >> f) & 1)

def has_any_call(uc, dc, cc, f):
    return has_up_call(uc, f) or has_down_call(dc, f) or has_car_call(cc, f)

def calls_above(uc, dc, cc, cf):
    return any(has_any_call(uc, dc, cc, f) for f in range(cf + 1, 5))

def calls_below(uc, dc, cc, cf):
    return any(has_any_call(uc, dc, cc, f) for f in range(cf))

def calls_exist(uc, dc, cc):
    return uc != 0 or dc != 0 or cc != 0


# ─────────────────────────────────────────────────────
# 4. TWO RL AGENTS
# ─────────────────────────────────────────────────────

class QLearningAgent:
    """
    Off-policy Q-learning (value-iteration style).

    Update rule:
        Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') - Q(s,a) ]

    The target uses max over ALL valid next actions,
    regardless of which action the agent actually picks next.
    This means:
      - It learns about the OPTIMAL policy
      - Even while following an exploratory (random) policy
      - The exploration policy and the learned policy are DIFFERENT
        (hence "off-policy")

    Analogy to value iteration:
      Value iteration: V(s) = max_a [ R(s,a) + γ Σ P(s'|s,a) V(s') ]
      Q-learning:      Q(s,a) ← r + γ max_a' Q(s',a')
      Both take the MAX over actions in the next state.
    """

    def __init__(self, alpha=0.1, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay_steps=500000, q_init=-5.0):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.Q = np.full((TOTAL_STATES, NUM_ACTIONS), q_init, dtype=np.float64)
        self.steps = 0
        self.rng = np.random.RandomState(0)
        self.name = "Q-learning"

    @property
    def epsilon(self):
        frac = min(1.0, self.steps / self.epsilon_decay_steps)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * frac

    def select_action(self, state, valid_actions, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return valid_actions[self.rng.randint(len(valid_actions))]
        s = state_to_index(state)
        best_q = max(self.Q[s, a] for a in valid_actions)
        best = [a for a in valid_actions if abs(self.Q[s, a] - best_q) < 1e-10]
        return best[self.rng.randint(len(best))]

    def update(self, state, action, reward, next_state, next_valid,
               next_action=None):
        """next_action is ignored — Q-learning always uses max."""
        s = state_to_index(state)
        ns = state_to_index(next_state)

        # VALUE-ITERATION STYLE: max over all valid next actions
        max_next_q = max(self.Q[ns, a] for a in next_valid)
        target = reward + self.gamma * max_next_q

        self.Q[s, action] += self.alpha * (target - self.Q[s, action])
        self.steps += 1

    def set_rng(self, rng):
        self.rng = rng

    def copy_q_table(self):
        return self.Q.copy()

    def load_q_table(self, qt):
        self.Q = qt.copy()


class SARSAAgent:
    """
    On-policy SARSA (policy-iteration style).

    Update rule:
        Q(s,a) ← Q(s,a) + α [ r + γ Q(s',a') - Q(s,a) ]

    The target uses Q(s', a') where a' is the action ACTUALLY CHOSEN
    in state s' (following the current epsilon-greedy policy).
    This means:
      - It learns about the policy IT IS CURRENTLY FOLLOWING
      - Including exploratory (random) actions
      - The exploration policy and the learned policy are THE SAME
        (hence "on-policy")

    Analogy to policy iteration:
      Policy iteration: evaluate current policy π, then improve
      SARSA:            Q(s,a) ← r + γ Q(s', π(s'))
      Both evaluate the CURRENT policy (including exploration).

    Key consequence:
      - SARSA's Q-values are slightly pessimistic because they
        account for future random exploration
      - Q-learning's Q-values are optimistic because they assume
        optimal behavior in the future
      - As ε → 0, both converge to the same values
      - During training with high ε, SARSA is more conservative
    """

    def __init__(self, alpha=0.1, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay_steps=500000, q_init=-5.0):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.Q = np.full((TOTAL_STATES, NUM_ACTIONS), q_init, dtype=np.float64)
        self.steps = 0
        self.rng = np.random.RandomState(0)
        self.name = "SARSA"

    @property
    def epsilon(self):
        frac = min(1.0, self.steps / self.epsilon_decay_steps)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * frac

    def select_action(self, state, valid_actions, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return valid_actions[self.rng.randint(len(valid_actions))]
        s = state_to_index(state)
        best_q = max(self.Q[s, a] for a in valid_actions)
        best = [a for a in valid_actions if abs(self.Q[s, a] - best_q) < 1e-10]
        return best[self.rng.randint(len(best))]

    def update(self, state, action, reward, next_state, next_valid,
               next_action=None):
        """next_action is REQUIRED — SARSA uses the actual next action."""
        s = state_to_index(state)
        ns = state_to_index(next_state)

        # POLICY-ITERATION STYLE: use the actual next action chosen
        next_q = self.Q[ns, next_action]
        target = reward + self.gamma * next_q

        self.Q[s, action] += self.alpha * (target - self.Q[s, action])
        self.steps += 1

    def set_rng(self, rng):
        self.rng = rng

    def copy_q_table(self):
        return self.Q.copy()

    def load_q_table(self, qt):
        self.Q = qt.copy()


# ─────────────────────────────────────────────────────
# 5. BASELINE POLICIES
# ─────────────────────────────────────────────────────

class RandomPolicy:
    def __init__(self, rng):
        self.rng = rng

    def select_action(self, state, valid_actions):
        return valid_actions[self.rng.randint(len(valid_actions))]


class SCANPolicy:
    def __init__(self):
        self.sweep_dir = DIR_UP

    def select_action(self, state, valid_actions):
        cf, cd, uc, dc, cc, np_ = state
        if STOP in valid_actions:
            if has_car_call(cc, cf):
                return STOP
            if np_ < 5:
                if self.sweep_dir == DIR_UP and has_up_call(uc, cf):
                    return STOP if cd in (DIR_UP, DIR_IDLE) else WAIT
                if self.sweep_dir == DIR_DOWN and has_down_call(dc, cf):
                    return STOP if cd in (DIR_DOWN, DIR_IDLE) else WAIT
        if self.sweep_dir == DIR_UP:
            if MOVE_UP in valid_actions:
                return MOVE_UP
            self.sweep_dir = DIR_DOWN
            if STOP in valid_actions:
                if has_car_call(cc, cf):
                    return STOP
                if np_ < 5:
                    if has_down_call(dc, cf):
                        return STOP if cd in (DIR_DOWN, DIR_IDLE) else WAIT
                    if has_up_call(uc, cf):
                        return STOP if cd in (DIR_UP, DIR_IDLE) else WAIT
            if MOVE_DOWN in valid_actions:
                return MOVE_DOWN
            return WAIT
        else:
            if MOVE_DOWN in valid_actions:
                return MOVE_DOWN
            self.sweep_dir = DIR_UP
            if STOP in valid_actions:
                if has_car_call(cc, cf):
                    return STOP
                if np_ < 5:
                    if has_up_call(uc, cf):
                        return STOP if cd in (DIR_UP, DIR_IDLE) else WAIT
                    if has_down_call(dc, cf):
                        return STOP if cd in (DIR_DOWN, DIR_IDLE) else WAIT
            if MOVE_UP in valid_actions:
                return MOVE_UP
            return WAIT


class LOOKPolicy:
    def __init__(self):
        self.direction = DIR_UP

    def _try_stop(self, state, valid_actions):
        cf, cd, uc, dc, cc, np_ = state
        if STOP not in valid_actions:
            return None
        if has_car_call(cc, cf):
            return STOP
        if np_ >= 5:
            return None
        hu = has_up_call(uc, cf)
        hd = has_down_call(dc, cf)
        if not hu and not hd:
            return None
        if cd == DIR_IDLE:
            return STOP
        if cd == DIR_UP and hu:
            return STOP
        if cd == DIR_DOWN and hd:
            return STOP
        return WAIT

    def select_action(self, state, valid_actions):
        cf, cd, uc, dc, cc, np_ = state
        if not calls_exist(uc, dc, cc):
            return WAIT
        sa = self._try_stop(state, valid_actions)
        if sa is not None:
            return sa
        if self.direction == DIR_UP:
            if calls_above(uc, dc, cc, cf) and MOVE_UP in valid_actions:
                return MOVE_UP
            self.direction = DIR_DOWN
            sa = self._try_stop(state, valid_actions)
            if sa is not None:
                return sa
            if calls_below(uc, dc, cc, cf) and MOVE_DOWN in valid_actions:
                return MOVE_DOWN
            return WAIT
        else:
            if calls_below(uc, dc, cc, cf) and MOVE_DOWN in valid_actions:
                return MOVE_DOWN
            self.direction = DIR_UP
            sa = self._try_stop(state, valid_actions)
            if sa is not None:
                return sa
            if calls_above(uc, dc, cc, cf) and MOVE_UP in valid_actions:
                return MOVE_UP
            return WAIT


class NearestCallPolicy:
    def __init__(self, rng):
        self.rng = rng

    def select_action(self, state, valid_actions):
        cf, cd, uc, dc, cc, np_ = state
        if STOP in valid_actions and has_car_call(cc, cf):
            return STOP
        if STOP in valid_actions and np_ < 5:
            hu = has_up_call(uc, cf)
            hd = has_down_call(dc, cf)
            if hu or hd:
                if cd == DIR_IDLE:
                    return STOP
                if cd == DIR_UP and hu:
                    return STOP
                if cd == DIR_DOWN and hd:
                    return STOP
                return WAIT
        if not calls_exist(uc, dc, cc):
            return WAIT
        target = None
        best_d = 999
        for f in range(5):
            if has_car_call(cc, f):
                d = abs(f - cf)
                if d < best_d:
                    best_d = d
                    target = f
        if target is None:
            for f in range(5):
                if has_up_call(uc, f) or has_down_call(dc, f):
                    d = abs(f - cf)
                    if d < best_d:
                        best_d = d
                        target = f
        if target is None:
            return WAIT
        if target > cf:
            if STOP in valid_actions and np_ < 5 and has_up_call(uc, cf):
                return STOP if cd in (DIR_UP, DIR_IDLE) else WAIT
            return MOVE_UP if MOVE_UP in valid_actions else WAIT
        elif target < cf:
            if STOP in valid_actions and np_ < 5 and has_down_call(dc, cf):
                return STOP if cd in (DIR_DOWN, DIR_IDLE) else WAIT
            return MOVE_DOWN if MOVE_DOWN in valid_actions else WAIT
        else:
            if STOP in valid_actions:
                if cd == DIR_IDLE:
                    return STOP
                hu = has_up_call(uc, cf)
                hd = has_down_call(dc, cf)
                if cd == DIR_UP and hu:
                    return STOP
                if cd == DIR_DOWN and hd:
                    return STOP
                return WAIT
            return WAIT


# ─────────────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────────────

def evaluate_policy(policy_fn, num_steps=10000, lam=0.05,
                    traffic='uniform', seed=42):
    env = ElevatorEnv(lam=lam, traffic=traffic, seed=seed)
    state = env.reset()
    total_reward = 0.0
    for _ in range(num_steps):
        va = env.get_valid_actions()
        action = policy_fn(state, va)
        state, reward = env.step(action)
        total_reward += reward
    cw = env.total_wait_time
    cb = env.total_boarded
    cs = env.total_system_time
    cv = env.total_served
    pw, pwn, ps, psn = env.get_pending_metrics()
    return {
        'avg_reward': total_reward / num_steps,
        'avg_wait_c': cw / cb if cb > 0 else float('nan'),
        'avg_sys_c': cs / cv if cv > 0 else float('nan'),
        'avg_wait_i': (cw + pw) / (cb + pwn) if (cb + pwn) > 0 else float('nan'),
        'avg_sys_i': (cs + ps + pw) / (cv + psn + pwn) if (cv + psn + pwn) > 0 else float('nan'),
        'served': cv,
        'boarded': cb,
        'still_waiting': pwn,
        'in_car': psn,
        'total_reward': total_reward,
    }


def fmt(v, w=10, d=2):
    if isinstance(v, float) and np.isnan(v):
        return "N/A".rjust(w)
    if isinstance(v, float):
        return f"{v:.{d}f}".rjust(w)
    return str(v).rjust(w)


def print_results(results):
    print()
    print(f"{'Policy':<16s} {'Rew/step':>9s} {'Wait(c)':>9s} "
          f"{'Sys(c)':>9s} {'Wait(i)':>9s} {'Sys(i)':>9s} "
          f"{'Served':>7s} {'Board':>7s} {'Queue':>7s} {'InCar':>7s}")
    print('-' * 112)
    for name, m in results.items():
        print(f"{name:<16s} "
              f"{fmt(m['avg_reward'],9,3)} "
              f"{fmt(m['avg_wait_c'],9)} {fmt(m['avg_sys_c'],9)} "
              f"{fmt(m['avg_wait_i'],9)} {fmt(m['avg_sys_i'],9)} "
              f"{fmt(m['served'],7)} {fmt(m['boarded'],7)} "
              f"{fmt(m['still_waiting'],7)} {fmt(m['in_car'],7)}")
    print("\n  (c)=completed  (i)=inclusive (+pending)")


# ─────────────────────────────────────────────────────
# 7. UNIFIED TRAINING LOOP (works for both agents)
# ─────────────────────────────────────────────────────

def train_agent(agent, total_steps=1000000, eval_interval=100000,
                eval_steps=10000, lam=0.05, traffic='uniform',
                reset_interval=50000, eval_seed=42):
    """
    Unified training loop that handles both Q-learning and SARSA.

    Q-learning loop:
        observe s → pick a → take a → observe s', r → update(s,a,r,s')
        next_action is irrelevant (uses max)

    SARSA loop:
        observe s → pick a → take a → observe s', r → pick a' → update(s,a,r,s',a')
        next_action IS the actual a' chosen for the next step
    """
    is_sarsa = isinstance(agent, SARSAAgent)
    algo_name = agent.name

    print(f"\nTraining {algo_name}: {total_steps} steps")
    print(f"  λ={lam}, traffic={traffic}, γ={agent.gamma}, α={agent.alpha}")
    print(f"  ε: {agent.epsilon_start} → {agent.epsilon_end} "
          f"over {agent.epsilon_decay_steps} steps")
    print(f"  Update: {'r + γ Q(s\',a\') [on-policy]' if is_sarsa else 'r + γ max Q(s\',a\') [off-policy]'}")
    print(f"  States: {TOTAL_STATES}, Q-table: "
          f"{TOTAL_STATES * NUM_ACTIONS * 8 / 1024 / 1024:.1f} MB")

    env = ElevatorEnv(lam=lam, traffic=traffic, seed=123)
    state = env.reset()
    t0 = time.time()

    best_reward = -float('inf')
    best_q = None
    best_step = 0

    # For SARSA, we need the first action before entering the loop
    va = env.get_valid_actions()
    action = agent.select_action(state, va)

    for step in range(total_steps):
        if step > 0 and step % reset_interval == 0:
            state = env.reset()
            va = env.get_valid_actions()
            action = agent.select_action(state, va)

        # Take action, observe next state and reward
        next_state, reward = env.step(action)
        next_valid = env.get_valid_actions()

        # Choose next action (needed for SARSA, computed anyway for both)
        next_action = agent.select_action(next_state, next_valid)

        # Update Q-table
        # Q-learning ignores next_action (uses max internally)
        # SARSA uses next_action
        agent.update(state, action, reward, next_state, next_valid,
                     next_action=next_action)

        # Transition
        state = next_state
        action = next_action  # SARSA reuses this; Q-learning doesn't care

        # Periodic evaluation
        if (step + 1) % eval_interval == 0:
            elapsed = time.time() - t0
            m = evaluate_policy(
                lambda s, va: agent.select_action(s, va, greedy=True),
                eval_steps, lam, traffic, eval_seed)
            cr = m['avg_reward']
            is_best = cr > best_reward
            if is_best:
                best_reward = cr
                best_q = agent.copy_q_table()
                best_step = step + 1
            total_pax = m['served'] + m['still_waiting'] + m['in_car']
            mark = " *** BEST" if is_best else ""
            print(f"  Step {step+1:>8d} | ε={agent.epsilon:.4f} | "
                  f"rew/s={cr:.3f} | wait_i={m['avg_wait_i']:.2f} | "
                  f"served={m['served']:>4d}/{total_pax:>4d} | "
                  f"{elapsed:.1f}s{mark}")

    print(f"\n{algo_name} done in {time.time()-t0:.1f}s")
    print(f"Best Q-table: step {best_step}, avg_reward={best_reward:.4f}")
    if best_q is not None:
        agent.load_q_table(best_q)
        print("Loaded best Q-table.\n")
    return agent


# ─────────────────────────────────────────────────────
# 8. COMPARISON
# ─────────────────────────────────────────────────────

def run_comparison(lam=0.05, traffic='uniform', eval_steps=10000,
                   eval_seed=42, agents=None):
    print(f"\n{'='*112}")
    print(f"COMPARISON: λ={lam}, traffic={traffic}, steps={eval_steps}")
    print(f"{'='*112}")

    results = {}

    rng1 = np.random.RandomState(eval_seed + 1)
    results['Random'] = evaluate_policy(
        lambda s, va: RandomPolicy(rng1).select_action(s, va),
        eval_steps, lam, traffic, eval_seed)

    sp = SCANPolicy()
    results['SCAN'] = evaluate_policy(
        lambda s, va: sp.select_action(s, va),
        eval_steps, lam, traffic, eval_seed)

    lp = LOOKPolicy()
    results['LOOK'] = evaluate_policy(
        lambda s, va: lp.select_action(s, va),
        eval_steps, lam, traffic, eval_seed)

    rng2 = np.random.RandomState(eval_seed + 2)
    nc = NearestCallPolicy(rng2)
    results['Nearest'] = evaluate_policy(
        lambda s, va: nc.select_action(s, va),
        eval_steps, lam, traffic, eval_seed)

    if agents:
        for name, ag in agents.items():
            ag.set_rng(np.random.RandomState(eval_seed + 10))
            results[name] = evaluate_policy(
                lambda s, va, a=ag: a.select_action(s, va, greedy=True),
                eval_steps, lam, traffic, eval_seed)

    print_results(results)
    return results


# ─────────────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 112)
    print("Q-LEARNING vs SARSA — ELEVATOR CONTROL")
    print("=" * 112)

    assert NUM_CC_NP == 112 and TOTAL_STATES == 430080
    print(f"State space: {TOTAL_STATES}, cc_np pairs: {NUM_CC_NP}")

    # ── Configuration ──
    LAM = 0.001
    TRAFFIC = 'uniform'
    GAMMA = 0.99
    ALPHA = 0.1
    TRAIN_STEPS = 5_000_000
    EVAL_STEPS = 10_000
    EVAL_SEED = 42
    EPS_START = 1.0
    EPS_END = 0.01
    EPS_DECAY = TRAIN_STEPS // 2

    agents = {}

    # ── Train Q-learning (off-policy, value-iteration style) ──
    print("\n" + "#" * 112)
    print("  Q-LEARNING (off-policy): target = r + γ max_a' Q(s',a')")
    print("  Learns optimal policy while exploring randomly")
    print("#" * 112)

    q_agent = QLearningAgent(
        alpha=ALPHA, gamma=GAMMA,
        epsilon_start=EPS_START, epsilon_end=EPS_END,
        epsilon_decay_steps=EPS_DECAY, q_init=-5.0)
    q_agent.set_rng(np.random.RandomState(456))

    train_agent(q_agent,
                total_steps=TRAIN_STEPS, eval_interval=200_000,
                eval_steps=EVAL_STEPS, lam=LAM, traffic=TRAFFIC,
                reset_interval=50_000, eval_seed=EVAL_SEED)
    agents['Q-learning'] = q_agent

    # ── Train SARSA (on-policy, policy-iteration style) ──
    print("\n" + "#" * 112)
    print("  SARSA (on-policy): target = r + γ Q(s',a')")
    print("  Learns value of current (exploratory) policy")
    print("#" * 112)

    sarsa_agent = SARSAAgent(
        alpha=ALPHA, gamma=GAMMA,
        epsilon_start=EPS_START, epsilon_end=EPS_END,
        epsilon_decay_steps=EPS_DECAY, q_init=-5.0)
    sarsa_agent.set_rng(np.random.RandomState(456))

    train_agent(sarsa_agent,
                total_steps=TRAIN_STEPS, eval_interval=200_000,
                eval_steps=EVAL_STEPS, lam=LAM, traffic=TRAFFIC,
                reset_interval=50_000, eval_seed=EVAL_SEED)
    agents['SARSA'] = sarsa_agent

    # ── Head-to-head comparison ──
    run_comparison(lam=LAM, traffic=TRAFFIC, eval_steps=EVAL_STEPS,
                   eval_seed=EVAL_SEED, agents=agents)

    # ── Compare on different traffic patterns ──
    for tp in ['up_peak', 'down_peak', 'mixed']:
        print(f"\n{'#'*112}")
        print(f"  Traffic: {tp}")
        print(f"{'#'*112}")

        q2 = QLearningAgent(alpha=ALPHA, gamma=GAMMA,
                            epsilon_start=EPS_START, epsilon_end=EPS_END,
                            epsilon_decay_steps=EPS_DECAY, q_init=-5.0)
        q2.set_rng(np.random.RandomState(456))
        train_agent(q2, total_steps=TRAIN_STEPS, eval_interval=200_000,
                    eval_steps=EVAL_STEPS, lam=LAM, traffic=tp,
                    reset_interval=50_000, eval_seed=EVAL_SEED)

        s2 = SARSAAgent(alpha=ALPHA, gamma=GAMMA,
                        epsilon_start=EPS_START, epsilon_end=EPS_END,
                        epsilon_decay_steps=EPS_DECAY, q_init=-5.0)
        s2.set_rng(np.random.RandomState(456))
        train_agent(s2, total_steps=TRAIN_STEPS, eval_interval=200_000,
                    eval_steps=EVAL_STEPS, lam=LAM, traffic=tp,
                    reset_interval=50_000, eval_seed=EVAL_SEED)

        run_comparison(lam=LAM, traffic=tp, eval_steps=EVAL_STEPS,
                       eval_seed=EVAL_SEED,
                       agents={f'Q-{tp}': q2, f'SARSA-{tp}': s2})

    # ── High traffic ──
    print(f"\n{'#'*112}")
    print(f"  High traffic: λ=0.1, uniform")
    print(f"{'#'*112}")

    q3 = QLearningAgent(alpha=ALPHA, gamma=GAMMA,
                        epsilon_start=EPS_START, epsilon_end=EPS_END,
                        epsilon_decay_steps=EPS_DECAY, q_init=-5.0)
    q3.set_rng(np.random.RandomState(456))
    train_agent(q3, total_steps=TRAIN_STEPS, eval_interval=200_000,
                eval_steps=EVAL_STEPS, lam=0.1, traffic='uniform',
                reset_interval=50_000, eval_seed=EVAL_SEED)

    s3 = SARSAAgent(alpha=ALPHA, gamma=GAMMA,
                    epsilon_start=EPS_START, epsilon_end=EPS_END,
                    epsilon_decay_steps=EPS_DECAY, q_init=-5.0)
    s3.set_rng(np.random.RandomState(456))
    train_agent(s3, total_steps=TRAIN_STEPS, eval_interval=200_000,
                eval_steps=EVAL_STEPS, lam=0.01, traffic='uniform',
                reset_interval=50_000, eval_seed=EVAL_SEED)

    run_comparison(lam=0.01, traffic='uniform', eval_steps=EVAL_STEPS,
                   eval_seed=EVAL_SEED,
                   agents={'Q-high': q3, 'SARSA-high': s3})