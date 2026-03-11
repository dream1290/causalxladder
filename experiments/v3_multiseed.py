"""
v3 Multi-Seed Replication
Run 8 seeds for both high-pressure and low-pressure regimes.
Plot mean ± std envelope to confirm equilibrium stability.
"""
import numpy as np
import matplotlib.pyplot as plt

# ── Environment ──────────────────────────────────────────────────────────────

class ConfoundedLeverWorld:
    def __init__(self, max_steps=1200, flip_mean=80, big_reward=10.0, big_penalty=-30.0,
                 step_cost=-0.01, pull_cost=-0.05, correct_streak=8, wrong_streak=6):
        self.max_steps = max_steps
        self.flip_mean = flip_mean
        self.big_reward = big_reward
        self.big_penalty = big_penalty
        self.step_cost = step_cost
        self.pull_cost = pull_cost
        self.correct_streak = correct_streak
        self.wrong_streak = wrong_streak

    def reset(self):
        self.C = np.random.randint(0, 2)
        self.steps = 0
        self.correct_consec = 0
        self.wrong_consec = 0
        self.total_reward = 0.0
        self.flip_timer = np.random.geometric(1.0 / self.flip_mean)

    def step(self, action):
        reward = self.step_cost
        done = False
        info = {'correct_pull': False}
        if action == 2:
            self.steps += 1
            self.correct_consec = self.wrong_consec = 0
            if self.steps >= self.max_steps: done = True
            self.total_reward += reward
            return reward, done, info
        reward += self.pull_cost
        is_correct = (action == self.C)
        if is_correct:
            reward += 0.2
            self.correct_consec += 1
            self.wrong_consec = 0
            info['correct_pull'] = True
            if self.correct_consec >= self.correct_streak:
                reward += self.big_reward
        else:
            reward += -0.2
            self.wrong_consec += 1
            self.correct_consec = 0
            if self.wrong_consec >= self.wrong_streak:
                reward += self.big_penalty
                done = True
        self.steps += 1
        self.total_reward += reward
        self.flip_timer -= 1
        if self.flip_timer <= 0:
            self.C = 1 - self.C
            self.flip_timer = np.random.geometric(1.0 / self.flip_mean)
        if self.steps >= self.max_steps:
            done = True
        return reward, done, info

# ── Agent ────────────────────────────────────────────────────────────────────

class ContinuousAgent:
    def __init__(self, genome):
        self.alpha = np.clip(genome[0], 0.01, 0.5)
        self.epsilon = np.clip(genome[1], 0.01, 0.3)
        self.belief = np.clip(genome[2], 0.1, 0.9)
        self.belief_rate_base = np.clip(genome[3], 0.1, 0.9)
        self.wait_prob_base = np.clip(genome[4], 0.0, 0.4)
        self.capacity = np.clip(genome[5], 0.0, 1.0)
        self.value = np.zeros(2)

    def act(self, last_correct=False, last_act=None):
        eff_rate = 0.05 + self.capacity * 0.85
        eff_wait = self.capacity * self.wait_prob_base
        if last_act is not None and last_act < 2:
            obs = last_act if last_correct else (1 - last_act)
            self.belief = (1 - eff_rate) * self.belief + eff_rate * obs
        if np.abs(self.belief - 0.5) < 0.35 and np.random.rand() < eff_wait:
            return 2
        if self.capacity < 0.25:
            if np.random.rand() < self.epsilon:
                return np.random.randint(0, 2)
            return np.argmax(self.value)
        return 0 if self.belief < 0.5 else 1

    def update(self, action, reward):
        if action < 2 and self.capacity < 0.3:
            self.value[action] += self.alpha * (reward - self.value[action])

# ── Evolution core ───────────────────────────────────────────────────────────

def mutate(genome):
    new_g = genome.copy()
    new_g += np.random.normal(0, 0.035, len(genome))
    new_g[5] = np.clip(new_g[5], 0.0, 1.0)
    return new_g

def run_episode(env, agent):
    env.reset()
    last_correct = False
    last_act = None
    done = False
    while not done:
        action = agent.act(last_correct, last_act)
        reward, done, info = env.step(action)
        last_correct = info.get('correct_pull', False)
        last_act = action
        agent.update(action, reward)
    meta_cost = 0.42 * (agent.capacity ** 1.75) * env.steps
    return env.total_reward - meta_cost

def evolve(name, flip_mean, big_penalty, gens=200, pop=80):
    env = ConfoundedLeverWorld(
        flip_mean=flip_mean, big_penalty=big_penalty,
        max_steps=1000 if flip_mean < 200 else 2000
    )
    pop_genomes = [
        np.array([0.12, 0.12, 0.5, 0.25, 0.08, np.random.uniform(0.0, 0.22)])
        for _ in range(pop)
    ]
    history_cap = []
    history_std = []
    for g in range(gens):
        agents = [ContinuousAgent(gm) for gm in pop_genomes]
        fits = [run_episode(env, a) for a in agents]
        best = np.argsort(fits)[-pop // 2:]
        new_pop = []
        for i in best:
            new_pop.append(pop_genomes[i].copy())
            new_pop.append(mutate(pop_genomes[i]))
        pop_genomes = new_pop[:pop]
        caps = [a.capacity for a in agents]
        history_cap.append(np.mean(caps))
        history_std.append(np.std(caps))
        if g % 100 == 0 or g == gens - 1:
            print(f"  {name} seed run gen {g:3d}: cap={np.mean(caps):.3f}±{np.std(caps):.3f}")
    return np.array(history_cap), np.array(history_std)

# ── Multi-seed experiment ────────────────────────────────────────────────────

NUM_SEEDS = 5
GENS = 200

high_runs = []
low_runs = []

for s in range(NUM_SEEDS):
    seed = s * 17 + 7  # spread seeds
    np.random.seed(seed)
    print(f"\n── Seed {seed} ──")
    print("  High pressure:")
    h_cap, h_std = evolve("HIGH", flip_mean=60, big_penalty=-30, gens=GENS)
    high_runs.append(h_cap)
    print("  Low pressure:")
    l_cap, l_std = evolve("LOW", flip_mean=2500, big_penalty=-2, gens=GENS)
    low_runs.append(l_cap)

high_all = np.array(high_runs)  # (NUM_SEEDS, GENS)
low_all = np.array(low_runs)

high_mean = high_all.mean(axis=0)
high_std = high_all.std(axis=0)
low_mean = low_all.mean(axis=0)
low_std = low_all.std(axis=0)

x = np.arange(GENS)

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 6))

# High pressure
ax.plot(x, high_mean, color='#1f77b4', linewidth=2.5, label='High pressure (flip=60, penalty=-30)')
ax.fill_between(x, high_mean - high_std, high_mean + high_std,
                color='#1f77b4', alpha=0.2)

# Low pressure
ax.plot(x, low_mean, color='#ff7f0e', linewidth=2.5, label='Low pressure (flip=2500, penalty=-2)')
ax.fill_between(x, low_mean - low_std, low_mean + low_std,
                color='#ff7f0e', alpha=0.2)

# Individual seed traces (faint)
for h in high_runs:
    ax.plot(x, h, color='#1f77b4', alpha=0.08, linewidth=0.8)
for l in low_runs:
    ax.plot(x, l, color='#ff7f0e', alpha=0.08, linewidth=0.8)

ax.set_xlabel('Generation', fontsize=13)
ax.set_ylabel('Population Average Causal Capacity [0–1]', fontsize=13)
ax.set_title(f'Emergence of Causal Capacity — {NUM_SEEDS}-Seed Replication', fontsize=15)
ax.legend(fontsize=12, loc='center right')
ax.set_ylim(-0.02, 1.02)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('multiseed_emergence.png', dpi=150)
print(f"\nSaved multiseed_emergence.png")

# Print summary
print(f"\n{'='*60}")
print(f"HIGH PRESSURE  final capacity: {high_mean[-1]:.3f} ± {high_std[-1]:.3f}")
print(f"LOW  PRESSURE  final capacity: {low_mean[-1]:.3f} ± {low_std[-1]:.3f}")
print(f"{'='*60}")

plt.show()
