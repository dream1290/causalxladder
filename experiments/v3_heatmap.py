"""
v3 Heatmap Sweep
Sweep (flip_mean, penalty) parameter space, measure equilibrium causal capacity.
Produce a heatmap showing the continuous crossover surface.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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

def evolve_quick(flip_mean, big_penalty, gens=150, pop=60):
    """Shorter evolution for sweep — 150 gens, 60 agents, average last 20 gens."""
    env = ConfoundedLeverWorld(
        flip_mean=flip_mean, big_penalty=big_penalty,
        max_steps=1000 if flip_mean < 200 else 2000
    )
    pop_genomes = [
        np.array([0.12, 0.12, 0.5, 0.25, 0.08, np.random.uniform(0.0, 0.22)])
        for _ in range(pop)
    ]
    late_caps = []
    for g in range(gens):
        agents = [ContinuousAgent(gm) for gm in pop_genomes]
        fits = [run_episode(env, a) for a in agents]
        best = np.argsort(fits)[-pop // 2:]
        new_pop = []
        for i in best:
            new_pop.append(pop_genomes[i].copy())
            new_pop.append(mutate(pop_genomes[i]))
        pop_genomes = new_pop[:pop]
        if g >= gens - 20:  # average last 20 generations
            late_caps.append(np.mean([a.capacity for a in agents]))
    return np.mean(late_caps)

# ── Sweep ────────────────────────────────────────────────────────────────────

flip_means = [40, 80, 200, 500, 1000, 2500]
penalties = [-2, -5, -10, -20, -25, -30]

NUM_SEEDS = 2

results = np.zeros((len(flip_means), len(penalties)))

total_cells = len(flip_means) * len(penalties)
cell = 0

for i, fm in enumerate(flip_means):
    for j, p in enumerate(penalties):
        cell += 1
        caps = []
        for s in range(NUM_SEEDS):
            np.random.seed(s * 31 + i * 7 + j * 13)
            c = evolve_quick(fm, p)
            caps.append(c)
        results[i, j] = np.mean(caps)
        print(f"[{cell:3d}/{total_cells}]  flip={fm:5d}  penalty={p:3d}  →  capacity={results[i,j]:.3f}")

# ── Heatmap ──────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 7))

# flip_means on Y (rows), penalties on X (columns)
# We want high flip_mean (rare flips = easy) at top, low at bottom
im = ax.imshow(results, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1.0,
               origin='lower')

ax.set_xticks(range(len(penalties)))
ax.set_xticklabels([str(p) for p in penalties], fontsize=11)
ax.set_yticks(range(len(flip_means)))
ax.set_yticklabels([str(fm) for fm in flip_means], fontsize=11)

ax.set_xlabel('Penalty Severity', fontsize=14)
ax.set_ylabel('Mean Flips Between Confound Changes (flip_mean)', fontsize=14)
ax.set_title('Equilibrium Causal Capacity Across Environmental Regimes', fontsize=15)

# Annotate cells
for i in range(len(flip_means)):
    for j in range(len(penalties)):
        val = results[i, j]
        color = 'white' if val > 0.55 or val < 0.15 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=10, fontweight='bold', color=color)

cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label('Evolved Causal Capacity [0–1]', fontsize=12)

fig.tight_layout()
fig.savefig('capacity_heatmap.png', dpi=150)
print(f"\nSaved capacity_heatmap.png")

# Print the raw matrix
print(f"\n{'='*60}")
print("Raw capacity matrix (rows=flip_mean, cols=penalty):")
print(f"flip_means: {flip_means}")
print(f"penalties:  {penalties}")
print(results.round(3))
print(f"{'='*60}")

plt.show()
