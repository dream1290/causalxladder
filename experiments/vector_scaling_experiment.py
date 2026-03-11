import numpy as np
import matplotlib.pyplot as plt

class ConfoundedLeverWorld:
    def __init__(self, hidden_dim=1, max_steps=1200, flip_mean=80, big_reward=10.0, big_penalty=-30.0,
                 step_cost=-0.01, pull_cost=-0.05, correct_streak=8, wrong_streak=6):
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        self.flip_mean = flip_mean
        self.big_reward = big_reward
        self.big_penalty = big_penalty
        self.step_cost = step_cost
        self.pull_cost = pull_cost
        self.correct_streak = correct_streak
        self.wrong_streak = wrong_streak

    def reset(self):
        self.C = np.random.randint(0, 2, self.hidden_dim)
        self.steps = 0
        self.correct_consec = 0
        self.wrong_consec = 0
        self.total_reward = 0.0
        self.flip_timers = np.random.geometric(1.0 / self.flip_mean, self.hidden_dim)
        return None

    def step(self, action):
        reward = self.step_cost
        done = False
        info = {'correct_pull': False}

        if action == 2:  # Wait
            self.steps += 1
            self.correct_consec = self.wrong_consec = 0
            if self.steps >= self.max_steps: done = True
            self.total_reward += reward
            return reward, done, info

        # Majority vote on hidden bits
        correct_action = 1 if np.sum(self.C) > self.hidden_dim // 2 else 0
        is_correct = (action == correct_action)
        reward += self.pull_cost

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

        # Independent flips
        self.flip_timers -= 1
        for i in range(self.hidden_dim):
            if self.flip_timers[i] <= 0:
                self.C[i] = 1 - self.C[i]
                self.flip_timers[i] = np.random.geometric(1.0 / self.flip_mean)

        if self.steps >= self.max_steps:
            done = True
        return reward, done, info

class VectorBeliefAgent:
    def __init__(self, genome, hidden_dim):
        self.alpha = np.clip(genome[0], 0.01, 0.5)
        self.epsilon = np.clip(genome[1], 0.01, 0.3)
        self.belief_rate_base = np.clip(genome[2], 0.1, 0.9)
        self.wait_prob_base = np.clip(genome[3], 0.0, 0.4)
        self.capacity = np.clip(genome[4], 0.0, 1.0)
        self.hidden_dim = hidden_dim
        self.beliefs = np.full(hidden_dim, 0.5)  # one scalar belief per hidden bit
        self.value = np.zeros(2)

    def act(self, last_correct=False, last_act=None):
        eff_rate = 0.05 + self.capacity * 0.85
        eff_wait = self.capacity * self.wait_prob_base
        if last_act is not None and last_act < 2:
            obs = 1.0 if last_correct else 0.0
            self.beliefs = (1 - eff_rate) * self.beliefs + eff_rate * obs   # same aggregate signal for every bit
        mean_belief = np.mean(self.beliefs)
        if np.abs(mean_belief - 0.5) < 0.35 and np.random.rand() < eff_wait:
            return 2
        if self.capacity < 0.25:
            if np.random.rand() < self.epsilon:
                return np.random.randint(0, 2)
            return np.argmax(self.value)
        return 1 if mean_belief > 0.5 else 0

    def update(self, action, reward):
        if action < 2 and self.capacity < 0.3:
            self.value[action] += self.alpha * (reward - self.value[action])

def mutate(genome):
    new_g = genome.copy()
    new_g += np.random.normal(0, 0.035, len(genome))
    new_g[4] = np.clip(new_g[4], 0.0, 1.0)
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

def evolve(name, hidden_dim, flip_mean, big_penalty, gens=150, pop=80):
    env = ConfoundedLeverWorld(hidden_dim=hidden_dim, flip_mean=flip_mean, big_penalty=big_penalty,
                               max_steps=1200 if flip_mean < 200 else 2500)
    pop_genomes = [np.array([0.12, 0.12, 0.25, 0.08, np.random.uniform(0.0, 0.22)]) for _ in range(pop)]
    history = []
    for g in range(gens):
        agents = [VectorBeliefAgent(gm, hidden_dim) for gm in pop_genomes]
        fits = [run_episode(env, a) for a in agents]
        best = np.argsort(fits)[-pop//2:]
        new_pop = []
        for i in best:
            new_pop.append(pop_genomes[i].copy())
            new_pop.append(mutate(pop_genomes[i]))
        pop_genomes = new_pop[:pop]
        avg_cap = np.mean([a.capacity for a in agents])
        history.append(avg_cap)
        if g % 50 == 0 or g == gens-1:
            print(f"{name} d={hidden_dim} gen {g:3d}: avg cap {avg_cap:.3f} | mean fit {np.mean(fits):6.0f}")
    return history

# === Run the vector-belief scaling sweep ===
np.random.seed(42)
dims = [1, 4, 8, 16]
flip_means = [40, 80, 200, 500, 1000, 2500]
penalties = [-2, -10, -30]

results = {}
for d in dims:
    print(f"\n=== Vector-belief scaling: hidden_dim = {d} ===")
    eq_caps = np.zeros((len(flip_means), len(penalties)))
    for i, fm in enumerate(flip_means):
        for j, p in enumerate(penalties):
            high = evolve(f"HIGH_d{d}_fm{fm}_p{p}", d, fm, p, gens=150)
            eq_caps[i, j] = high[-1]
    results[d] = eq_caps

# Plot
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.ravel()
for idx, d in enumerate(dims):
    im = axs[idx].imshow(results[d], cmap='RdYlBu_r', origin='lower', vmin=0.0, vmax=0.65)
    axs[idx].set_title(f'hidden_dim = {d} (vector belief)')
    axs[idx].set_xticks(range(len(penalties)))
    axs[idx].set_xticklabels(penalties)
    axs[idx].set_yticks(range(len(flip_means)))
    axs[idx].set_yticklabels(flip_means)
    axs[idx].set_xlabel('Penalty Severity')
    axs[idx].set_ylabel('Flip Mean')
    plt.colorbar(im, ax=axs[idx], label='Equilibrium Causal Capacity')
plt.suptitle('Vector Belief Scaling — Phase Transition Recovery?')
plt.tight_layout()
plt.savefig('vector_scaling_heatmap.png')
plt.show()

print("Vector-belief scaling complete. Check vector_scaling_heatmap.png")