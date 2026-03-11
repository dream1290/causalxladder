import numpy as np
import matplotlib.pyplot as plt

from REPL import ConfoundedLeverWorld, run_episode

class EvolvableAgent:
    def __init__(self, genome):
        self.genome = np.array(genome, dtype=float)
        self.has_hidden = bool(genome[-1] > 0.5)
        self.value = np.zeros(2)
        self.belief = genome[2]
        self.alpha = np.clip(genome[0], 0.01, 0.5)
        self.epsilon = np.clip(genome[1], 0.01, 0.3)
        self.belief_rate = np.clip(genome[3], 0.1, 0.9)
        self.wait_prob = np.clip(genome[4], 0.0, 0.4)
        self.meta_cost = np.clip(genome[5], 0.0, 0.02)

    def reset(self):
        self.belief = self.genome[2]
        self.value = np.zeros(2)

    def act(self, last_correct=False, last_act=None):
        if not self.has_hidden:
            if np.random.rand() < self.epsilon:
                return np.random.randint(0, 2)
            return np.argmax(self.value)
        # causal path
        if last_act is not None and last_act < 2:
            obs = last_act if last_correct else (1 - last_act)
            self.belief = (1 - self.belief_rate) * self.belief + self.belief_rate * obs
        if np.abs(self.belief - 0.5) < 0.3 and np.random.rand() < self.wait_prob:
            return 2  # Wait
        return 0 if self.belief < 0.5 else 1

    def update(self, action, reward):
        if not self.has_hidden:
            self.value[action] += self.alpha * (reward - self.value[action])

def mutate(genome):
    new_g = genome.copy()
    new_g[:6] += np.random.normal(0, 0.05, 6)
    if np.random.rand() < 0.08:
        new_g[-1] = 1 - new_g[-1]  # flip has_hidden
    return new_g

def evolve(regime_name, flip_mean, big_penalty, generations=500, pop_size=80):
    env = ConfoundedLeverWorld(flip_mean=flip_mean, big_penalty=big_penalty, max_steps=800)
    population = [np.random.uniform([0.01,0.01,0.1,0.1,0.0,0.0,0,0,0], [0.5,0.3,0.9,0.9,0.4,0.02,1,1,1]) for _ in range(pop_size)]
    history_hidden = []

    for gen in range(generations):
        agents = [EvolvableAgent(g) for g in population]
        fitness = []
        for a in agents:
            score = run_episode(env, a, causal=a.has_hidden)   # reuse run_episode but ignore its internal causal flag
            # subtract metabolic cost already handled inside step via agent.meta_cost, but we add it explicitly here for clarity
            fitness.append(score - a.meta_cost * 800)  # rough per-episode cost
        # selection + reproduction
        idx = np.argsort(fitness)[-pop_size//2:]
        new_pop = []
        for i in idx:
            new_pop.append(population[i].copy())
            new_pop.append(mutate(population[i]))
        population = new_pop[:pop_size]
        frac_hidden = np.mean([a.has_hidden for a in agents])
        history_hidden.append(frac_hidden)
        if gen % 50 == 0:
            print(f"{regime_name} gen {gen}: {frac_hidden:.1%} hidden")

    return history_hidden

# Run both regimes
print("High-pressure evolution...")
high = evolve("HIGH", flip_mean=80, big_penalty=-25)
print("\nLow-pressure evolution...")
low = evolve("LOW", flip_mean=500, big_penalty=-5)

plt.figure(figsize=(10,5))
plt.plot(high, label='High pressure (should force causal)', linewidth=2)
plt.plot(low, label='Low pressure (control)', linewidth=2)
plt.xlabel('Generation')
plt.ylabel('% population with hidden causal tracker')
plt.title('Emergence of Causal Representation under Selection Pressure')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('evo_emergence.png')
plt.show()