import numpy as np
import matplotlib.pyplot as plt

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
        if action < 2 and self.capacity < 0.3:  # only Q-learning for low capacity
            self.value[action] += self.alpha * (reward - self.value[action])

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

def evolve(name, flip_mean, big_penalty, gens=300, pop=100):
    env = ConfoundedLeverWorld(flip_mean=flip_mean, big_penalty=big_penalty, max_steps=1200 if flip_mean < 200 else 2500)
    pop_genomes = [np.array([0.12, 0.12, 0.5, 0.25, 0.08, np.random.uniform(0.0, 0.22)]) for _ in range(pop)]
    history = []
    for g in range(gens):
        agents = [ContinuousAgent(gm) for gm in pop_genomes]
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
            print(f"{name} gen {g:3d}: avg capacity {avg_cap:.3f}  |  mean fitness {np.mean(fits):6.0f}")
    return history

np.random.seed(42)
print("=== High pressure ===")
high = evolve("HIGH", 60, -30)
print("\n=== Low pressure ===")
low = evolve("LOW", 2500, -2)

plt.figure(figsize=(10, 5))
plt.plot(high, label='High pressure (forces causal)', linewidth=3)
plt.plot(low, label='Low pressure (associative wins)', linewidth=3)
plt.xlabel('Generation')
plt.ylabel('Population Average Causal Capacity [0–1]')
plt.title('Continuous Emergence of Causal Capacity — v3')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('continuous_emergence_v3.png')
plt.show()