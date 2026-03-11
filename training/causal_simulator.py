import numpy as np

class CausalSimulator:
    def __init__(self, hidden_dim=8, flip_mean=80, max_steps=200,
                 big_reward=10.0, big_penalty=-30.0,
                 step_cost=-0.01, pull_cost=-0.05,
                 correct_streak=8, wrong_streak=6):
        self.hidden_dim = hidden_dim
        self.flip_mean = flip_mean
        self.max_steps = max_steps
        self.big_reward = big_reward
        self.big_penalty = big_penalty
        self.step_cost = step_cost
        self.pull_cost = pull_cost
        self.correct_streak = correct_streak
        self.wrong_streak = wrong_streak

    def reset(self):
        """Reset hidden state C and return initial observation"""
        self.C = np.random.randint(0, 2, self.hidden_dim).astype(float)
        self.steps = 0
        self.correct_consec = 0
        self.wrong_consec = 0
        self.flip_timers = np.random.geometric(1.0 / self.flip_mean, self.hidden_dim)
        return self._get_obs()

    def step(self, action):  # action: 0=Red, 1=Blue, 2=Wait
        """Return (obs, reward, done)"""
        reward = self.step_cost
        done = False

        if action == 2:  # Wait is cheap observation
            self.steps += 1
            self.correct_consec = self.wrong_consec = 0
            if self.steps >= self.max_steps:
                done = True
            return self._get_obs(), reward, done

        # === Core causal rule: majority + correlation ===
        majority = 1 if np.sum(self.C) > self.hidden_dim // 2 else 0
        correlated_flip = (self.C[0] == 1 and self.C[1] == 1)  # extra rule
        correct_action = 1 - majority if correlated_flip else majority
        is_correct = (action == correct_action)

        reward += self.pull_cost
        if is_correct:
            reward += 0.2
            self.correct_consec += 1
            self.wrong_consec = 0
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

        # Independent geometric flips
        self.flip_timers -= 1
        for i in range(self.hidden_dim):
            if self.flip_timers[i] <= 0:
                self.C[i] = 1 - self.C[i]
                self.flip_timers[i] = np.random.geometric(1.0 / self.flip_mean)

        if self.steps >= self.max_steps:
            done = True
        return self._get_obs(), reward, done

    def _get_obs(self):
        """4-dimensional observation vector (never reveals C)"""
        last_correct = 0.0  # will be filled by training loop
        steps_hint = min(self.steps / 50.0, 1.0)  # normalized hint
        noise = np.random.randn()
        return np.array([0.0, float(last_correct), steps_hint, noise], dtype=np.float32)

# === Quick verification ===
if __name__ == "__main__":
    env = CausalSimulator(hidden_dim=8)
    obs = env.reset()
    print("Simulator ready — hidden_dim=8")
    print("Example obs shape:", obs.shape)
    r_total = 0.0
    for _ in range(50):
        a = np.random.randint(0, 3)
        obs, r, done = env.step(a)
        r_total += r
        if done:
            break
    print(f"Sample episode total reward: {r_total:.1f} (should vary wildly — that's the pressure)")