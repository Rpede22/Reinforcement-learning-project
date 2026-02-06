
import random
import numpy as np

class Q_ENSEMBLE_MIN:

    def __init__(self, env , ensemble_size = 5,alpha = 0.9, gamma = 0.95, epsilon = 1, epsilon_decay = 0.99, min_epsilon = 0.01): 
        self.env = env

        self.num_states = env.num_states
        self.num_actions = env.num_actions

        self.alpha = alpha
        self.gamma = gamma

        #M number of q_tables
        self.M = ensemble_size
        self.q = [np.zeros((self.num_states, self.num_actions)) for _ in range(self.M)]

        # epsilon schedule
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon   = min_epsilon

    def choose_action(self, state, eps):
        if random.random() < eps:
            a = random.randint(0, self.num_actions - 1)
        else:
            #getting sum of all q_tables to determine action. 
            total_q = sum(q_table[state] for q_table in self.q)
            a = int(np.argmax(total_q))
        return a
    
    def train(self, num_episodes=10000, max_steps=1000, window_size=50):
        eps = self.epsilon
        
        episode_returns = []

        for ep in range(num_episodes):
            state = self.env.reset()
            total_r    = 0.0

            for step in range(max_steps):
                action = self.choose_action(state, eps)
                next_state, reward, done = self.env.step(action)
                
                total_r   += reward  

                # best action via sum
                sums = sum(q[next_state] for q in self.q)
                a_star = int(np.argmax(sums))

                # minimum value at a_star
                clipped = min(q[next_state, a_star] for q in self.q)

                #build target
                target = reward + self.gamma * clipped

                # Choose randomm q_table to update. 
                k = random.randrange(self.M)
                self.q[k][state, action] += self.alpha * (
                    target - self.q[k][state, action]
                )

                state = next_state

                if done:
                    break
                
            episode_returns.append(total_r)  
            # decay epsilon
            eps = max(self.min_epsilon, eps * self.epsilon_decay)
            
            if ep % window_size == 0:
                recent_sum = np.sum(episode_returns[-window_size:])
                print(f"[Episode {ep:5d}/{num_episodes}] "
                        f"sum return(last {window_size}) = {recent_sum:.2f}, "
                        f"ε = {eps:.3f}"
                        )
    
        return  episode_returns

    def get_Q_values(self, state):
        return sum(q_table[state] for q_table in self.q)

    def evaluate_return(self, state, max_steps=200):
        # Run a greedy rollout from a given state to compute the actual return
        total_reward = 0.0
        gamma_t = 1.0
        current_state = state

        for step in range(max_steps):
            # Use summed Q-values to select greedy action
            total_q = sum(q[current_state] for q in self.q)
            action = np.argmax(total_q)
            next_state, reward, done = self.env.step(action)

            total_reward += gamma_t * reward
            gamma_t *= self.gamma
            current_state = next_state

            if done:
                break

        return total_reward
