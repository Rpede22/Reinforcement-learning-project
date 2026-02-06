
import random
import numpy as np

class DDQN_MIN:

    def __init__(self, env , alpha = 0.9, gamma = 0.95, epsilon = 1, epsilon_decay = 0.99, min_epsilon = 0.01): 
        self.env = env

        self.num_states = env.num_states
        self.num_actions = env.num_actions

        #  allocate two Q‐tables: states × action_space
        self.q1 = np.zeros((self.num_states, self.num_actions))
        self.q2 = np.zeros((self.num_states, self.num_actions))

        self.alpha = alpha
        self.gamma = gamma

        # epsilon schedule
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon   = min_epsilon

    def choose_action(self, state, eps):
        if random.random() < eps:
            a = random.randint(0, self.num_actions - 1)
        else:
            a = int(np.argmax(self.q1[state] + self.q2[state]))
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
                
                total_r  += reward  

                #a* = arg MAX [Q1,Q2]
                a_star = int(np.argmax(self.q1[next_state] + self.q2[next_state]))

                #Clipped finding the minimum value, between two q_tables. 
                clipped = min(self.q1[next_state, a_star],
                                  self.q2[next_state, a_star])
                
                #forming target to update q_tables 
                target = reward + self.gamma * clipped

                #Now updating towards the same target. 
                if random.random() < 0.5:
                    # UPDATE(q1):
                    #   Q1(s,a) += α [ r + γ Q2(s',a*) − Q1(s,a) ]
                    self.q1[state, action] += self.alpha * (target - self.q1[state, action])

                else: 
                    # UPDATE(q2):
                    #   Q2(s,a) += α [ r + γ Q1(s',b*) − Q2(s,a) ]
                    self.q2[state, action] += self.alpha * (target - self.q2[state, action])

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
    
        return self.q1, self.q2, episode_returns

    def get_Q_values(self, state):
        return self.q1[state] + self.q2[state]

    def evaluate_return(self, state, max_steps=200):
        # Run a greedy rollout from a given state to compute the actual return
        total_reward = 0.0
        gamma_t = 1.0
        current_state = state

        for step in range(max_steps):
            action = np.argmax(self.q1[current_state])
            next_state, reward, done = self.env.step(action)

            total_reward += gamma_t * reward
            gamma_t *= self.gamma
            current_state = next_state

            if done:
                break

        return total_reward
