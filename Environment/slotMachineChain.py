import numpy as np

class SlotMachineChain:
    """
    SlotMachine Chain MDP:
      - States: 0,1,...,N-1 (state N-1 is terminal finish line)
      - Actions: "WORK", "GAMBLE"
      - Transitions:
            WORK at s < terminal → s+1
            WORK at terminal  → stays at N-1
            GAMBLE at any s    → stays at s
      - Rewards:
            WORK: r_work each step; additionally +R_goal upon first entering N-1
            GAMBLE: with prob p_win → +B; else → -b
      - Episode ends when you hit state N-1.
    """

    def __init__(self, num_states=10, r_work=1.0, R_goal=10.0, p_win=0.1, B=20.0, b=1.0):
        self.num_states = num_states
        self.terminal_state = num_states - 1
        self.r_work = r_work
        self.R_goal = R_goal
        self.p_win = p_win
        self.B = B
        self.b = b
        self.num_actions = 2 

        self.transition_probs = self._init_transitions()

        # start out at state 0
        self.reset()

    def _init_transitions(self):
        """
        Build dict mapping (state, action) → list of (next_state, prob).
        """
        probs = {}
        for s in range(self.num_states):
            # GAMBLE: stay in same state
            probs[(s, 0)] = [(s, 1.0)]
            
            if s < self.terminal_state:
                # WORK: move forward if not terminal, else stay
                probs[(s, 1)] = [(s + 1, 1.0)]

            else:
                probs[(s, 1)] = [(s, 1.0)]
        return probs

    def step(self, action):
        """
        action ∈ {"WORK", "GAMBLE"}
        Returns (next_state, reward, done)
        """
        trans = self.transition_probs[(self.state, action)]
        next_states, probs = zip(*trans)
        self.state = np.random.choice(next_states, p=probs)
        done = False

        if action == 1:
            reward = self.r_work
            # bonus for reaching the goal
            if self.state == self.terminal_state:
                reward += self.R_goal
                done = True
        # GAMBLE
        else:  
            if np.random.rand() < self.p_win:
                reward = self.B
            else:
                reward = -self.b

        return self.state, reward, done

    def reset(self):
        #resets state to 0 
        self.state = 0
        return self.state
