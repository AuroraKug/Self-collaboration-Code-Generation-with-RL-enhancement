import random
import collections
import json

class RLAgent:
    def __init__(self, action_space=None, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        """
        Initialize the RL Agent with Q-learning parameters.
        action_space: A list of possible actions.
        learning_rate (alpha): How much new information overrides old information.
        discount_factor (gamma): Importance of future rewards.
        exploration_rate (epsilon): Probability of choosing a random action.
        """
        self.action_space = action_space if action_space else ["default_action"]
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        
        # Q-table: stores Q-values for (state, action) pairs
        # Using collections.defaultdict for easier handling of new states
        self.q_table = collections.defaultdict(lambda: collections.defaultdict(float))

    def _get_state_str(self, state):
        """Converts a state dictionary to a hashable string for Q-table keys."""
        # Simple implementation: sort by key to ensure consistency
        if isinstance(state, dict):
            return json.dumps(state, sort_keys=True)
        return str(state) # Fallback for non-dict states

    def get_action(self, state):
        """
        Choose an action based on the current state using epsilon-greedy strategy.
        """
        state_str = self._get_state_str(state)
        
        # Exploration vs. Exploitation
        if random.random() < self.epsilon:
            # Explore: choose a random action
            action = random.choice(self.action_space)
            print(f"RLAgent (Explore): Choosing random action: {action} for state: {state_str}")
        else:
            # Exploit: choose the best known action for the current state
            # If state not in q_table or all actions have Q-value 0, pick randomly to encourage exploration
            if not self.q_table[state_str] or all(v == 0 for v in self.q_table[state_str].values()):
                action = random.choice(self.action_space)
                print(f"RLAgent (Exploit - New State/All Zero Q): Choosing random action: {action} for state: {state_str}")
            else:
                action = max(self.q_table[state_str], key=self.q_table[state_str].get)
                print(f"RLAgent (Exploit): Choosing best action: {action} with Q-value {self.q_table[state_str][action]} for state: {state_str}")
        return action

    def update(self, state, action, reward, next_state, done):
        """
        Update the Q-value for the state-action pair using the Q-learning formula.
        Q(s, a) = Q(s, a) + lr * [reward + gamma * max_a' Q(s', a') - Q(s, a)]
        """
        state_str = self._get_state_str(state)
        next_state_str = self._get_state_str(next_state)
        
        # Current Q-value
        old_q_value = self.q_table[state_str][action]
        
        # Max Q-value for the next state (0 if next state is terminal/done)
        next_max_q = 0
        if not done and self.q_table[next_state_str]:
            next_max_q = max(self.q_table[next_state_str].values())
        
        # Q-learning formula
        new_q_value = old_q_value + self.lr * (reward + self.gamma * next_max_q - old_q_value)
        self.q_table[state_str][action] = new_q_value
        
        print(f"RLAgent: Updating Q-table for state: {state_str}, action: {action}")
        print(f"         Old Q: {old_q_value:.2f}, New Q: {new_q_value:.2f}, Reward: {reward}, NextMaxQ: {next_max_q:.2f}")

# Example of how an action space could be defined for the Coder
CODER_ACTION_SPACE = [
    "No_Modification",
    "Instruction_Focus_Error_Type_{error_type}", # Placeholder, {error_type} would be filled
    "Instruction_Simplify_Approach",
    "Instruction_Check_Definitions"
]

# It's good practice to be able to save/load the Q-table for persistence
# For simplicity, these are not implemented here but would be important for real use.
# def save_q_table(q_table, filename="q_table.json"):
#     with open(filename, 'w') as f:
#         # Convert defaultdict to dict for JSON serialization
#         serializable_q_table = {k: dict(v) for k, v in q_table.items()}
#         json.dump(serializable_q_table, f)

# def load_q_table(filename="q_table.json"):
#     try:
#         with open(filename, 'r') as f:
#             loaded_data = json.load(f)
#             q_table = collections.defaultdict(lambda: collections.defaultdict(float))
#             for k, v_dict in loaded_data.items():
#                 for action, value in v_dict.items():
#                     q_table[k][action] = value
#             return q_table
#     except FileNotFoundError:
#         return collections.defaultdict(lambda: collections.defaultdict(float))

# We need json for _get_state_str if state is a dict
import json

# Example of how an action space could be defined for the Coder
CODER_ACTION_SPACE = [
    "No_Modification",
    "Instruction_Focus_Error_Type_{error_type}", # Placeholder, {error_type} would be filled
    "Instruction_Simplify_Approach",
    "Instruction_Check_Definitions"
] 