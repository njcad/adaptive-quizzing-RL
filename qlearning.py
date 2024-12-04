"""
Q Learning on preprocessed RIIID data.
"""

import pandas as pd
from tqdm import tqdm
import time


def build_q_table(df):
    """
    Q-Table has a row for each state and a column for each action.
    We will initialize it to 0s here and update it in the q_learning algorithm.
    I'm implementing this as a nested dictionary:
        q_table = {state : { action : q_value }}
    """
    actions = [1, 2, 3, 4, 5]
    states = df['s'].unique()
    next_states = df['sp'].unique()
    states = set(states).union(set(next_states))

    q_table = {state : {action : 0 for action in actions} for state in states}
    return q_table
    

def q_learning(num_epochs, alpha, gamma, q_table, data, policy_path):
    """
    Q-Learning algorithm according to alg 17.10
    Do num_epochs of passes over the dataset using alpha learning rate and gamma discount factor.
    Update q_table as we go.
    Use a decaying value of alpha.
    """
    # save initial alpha since we will decay it
    alpha_0 = alpha
    min_alpha = 0.001
    decay = 0.9

    for i in range(num_epochs):
        # calculate alpha
        alpha_i = max(min_alpha, alpha_0 * (decay ** i))

        # iterate over each data point
        for _, row in tqdm(data.iterrows()):
            s = row['s']
            a = row['a']
            r = row['r']
            sp = row['sp']

            # identify current value in q table for s and a
            curr_q = q_table[s][a]

            # identify max q value for next state
            max_next_q = max(q_table[sp].values())

            # calculate update according to 17.10
            update = r + (gamma * max_next_q) - curr_q
            q_table[s][a] = curr_q + (alpha_i * update)

    # extract policy as max q-value action for each state
    policy = {}
    for state in q_table.keys():
        best_action = max(q_table[state], key=q_table[state].get)
        policy[state] = best_action   
    
    # save the policy
    write_policy(policy, policy_path)


def write_policy(policy, policy_path):
    """
    Given policy dict {s: a}, write each line to policy path.
    """
    with open(policy_path, 'w') as f:
        for state in sorted(policy.keys()):
            f.write(f"{state} : {policy[state]}\n")
    

def main():
    # track runtime
    start = time.time()

    # declare paths
    data_path = "../sar_spaces/sar_space.csv"
    policy_path = "../polcies/v4.policy"

    # load the data
    df = pd.read_csv(data_path)

    # define hyperparameters  
    alpha = 0.5
    num_epochs = 35
    gamma = 0.99
   
    # initialize q table
    q_table = build_q_table(df)

    # call q-learning fn
    q_learning(num_epochs, alpha, gamma, q_table, df, policy_path)

    # runtime
    end = time.time()
    print(f"\nTotal runtime = {end - start} seconds.")


if __name__ == '__main__':
    main()
