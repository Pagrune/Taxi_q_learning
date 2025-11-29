import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    """
    Update Q(s,a) using the Q-learning update rule.
    """
    max_next_Q = np.max(Q[sprime, :])

    Q[s, a] = Q[s, a] + alpha * (r + gamma * max_next_Q - Q[s, a])

    return Q


def epsilon_greedy(Q, s, epsilone):
    """
    Choose an action following epsilon-greedy policy.
    """
    if np.random.rand() < epsilone:
        return np.random.randint(Q.shape[1])
    else:
        return np.argmax(Q[s, :])



if __name__ == "__main__":
    #env = gym.make("Taxi-v3")
    env = gym.make("Taxi-v3", render_mode="human")


    env.reset()
    env.render()

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.3 # choose your own

    gamma = 0.95 # choose your own

    epsilon = 0.3 # choose your own

    n_epochs = 2000 # choose your own
    max_itr_per_epoch = 200 # choose your own
    rewards = []

    for e in range(n_epochs):
        r = 0
        S, _ = env.reset()

        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q=Q, s=S, epsilone=epsilon)

            Sprime, R, done, _, info = env.step(A)

            r += R

            Q = update_q_table(
                Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma
            )

            S = Sprime
            if done:
                break

        print("episode #", e, " : r = ", r)
        rewards.append(r)

    print("Average reward = ", np.mean(rewards))

    # plot the rewards in function of epochs

    print("Training finished.\n")

    
    """
    
    Evaluate the q-learning algorihtm
    
    """

    env.close()