import numpy as np
import matplotlib.pyplot as plt

import torch


def distance(sim, obs):
    diff = sim - obs
    if not(np.all(np.isfinite(sim)) and np.all(np.isfinite(obs))):
        print("Non-finite simulation/obs values!", sim, obs)
        return np.inf
    return np.linalg.norm(diff)

def abc(prior, simulate, obs_summary, N_steps, eps = 1e-1):
    distances=[]
    accepted=[]
    accepted_summaries=[]
    rejected=[]
    rejected_summaries=[]

    for _ in range(N_steps):
        theta = prior.sample()
        sim_summary = simulate(theta)
        d = distance(sim_summary, obs_summary)
        distances.append(d)
        if d < eps:
            accepted.append(theta)
            accepted_summaries.append(sim_summary[0, :])
        else:
            rejected.append(theta)
            rejected_summaries.append(sim_summary[0, :])
    
    distances = np.array(distances)
    plt.hist(distances)
    plt.title('Histogram of ABC distances')
    plt.show()

    print(f"Accepted samples: {len(accepted)}, Rejected samples: {len(rejected)}")

    # print(accepted, rejected)
    plt.scatter(rejected_summaries, rejected, color='orange', alpha=0.5, label='Rejected')
    plt.scatter(accepted_summaries, accepted, color='green', alpha=1.0, label='Accepted')
    # plt.scatter(model(obs_data).detach().numpy(), [true_theta], color='purple', label='Observed Data', s=100, marker='D')
    plt.xlabel('Compressed Summary')
    plt.ylabel('Theta')
    plt.legend()
    plt.show()
    return accepted, accepted_summaries, rejected, rejected_summaries

def pmc_abc(n_sims, prior, eps = 1e-1):
    theta_prior = lambda n : prior.sample((n,))
    theta_particles = theta_prior(n_sims)
    weights = torch.ones(n_sims) / n_sims
    epsilons = []
    epsilon = np.inf
    iteration = 0
    stds = []

    while epsilon > eps and iteration < 8:
        iteration += 1
        simulated_summaries = np.zeros((n_sims, 1))
        distances = np.zeros(n_sims)
        for i, theta in enumerate(theta_particles):
            sim_summary = simulate(theta)
            # sim_summary = true_model(sim_data.numpy())
            simulated_summaries[i] = sim_summary
            if (np.isnan(sim_summary).any() or np.isinf(sim_summary).any()):
                print("Non-finite simulated summary for theta:", theta)
                continue
            distances[i] = distance(sim_summary, obs_summary)

        epsilon = np.percentile(distances, 75)  # 75 percentile is the threshold
        epsilons.append(epsilon)

        accepted_indices = distances < epsilon
        accepted_thetas = theta_particles[accepted_indices]
        accepted_summaries = simulated_summaries[accepted_indices]
        accepted_weights = weights[accepted_indices]

        rejected_indices = distances >= epsilon
        rejected_thetas = theta_particles[rejected_indices]
        rejected_summaries = simulated_summaries[rejected_indices]
        rejected_weights = weights[rejected_indices]

        mean = np.average(accepted_thetas, weights = accepted_weights+1e-3)
        std = np.sqrt(np.cov(accepted_thetas, aweights = accepted_weights))
        stds.append(std)

        theta_particles_new = torch.normal(mean, std, (n_sims,))
        # theta_particles_new[theta_particles_new < prior_min] = prior_min
        pdf = lambda x : np.exp(-0.5 * ((x - mean) / std)**2) / (std * np.sqrt(2 * np.pi))
        number_func = lambda theta_new, theta_old, std_old : np.exp(-0.5 * ((theta_new - theta_old) / std_old)**2) / (std_old * np.sqrt(2 * np.pi))
        deno = (number_func(theta_particles_new, theta_particles, std) * weights)

        weights_new = pdf(theta_particles) / deno.sum()

        theta_particles = theta_particles_new
        weights = weights_new / weights_new.sum()

        print(f"Iter {iteration}, {mean=}, {std=}, Epsilon: {epsilon:4f}, Accepted Particles: {accepted_thetas.shape[0]}")
    plt.plot(stds)
    return accepted_summaries, rejected_summaries, accepted_thetas, rejected_thetas