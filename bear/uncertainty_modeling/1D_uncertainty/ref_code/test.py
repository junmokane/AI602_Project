# imports!
import numpy as np
import matplotlib.pyplot as plt


# Set up the prior
mu_w = 0
mu_b = 0

sigma_w = 0.2
sigma_b = 0.2

w_0 = np.hstack([mu_b, mu_w])[:, None]
V_0 = np.diag([sigma_b, sigma_w])**2

# Get observations
true_sigma_y = 0.1
true_w = np.array([[2, 0.3]]).T

X_in = 2 * np.random.rand(11, 1) - 1

Phi_X_in = np.hstack((
    np.ones((X_in.shape[0], 1)),  # pad with 1s for the bias term
    X_in
))

true_sigma_y = 0.05
noise = true_sigma_y * np.random.randn(X_in.shape[0], 1)

y = Phi_X_in @ true_w + noise

# Compute the posterior
sigma_y = true_sigma_y  # I'm going to guess the noise correctly

V0_inv = np.linalg.inv(V_0)
V_n = sigma_y**2 * np.linalg.inv(sigma_y**2 * V0_inv + (Phi_X_in.T @ Phi_X_in))
w_n = V_n @ V0_inv @ w_0 + 1 / (sigma_y**2) * V_n @ Phi_X_in.T @ y



# hrm, plotting the matrix made for `N` duplicate labels.
# https://stackoverflow.com/questions/26337493/pyplot-combine-multiple-line-labels-in-legend
def get_dedup_labels(plt):
    handles, labels = plt.gca().get_legend_handles_labels()
    new_handles = []
    new_labels = []
    for handle, label in zip(handles, labels):
        if label not in new_labels:
            new_handles.append(handle)
            new_labels.append(label)
    return new_handles, new_labels

grid_size = 0.01
x_grid = np.arange(-1, 1, grid_size)[:, None]
N = 100

Phi_X = np.hstack((
    np.ones((x_grid.shape[0], 1)),  # pad with 1s for the bias term
    x_grid
))

w = np.random.randn(N, 2) @ np.linalg.cholesky(V_n) + w_n.T

plt.clf()
plt.figure(figsize=(8, 6))
plt.plot(x_grid, Phi_X @ w.T, '-m', alpha=.2, label='weights sampled from posterior')
plt.plot(X_in, y, 'xk', label='observations')
plt.legend(*get_dedup_labels(plt))
# maybe_save_plot('2018-01-10-samples')  # Graph showing x's for observations, a line from the mean Bayesian prediction, and shaded area of uncertainty.
plt.show()


w = np.random.randn(1, 2) @ np.linalg.cholesky(V_n) + w_n.T

mean_pred = Phi_X @ w.T

plt.clf()
plt.figure(figsize=(8, 6))

upper_bound = mean_pred[:, 0] + 2 * sigma_y
lower_bound = mean_pred[:, 0] - 2 * sigma_y

plt.plot(x_grid, mean_pred[:, 0], '-m', label='weight sampled from posterior')
plt.fill_between(x_grid[:, 0], lower_bound, upper_bound, color='m', alpha=0.2, label='two standard deviations')

plt.plot(X_in, y, 'xk', label='observations')
# maybe_save_plot('2018-01-10-sample-with-error')  # Graph showing x's for observations, a line for one sample of the weights, and shaded area for uncertainty.
plt.show()

