import numpy.random as rn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# print parameters
def print_params(cfg):

    print('Dataset Parameters:')
    print('     p =', cfg['dim'], '(input dimension)')
    print('     n =', cfg['n_train'], '(number of training examples)')
    print('     gamma =', cfg['gamma'], '(1 / |w*|')
    print('     pi_nonsep =', cfg['pi_nonsep'], '(fraction of non-separable examples)')
    print('Network Parameters:')
    print('     k =', cfg['k'], ' (hidden layer width is 2k)')
    print('     alpha =', cfg['alpha'], '(leaky relu parameter)')
    print('Optimization Parameters:')
    print('     eta =', cfg['learning_rate'], '(learning rate)')

    print('\nComponents of bound for the number of non-zero steps M:')
    print('     |w*|^2 / alpha^2 = {:.3f}'.format(1 / cfg['gamma']**2 / cfg['alpha']**2))
    print('     |w*|^2 / min(eta, sqrt(eta)) = {:.3f}'.format(1 / cfg['gamma']**2 /
                                                              min(cfg['learning_rate'], np.sqrt(cfg['learning_rate']))))
    print('Value for R, bound on initial weights norm:')
    print('     1/sqrt(2k) = {:.3f}'.format(1/np.sqrt(2*cfg['k'])))
    print('To guarantee expressiveness, k should be at least:')
    print('     2*(n/(2d-2)) = {}'.format(int(np.ceil(2*cfg['n_train']/(2*cfg['dim']-2)))))
    print('     k = ', cfg['k'])

def plot_data(x, y, w_star_hat, gamma):

    n = x.shape[0]
    dim = x.shape[1]
    w_star_norm = 1 / gamma

    w_rand_hat = rn.normal(0, 1, dim)
    w_rand_hat = w_rand_hat - np.dot(w_rand_hat, w_star_hat) * w_star_hat
    w_rand_hat = w_rand_hat / np.sqrt(np.sum(w_rand_hat**2))
    w_rand_hat_tile = np.tile(w_rand_hat, (n, 1))
    x_rand_proj = np.sum(x * w_rand_hat_tile, axis=1)

    w_star_hat_tile = np.tile(w_star_hat, (n, 1))
    x_star_proj = np.sum(x * w_star_hat_tile, axis=1)

    plot_x, plot_y = 8, 8
    n0, n1 = 1, 1
    fig, axes = plt.subplots(figsize=(plot_x*n1, plot_y*n0), nrows=n0, ncols = n1)
    fig.tight_layout(h_pad = 5)

    ax = axes
    ax.plot(x_star_proj[np.argwhere(y==0)], x_rand_proj[np.argwhere(y==0)],'.',label='y=0')
    ax.plot(x_star_proj[np.argwhere(y==1)], x_rand_proj[np.argwhere(y==1)],'.',label='y=1')
    ax.plot([-1/w_star_norm, -1/w_star_norm], [-1,1], '-k')
    ax.plot([1/w_star_norm, 1/w_star_norm], [-1,1], '-k')
    ax.legend(loc=4)

def plot_data_2d(x, y):

    plot_x, plot_y = 8, 8
    n0, n1 = 1, 1
    fig, axes = plt.subplots(figsize=(plot_x*n1, plot_y*n0), nrows=n0, ncols = n1)
    fig.tight_layout(h_pad = 5)

    ax = axes
    ax.plot(x[np.argwhere(y==0),0], x[np.argwhere(y==0),1],'.',label='y=0')
    ax.plot(x[np.argwhere(y==1),0], x[np.argwhere(y==1),1],'.',label='y=1')
    ax.legend(loc=4)

def compute_f(w, x, alpha):
    import numpy as np

    k = int(w.shape[0]/2)
    l0 = np.dot(w, x.T)
    l1 = l0 * (l0 > 0) + alpha * l0 * (l0 <= 0)
    f = (np.sign( (np.sum(l1[:k], axis=0) - np.sum(l1[k:], axis=0))) + 1) / 2
    return f
