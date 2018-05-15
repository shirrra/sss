import numpy as np
import numpy.random as rn
from scipy.linalg import qr

def gen_data_switch(cfg):

    def gen_xy(n, dim, gamma, U, make_switch):
        if make_switch:
            switch = -1
        else:
            switch = 1
        y = np.sign(rn.uniform(0, 1, n) - 0.5) # -1 or 1
        x_w_proj = rn.uniform(gamma, 1, n)
        x_w_proj = switch * y * x_w_proj
        x_rec_norm = (rn.uniform(low=0, high=1, size=n))**(1/(dim-1)) * np.sqrt((1 - x_w_proj**2))
        x_rec_hat = rn.normal(0, 1, (dim-1, n)) # shape p x n
        x_rec_hat = x_rec_hat / np.sqrt(np.sum(x_rec_hat**2, axis=0))
        x_rec = x_rec_hat * x_rec_norm
        y = (y + 1) / 2 # 0 or 1
        x = np.zeros([dim, n])
        x[0, :] = x_w_proj
        x[1:, :] = x_rec
        x = np.matmul(U, x).T

        return x, y

    print(cfg)
    # dataset
    dim = cfg['dim']
    n_train = cfg['n_train']
    n_test = cfg['n_test']
    gamma = cfg['gamma']
    pi_nonsep = cfg['pi_nonsep']

    # split the training set into sep and nonsep
    n_train_nonsep = int(pi_nonsep * n_train)
    n_train_sep = n_train - n_train_nonsep

    A = rn.randn(dim, dim)
    U, R = qr(A)
    w_star_hat = U[:, 0]

    x_sep, y_sep = gen_xy(n_train_sep, dim, gamma, U, False)
    x_nonsep, y_nonsep = gen_xy(n_train_nonsep, dim, gamma, U, cfg['make_switch'])
    x_train = np.concatenate([x_sep, x_nonsep])
    y_train = np.concatenate([y_sep, y_nonsep])
    x_test, y_test = gen_xy(n_test, dim, gamma, U, False)

    data = {
        'x_train' : x_train,
        'y_train' : y_train,
        'x_test' : x_test,
        'y_test' : y_test,
        'w_star_hat' : w_star_hat,
        'n_train_nonsep' : n_train_nonsep,
        'gamma' : gamma,
        'U' : U,
    }

    return data

def gen_data_d(cfg):

    def gen_xy(n, dim, gamma, U, make_switch):
        if make_switch:
            switch = -1
        else:
            switch = 1
        y = np.sign(rn.uniform(0, 1, n) - 0.5) # -1 or 1
        x_w_proj = rn.uniform(gamma, 1, n)
        x_w_proj = switch * y * x_w_proj
        x_rec_norm = (rn.uniform(low=0, high=1, size=n))**(1/(dim-1)) * np.sqrt((1 - x_w_proj**2))
        x_rec_hat = rn.normal(0, 1, (dim-1, n)) # shape p x n
        x_rec_hat = x_rec_hat / np.sqrt(np.sum(x_rec_hat**2, axis=0))
        x_rec = x_rec_hat * x_rec_norm
        y = (y + 1) / 2 # 0 or 1
        x = np.zeros([dim, n])
        x[0, :] = x_w_proj
        x[1:, :] = x_rec
        x = np.matmul(U, x).T

        return x, y

    print(cfg)
    # dataset
    dim = cfg['dim']
    n_train = cfg['n_train']
    n_test = cfg['n_test']
    gamma = cfg['gamma']
    pi_nonsep = cfg['pi_nonsep']

    # split the training set into sep and nonsep
    n_train_nonsep = int(pi_nonsep * n_train)
    n_train_sep = n_train - n_train_nonsep

    A = rn.randn(dim, dim)
    U, R = qr(A)
    w_star_hat = U[:, 0]

    x_sep, y_sep = gen_xy(n_train_sep, dim, gamma, U, False)
    x_nonsep, y_nonsep = gen_xy(n_train_nonsep, dim, gamma, U, cfg['make_nonsep'])
    x_train = np.concatenate([x_sep, x_nonsep])
    y_train = np.concatenate([y_sep, y_nonsep])
    x_test, y_test = gen_xy(n_test, dim, gamma, U, False)

    data = {
        'x_train' : x_train,
        'y_train' : y_train,
        'x_test' : x_test,
        'y_test' : y_test,
        'w_star_hat' : w_star_hat,
        'n_train_nonsep' : n_train_nonsep,
        'gamma' : gamma,
        'U' : U,
    }

    return data

def gen_data_xor(cfg):

    # dataset
    dim = 2
    n_train = cfg['n_train']
    n_test = cfg['n_test']
    gamma = cfg['gamma']

    def gen(n, gamma):
        x = rn.uniform(gamma, 1, (n, 2))
        y2 = np.sign(rn.uniform(0, 1, (n, 2)) - 0.5)
        x = x * y2
        y = (np.sign(x[:,0])*np.sign(x[:,1]) + 1) / 2

        # x_hat = rn.normal(0, 1, (2, n))
        # x_hat = x_hat / np.sqrt(np.sum(x_hat**2, axis=0))
        # x_norm = np.sqrt(rn.uniform(low=0, high=1, size=n))
        # x = x_hat * x_norm
        # x = x[:,abs(x[0,:]) > gamma]
        # x = x[:,abs(x[1,:]) > gamma]
        #
        # x = x.T
        # y = (np.sign(x[:,0])*np.sign(x[:,1]) + 1) / 2

        return x, y

    x_train, y_train = gen(n_train, gamma)
    x_test, y_test = gen(n_test, gamma)

    w_star_hat = np.asarray([1, 0])
    n_train_nonsep = int(n_train/2)

    data = {
        'x_train' : x_train,
        'y_train' : y_train,
        'x_test' : x_test,
        'y_test' : y_test,
        'w_star_hat' : w_star_hat,
        'n_train_nonsep' : n_train_nonsep
    }

    return data

def gen_data_w(cfg, w):
    from helper import compute_f

    dim = cfg['dim']
    n_train = cfg['n_train']
    n_test = cfg['n_test']
    alpha = cfg['alpha']

    x = rn.normal(0, 1, [n_train + n_test, dim])
    y = compute_f(w, x, alpha)
    w_star_hat = rn.normal(0, 1, dim)
    w_star_hat = w_star_hat / np.sqrt(np.sum(w_star_hat**2))

    data = {
        'x_train' : x[:n_train, :],
        'y_train' : y[:n_train],
        'x_test' : x[n_train:, :],
        'y_test' : y[n_train:],
        'w_star_hat' : w_star_hat,
        'n_train_nonsep' : 2,
        'gamma' : 1,
        'w' : w,
    }

    return data

def gen_data_theta(cfg, num0=2, method='equal', theta=[]):
    from helper import compute_f

    if method == 'equal':
        theta = np.linspace(0,1,(2*num0+1))
    elif method == 'random' or method == 'rand':
        theta = np.sort(rn.rand(num0*2))
        #theta = np.append(theta, 1)
    elif method == 'input':
        theta.sort()
        #if theta[-1] < 1:
        #    np.append(theta, 1)
    else:
        print("\n*** no such method! please use 'equal'/'rand'/'input' ***\n")

    dim = cfg['dim']
    n_train = cfg['n_train']
    n_test = cfg['n_test']

    n = n_train + n_test
    x = rn.normal(0, 1, [n, dim])
    y = np.zeros(n)
    for i in range(n):
        angle = np.arctan2(x[i,1], x[i,0]) / (2*np.pi)
        angle += (angle < 0)
        y[i] = (np.argmax(angle < theta))%2

    data = {
        'x_train' : x[:n_train, :],
        'y_train' : y[:n_train],
        'x_test' : x[n_train:, :],
        'y_test' : y[n_train:],
        'n_train_nonsep' : 2,
        'theta' : theta,
    }

    return data
