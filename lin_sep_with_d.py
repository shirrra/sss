'''
FEB18


Connecting and configuring the queue
-------
ssh -X mcluster01
qlogin -q gpuint.q
setenv PATH /usr/wisdom/python3/bin:$PATH
setenv PYTHONPATH /usr/wisdom/python3
setenv LD_LIBRARY_PATH /usr/local/cudnn-v6/lib64

'''

from __future__ import print_function

import tensorflow as tf
#import model
from numpy import random as rn
import numpy as np
from scipy.linalg import qr
import os

def sim(cfg):

    #################################
    '''      configurations       '''
    #################################

    print('\n\n*** Configuring')
    # choosing GPU device
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    session_config = tf.ConfigProto()
    session_config.gpu_options.visible_device_list = '0' # which GPUs to use

    print('\nConfigs:\n', cfg)
    cfg_name = cfg['name']
    learning_rate = cfg['learning_rate']
    beta = cfg['beta']
    batch_size = cfg['batch_size']
    save_step = cfg['save_step']
    print_step = cfg['print_step']
    break_thresh = cfg['break_thresh']
    training_epochs = cfg['training_epochs']
    n_train = cfg['n_train']
    n_test = cfg['n_test']
    dim = cfg['dim']
    k = cfg['k']
    w_star_norm = cfg['w_star_norm']
    alpha = cfg['alpha']
    d_inf = cfg['d_inf']

    n_all = n_train + n_test
    n_classes = 2
    if (n_train % batch_size != 0):
        print("\n*** Warning! batch size doesn't divide n_train *** \n")
        input("Press enter to continue")
    total_batch = int(n_train/batch_size)
    vi = 1 / np.sqrt(2*k)
    w_max = 1 / np.sqrt(2*k)

    # paths
    data_path = '../data/'
    sim_path = ''
    logs_path = sim_path + 'logs/' + cfg_name + '/'
    res_path = sim_path + 'res/'
    res_name = res_path + 'all_' + cfg_name
    #model_name = 'C:/Users/Shira/Documents/TF/model_1_' + cfg_name + '.ckpt'

    print('\n*** Preparing data\n')

    # split the training set into sep and nonsep
    n_train_nonsep = max(1, int(cfg['percent_nonsep'] / 100 * n_train))
    n_train_sep = n_train - n_train_nonsep

    def create_data(n, dim, U, x_w_proj):
        x_rec_norm = (rn.uniform(low=0, high=1, size=n))**(1/(dim-1)) * np.sqrt((1 - x_w_proj**2))
        x_rec_direction = rn.normal(0, 1, (dim-1, n)) # shape p x n
        x_rec_direction = x_rec_direction / np.sqrt(np.sum(x_rec_direction**2, axis=0))
        x_rec = x_rec_direction * x_rec_norm

        x = np.zeros([dim, n])
        x[0, :] = x_w_proj
        x[1:, :] = x_rec
        x = np.matmul(U, x).T

        return x

    A = rn.randn(dim, dim)
    U, R = qr(A)
    w_star_direction = U[:, 0]
    gamma = 1 / w_star_norm

    y_sep = np.sign(rn.uniform(0, 1, n_train_sep) - 0.5)
    x_sep_w_proj = rn.uniform(gamma, 1, n_train_sep) # for sep points, the projection is at least gamma
    x_sep_w_proj = y_sep * x_sep_w_proj
    y_sep = (y_sep + 1) / 2
    x_sep = create_data(n_train_sep, dim, U, x_sep_w_proj)

    y_nonsep = np.sign(rn.uniform(0, 1, n_train_nonsep) - 0.5)
    x_nonsep_w_proj = rn.uniform(gamma-gamma*d_inf, gamma, n_train_nonsep)
    d_inf_chosen = 1 - np.min(x_nonsep_w_proj) / gamma
    sum_d = gamma - np.sum(x_nonsep_w_proj) / gamma
    x_nonsep_w_proj = y_nonsep * x_nonsep_w_proj
    y_nonsep = (y_nonsep + 1) / 2
    x_nonsep = create_data(n_train_nonsep, dim, U, x_nonsep_w_proj)

    x_train = np.concatenate([x_sep, x_nonsep])
    y_train = np.concatenate([y_sep, y_nonsep])
    d_train = gamma - np.concatenate([x_sep_w_proj, x_nonsep_w_proj]) / gamma

    y_test = np.sign(rn.uniform(0, 1, n_test) - 0.5)
    x_test_w_proj = rn.uniform(gamma, 1, n_test) # for sep points, the projection is at least gamma
    x_test_w_proj = y_test * x_test_w_proj
    y_test = (y_test + 1) / 2
    x_test = create_data(n_test, dim, U, x_test_w_proj)

    I_train_sep = range(0, n_train_sep)
    I_train_nonsep = range(n_train_sep, n_train)
    I_train_is_sep = np.zeros(n_train)
    I_train_is_sep[I_train_sep] = 1

    print('\n*** Configured according to', cfg_name, ', with ', \
       n_train_sep, 'sep labels and', n_train_nonsep, \
       'nonsep labels')
    # array for holding the accuracy results
    n_max_epochs = training_epochs + 1
    train_acc = np.zeros((n_max_epochs, 3))
    ind_all, ind_sep, ind_nonsep = 0, 1, 2
    test_acc = np.zeros(n_max_epochs)
    avg_costs = np.zeros(n_max_epochs)
    non_zero_steps = np.zeros((n_max_epochs, 3))
    G_w = np.zeros(n_max_epochs)
    F_w = np.zeros(n_max_epochs)
    sum_d_nzs = np.zeros(n_max_epochs)

    print('\n*** Building Computation Graph')

    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, dim], name='InputData')
    y = tf.placeholder(tf.float32, [None], name='LabelData')

    weights = {}
    weights_out = tf.concat((vi * tf.ones(k), -vi * tf.ones(k)), axis=0)

    weights_init = rn.normal(0, 1, [dim, 2*k])
    for i in range(2*k):
        weights_init[:,i] /= np.sqrt(np.sum(weights_init[:,i]**2))
    weights_init *= w_max

    def degenerate_multilayer_perceptron(x, weights, alpha):
        layer = x
        n_curr = dim
        w_name = 'w0'
        weights[w_name] = tf.Variable(tf.cast(weights_init, tf.float32), name=w_name)
        layer = tf.matmul(layer, weights[w_name])
        layer = leaky_relu(layer, alpha)
        # Output layer
        out_layer = tf.tensordot(layer, weights_out, axes=1)
        return out_layer

    def leaky_relu(x, alpha):
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

    # Encapsulating all ops into scopes, making Tensorboard's Graph
    # Visualization more convenient
    with tf.name_scope('Model'):
        # Build model
        pred = degenerate_multilayer_perceptron(x, weights, alpha)

    with tf.name_scope('Loss'):
        loss = tf.losses.hinge_loss(labels=y, logits=pred)

    with tf.name_scope('SGD'):
        # Gradient Descent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # Op to calculate every variable gradient
        grads = tf.gradients(loss, tf.trainable_variables())
        grads_and_vars = list(zip(grads, tf.trainable_variables()))
        # Op to update all variables according to their gradient
        apply_grads = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

    with tf.name_scope('Accuracy'):
        # Accuracy
        acc = tf.equal(tf.sign(pred), tf.sign(y-.5))
        acc = tf.reduce_mean(tf.cast(acc, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    print('\n*** Check problem parameters:')
    print('Components of bound for the number of non-zero steps M:')
    print('     |w*|^2 / alpha^2 = {:.3f}'.format(np.sum(w_star_norm**2) / alpha**2))
    print('     |w*|^2 / min(eta, sqrt(eta)) = {:.3f}'.format(np.sum(w_star_norm**2) / min(learning_rate, np.sqrt(learning_rate))))
    print('Value for R, bound on initial weights norm:')
    print('     1/sqrt(2k) = {:.3f}'.format(1/np.sqrt(2*k)))
    print('Maximal initial weights norm (per neuron):')
    print('     max |u_0^i|, |v_0^i| = {:.3f}'.format(
        max([np.sqrt(np.sum(weights_init[:,i]**2)) for i in range(2*k)])))
    print('Value for v, which should be set equal to R:')
    print('     v = {:.3f}'.format(vi))
    print('To guarantee expressiveness, k should be at least:')
    print('     2*(n/(2d-2)) = {}'.format(int(np.ceil(2*n_train/(2*dim-2)))))
    print('     k = ', k)
    #return

    # Start training
    with tf.Session(config=session_config) as sess:

        print('\n*** Training')
        # Run the initializer
        sess.run(init)

        epoch = 0

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            epoch_perm = rn.permutation(n_train)

            train_acc[epoch, ind_all] = acc.eval({x: x_train, y: y_train})
            train_acc[epoch, ind_sep] = acc.eval({x: x_train[I_train_sep], y: y_train[I_train_sep]})
            train_acc[epoch, ind_nonsep] = acc.eval({x: x_train[I_train_nonsep], y: y_train[I_train_nonsep]})
            test_acc[epoch] = acc.eval({x: x_test, y: y_test})
            w_learned = np.array(weights['w0'].eval()).transpose()
            G_w[epoch] = np.sqrt(np.sum(w_learned**2))
            F_w[epoch] = (np.sum(w_learned[:k,:]*w_star_direction-w_learned[k:,:]*w_star_direction)) * w_star_norm

            for ind in epoch_perm:
                batch_xs, batch_ys = x_train[ind,:].reshape(-1, dim), y_train[ind].reshape(1)

                # Run optimization op (backprop), cost op (to get loss value)
                _, loss_on_batch = sess.run([apply_grads, loss],
                                            feed_dict={x: batch_xs, y: batch_ys})

                # Compute average loss
                avg_cost += loss_on_batch / total_batch
                if loss_on_batch > 0:
                    sum_d_nzs[epoch] += d_train[ind]
                    non_zero_steps[epoch, ind_all] += 1
                    if I_train_is_sep[ind]:
                        non_zero_steps[epoch, ind_sep] += 1
                    else:
                        non_zero_steps[epoch, ind_nonsep] += 1

            avg_costs[epoch] = avg_cost

            stopping = (non_zero_steps[epoch, ind_all] <= break_thresh)

            if (epoch / print_step < 100) and ((epoch % print_step == 0) or (epoch < 30)) or stopping:
                print('\nEpoch: {}'.format(epoch))
                print('\nConfigured according to', cfg_name, ', with ', \
                   n_train_sep, 'sep labels and', n_train_nonsep,\
                   'nonsep labels')
                print('Configs:\n', cfg)
                print('\nBefore training,')
                print('Train Accuracy: {:.3f}'.format(train_acc[epoch, ind_all]))
                print('     Train sep Accuracy: {:.3f}'.format(train_acc[epoch, ind_sep]))
                print('     Train nonsep Accuracy: {:.3f}'.format(train_acc[epoch, ind_nonsep]))
                print('Test Accuracy: {:.3f}'.format(test_acc[epoch]))
                print('While training, cost =', '{:.9f}'.format(avg_cost), '({:.3f})'.format(np.exp(-avg_cost)))
                print('Number of non-zero steps (all, sep, nonsep): ', non_zero_steps[epoch, :])
                if (epoch / print_step >= 99) and not stopping:
                    print('\n*** Stops printing to avoid too much output! Still running')

            if (epoch % save_step == 0) or stopping:
                #print('\n*** Saving')
                ind_try = 0
                while True :
                    ind_try+=1
                    try :
                        np.savez(res_name,
                            train_acc=train_acc[:epoch+1,:],
                            test_acc=test_acc[:epoch+1],
                            avg_costs=avg_costs[:epoch+1],
                            non_zero_steps=non_zero_steps[:epoch+1,:],
                            G_w=G_w[:epoch+1],
                            F_w=F_w[:epoch+1],
                            w_learned=w_learned,
                            x_train=x_train,
                            y_train=y_train,
                            w_star_direction=w_star_direction,
                            d_inf_chosen=d_inf_chosen,
                            sum_d=sum_d,
                            sum_d_nzs=sum_d_nzs[:epoch+1],
                            config=cfg)
                        break
                    except PermissionError :
                        print('\n#'*20, end='')
                        print('\n<<< Saving attempt {} failed, trying again >>> \n'.format(ind_try))
                    except KeyboardInterrupt :
                        print('\n<<< Simulation interrupted, cannot save')
                        break

            if stopping:
                print('\n*** Training reached {} non-zero updates and is stopping'.format(break_thresh))
                break

            # print('\n*** Saving model')
            # saver.save(sess, model_name)
            # #saver_b.save(sess, biases_name)
            # print('w0[1,2] = ', weights['w0'][1,2].eval())
            # print('b0[3] = ', biases['b0'][3].eval())

        print('\n*** Optimization Finished!')
        print('\n*** Configured according to', cfg_name, ', with ', \
            n_train_sep, 'sep labels and', n_train_nonsep,\
            'nonsep labels')    # Test model
        # Calculate accuracy
        print('*** Train Accuracy: {:.3f}'.format(acc.eval({x: x_train, y: y_train})))
        print('*** Accuracy: {:.3f}'.format(acc.eval({x: x_test, y: y_test})))

        print('\n*** Run the command line:' \
              '\n      --> tensorboard --logdir=', logs_path, \
              '\n      Then open http://0.0.0.0:6006/ into your web browser\n')
