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
import numpy as np
import os
import numpy.random as rn

def optimize(cfg, data):

    #################################
    '''      configurations       '''
    #################################

    print('\n\n*** Configuring')
    # choosing GPU device
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    session_config = tf.ConfigProto()
    session_config.gpu_options.visible_device_list = '0' # which GPUs to use

    print('\nConfigs:\n', cfg)

    # network
    k = cfg['k']
    alpha = cfg['alpha']

    # optimization
    learning_rate = cfg['learning_rate']
    beta = cfg['beta']
    batch_size = cfg['batch_size']
    break_thresh = cfg['break_thresh']
    training_epochs = cfg['training_epochs']

    # meta
    cfg_name = cfg['name']
    save_step = cfg['save_step']
    print_step = cfg['print_step']

    # paths
    data_path = '../data/'
    res_path = cfg['res_path']
    logs_path = res_path + 'logs/' + cfg_name + '/'
    data_name = res_path + cfg_name + '_data'
    res_name = res_path + cfg_name + '_res'
    ckpt_name = res_path + cfg_name + '_model'
    #model_name = 'C:/Users/Shira/Documents/TF/model_1_' + cfg_name + '.ckpt'


    np.save(data_name, data)

    print('\n*** Preparing data\n')
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    n_train_nonsep = data['n_train_nonsep']
    n_train = x_train.shape[0]
    dim = x_train.shape[1]
    n_train_sep = n_train - n_train_nonsep
    n_test = x_test.shape[0]

    n_all = n_train + n_test
    n_classes = 2
    if (n_train % batch_size != 0):
        print("\n*** Warning! batch size doesn't divide n_train *** \n")
        input("Press enter to continue")
    total_batch = int(n_train/batch_size)
    vi = 1 / np.sqrt(2*k)
    w_max = 1 / np.sqrt(2*k)

    I_train_sep = range(0, n_train_sep)
    I_train_nonsep = range(n_train_sep, n_train)
    I_train_is_sep = np.zeros(n_train, dtype=np.int8)
    I_train_is_sep[I_train_sep] = 1

    print('\n*** Configured according to', cfg_name, ', with ', \
       n_train_sep, 'sep labels and', n_train_nonsep, \
       'nonsep labels')
    # array for holding the accuracy results
    n_max_epochs = training_epochs + 1
    ind_all, ind_sep, ind_nonsep = 0, 1, 2
    avg_costs = np.zeros(n_max_epochs)
    G_w = np.zeros(n_max_epochs)
    F_w = np.zeros(n_max_epochs)
    nzs_per_epoch = [[],[],[]]

    iszero_sbs = []
    issep_sbs = []

    train_acc_list = []
    train_acc_sep_list = []
    train_acc_nonsep_list = []
    test_acc_list = []
    w_learned_list = []

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

    #saver = tf.train.Saver([weights['w0']])

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

            nzs_per_epoch[0].append([0])
            nzs_per_epoch[1].append([0])
            nzs_per_epoch[2].append([0])

            first_step_in_epoch = True

            for ind in epoch_perm:

                #save_path = saver.save(sess, ckpt_name)
                if cfg['save_sbs'] or first_step_in_epoch:
                    train_acc_list.append(acc.eval({x: x_train, y: y_train}))
                    train_acc_sep_list.append(acc.eval({x: x_train[I_train_sep], y: y_train[I_train_sep]}))
                    train_acc_nonsep_list.append(acc.eval({x: x_train[I_train_nonsep], y: y_train[I_train_nonsep]}))
                    test_acc_list.append(acc.eval({x: x_test, y: y_test}))
                    w_learned_list.append(np.array(weights['w0'].eval()).T)

                batch_xs, batch_ys = x_train[ind,:].reshape(-1, dim), y_train[ind].reshape(1)

                # Run optimization op (backprop), cost op (to get loss value)
                _, loss_on_batch = sess.run([apply_grads, loss],
                                            feed_dict={x: batch_xs, y: batch_ys})

                # Compute average loss
                avg_cost += loss_on_batch / total_batch
                zs = np.int8(loss_on_batch==0)
                ss = I_train_is_sep[ind]
                if cfg['save_sbs']:
                    issep_sbs.append(ss)
                    iszero_sbs.append(zs)

                #print('zs=',zs,', ss=',ss)
                nzs_per_epoch[ind_all][-1] += 1-zs
                nzs_per_epoch[ind_sep][-1] += (1-zs)*ss
                nzs_per_epoch[ind_nonsep][-1] += (1-zs)*(1-ss)

                first_step_in_epoch = False

            avg_costs[epoch] = avg_cost

            #stopping = (nzs_per_epoch[ind_all][-1] <= break_thresh)
            stopping = False

            if (epoch / print_step == 100):
                print_step *= 10
                print('\nprint step grows by a factor of 10 and is now equal to', print_step)

            if (epoch % print_step == 0) or (epoch < 10) or stopping:
                print('\n\nEpoch: {}'.format(epoch))
                #print('\nConfigured according to', cfg_name, ', with ', \
                #   n_train_sep, 'sep labels and', n_train_nonsep,\
                #   'nonsep labels')
                #print('Configs:\n', cfg)
                print('Before training on epoch, train Accuracy (all, sep, nonsep): {:.3f}, {:.3f}, {:.3f}'.format(
                    train_acc_list[-1], train_acc_sep_list[-1], train_acc_nonsep_list[-1]))
                print('Test Accuracy: {:.3f}'.format(test_acc_list[-1]))
                print('While training, cost =', '{:.9f}'.format(avg_cost), '({:.3f})'.format(np.exp(-avg_cost)))
                print('Number of non-zero steps (all, sep, nonsep): ',
                    nzs_per_epoch[0][-1], nzs_per_epoch[1][-1], nzs_per_epoch[2][-1])
                #if (epoch / print_step >= 99) and not stopping:
                #    print('\n*** Stops printing to avoid too much output! Still running')
                #print()
            else:
                print('{}, '.format(epoch), end = '')

            if (epoch % save_step == 0) or stopping:
                print('\n*** Saving')
                ind_try = 0
                while True :
                    ind_try+=1
                    try :
                        np.savez(res_name,
                            avg_costs=avg_costs[:epoch+1],
                            x_train=x_train,
                            y_train=y_train,
                            nzs_per_epoch=nzs_per_epoch,
                            iszero_sbs=iszero_sbs,
                            issep_sbs=issep_sbs,
                            train_acc_list=train_acc_list,
                            train_acc_sep_list=train_acc_sep_list,
                            train_acc_nonsep_list=train_acc_nonsep_list,
                            test_acc_list=test_acc_list,
                            w_learned_list=w_learned_list,
                            config=cfg)
                        break
                    except PermissionError :
                        print('\n#'*20, end='')
                        print('\n<<< Saving attempt {} failed, trying again >>> \n'.format(ind_try))
                    except KeyboardInterrupt :
                        print('\n<<< Simulation interrupted, cannot save')
                        stopping = True
                        break

            if stopping:
                print('\n*** Training reached {} non-zero updates and is stopping'.format(break_thresh))
                break

            # print('\n*** Saving model')
            # saver.save(sess, model_name)
            # #saver_b.save(sess, biases_name)

        print('\n*** Optimization Finished!')
        print('\n*** Configured according to', cfg_name, ', with ', \
            n_train_sep, 'sep labels and', n_train_nonsep,\
            'nonsep labels')
        print('Configs:\n', cfg)
        # Calculate accuracy
        print('*** Train Accuracy: {:.3f}'.format(acc.eval({x: x_train, y: y_train})))
        print('*** Accuracy: {:.3f}'.format(acc.eval({x: x_test, y: y_test})))

        print('\n*** Run the command line:' \
              '\n      --> tensorboard --logdir=', logs_path, \
              '\n      Then open http://0.0.0.0:6006/ into your web browser\n')

        return
