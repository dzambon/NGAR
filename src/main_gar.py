from itertools import product

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn as sns
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import TimeDistributed, Input, LSTM, Lambda, Dense, Reshape, Concatenate, Dropout
from keras.models import Model
from keras.regularizers import l2
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from spektral.layers import GraphConv, GlobalAttentionPool
from spektral.utils import localpooling_filter, init_logging, log
from spektral.utils.plotting import plot_numpy

from data_generator import get_peter_graphs, get_rotation_graphs, get_input_sequences, get_targets
from graph_distances import NX_GED


def get_recerr(gl_test, gl_pred, gl_var_pred, reduced=None, nxged=None, methods=None):
    if reduced is None:
        n = len(gl_test)
    else:
        n = reduced

    data = []
    labels = []
    
    # Compute the full distance matrix needed for the moving average
    if 'mavg' in methods:
        dm = nxged.distmat(gl_test, gl_test, symmetric=True, n_jobs=20)
    else:
        dm = None

    if 'iid' in methods:
        # i.i.d. process
        # recerr distance of graph gt from the best stationary guess (the mean graph)
        print('i.i.d. baseline')
        if dm is None:
            _, residuals_mean = nxged.get_mean(gl_test, n_jobs=20)
        else:
            g_medoid = np.argmin(np.sum(dm**2))
            residuals_mean = dm[g_medoid]
        data.append(residuals_mean)
        labels.append('Mean')

    if 'mart' in methods:
        # Martingale process
        # recerr distance of graph g(t) from the g(t-1)
        print('Martingale baseline')
        residuals_mart = nxged.distmat(gl_test[1:], gl_test[:-1], paired=True, n_jobs=20)
        data.append(residuals_mart)
        labels.append('Mart.')

    if 'mavg' in methods:
        # Moving average
        print('moving average baseline')
        residuals_mavg = []
        p = 20
        for t in range(n-p-1):
            g_medoid = np.argmin(np.sum(dm[:, t:t+p]**2))
            residuals_mavg.append(dm[g_medoid, t+p])
        residuals_mavg = np.array(residuals_mavg)
        data.append(residuals_mavg)
        labels.append('Mov.avg.')

    
    if 'var' in methods:
        # Multivariate AR
        print('VAR baseline')
        residuals_var = nxged.distmat(gl_test[-len(gl_var_pred):], gl_var_pred, paired=True)
        data.append(residuals_var)
        labels.append('VAR')
    
    if 'ngar' in methods:
        # AR process
        # recerr distance of graph gt from the gt_pred
        print('NGAR')
        residuals_ngar = nxged.distmat(gl_test, gl_pred, paired=True)
        data.append(residuals_ngar)
        labels.append('NGAR')

    if 'mean' in methods and 'ngar' in methods:
        # Validate the models
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot([0, n], [np.mean(residuals_mean)] * 2, label='E[e]')
        plt.scatter(list(range(n)), residuals_mean, label='e')
        plt.title('Graphical check of model assumption H0')
        plt.legend()
    
        plt.subplot(1, 2, 2)
        plt.plot([0, n], [np.mean(residuals_ngar)] * 2, label='E[e\']')
        plt.scatter(list(range(n)), residuals_ngar, label='e\'')
        plt.title('Graphical check of model assumption H1')
        plt.legend()

    # Compare models
    plt.figure().add_subplot(111)
    ax = sns.violinplot(data=data)
    plt.ylabel('GED')
    ax.set_xticklabels(labels)
    plt.xticks(rotation=45)
    ax.set(ylim=(0, 35))
    ax.yaxis.grid()

    plt.figure(figsize=[10, 4])
    grid = plt.GridSpec(1, 3, wspace=0.4, hspace=0.)
    sax1 = plt.subplot(grid[0, :2])
    if 'mean' in methods:
        plt.plot([0, n], [np.mean(residuals_mean)] * 2, c='C0')
        plt.scatter(list(range(n)), residuals_mean, label='Mean', marker='.', c='C0')
    if 'mart' in methods:
        plt.plot([0, n], [np.mean(residuals_mart)] * 2, c='C1')
        plt.scatter(list(range(1, n)), residuals_mart, label='Martingale', marker='.', c='C1')
    if 'ngar' in methods:
        plt.plot([0, n], [np.mean(residuals_ngar)] * 2, c='C2')
        plt.scatter(list(range(n)), residuals_ngar, label='NGAR', marker='.', c='C2')
    if 'mavg' in methods:
        plt.plot([0, n], [np.mean(residuals_mavg)] * 2, c='C3')
        plt.scatter(list(range(len(residuals_mavg))), residuals_mavg, label='M.AVG.', marker='.', c='C3')
    if 'var' in methods:
        plt.plot([0, n], [np.mean(residuals_var)] * 2, c='C4')
        plt.scatter(list(range(len(residuals_var))), residuals_var, label='VAR', marker='.', c='C4')
        
    plt.ylabel('GED')
    plt.xlabel('Timestep')
    plt.legend()

    sax2 = plt.subplot(grid[0, 2], sharey=sax1)
    # data = [residuals_mean, residuals_mart, residuals_ngar]
    # labels = ['Mean', 'Mart.', 'NGAR']
    ax = sns.violinplot(data=data, cut=0)
    plt.ylabel('GED')
    ax.set_xticklabels(labels)
    plt.xticks(rotation=45)

    if 'mean' in methods and 'ngar' in methods:
        a = scipy.stats.wilcoxon(residuals_mean - residuals_ngar)
        log('p-value wicoxon Mean - NGAR = {}'.format(a.pvalue))
    if 'mart' in methods and 'ngar' in methods:
        b = scipy.stats.wilcoxon(residuals_mart - residuals_ngar[1:])
        log('p-value wicoxon Martingale - NGAR = {}'.format(b.pvalue))

    # if a.pvalue < 0.05:
    #     plt.arrow(.05, -.75, .9, 0, color='black')
    # if b.pvalue < 0.05:
    #     plt.arrow(1.05, -.75, .9, 0, color='black')
    # plt.tight_layout()

    return data, dm


# Parameters
np.random.seed(20180114)      # seed for replicability
problem = 'rotation'          # 'peter' or 'rotation'
rotation_type = 'simple'   # 'simple' or 'dynamic'
T_main = 100000            # Length of main sequence to split randomly
T_seq = int(T_main * 0.1)  # Length of sequence for testing in time
T = T_main + T_seq         # Length of full sequence
distortion = .01           # Distortion of the rotation system
N = 5                      # Number of nodes
F = 2                      # Dimension of the node attributes
ts = 20                    # Length of a single sequence (must be proportional to 2 * (complexity - N*F))
l2_reg = 5e-4              # Weight for l2 regularization
batch_size = 256           # Size of the minibatches
dropout_rate = 0.0         # Dropout rate for whole network
epochs = 2000              # Training epochs
es_patience = 20           # Patience for early stopping
log_dir = init_logging()   # Directory for logging
reduced = 1000
methods = []
methods += ['iid']
methods += ['mart']
methods += ['mavg']
methods += ['var']
methods += ['ngar']

# Tuneables
complexity = [10, 15, 20, 30, 60, 110]   # Complexity of the peter system
memory_order = [1, 5, 10, 20, 50, 100]  # Memory order for rotation system
tuneables = [complexity if problem == 'peter' else memory_order]

for c_m_o_, in product(*tuneables):
    log('Problem: {}; Complexity/Memory order: {}'.format(problem, c_m_o_))
    # Create all graphs
    if problem == 'peter':
        nfeat, adjacency = get_peter_graphs(N, F, T, c_m_o_, distortion)
    elif problem == 'rotation':
        nfeat, adjacency = get_rotation_graphs(N, F, T, c_m_o_, distortion, rot_type=rotation_type)
    else:
        raise ValueError('Problem can be: peter, rotation')
    np.savez(log_dir + '{}_{}_original_graph'.format(problem, c_m_o_),
             nfeat=nfeat,
             adjacency=adjacency)

    # Create filters (Laplacian)
    fltr = localpooling_filter(adjacency.copy())

    # Create regressors and targets
    adj_target = get_targets(adjacency, T, ts)
    nf_target = get_targets(nfeat, T, ts)
    fltr = get_input_sequences(fltr, T, ts)
    node_features = get_input_sequences(nfeat, T, ts)

    # Split data for sequential tests
    adj_target_seq = adj_target[T_main:]
    adj_target = adj_target[:T_main]
    nf_target_seq = nf_target[T_main:]
    nf_target = nf_target[:T_main]
    adj_seq = fltr[T_main:]
    fltr = fltr[:T_main]
    nf_seq = node_features[T_main:]
    node_features = node_features[:T_main]

    # Train, test, val split (randomized)
    adj_train, adj_test, \
    nf_train, nf_test, \
    adj_target_train, adj_target_test, \
    nf_target_train, nf_target_test = train_test_split(fltr, node_features,
                                                       adj_target, nf_target,
                                                       test_size=int(T_main * 0.1))
    adj_train, adj_val, \
    nf_train, nf_val, \
    adj_target_train, adj_target_val, \
    nf_target_train, nf_target_val = train_test_split(adj_train, nf_train,
                                                      adj_target_train, nf_target_train,
                                                      test_size=int(T_main * 0.1))

    matplotlib.rcParams.update({'font.size': 14})
    sns.set(style='whitegrid', palette="pastel", color_codes=True)

    if 'ngar' in methods:
        # Model definition
        # Note: TimeDistributed does not work for multiple inputs, so we need to
        # concatenate X and A before feeding the layers
        X_in = Input(shape=(ts, N, F))
        filter_in = Input(shape=(ts, N, N))
    
        # Convolutional block
        conc1 = Concatenate()([X_in, filter_in])
        gc1 = TimeDistributed(
            Lambda(lambda x_:
                   GraphConv(128, activation='relu', kernel_regularizer=l2(l2_reg), use_bias=True)([x_[..., :-N], x_[..., -N:]])
                   )
        )(conc1)
        gc1 = Dropout(dropout_rate)(gc1)
        conc2 = Concatenate()([gc1, filter_in])
        gc2 = TimeDistributed(
            Lambda(lambda x_:
                   GraphConv(128, activation='relu', kernel_regularizer=l2(l2_reg), use_bias=True)([x_[..., :-N], x_[..., -N:]])
                   )
        )(conc2)
    
        # pool = TimeDistributed(NodeAttentionPool())(gc2)
        pool = TimeDistributed(GlobalAttentionPool(128))(gc2)
        # pool = Lambda(lambda x_: K.reshape(x_, (-1, ts, N * 128)))(gc2)
    
        # Recurrent block
        lstm = LSTM(256, return_sequences=True)(pool)
        lstm = LSTM(256)(lstm)
    
        # Dense block
        # dense1 = BatchNormalization()(lstm)
        # dense1 = Dropout(dropout_rate)(dense1)
        dense1 = Dense(256, activation='relu', kernel_regularizer=l2(l2_reg))(lstm)
        # dense2 = BatchNormalization()(dense1)
        # dense2 = Dropout(dropout_rate)(dense2)
        dense2 = Dense(512, activation='relu', kernel_regularizer=l2(l2_reg))(dense1)
    
        adj_out = Dense(N * N, activation='sigmoid')(dense2)
        adj_out = Reshape((N, N), name='ADJ')(adj_out)
        nf_out = Dense(N * F, activation='linear')(dense2)
        nf_out = Reshape((N, F), name='NF')(nf_out)
    
        # Callbacks
        es_callback = EarlyStopping(monitor='val_loss', patience=es_patience, verbose=1)
        mc_callback = ModelCheckpoint(log_dir + '{}_{}_best_model.h5'.format(problem, c_m_o_),
                                      save_best_only=True, save_weights_only=True,
                                      verbose=1)
        tb_callback = TensorBoard(log_dir)
    
        # Build model
        model = Model(inputs=[X_in, filter_in],
                      outputs=[nf_out, adj_out])
        model.compile('adam',
                      ['mse', 'binary_crossentropy'],
                      metrics=['acc'])
        plot_model(model,
                   to_file=log_dir + '{}_{}_model.png'.format(problem, c_m_o_),
                   show_shapes=True)
    
        # Train model
        validation_data = [[nf_val, adj_val], [nf_target_val, adj_target_val]]
        model.fit([nf_train, adj_train],
                  [nf_target_train, adj_target_train],
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_data=validation_data,
                  callbacks=[es_callback, mc_callback, tb_callback])
    
        # Evaluate model
        eval_results = model.evaluate([nf_test, adj_test],
                                      [nf_target_test, adj_target_test],
                                      batch_size=batch_size,
                                      verbose=True)
        log('Done\nLoss: {}\nNF loss: {}\nADJ loss: {}\nNF acc: {}\nADJ acc: {}\n'
            .format(*eval_results))
        log('Problem: {} Order: {} Loss: {} NF loss: {} ADJ loss: {} NF acc: {} ADJ acc: {}\n'
            .format(*[problem, c_m_o_] + eval_results))
        nf_pred, adj_pred = model.predict([nf_test, adj_test],
                                          batch_size=batch_size)
        adj_pred = np.round(adj_pred + 1e-6)
    
        nf_pred_seq, adj_pred_seq = model.predict([nf_seq, adj_seq],
                                                  batch_size=batch_size)
        adj_pred_seq = np.round(adj_pred_seq + 1e-6)
    
        # Save data for later
        np.savez(log_dir + '{}_{}_predicted_and_target_graphs'.format(problem, c_m_o_),
                 nf_target_test=nf_target_test,
                 adj_target_test=adj_target_test,
                 nf_pred=nf_pred,
                 adj_pred=adj_pred)
        np.savez(log_dir + '{}_{}_predicted_and_target_graphs_seq'.format(problem, c_m_o_),
                 nf_seq=nf_seq,
                 adj_seq=adj_seq,
                 nf_target_seq=nf_target_seq,
                 adj_target_seq=adj_target_seq,
                 nf_pred_seq=nf_pred_seq,
                 adj_pred_seq=adj_pred_seq)

        ############################################################################
        # PLOTS
        ############################################################################
        # Plotting params
        n_plots = 10
        wspace = 0.2
        hspace = 0.7
    
        # Plot target-prediction pairs
        plt.figure(figsize=(20, 3.5))
        # losses = [model.evaluate([nf_test[i:i+1], adj_test[i:i+1]],
        #                          [nf_target_test[i:i+1], adj_target_test[i:i+1]],
        #                          verbose=False)[0]
        #           for i in range(nf_test.shape[0])]
        idxs = np.random.permutation(nf_test.shape[0])[:n_plots]
        for idx, i in enumerate(idxs):
            plt.subplot(2, n_plots, idx + 1)
            plot_numpy(adj_target_test[i], nf_target_test[i],
                       labels=False, node_size=20, layout='delaunay',
                       node_color=sns.color_palette('bright')[:N])
            plt.subplot(2, n_plots, idx + 1 + n_plots)
            plot_numpy(adj_pred[i], nf_pred[i],
                       labels=False, node_size=20, layout='delaunay',
                       node_color=sns.color_palette('bright')[:N])
    
        plt.tight_layout()
        plt.subplots_adjust(wspace=wspace, hspace=hspace)
        plt.savefig(log_dir + '{}_{}_target_prediction_pairs.pdf'.format(problem, c_m_o_),
                    dpi=500, bbox_inches='tight')
    
        # Plot sequence + target/prediction
        plt.figure(figsize=(24, 2.5))
        idx = np.random.randint(0, nf_test.shape[0])
        for i in range(n_plots):
            plt.subplot(1, n_plots + 2, i + 1)
            plt.title('t - {}'.format(n_plots - i))
            plot_numpy(adj_test[idx, i - n_plots], nf_test[idx, i - n_plots],
                       labels=False, node_size=20, layout='delaunay',
                       node_color=sns.color_palette('bright')[:N])
        plt.subplot(1, n_plots + 2, n_plots + 1)
        plt.title('True')
        plot_numpy(adj_target_test[idx], nf_target_test[idx],
                   labels=False, node_size=20, layout='delaunay',
                   node_color=sns.color_palette('bright')[:N])
        plt.subplot(1, n_plots + 2, n_plots + 2)
        plt.title('Pred.')
        plot_numpy(adj_pred[idx], nf_pred[idx],
                   labels=False, node_size=20, layout='delaunay',
                   node_color=sns.color_palette('bright')[:N])
    
        plt.tight_layout()
        plt.subplots_adjust(wspace=wspace, hspace=hspace)
        plt.savefig(log_dir + '{}_{}_seq+target+prediction.pdf'.format(problem, c_m_o_),
                    dpi=500, bbox_inches='tight')
    
        # Plot true sequence vs. purely AR prediction
        plt.figure(figsize=(20, 3.5))
        idx = np.random.randint(0, nf_seq.shape[0] - n_plots)
        nf_seed = nf_seq[idx][None, ...]
        adj_seed = adj_seq[idx][None, ...]
        for i in range(n_plots):
            nf_pred_, adj_pred_ = model.predict([nf_seed, adj_seed])
            nf_seed = np.concatenate((nf_seed[:, 1:, ...], nf_pred_[:, None, ...]), axis=1)
            adj_seed = np.concatenate((adj_seed[:, 1:, ...], adj_pred_[:, None, ...]), axis=1)
            plt.subplot(2, n_plots, i + 1)
            plt.title('t + {}'.format(i + 1))
            plot_numpy(adj_target_seq[idx + i], nf_target_seq[idx + i],
                       labels=False, node_size=20, layout='delaunay',
                       node_color=sns.color_palette('bright')[:N])
            plt.subplot(2, n_plots, i + 1 + n_plots)
            plot_numpy(adj_pred_[0], nf_pred_[0],
                       labels=False, node_size=20, layout='delaunay',
                       node_color=sns.color_palette('bright')[:N])
    
        plt.tight_layout()
        plt.subplots_adjust(wspace=wspace, hspace=hspace)
        plt.savefig(log_dir + '{}_{}_true_seq_v_AR_seq.pdf'.format(problem, c_m_o_),
                    dpi=500, bbox_inches='tight')
    
    if 'var' in methods:
        from statsmodels.tsa.vector_ar.var_model import VAR

        # unroll the nodefeatures and adjacency
        nfeat_unrolled = nfeat.reshape(T, -1)
        adj_unrolled = adjacency.reshape(T, -1)

        # deal with zero variance components
        zero_var_method = 'reduced'
        if zero_var_method == 'noise':
            adj_tmp = adj_unrolled + np.random.randn(*adj_unrolled.shape)*.0001
        elif zero_var_method == 'reduced':
            variances = np.var(adj_unrolled , axis=0)
            variances_null = np.where(variances==0)
            variances_not_null = np.where(variances!=0)
            adj_tmp = adj_unrolled[:, variances_not_null[0]]

        # create sequence of vectors
        x = np.concatenate((nfeat_unrolled, adj_tmp), axis=1)

        # train the VAR model
        T_train = T_main
        x_train = x[:T_train]
        model = VAR(x_train)
        model_fit = model.fit(maxlags=ts)

        # test the model
        len_test = nf_target_seq.shape[0]
        x_test = np.concatenate((nf_target_seq.reshape(len_test, -1),
                                 adj_target_seq.reshape(len_test, -1)[:, variances_not_null[0]]),
                                axis=1)
        x_pred = np.empty((len_test-ts, x_train.shape[1])) # the first ts vectors are the regressors
        for t in range(ts, len_test):
            x_pred[t-ts] = model_fit.params[0]
            x_pred[t-ts] += model_fit.params[1:].transpose().dot(x_test[t-model_fit.k_ar:t][::-1].ravel())
        # clip the adjacency value in {0,1}
        x_pred[:, N*F:] = np.round(x_pred[:, N*F:])
        
        # deal with zero variance components
        if zero_var_method == 'noise':
            x_adj = x_pred[:, N * F:]
        elif zero_var_method == 'reduced':
            x_adj = np.empty((x_pred.shape[0], N*N))
            x_adj[:, variances_not_null[0]] = x_pred[:, N * F:]
            x_adj[:, variances_null[0]] = adj_unrolled[:1, variances_null[0]].repeat(x_adj.shape[0], axis=0)

        # re-assemble node features and adjacency matrices
        nf_var_pred = x_pred[:, :N*F].reshape(-1, N, F)
        adj_var_pred = x_adj.reshape(-1, N, N)



    # Plot residual GED
    nxged = NX_GED()
    gl_t = NX_GED.npy_to_nx(adj_target_seq, nf_target_seq)[:reduced]
    if 'ngar' in methods:
        gl_p = NX_GED.npy_to_nx(adj_pred_seq, nf_pred_seq)[:reduced]
    else:
        gl_p = None
    if 'var' in methods:
        gl_var = NX_GED.npy_to_nx(adj_var_pred, nf_var_pred)[:reduced]
    else:
        gl_var = None
    _, dissimilarity_matrix = get_recerr(gl_test=gl_t, gl_pred=gl_p, gl_var_pred=gl_var, nxged=nxged, reduced=reduced, methods=methods)
    np.savez(log_dir + '{}_{}_dissimilarity_matrix_seq'.format(problem, c_m_o_),
             dissimilarity_matrix=dissimilarity_matrix)

    plt.tight_layout()
    figs = [plt.figure(n) for n in plt.get_fignums()]
    for i, fig in enumerate(figs):
        fig.savefig(log_dir + '{}_{}_{}_{}.pdf'.format(problem, c_m_o_, fig, i),
                    dpi=500, bbox_inches='tight')

    plt.close('all')
    K.clear_session()
