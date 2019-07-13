import numpy as np
import scipy.spatial
import os
import networkx as nx
from tqdm import tqdm
import joblib
"""
see also
    https://github.com/chrsmrrs/hashgraphkernel
    https://github.com/Jacobe2169/GMatch4py
"""

class NX_GED():
    """
    see also
        https://github.com/chrsmrrs/hashgraphkernel
        https://github.com/Jacobe2169/GMatch4py
        https://github.com/dbblumenthal/gedlib
    """
    @staticmethod
    def pos_corr(n1, n2):
        # return np.abs(np.dot(n1['pos'], n2['pos']))
        return scipy.spatial.distance.euclidean(n1['pos'], n2['pos'])

    node_match = pos_corr

    def distmat(self, gl1, gl2, verbose=False, symmetric=False, paired=False, n_jobs=1):
        n1, n2 = len(gl1), len(gl2)

        if paired:
            dm = - np.ones(n1)
            for i in tqdm(range(n1)):
                dm[i] = self.ged(gl1[i], gl2[i], verbose=verbose)
        else:
            dm = - np.ones((n1, n2))
            for i in tqdm(range(n1)):
                if symmetric:
                    jmin = i
                else:
                    jmin = 0

                if n_jobs == 1:
                    for j in range(jmin, n2):
                        dm[i, j] = self.ged(gl1[i], gl2[j], verbose=verbose)
                else:
                    def fun(g1, g2):
                        return self.ged(g1=g1,g2=g2, verbose=False)

                    dis_mat_list = joblib.Parallel(n_jobs=n_jobs)(
                        joblib.delayed(fun)(g1=gl1[i], g2=gl2[j])
                        for j in range(jmin, n2)
                    )
                    dm[i, jmin:n2] = dis_mat_list

                if symmetric:
                    for j in range(jmin, n2):
                        dm[j, i] = dm[i, j]
        return dm

    def ged(self, g1, g2, verbose=True):
        for v in nx.optimize_graph_edit_distance(g1, g2, node_subst_cost=self.node_match):
            if verbose:
                print(v)
        return v

    @staticmethod
    def npy_to_nx(adjmat, vertex_attr):
        if len(adjmat.shape) == 2:
            raise ValueError('pass with [None, ...] if you have only one graph')
        gg = []
        for n in range(adjmat.shape[0]):
            g = nx.Graph(adjmat[n])
            for i in range(len(g.nodes)):
                g.nodes[i]['pos'] = vertex_attr[n][i]
            gg.append(g)
        return gg

    def distmat_npy(self, adj1, adj2, pt1, pt2, verbose=False, symmetric=False, paired=False):
        g1, g2 = NX_GED.npy_to_nx(adj1, pt1), NX_GED.npy_to_nx(adj2, pt2)
        return self.distmat(gl1=g1, gl2=g2, verbose=verbose, symmetric=symmetric,paired=paired)


    # Find the mean graph
    def get_mean(self, gl_test, n_jobs=1):
        n = len(gl_test)
        p = np.random.permutation(list(range(n)))
        g = [gl_test[i] for i in p]
        """
        complete diss_mat requires n**2 / 2
        here I decompose firstly in K subgroups of size S, with S * K = n
        the cost becomes n * sqrt(n) * 3/2
        in fact
        S**2 /2 * K   + K*n
        S**2 /2 * n/S + n/S*n
        S*n/2         + n**2/S
        n * (S/2 + n/S)
        and taking S = sqrt(n).
        n * sqrt(n) * 3/2

        The gain is sqrt(n) / 3
        """
        
        K = int(n / np.sqrt(n))
        S = int(n / K)
        imus = np.zeros(S)
        for k in range(K):
            gk = g[k * S:(k + 1) * S + 1]
            dk = self.distmat(gk, gk, symmetric=True, n_jobs=n_jobs)
            imus[k] = np.argmax(np.sum(dk ** 2, axis=1)) + k * S

        imus = imus.astype(int)
        gkmu = [g[imus[i]] for i in range(imus.shape[0])]
        dkmu = self.distmat(gkmu, g, symmetric=False, n_jobs=n_jobs)
        imuk = np.argmax(np.sum(dkmu ** 2, axis=1))
        
        pinv = p * 0
        for i in range(n):
            pinv[p[i]] = i
        pinv = pinv.astype(int)

        imu_final = pinv[imus[imuk]]
        return imu_final, dkmu[imuk][pinv]