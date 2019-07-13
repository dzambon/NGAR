import numpy as np
from scipy.spatial import Delaunay
from scipy.stats import ortho_group


def node_features_to_delaunay_adj(node_features):
    N = node_features.shape[-2]
    # Compute Delaunay triangulations
    adjacency = []
    for nf in node_features:
        tri = Delaunay(nf)
        edges_explicit = np.concatenate((tri.vertices[:, :2],
                                         tri.vertices[:, 1:],
                                         tri.vertices[:, ::2]), axis=0)
        adj = np.zeros((N, N))
        adj[edges_explicit[:, 0], edges_explicit[:, 1]] = 1.
        adj = np.clip(adj + adj.T, 0, 1)
        adjacency.append(adj)
    return np.array(adjacency)


def get_input_sequences(x, T, ts):
    x_input = - np.ones((T - ts - 1, ts) + x.shape[1:])
    for i in range(T - ts - 1):
        x_input[i] = x[i:(i + ts)]
    return x_input


def get_targets(x, T, ts):
    x_target = - np.ones((T - ts - 1, ) + x.shape[1:])
    for i in range(T - ts - 1):
        x_target[i] = x[i + ts]
    return x_target


def get_peter_graphs(N, F, T, complexity, sigma_distortion):
    timesteps = dynamic_peter(T, complexity, sigma_distortion)
    # Node features are consecutive F-dimensional slices of the state
    node_features = np.stack((timesteps[..., i:i + F] for i in range(0, N * F, F)), axis=1)
    adjacency = node_features_to_delaunay_adj(node_features)
    return node_features, adjacency


def dynamic_peter(T, complexity, sigma_distortion):
    # Evolve dynamic system
    R = ortho_group.rvs(dim=complexity)  # Create random orthonormaml matrix
    X = np.ones((complexity,))  # Initialize state
    timesteps = []
    for t in range(T):
        X = R.dot(X)
        X += np.random.randn(*X.shape) * sigma_distortion
        timesteps.append(X)
    return np.stack(timesteps)


def get_rotation_graphs(N, F, T, memory_order, sigma_distortion, rot_type='dynamic', memory_points=None,
                        update_memory_points=True, no_distortion_in_the_first=False):

    if rot_type == 'dynamic':
        node_features = dynamic_rotation(N, F, T, memory_order, sigma_distortion, memory_points=memory_points,
                        update_memory_points=update_memory_points,
                        no_distortion_in_the_first=no_distortion_in_the_first)
    else:
        node_features = simple_rotation(N, F, T, memory_order, sigma_distortion)
    adjacency = node_features_to_delaunay_adj(node_features)
    return node_features, adjacency


def dynamic_rotation(N, F, T, memory_order, sigma_distortion, memory_points=None,
                     update_memory_points=True, no_distortion_in_the_first=False):
    
    theta_inc = 0.1 # starting angle
    direc = 1   # direction of rotation
    scale = F * 3 # scaling factor for the node features
    # weight to the past time steps
    if memory_order == 1:
        weight = [1]
    else:
        weight = [2 ** (-k - 1) for k in range(memory_order)]
        weight[-1] = weight[-2]

    # list of matrices of the global map
    R = [np.zeros((0, F)) for n in range(N)]
    for n in range(N):
        theta = theta_inc + n * 0.05
        for i in range(memory_order):
            th = theta * direc
            # rotation of the points at t-i
            Ri = weight[i] * np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
            # construct single matrix
            R[n] = np.concatenate((R[n], Ri), axis=0)
            # update angle
            direc *= -1
            theta *= 2

    # init memory points
    if memory_points is None:
        memory_points_current = np.random.rand(np.max([memory_order, 1]), N, F) - .5
        memory_points_current *= scale
    else:
        memory_points_current = memory_points
        
    # apply the map
    points_list = []
    for t in range(T):
        #arrange mem_points as [ NxF | NxF | NxF | NxF ] and normalise + scale
        mp = np.hstack(list(memory_points_current))
        # init points
        new_points = np.zeros((N,F))
        for n in range(N):
            # rotate
            new_points[n] = np.dot(mp[n], R[n]) / np.linalg.norm(mp[n]) * scale
        # perturb with gaussian noise
        if no_distortion_in_the_first and t==0:
            pass
        else:
            new_points += np.random.randn(N, F) * sigma_distortion
        # store
        points_list.append(new_points)
        # update mem_points
        if update_memory_points:
            memory_points_current[1:] = memory_points_current[:-1].copy()
            memory_points_current[0] = new_points.copy()
    return np.array(points_list)


def simple_rotation(N, F, T, memory_order, sigma_distortion):
    starting_point = np.random.uniform(-1, 1, size=(N, F))
    output = [starting_point for _ in range(memory_order)]
    c = np.random.uniform(-1, 1, size=(N, ))
    omegas = c
    for t in range(T):
        new_point = np.empty((N, F))
        for n in range(N):
            omegas[n] = c[n] + 0.01 * np.cos(np.sum([_[n, ...] for _ in output[-memory_order:]]))
            th = omegas[n]
            R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
            new_point[n, ...] = output[-1][n].dot(R)
        new_point += np.random.randn(N, F) * sigma_distortion
        output.append(new_point)

    return np.array(output)[memory_order:, ...]
