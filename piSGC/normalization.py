import numpy as np
import networkx as nx
import scipy.linalg
import scipy.sparse as sp
from scipy.linalg import pinvh
from tqdm import tqdm
from scipy import optimize
import scipy.io as sio
import os

np.set_printoptions(formatter={'all': lambda x: str(x)}, threshold=100)


def aug_normalized_adjacency(adj, ectd_data, args):
    # self-loops
    # adj = adj + sp.eye(adj.shape[0])

    # temp adj + self loop
    temp = (adj + sp.eye(adj.shape[0])).toarray()
    # mean_first_passage_time(temp)

    if args.adj == 'A1':
        adj_2nd = temp
    elif args.adj == 'A1_2':
        adj_2nd = temp @ temp
    elif args.adj == 'A1_3':
        adj_2nd = temp @ temp @ temp
    elif args.adj == 'A1_4':
        adj_2nd = temp @ temp @ temp @ temp
    elif args.adj == 'A1_5':
        adj_2nd = temp @ temp @ temp @ temp @ temp
    elif args.adj == 'A1_6':
        adj_2nd = temp @ temp @ temp @ temp @ temp @ temp

    adj_2nd = np.where(adj_2nd <= 0, adj_2nd, 1.)

    # 3-layer hierachical
    adj = adj + sp.eye(adj.shape[0])
    # 2-pow adj?
    adj = adj.toarray()
    split_list = [adj, adj @ adj, adj @ adj @ adj]
    split_list = [np.where(a <= 0, a, 1.) for a in split_list]
    for index in range(len(split_list) - 1, -1, -1):
        if index > 0:
            split_list[index] -= split_list[index - 1]

    value_list = []
    for adj in split_list:
        print(np.count_nonzero(adj))
        i, j = np.nonzero(adj)
        value_list.append(zip(i, j))

    '''
    ectd_data_path:
        ectd + 
        {dataset} +
        {is_vg} +
        {fun} +
        {An}
    '''
    # ectd_data = './pinv-dataset/ectd_cora_vg_log_A1.txt'
    # adj[i, j] = 1. record indexes
    print(np.count_nonzero(adj_2nd))
    # print(adj_2nd)
    i, j = np.nonzero(adj_2nd)
    values = zip(i, j)
    # print("values: {}".format(values))

    # print(calculate_rd(nnodes=adj_2nd.shape[0], values=values, adj=adj_2nd, save='./test.txt'))

    # deg = np.diag(adj_2nd.sum(1))
    deg = np.diag(adj.sum(1))
    if args.using_vg:
        print("using vg")
        if args.vg == 0.:
            vg = deg.sum().sum()
        else:
            vg = args.vg
    else:
        vg = 1.

    # vg = deg.sum().sum()
    # standard laplacian
    # lap = deg - adj
    # print("laplacian:\n {}".format(lap))
    # print("pinv lap:\n {}".format(pinvh(lap)))
    # print("L-1: {}".format(np.linalg.inv(lap)))

    # symmetric laplacian
    d_inv_sqrt = np.power(deg, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.

    # d_inv = np.power(deg, -1.)
    # d_inv[np.isinf(d_inv)] = 0.

    lap = np.identity(adj.shape[0]) - d_inv_sqrt.dot(adj).dot(d_inv_sqrt)
    # lap = np.around(lap, 3)
    # sio.savemat('lap_A3.mat', {'lap': lap})

    # print("L-1: {}".format(np.linalg.inv(lap)))
    # lap = np.identity(adj.shape[0]) - d_inv_sqrt.dot(adj_2nd).dot(d_inv_sqrt)
    # lap = np.identity(adj.shape[0]) - d_inv.dot(adj)

    # print("laplacian:\n {}, {}, {}".format(lap.dtype, type(lap), lap.shape))

    # print("symmetric pinv lap numpy:\n {}".format(np.linalg.pinv(np.mat(lap))))
    # print("symmetric pinv lap:\n {}".format(scipy.linalg.pinv(np.mat(lap))))
    # print("are they the same:\n {}".format(np.allclose(scipy.linalg.pinv(lap), pinvh(lap))))
    # print("values < 0: {}".format(np.sum(pinvh(lap) < 0.)))

    pinv = pinvh(lap)
    # pinv = np.around(pinv, 3)
    # sio.savemat('lapinv_A6.mat', {'lapinv': lap})

    # pinv = np.linalg.pinv(lap, hermitian=True)
    # print("np pinv: {}".format(pinv))
    # print("AGA: {}".format(lap @ pinv @ lap))

    # ectd = calculate_ectd_hie(nnodes=adj.shape[0], values=values, pinv=pinv, vg=vg,
    #                       save=ectd_data, split=value_list, full_adj=adj_2nd)

    ectd = calculate_ectd(nnodes=adj.shape[0], values=values, pinv=pinv, vg=vg, save=ectd_data)
    ectd = np.around(ectd, 3)

    adj = ectd
    adj = np.where(ectd > 0, adj_2nd, 0.)
    adj = sp.coo_matrix(adj)

    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()  # D^-1/2*A*D^-1/2
    # return d_mat_inv.dot(adj).tocoo()  # A*D^-1
    # return adj.tocoo()


def rw_normalized_adjacency(adj):
    # self-loops
    adj = adj + sp.eye(adj.shape[0])

    # 2-pow adj?
    adj = adj.toarray()

    adj_2nd = adj
    adj_2nd = np.where(adj_2nd <= 0, adj_2nd, 1.)
    # adj[i, j] = 1. record indexes
    print(np.count_nonzero(adj_2nd))
    i, j = np.nonzero(adj_2nd)
    values = zip(i, j)

    deg = np.diag(adj_2nd.sum(1))
    vg = deg.sum().sum()

    # standard laplacian
    lap = deg - adj_2nd
    print("laplacian:\n {}".format(lap))

    # symmetric laplacian
    d_inv_sqrt = np.power(deg, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.

    lap = np.identity(adj.shape[0]) - d_inv_sqrt.dot(adj_2nd).dot(d_inv_sqrt)
    print("symmetric laplacian:\n {}".format(lap))

    pinv = pinvh(lap)
    # ectd = calculate_ectd(nnodes=adj.shape[0], values=values, pinv=pinv, vg=vg)
    ectd = calculate_ectd_hie(nnodes=adj.shape[0], values=values, pinv=pinv, vg=vg)

    adj = ectd
    adj = sp.coo_matrix(adj)

    row_sum = np.array(adj.sum(1))

    # D^-1
    d_inv = np.power(row_sum, -1.).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)

    return adj.dot(d_mat_inv).tocoo()  # A*D^-1


def fetch_normalization(type):
    switcher = {
        'AugNormAdj': aug_normalized_adjacency,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
        'RWNormAdj': rw_normalized_adjacency,  # A' = ( A + I ) * (D + I)^-1
    }
    func = switcher.get(type, lambda: "Invalid normalization technique.")
    return func


def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sys_normalized_adjacency(adj):
    # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def cal_pinv(lap):
    pass


def sigmoid(vec):
    # return 1 / (1 + np.exp(-vec))
    alpha = 1.
    beta = 0.
    return 1 / (1 + np.exp(-alpha * (vec - beta)))


def arctan(vec):
    return 2 * np.arctan(vec) / np.pi


def cos(vec):
    return 2 * np.cos(vec) / np.pi


def sigmoid_kct(vec, std):
    alpha = 1.5
    return 1 / (1 + np.exp(-alpha * vec / std))


def alpha_exp(vec, alpha=0.3):
    return (1 - alpha) + alpha * np.exp(-vec)


def calculate_rd(nnodes, values, adj, save=None):
    # make adj a strong-connect graph
    # standard adj
    adj = np.where(adj != 0, adj, 1e10)
    # print(adj)
    # normalized adj
    # adj = np.where(adj <= 0, adj, 1e-10)
    G = nx.from_numpy_array(adj)
    G = G.to_directed()

    if os.path.exists(save):
        rd = np.loadtxt(save, delimiter=',')
    else:
        rd = np.zeros((nnodes, nnodes), )
        for i, j in tqdm(values):
            rd[i, j] = nx.resistance_distance(G, i, j) if i != j else 0.
        np.savetxt(save, delimiter=',')
    return rd


def calculate_ectd(nnodes, values, pinv, vg=1., save=None):
    if os.path.exists(save): # no save
        ectd = np.loadtxt(save, delimiter=',')
    else:
        ectd = np.zeros((nnodes, nnodes), )

        for i, j in tqdm(values):
            eij = np.zeros((nnodes, 1), )
            eij[i, 0] = 1.
            eij[j, 0] = -1. if i != j else 0.

            ectd[i, j] = vg * eij.T @ pinv @ eij

        np.savetxt(save, ectd, fmt='%f', delimiter=',')

    # eps = 3/6/10
    eps = 10
    ectd[ectd > eps] = 0.
    ectd_norm = ectd
    ectd_norm = np.power(ectd, -1.)
    ectd_norm[np.isinf(ectd_norm)] = 0.

    return ectd_norm


def calculate_ectd_hie(nnodes, values, pinv, vg=1., save=None, split=None, full_adj=None):
    # G = nx.from_numpy_array(full_adj)
    # spd = dict(nx.all_pairs_shortest_path_length(G))

    if os.path.exists(save): # no save
        ectd = np.loadtxt(save, delimiter=',')
    else:
        ectd = np.zeros((nnodes, nnodes), )
        adj_1, adj_2, adj_3 = list(split[0]), list(split[1]), list(split[2])

        for i, j in tqdm(values):
            eij = np.zeros((nnodes, 1), )
            eij[i, 0] = 1.
            eij[j, 0] = -1. if i != j else 0.

            ectd[i, j] = vg * eij.T @ pinv @ eij

            if (i, j) in adj_1:
                # print('\n1-a\n')
                ectd[i, j] = np.power((1 / ectd[i, j]), np.log(1.))
                # ectd[i, j] *= 100.
                # ectd[i, j] += spd[i][j]
            elif (i, j) in adj_2:
                # print('\n2-a\n')
                ectd[i, j] = np.power((1 / ectd[i, j]), np.log(2.))
                # ectd[i, j] *= 10.
                # ectd[i, j] = ectd[i, j] + 2 * spd[i][j]
            elif (i, j) in adj_3:
                # print('\n3-a\n')
                ectd[i, j] = np.power((1 / ectd[i, j]), np.log(3.))
                # ectd[i, j] = ectd[i, j] + 3 * spd[i][j]
            # ectd[i, j] = np.exp(-ectd[i, j]) if i != j else 0.
            # ectd[i, j] = eij.T @ pinv @ eij - vg if i != j else 0.
            # ectd[i, j] = np.log(ectd[i, j])
            # a - [0.4, 0.9]
            # a = 2 * np.e
            # ectd[i, j] = np.power(a, np.log(ectd[i, j]))
            # ectd[i, j] = np.power(a, ectd[i, j])
            # ectd[i, j] = np.exp(-np.log2(ectd[i, j]))
            # ectd[i, j] = np.sqrt(1. / ectd[i, j])
            # ectd[i, j] = alpha_exp(ectd[i, j])

            # sigmoid
            # ectd[i, j] = sigmoid(-ectd[i, j])
            # ectd[i, j] = sigmoid(1. / ectd[i, j])

            # tanh
            # ectd[i, j] = np.tanh(1. / ectd[i, j])
            # ectd[i, j] = 1. / np.tanh(ectd[i, j])

            # arctan
            # ectd[i, j] = arctan(1. / ectd[i, j])
            # ectd[i, j] = cos(ectd[i, j])
            # ectd[i, j] = arctan(1. / ectd[i, j])
        # np.savetxt(save, ectd, fmt='%f', delimiter=',')

    min = ectd[ectd.nonzero()].min()
    # print("init ectd:\n {}".format(ectd))
    # print("init min: {}".format(min))
    # print("init max: {}".format(ectd.max()))

    # reciprocal
    # ectd = np.power(ectd, -1.)

    ectd[np.isinf(ectd)] = 0.
    # set diagonal to 1
    # np.fill_diagonal(ectd, 1.)
    # print(np.max(ectd, axis=1))
    np.fill_diagonal(ectd, np.max(ectd, axis=1))

    # print("reciprocal of ectd:\n {}".format(ectd))

    min = ectd[ectd.nonzero()].min()
    # print("rep min: {}".format(min))
    # print("rep max: {}".format(ectd.max()))
    # print(ectd[100, 100], ectd[100, 1602], ectd[100, 2056])
    # print(ectd[0, 633])
    # min-max normalization(break symmetric) global?

    # ectd_norm = (ectd - min) / (ectd.max() - min)
    # ectd_norm[ectd_norm < 0] = 0.
    # max-min normalization
    # ectd_norm = (ectd - ectd.max()) / (min - ectd.max())
    # ectd_norm[ectd_norm == (ectd.max() / (ectd.max() - min))] = 0.

    # row-sum normalization
    # ectd_norm = row_normalize(ectd)

    # no normalization
    ectd_norm = ectd

    # print("symmetric matrix: {}".format(np.allclose(ectd_norm, ectd_norm.T)))
    # print("row-normed ectd:\n {}".format(ectd_norm))
    # max values
    # print(np.argmax(ectd, axis=1))
    # print("values smaller than 0: {}".format(np.sum(ectd_norm < 0)))

    return ectd_norm


def mean_first_passage_time(adj, tol=1e-3):
    # P = D^-1 * A
    nnodes = adj.shape[0]
    p = row_normalize(adj)
    e_vals, e_vecs = np.linalg.eig(p.T)
    aux = np.abs(e_vals - 1.)
    index = np.argmin(aux)
    w = e_vecs[:, index].T / e_vecs[:, index].sum()

    w_tile = np.tile(w, (nnodes, 1))
    I = np.identity(nnodes)

    Z = np.linalg.inv(I - p + w)
    mfpt = (np.tile(np.diag(Z).T, (nnodes, 1)) - Z) / w_tile

    print("mfpt: {}".format(mfpt))
    pass


def cal_ets(adj, ns=4):
    '''
    adj - numpy array of adjacency matrix
    ns - number of steps to consider
    '''
    nnodes = adj.shape[0]
    p = row_normalize(adj)

    delta = np.identity(nnodes)
    phi_list = np.empty([ns + 1, nnodes, nnodes])
    et = np.zeros([nnodes, nnodes])
    et = delta
    phi_list[0] = delta

    for i in range(0, ns):
        phi_list[i + 1] = delta + np.multiply((1 - delta), p.dot(phi_list[i]))
        et += (i + 1) * (np.subtract(phi_list[i + 1], phi_list[i]))
    ectd = np.reciprocal(et + et.T)
    ectd[np.isinf(ectd)] = 0.
    print(np.count_nonzero(ectd))

    return ectd


def getHittingTimeByTarget(b, t, Nstep=3, epsilon=.1):
    power = np.ones((b.shape[0])) / b.shape[0]
    n_t = np.zeros((1, b.shape[0]))
    b_ = b.copy()
    b_[t, :] = 0

    for i in range(1, Nstep):
        power = np.matmul(power, b_)
        n_t += power
        if np.linalg.norm(i * power) < epsilon: break
    return n_t


def getHitttingTime(adj):
    b = (adj.T / np.sum(adj, axis=1)).T
    h = np.zeros(adj.shape)

    for target in range(adj.shape[0]):
        h[:, target] = getHittingTimeByTarget(b, target)
    print(h)
    print(np.count_nonzero(h))

    ECTD = h + h.T
    return h
