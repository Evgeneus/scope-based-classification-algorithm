def input_adapter(psi_w, n_papers):
    '''
    :param psi_w: 
    :param n_papers: 
    :return: 
    '''

    Psi = [[] for _ in range(n_papers)]
    for w_id, worker_data in enumerate(psi_w):
        for obj, val in worker_data.iteritems():
            Psi[obj].append((w_id, val))
    return Psi


def invert(N, M, Psi):
    """
    Inverts the observation matrix. Need for performance reasons.
    :param N:
    :param M:
    :param Psi:
    :return:
    """
    inv_Psi = [[] for s in range(N)]
    for obj in range(M):
        for s, val in Psi[obj]:
            inv_Psi[s].append((obj, val))
    return inv_Psi


def prob_binary_convert(data):
    '''
    :param data: 
    :return: 
    '''
    data_b = []
    for obj in data:
        values = obj.keys()
        probs = obj.values()
        value_id = probs.index(max(probs))
        data_b.append(values[value_id])
    return data_b


def get_theta(data):
    theta = 0.
    for obj_p in data:
        if len(obj_p) == 1:
            if obj_p.keys()[0] == 1:
                theta += obj_p[1]
        else:
            theta += obj_p[1]
    return theta/len(data)
