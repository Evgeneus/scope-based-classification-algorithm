import numpy as np
import random


def synthesize(acc_distribution, n_papers, papers_page, J, theta):
    '''
    
    :param acc_distribution: 
    :param n_papers: 
    :param papers_page: 
    :param J: 
    :param theta: 
    :return:GT- ground truth values
            GT = [obj_1_val, obj_2_val, ..]
            psi_obj - observations 
            psi_obj = [[obj_1 values], []]
            psi_w - workers' judgments on papers
            psi_w = [{obj_id: val,..}, {}], index === worker's id
    '''

    GT = [np.random.binomial(1, theta) for _ in range(n_papers)]

    # generate observations
    pages_n = n_papers / papers_page
    psi_obj = [[] for obj in range(n_papers)]
    psi_w = [{} for _ in range(pages_n*J)]
    for page_id in range(pages_n):
        for _pointer in range(J):
            worker_id = page_id * J + _pointer
            worker_acc = random.choice(acc_distribution)
            for obj_id in range(page_id*papers_page, page_id*papers_page+papers_page, 1):
                gold_value = GT[obj_id]
                if gold_value:
                    worker_judgment = np.random.binomial(1, worker_acc)
                else:
                    worker_judgment = np.random.binomial(1, 1-worker_acc)
                psi_obj[obj_id].append(worker_judgment)
                psi_w[worker_id].update({obj_id: worker_judgment})
    return GT, psi_obj, psi_w
