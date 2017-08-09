import numpy as np


def do_quiz_scope(trust_min, quiz_papers_n, cheaters_prop, easy_add_val):
    correct_judgments = 0
    # decide if a worker a cheater
    if np.random.binomial(1, cheaters_prop):
        # decide if a worker a random cheater or smart one
        rand_cheater_prop = 1.
        if np.random.binomial(1, rand_cheater_prop):
            # random cheater
            worker_type = 'rand_ch'
            worker_accuracy = 0.5
        else:
            # smart cheater
            # TO DO
            worker_type = 'smart_ch'
            worker_accuracy = 0.6
    else:
        worker_type = 'worker'
        # worker_accuracy = 0.5 + (np.random.beta(1, 1) * 0.5)
        worker_accuracy = np.random.uniform(0.5, 0.7)

    for paper_id in range(quiz_papers_n):
        # decide if paper in/out of scope
        # bernoulli trial with p=0.5
        # gold_value = np.random.binomial(1, 0.5)

        worker_accuracy_new = worker_accuracy
        # if a paper is easy, otherwise it is avg
        if worker_type is not 'rand_ch':
            if np.random.binomial(1, 0.5):
                if worker_accuracy + easy_add_val > 1.:
                    worker_accuracy_new = 1.
                else:
                    worker_accuracy_new = worker_accuracy + easy_add_val
        if np.random.binomial(1, worker_accuracy_new):
            correct_judgments += 1
    worker_trust = float(correct_judgments)/quiz_papers_n
    if worker_trust >= trust_min:
        return [worker_trust, worker_accuracy, worker_type]
    return [worker_type]
