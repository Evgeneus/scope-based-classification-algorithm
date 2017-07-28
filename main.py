'''
run experiments
'''

import numpy as np
import pandas as pd
from cf_simulation_synthetic import synthesizer
from quiz_simulation import do_quiz_scope
from task_simulation import do_task_scope, get_metrics
from generator import synthesize
from classifier_utils import classifier, estimate_accuracy, find_jt, get_loss
from fusion_utils import input_adapter, get_theta
from em import expectation_maximization


def run_quiz_scope(trust_min=0.75, quiz_papers_n=4, cheaters_prop=0.5,  easy_add_acc = 0.2):
    statistic_passed = {
        'rand_ch': 0,
        'smart_ch': 0,
        'worker': 0
    }
    statistic_total = {
        'rand_ch': 0,
        'smart_ch': 0,
        'worker': 0
    }
    # a value: (user_trust, user_accuracy)
    user_population = {
        'rand_ch': [],
        'smart_ch': [],
        'worker': []
    }
    acc_passed_distr = []
    for _ in range(10000):
        result = do_quiz_scope(trust_min, quiz_papers_n, cheaters_prop, easy_add_acc)
        if len(result) > 1:
            statistic_passed[result[2]] += 1
            statistic_total[result[2]] += 1
            user_population[result[2]].append(result[:2])
            acc_passed_distr.append(result[1])
        else:
            statistic_total[result[0]] += 1
    # pd.DataFrame(acc_passed_distr).to_csv('visualisation/data/quiz_acc_passed_z03_t5.csv', header=False, index=False)
    rand_cheaters_passed = statistic_passed['rand_ch'] / float(statistic_total['rand_ch']) * 100
    # smart_cheaters_passed = statistic_passed['smart_ch'] / float(statistic_total['smart_ch']) * 100
    smart_cheaters_passed = 0.
    workers_passed = statistic_passed['worker'] / float(statistic_total['worker']) * 100

    # print '*** Quiz ***'
    # print 'random cheaters passed: {}%'.format(rand_cheaters_passed)
    # print 'smart cheaters passed: {}%'.format(smart_cheaters_passed)
    # print 'workers passed: {}%'.format(workers_passed)

#   calculate the proportion of types of users passed the quiz
    user_prop = []
    users_passed = float(sum(statistic_passed.values()))
    for user_t in ['rand_ch', 'smart_ch', 'worker']:
        user_prop.append(statistic_passed[user_t]/users_passed)
    return [user_prop, user_population, acc_passed_distr]


def run_task_scope(trust_trsh, user_prop, user_population, easy_add_acc, n_papers, quiz_papers_n):
    tests_page = 0
    papers_page = 10
    price_row = 0.2
    fp_cost = 5
    data = []
    theta = 0.5
    # do simulation
    # for N in range(3, 21, 2):
    for N in [2, 3, 5, 10]:
        Nj_params = range(N/2 + 1, N + 1, 1)
        loss_dict = {}
        for key in Nj_params:
            loss_dict.update({key: []})  # {Nj: [loss]}
        for _ in range(1000):
            gold_data, trusted_judgment, budget_spent = do_task_scope(trust_trsh, tests_page, papers_page,
                                                                      n_papers, price_row, N, user_prop,
                                                                      user_population, easy_add_acc, quiz_papers_n, theta)
            for Nj in Nj_params:
                loss = get_metrics(gold_data, trusted_judgment, fp_cost, Nj)
                loss_dict[Nj].append(loss)

        loss_avg_list, loss_std_list = [], []
        for Nj in Nj_params:
            loss_avg_list.append(np.average(loss_dict[Nj]))
            loss_std_list.append(np.std(loss_dict[Nj]))
        Nj_ind_opt = loss_avg_list.index(min(loss_avg_list))
        Nj_opt = Nj_params[Nj_ind_opt]
        budget_spent = N*float(papers_page+quiz_papers_n)/(papers_page)
        data_iem = [quiz_papers_n, N, Nj_opt, loss_avg_list[Nj_ind_opt],
                    loss_std_list[Nj_ind_opt], budget_spent, fp_cost, papers_page, theta]
        data.append(data_iem)

    df = pd.DataFrame(data=data, columns=['tests_page', 'N', 'Nj', 'loss_avg',
                                          'loss_std', 'budget', 'cr', 'papers_page', 'theta'])

    with open('visualisation/data/loss_tests.csv', 'a') as f:
        df.to_csv(f, index=False, header=False)


def run_task_criteria():
    tests_page_params = [1, 1, 1, 2, 2, 3]
    papers_page_params = [1, 2, 3, 2, 3, 3]
    for test_page, papaers_page in zip(tests_page_params, papers_page_params):
        job_accuracy_list = []
        budget_spent_list = []
        for _ in range(10000):
            job_accuracy, budget_spent, paid_pages_n = synthesizer(trust_min=1., n_criteria=3,
                                                                   test_page=test_page, papers_page=papaers_page,
                                                                   quiz_papers_n=4, n_papers=18, budget=50,
                                                                   price_row=0.4, judgment_min=3, judgment_max=5,
                                                                   cheaters_prop=0.1)
            job_accuracy_list.append(job_accuracy)
            budget_spent_list.append(budget_spent)

        job_accuracy_avg = np.mean(job_accuracy_list)
        job_accuracy_std = np.std(job_accuracy_list)
        budget_spent_avg = np.mean(budget_spent_list)
        budget_spent_std = np.std(budget_spent_list)

        print '*********************'
        print 'tests_page: {}'.format(test_page)
        print 'papaers_page: {}'.format(papaers_page)
        print '---------------------'
        print 'job_accuracy_avg={}\n' \
              'job_accuracy_std={}\n' \
              'budget_spent_avg={}$\n' \
              'budget_spent_std={}$\n'.format(job_accuracy_avg, job_accuracy_std, budget_spent_avg, budget_spent_std)


def postProc_algorithm():
    trusts_trsh = 1.
    cheaters_prop = 0.3
    n_papers = 900
    quiz_papers_n = 5
    papers_page = 10
    theta_params = np.arange(0.1, 1, 0.1)
    # theta_params = [0.1, 0.5, 0.9]
    # cost_params = range(1, 10)
    cost = 10

    # print 'theta: {}'.format(theta)
    # loss_real_dict = {0:[], 1:[], 2:[], 3:[], 4:[]}
    # for i in range(100):
    data = []
    # for theta in theta_params:
    for theta in [0.8, 0.9]:
        print 'theta: {}'.format(theta)
        # for cost in cost_params:
        for J in [5]:
            print 'J: {}'.format(J)
            Jt_mv = J / 2 + 1
            # print 'cost: {}'.format(cost)
            theta_mv_list = []
            theta_em_list = []
            Jt_mv_list = []
            Jt_em_list = []
            loss_mv = []
            loss_mv_theta = []
            loss_em = []
            acc_em_list = []
            loss_clmv = []
            for _ in range(30):
                user_prop, user_population, acc_distribution = run_quiz_scope(trusts_trsh, quiz_papers_n,
                                                                              cheaters_prop, 0.0)
                GT, psi_obj, psi_w = synthesize(acc_distribution, n_papers, papers_page, J, theta)
                # MV estimation
                agg_values, theta_mv = classifier(psi_obj, Jt_mv)
                acc_mv = estimate_accuracy(agg_values, psi_w)
                # acc_mv_avg = np.mean(acc_mv)
                # loss_mv.append(get_loss(GT, psi_obj, cost, Jt_mv))
                # # Class func: MV theta=0.5
                # Nt = quiz_papers_n
                # z = cheaters_prop
                # Zs = (z * 0.5 ** Nt) / (z * 0.5 ** Nt + (2. * (1 - z) / (Nt + 1)) * (1 - 1. / (2 ** Nt + 1)))
                # acc_tw_avg = 2 ** (Nt + 1) * (Nt + 1) * (1 - 0.5 ** (Nt + 2)) / ((2 ** (Nt + 1) - 1) * (Nt + 2))
                # acc_avg = Zs * 0.5 + (1 - Zs) * acc_tw_avg
                # Jt_mv_theta = find_jt(0.5, J, acc_avg, cost)
                # loss_mv_theta.append(get_loss(GT, psi_obj, cost, Jt_mv_theta))
                # print 'Jt_mv_theta: {}'.format(Jt_mv_theta[0])
                # print 'loss: {}'.format(loss_mv_theta)
                # # Class func: MV
                # Jt_mv_opt = find_jt(theta_mv, J, acc_mv_avg, cost)
                # loss_clmv.append(get_loss(GT, psi_obj, cost, Jt_mv_opt))
                # theta_mv_list.append(theta_mv)
                # Jt_mv_list.append(Jt_mv_opt)

                # Class func: EM
                Psi = input_adapter(psi_w, n_papers)
                acc_em, em_p = expectation_maximization(len(psi_w), n_papers, Psi, acc_mv)
                theta_em = get_theta(em_p)
                acc_em_avg = np.mean(acc_em)
                acc_em_list.append(acc_em_avg)
                # agg_em = prob_binary_convert(em_p)
                # theta_em = sum(agg_em) / float(len(agg_em))
                Jt_em = find_jt(theta_em, J, acc_em_avg, cost)
                loss_em.append(get_loss(GT, psi_obj, cost, Jt_em))
                theta_em_list.append(theta_em)
                # Jt_em_list.append(Jt_em)
            print 'losss: {}'.format(np.mean(loss_em))


    #
    #         data.append([theta, np.mean(loss_em),
    #                      np.mean(theta_em), np.mean(acc_em_list),
    #                      ])
    # df = pd.DataFrame(data, columns=['theta', 'loss', 'theta', 'acc'])
    # df.to_csv('visualisation/data/round1_theta.csv', index=False)

            # data.append([theta, np.mean(theta_mv_list), np.std(theta_mv_list),
            #              np.mean(theta_em_list), np.std(theta_em_list),
            #              J, np.mean(Jt_mv_list), np.std(Jt_mv_list),
            #              np.mean(Jt_em_list), np.std(Jt_em_list), cost,
            #              np.mean(loss_clmv), np.std(loss_clmv),
            #              np.mean(loss_mv), np.std(loss_mv),
            #              np.mean(loss_em), np.std(loss_em),
            #              np.mean(loss_mv_theta), np.std(loss_mv_theta)])
    # df = pd.DataFrame(data, columns=['theta', 'theta_mv_avg', 'theta_mv_std',
    #                                  'theta_em_avg', 'theta_em_std',
    #                                  'J', 'Jt_mv_avg', 'Jt_mv_std',
    #                                  'Jt_em_avg', 'Jt_em_std', 'cost',
    #                                  'loss_clmv_avg', 'loss_clmv_std',
    #                                  'loss_mv_avg', 'loss_mv_std',
    #                                  'loss_em_avg', 'loss_em_std', 'loss_mv_theta'])
    # df.to_csv('visualisation/data/loss_theta_test.csv', index=False)
    # df = pd.DataFrame(data, columns=['theta', 'theta_mv_avg', 'theta_mv_std',
    #                                  'theta_em_avg', 'theta_em_std',
    #                                  'J', 'Jt_mv_avg', 'Jt_mv_std',
    #                                  'Jt_em_avg', 'Jt_em_std', 'cost',
    #                                  'loss1_avg', 'loss1_std',
    #                                  'loss_avg', 'loss_std',
    #                                  'loss2_avg', 'loss2_std',
    #                                  'loss3_avg', 'loss3_std'])
    # df.to_csv('visualisation/data/loss_theta_mv22.csv', index=False)
    #
    # df = pd.DataFrame(data, columns=['theta', 'theta_mv_avg', 'theta_mv_std',
    #                                  'theta_em_avg', 'theta_em_std',
    #                                  'J', 'Jt_mv_avg', 'Jt_mv_std',
    #                                  'Jt_em_avg', 'Jt_em_std', 'cost',
    #                                  'loss_avg', 'loss_std',
    #                                  'loss1_avg', 'loss1_std',
    #                                  'loss2_avg', 'loss2_std',
    #                                  'loss3_avg', 'loss3_std'])
    # df.to_csv('visualisation/data/loss_theta_clmv22.csv', index=False)
    #
    # df = pd.DataFrame(data, columns=['theta', 'theta_mv_avg', 'theta_mv_std',
    #                                  'theta_em_avg', 'theta_em_std',
    #                                  'J', 'Jt_mv_avg', 'Jt_mv_std',
    #                                  'Jt_em_avg', 'Jt_em_std', 'cost',
    #                                  'loss1_avg', 'loss1_std',
    #                                  'loss2_avg', 'loss2_std',
    #                                  'loss_avg', 'loss_std',
    #                                  'loss3_avg', 'loss3_std'])
    # df.to_csv('visualisation/data/loss_theta_em22.csv', index=False)
    #
    # df = pd.DataFrame(data, columns=['theta', 'theta_mv_avg', 'theta_mv_std',
    #                                  'theta_em_avg', 'theta_em_std',
    #                                  'J', 'Jt_mv_avg', 'Jt_mv_std',
    #                                  'Jt_em_avg', 'Jt_em_std', 'cost',
    #                                  'loss1_avg', 'loss1_std',
    #                                  'loss2_avg', 'loss2_std',
    #                                  'loss3_avg', 'loss3_std',
    #                                  'loss_avg', 'loss_std'])
    # df.to_csv('visualisation/data/loss_theta_mvt22.csv', index=False)



if __name__ == '__main__':
    postProc_algorithm()
    # trusts_trsh = 1.
    # cheaters_prop = 0.3
    # easy_add_acc = 0.0
    # n_papers = 500
    #
    # for quiz_papers_n in range(1, 11, 1):
    #     user_prop, user_population, acc_distr = run_quiz_scope(trusts_trsh, quiz_papers_n, cheaters_prop, easy_add_acc)
    #     d_item = run_task_scope(trusts_trsh, user_prop, user_population, easy_add_acc, n_papers, quiz_papers_n)
