# -*- coding: utf-8 -*-
# @Desc  :

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def write_sel_clean(pvals_one_class, alpha=0.1, lambda_par=0.5):
    # pi0_true = 1.0 - np.mean(is_nonnull)

    results_fdr = pd.DataFrame()
    results_reject = {}

    for m in list(pvals_one_class):
        results_reject[m] = {}
        pval = pvals_one_class[m]

        pi = 1.0  # we set it as 1.

        alpha_eff = alpha / pi
        reject, pvals_adj, _, _ = multipletests(pval, alpha=alpha_eff,
                                                method='fdr_bh')  # BH multiple testing method

        rejections = np.sum(reject)

        pd.set_option('precision', 5)
        # res_tmp = pd.DataFrame(res_tmp, index=[0])
        # results_fdr = pd.concat([results_fdr, res_tmp])
        results_reject[m][False] = reject
    print(results_fdr)

    return results_fdr, results_reject


def evaluate_all_methods(pvals_one_class, is_nonnull, alpha=0.1, lambda_par=0.5):
    pi0_true = 1.0 - np.mean(is_nonnull)

    results_fdr = pd.DataFrame()
    results_reject = {}

    for m in list(pvals_one_class):
        results_reject[m] = {}
        pval = pvals_one_class[m]

        pi = 1.0  # we set it as 1.

        alpha_eff = alpha / pi
        reject, pvals_adj, _, _ = multipletests(pval, alpha=alpha_eff,
                                                method='fdr_bh')  # BH multiple testing method

        rejections = np.sum(reject)
        if rejections > 0:
            fdr = 1 - np.mean(is_nonnull[np.where(reject)[0]])
            recall = np.sum(is_nonnull[np.where(reject)[0]]) / np.sum(is_nonnull)
        else:
            fdr = 0
            recall = 0
        F1 = 0
        if fdr == 1 or recall == 0:
            F1 = 0
            print('FDR==1 or Recall==0')
        else:
            F1 = 2 / (1 / (1 - fdr) + 1 / recall)
        res_tmp = {'Method': m, 'Pi0-true': pi0_true,
                   'Alpha': alpha, 'Rejections': rejections, 'FDR': fdr, 'Recall': recall, 'F1': F1}

        pd.set_option('precision', 5)
        res_tmp = pd.DataFrame(res_tmp, index=[0])
        results_fdr = pd.concat([results_fdr, res_tmp])
        results_reject[m][False] = reject

    print(results_fdr)

    return results_fdr, results_reject
