# -*- coding: utf-8 -*-
# @Desc  :

import copy

import numpy as np


class Pvalues1:
    def __init__(self, bbox, scores_cal=None, scores_eval=None, delta=0.05):
        self.bbox = copy.deepcopy(bbox)

        # Calibrate
        self.scores_cal = scores_cal
        self.n_cal = len(self.scores_cal)
        self.delta = delta

        # Evaluation
        self.scores_eval = scores_eval

    def predict(self):
        scores_test = self.scores_eval
        scores_mat = np.tile(self.scores_cal, (len(scores_test), 1))
        tmp = np.sum(scores_mat <= scores_test.reshape(len(scores_test), 1), 1)
        pvals = (1.0 + tmp) / (1.0 + self.n_cal)

        # Collect results
        output = {"pvalue": pvals}

        return output
