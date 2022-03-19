import numpy as np
import itertools
import uuid
import warnings
from typing import Tuple, List, Any

from src.thspace import THSpace


def _add_uuid_udf():
    return str(uuid.uuid4())[:8]


def gen_th_space(space_params):
    """
    Generate threshold space from 
    given space parameters. Sort
    in descending order.
    """
    res = []
    for param in space_params:
        min_, max_, n_ = param[0], param[1], param[2]
        nps = np.linspace(min_, max_, n_)
        nps = np.round(nps, 4)
        res.append(nps)
    return list(sorted(itertools.product(*res)))

  
class ThresholdOptimizationException(Exception):
    def __init__(self, message="threshold opt failed."):
        self.message = message
        super().__init__(self.message)
        

def is_th_in_space_edges(ths, spaces):
    for th, space in zip(ths, spaces):
        if th == min(space) or th == max(space):
            warnings.warn('optimized threshold is on defined interval edge.')


class BayesianThreshold:
    """
    Calculates the conditional joint probability
    of the labels being equal to 1, given the estimators.
    In this module threshold is applied on the higher
    side, meaning that it is the upper threshold.
    Hence if the threshold needs to be applied on the lower
    end, it is advised to multiply the estimators with -1.

    Parameters:
        y: np.ndarray
            one dimensional array of labels (0, 1)
        x: np.ndarray
            2 dimensional array of features and feature values.
    """
    def __init__(self, y, x, prior=None):
        self.y = y
        self.k = len(self.y)
        self.x = x
        self._prior = prior
        self.n_x = self.x.shape[1]
        self.data = np.column_stack([self.y, self.x])

    @property
    def prior(self) -> float:
        """
        Prior probability.

        Returns
        -------
        float:
            prior probability
        """
        if self._prior is None:
            return sum(self.y)/self.k
        else:
            return self._prior
        
    def cond_theta(self, ths, data=None) -> np.array:
        """
        slice sample that meets the theta
            conditions (thresholds). 
        """
        if data is None:
            data = self.data.copy()
        for idx in range(self.n_x):
            data = data[data[:, idx+1] <= ths[idx]]
        return data
    
    @staticmethod
    def _calc_cond_proba(likelihood, prior, marginal_proba) -> float:
        """
        Calculate condition probability given likelihood,
        prior probability and marginal probability with:

        P(y=1|θ) = P(θ|y=1)*P(y=1)/P(θ)
            where θ is P(T1<x1, T2<x2...)
            X : {xEX | x1,x2,x3...xn} and T is thresholds (TER||)


        Parameters
        ----------
        likelihood: float
            likelihood ( P(θ|y=1) )
        prior: float
            prior probability ( P(y=1) )
        marginal_proba: float
            marginal probability ( P(θ) )

        Returns
        -------
        float:
            condition probability
        """
        if marginal_proba > 0:
            return round(likelihood * prior / marginal_proba, 4)
        else:
            return 0.0
    
    def proba(self, ths: List[float]) -> float:
        """
        Calculate condition probability in threshold θ,
        with given formula:

        P(y=1|θ) = P(θ|y=1)*P(y=1)/P(θ)
            where θ is P(T1<x1, T2<x2...)
            X : {xEX | x1,x2,x3...xn} and T is thresholds (TER||)


        Parameters
        ----------
        ths: List[float]
            thresholds

        Returns
        -------
        float:
            condition probability
        """
        cond_y = self.data[self.data[:, 0] == 1]
        cond_y_theta = self.cond_theta(ths, cond_y)
        cond_theta = self.cond_theta(ths, self.data)
        p_y_theta = cond_y_theta.shape[0] / cond_y.shape[0]
        p_theta = cond_theta.shape[0] / self.k
        return self._calc_cond_proba(p_y_theta, self.prior, p_theta)

    
class THOpt(BayesianThreshold):
    """
    Optimize thresholds by comparing the
        probability at threshold with 
        bayesian modelling.
        Comparisons are done per each
        X, that is given a threshold in 
        threshold space.

    Parameters:
        y: np.ndarray
            one dimensional array of labels (0, 1)
        x: np.ndarray
            2 dimensional array of features and feature values.
    """
    
    def __init__(self, y, x, th_space, prior=None):
        super().__init__(y, x, prior)
        self.ts = THSpace(th_space)
        self.err = 1
        self.thresholds = 0
        self.p = 0
        self.hist = []
        
    def log(self, ths, p, err):
        self.hist.append(
            {
                'ths': ths,
                'proba': p,
                'error': err
            }
        )
        
    def opt(self, opt_proba: float = 0.9) -> Tuple[Any, float, float]:
        """
        optimization is done to get the
        probability at thresholds that
        is close to given optimal
        probability (1 - err_margin). It's advised to
        keep a low error margin. Having this value too high
        or too low might lead to over-fitting.
        We'd like to keep discovering new
        thresholds and optimizing, hence
        a value of 0.9 is proposed.

        Parameters
        ----------
        opt_proba: float
            desired optimal probability.

        Returns
        -------
        Tuple[float]
        """
        # opt
        while True:
            ths = self.ts.take(randomly=True)
            if ths:
                p = self.proba(ths)
                err = abs(opt_proba - p)
                if err < self.err:
                    self.err = err
                    self.p = p
                    self.thresholds = ths
            else:
                break
            self.log(ths, p, err)
        # out 
        if self.thresholds and self.p:
            is_th_in_space_edges(self.thresholds, self.ts.space)
            # in the future, we can return defaults here
            return self.thresholds, self.p, self.err
        else:
            return [None], 0.0, 1.0
        
