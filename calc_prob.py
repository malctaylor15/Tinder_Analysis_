from math import lgamma
from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind, ttest_ind_from_stats, ttest_1samp
from scipy.stats import beta
from mpmath import betainc




#defining the functions used
@jit
def h(a, b, c, d):
    num = lgamma(a + c) + lgamma(b + d) + lgamma(a + b) + lgamma(c + d)
    den = lgamma(a) + lgamma(b) + lgamma(c) + lgamma(d) + lgamma(a + b + c + d)
    return np.exp(num - den)

@jit
def g0(a, b, c):    
    return np.exp(lgamma(a + b) + lgamma(a + c) - (lgamma(a + b + c) + lgamma(a)))

@jit
def hiter(a, b, c, d):
    while d > 1:
        d -= 1
        yield h(a, b, c, d) / d

def g(a, b, c, d):
    return g0(a, b, c) + sum(hiter(a, b, c, d))

def calc_prob_between(beta1, beta2):
    return g(beta1.args[0], beta1.args[1], beta2.args[0], beta2.args[1])

def calc_beta_mode(a, b):
    '''this function calculate the mode (peak) of the Beta distribution'''
    return (a-1)/(a+b-2)

def dist_plot(betas, names, linf=0, lsup=0.7):
    '''this function plots the Beta distribution'''
    x=np.linspace(linf,lsup, 100)
    for f, name in zip(betas,names) :
        y=f.pdf(x) #this for calculate the value for the PDF at the specified x-points
        y_mode=calc_beta_mode(f.args[0], f.args[1])
        y_var=f.var() # the variance of the Beta distribution
        plt.plot(x,y, label=f"{name} sample, conversion rate: {y_mode:0.1E} $\pm$ {y_var:0.1E}")
        plt.yticks([])
    plt.legend()
    plt.show()


def get_cnt_pct(series_raw):
    series = series_raw.copy()
    series = series.fillna('Missing')
    pct = np.round(100 * series.value_counts(normalize=True), 2)
    cnt = series.value_counts()
    out = pd.concat([cnt, pct], axis=1, keys=['Count', 'Percent'])
    return (out)

class ABTest_1Group():
    def __init__(self, series ):
        self.series = series.copy()
        vc = get_cnt_pct(self.series)
        print("Value Counts for Series")
        print(vc)

        assert(len(vc) == 2), "More than 2 levels in series"
        self.uniques = vc.index.tolist()

    def t_test(self, compare_val=0.5, unique_val=''):
        if unique_val not in self.uniques:
            unique_val = self.uniques[0]

        print(f"Using {unique_val} as 1")
        binary = np.where(self.series == unique_val, 1, 0)
        t, p = ttest_1samp(binary, compare_val)

        print("ttest_ind:            t = %g  p = %g" % (t, p))

        if p < 0.05:
            print(f"P value is low, reject H0 that mean == {compare_val}")
        else:
            print(f"P value is high, do not reject H0, mean =/= {compare_val}")

    def bayesian_compare(self, unique_val='', lower_test_bound = 0.45, upper_test_bound = 0.55, **kwargs):
        if unique_val not in self.uniques:
            unique_val = self.uniques[0]

        print(f"Using {unique_val} as 1")
        binary = np.where(self.series == unique_val, 1, 0)

        grp1_size = len(binary)
        grp1_sum = sum(binary)

        a_T, b_T = grp1_sum + 1, grp1_size - grp1_sum + 1
        beta_T = beta(a_T, b_T)

        linf = kwargs.get('linf', 0.2)
        lsup = kwargs.get('lsup', 0.8)
        dist_plot([beta_T], names=[unique_val], linf=linf, lsup=lsup)


        p=betainc(a_T, b_T, lower_test_bound ,upper_test_bound, regularized=True)
        print(f"Likelihood p is between {lower_test_bound} and {upper_test_bound}\
            is {np.round(float(p), 4)}")

class ABTest_2Group():
    def __init__(self, df, group_col_name:str, metric_col_name:str):

        self.group_col_name = group_col_name
        self.metric_col_name = metric_col_name

        if group_col_name not in df.columns:
            print(group_col_name, " not in columns: ", df.columns)
            error = df[group_col_name]
        if metric_col_name not in df.columns:
            print(metric_col_name, " not in columns: ", df.columns)
            error = df[metric_col_name]

        self.df = df[[group_col_name, metric_col_name]].copy()
        print('Group VC:')
        grp_vc = get_cnt_pct(df[group_col_name])
        self.unique_grps = grp_vc.index.tolist()
        print(grp_vc)

        print("Metric VC:")
        metric_vc = get_cnt_pct(df[metric_col_name])
        self.unique_metrics = metric_vc.index.tolist()
        print(metric_vc)

    def _load_binary_data(self, metric_val):
        if metric_val not in self.unique_metrics:
            metric_val = self.unique_metrics[0]
        group1 = self.df[self.df[self.group_col_name] == self.unique_grps[0]][self.metric_col_name]
        group2 = self.df[self.df[self.group_col_name] == self.unique_grps[1]][self.metric_col_name]
        group1_binary = np.where(group1 == metric_val, 1, 0)
        group2_binary = np.where(group2 == metric_val, 1, 0)
        print(f"Comparing mean of metric f{metric_val} between groups: f{self.unique_grps}")

        return(group1_binary, group2_binary)


    def t_test(self, metric_val = ''):


        print(f"Comparing mean of metric f{metric_val} between groups: f{self.unique_grps}")

        group1_binary, group2_binary = self._load_binary_data(metric_val)

        t, p = ttest_ind(group1_binary, group2_binary, equal_var=False)
        print("ttest_ind:            t = %g  p = %g" % (t, p))
        if p < 0.05:
            print(f"P value is low, reject H0 that means are the same between the two groups")
        else:
            print(f"P value is high, do not reject H0, means are different between the groups")

    def bayesian_compare(self, metric_val ='', **kwargs):

        group1_binary, group2_binary = self._load_binary_data(metric_val)

        grp1_size = len(group1_binary)
        grp1_sum = sum(group1_binary)

        grp2_size = len(group2_binary)
        grp2_sum = sum(group2_binary)

        # here we create the Beta functions for the two sets
        a_C, b_C = grp1_sum + 1, grp1_size - grp1_sum + 1
        beta_C = beta(a_C, b_C)
        a_T, b_T = grp2_sum + 1, grp2_size - grp2_sum + 1
        beta_T = beta(a_T, b_T)

        # calculating the lift
        lift = (beta_T.mean() - beta_C.mean()) / beta_C.mean()

        # calculating the probability for Test to be better than Control
        prob = calc_prob_between(beta_T, beta_C)
        linf = kwargs.get('linf', 0.2)
        lsup = kwargs.get('lsup', 0.8)
        dist_plot([beta_C, beta_T], names=self.unique_grps, linf=linf, lsup=lsup)

        print(
            f"Short messeages option lift Long Conversation Rate Rates by {lift * 100:2.2f}% with {prob * 100:2.1f}% probability.")


    def beta_within_range(self, grp, compare_val_low, compare_val_high=1, metric_val=''):

        if grp not in self.unique_grps:
            grp = self.unique_grps[0]
        if metric_val not in self.unique_metrics:
            metric_val = self.unique_metrics[0]

        print(f"Using distribution for group: {grp}")

        # Probability that test is greater than value
        group1 = self.df[self.df[self.group_col_name] == grp][self.metric_col_name]
        group1_binary = np.where(group1 == metric_val, 1, 0)

        grp1_size = len(group1_binary)
        grp1_sum = sum(group1_binary)

        a_C, b_C = grp1_sum + 1, grp1_size - grp1_sum + 1
        p = betainc(a_C, b_C, compare_val_low, compare_val_high, regularized=True)
        print(f"Likelihood p is between {compare_val_low} and {compare_val_high}  is  {np.round(float(p), 4)}")
        return(float(p))

