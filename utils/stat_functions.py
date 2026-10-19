import numpy as np

def confusion_matrix(p_racmo, p_obs, thr):
    ''' 
    Computes confusing matrix for variable 
    above certain threshold thr
    '''
    assert np.shape(p_racmo) == np.shape(p_obs), f"first convert racmo data to obs shape {np.shape(p_racmo)}, {np.shape(p_obs)}"
    val = np.empty(np.shape(p_obs))
    val[np.where((p_racmo <= thr) & (p_obs <= thr))] = 0.5 # True negative
    val[np.where((p_racmo > thr) & (p_obs > thr))] = 1.5 # True positive
    val[np.where((p_racmo <= thr) & (p_obs > thr))] = 2.5 # False negative
    val[np.where((p_racmo > thr) & (p_obs <= thr))] = 3.5 # False positive
    val[np.isnan(p_racmo) | np.isnan(p_obs)] =  np.nan
    return val

def skill_scores(dval, weights=None, printr=True):
    '''
    Computes skill scores based on confusion matrix
    '''
    if weights is None:
        TN = np.sum(dval==0.5)
        TP = np.sum(dval==1.5)
        FN = np.sum(dval==2.5)
        FP = np.sum(dval==3.5)
    else:
        TN = np.sum(dval==0.5)*np.nansum(weights[dval==0.5])/np.nansum(weights)
        TP = np.sum(dval==1.5)*np.nansum(weights[dval==1.5])/np.nansum(weights)
        FN = np.sum(dval==2.5)*np.nansum(weights[dval==2.5])/np.nansum(weights)
        FP = np.sum(dval==3.5)*np.nansum(weights[dval==3.5])/np.nansum(weights)
    csi = TP/(TP+FP+FN)
    pod = TP/(TP+FN)
    far = FP/(TN+FP)
    if printr:
        print('CSI=',csi, ', POD=', pod, ' and FAR=', far)
    return TN, TP, FN, FP
            
def compute_bias_se_r2_skill(mod_in, obs_in, thr, 
                             thr2=None, skill=True, 
                             weights=None):
    ''' 
    Computes bias, squared error, r^2 and
    skill scores for model (mod_in) compared
    to observations (obs_in) when above certain
    threshold thr.
    '''
    mod = np.array(mod_in, copy=True)
    obs = np.array(obs_in, copy=True)
    mod[mod <= thr] = np.nan
    obs[obs <= thr] = np.nan
    log_mod = np.log10(mod)
    log_obs = np.log10(obs)
    bias = np.nanmean(mod - obs)
    se = np.nansum((mod - obs)**2)
    log_bias = np.nanmean(log_mod - log_obs)
    log_se = np.nansum((log_mod - log_obs)**2)
    sum_obs = np.nansum(obs[np.isfinite(mod)])
    sum_mod = np.nansum(mod[np.isfinite(obs)])
    sum_obs_mod = np.nansum(obs[np.isfinite(mod-obs)]*mod[np.isfinite(mod-obs)])
    sum_obs2 = np.nansum(obs[np.isfinite(mod)]**2)
    sum_mod2 = np.nansum(mod[np.isfinite(obs)]**2)
    log_obs       = np.nansum(np.log10(obs[np.isfinite(mod)]))
    log_mod       = np.nansum(np.log10(mod[np.isfinite(obs)]))
    log_obs_mod   = np.nansum(np.log10(obs[np.isfinite(mod-obs)]) * np.log10(mod[np.isfinite(mod-obs)]))
    log_obs2      = np.nansum(np.log10(obs[np.isfinite(mod)])**2)
    log_mod2      = np.nansum(np.log10(mod[np.isfinite(obs)])**2)
    if skill:
        c1 = confusion_matrix(mod_in, obs_in, thr)
        TN1, TP1, FN1, FP1 = skill_scores(c1, weights=weights, printr=False)
        if thr2 == None:
            return bias, se, log_bias, log_se, \
                sum_obs, sum_mod, sum_obs_mod, \
                sum_obs2, sum_mod2, log_obs, log_mod, \
                    log_obs_mod, log_obs2, log_mod2, \
                    TP1, FN1, FP1, TN1
        else:
            c2 = confusion_matrix(mod, obs, thr2)
            TN2, TP2, FN2, FP2 = skill_scores(c2, weights=weights, printr=False)
            return bias, se, log_bias, log_se, \
                sum_obs, sum_mod, sum_obs_mod, \
                sum_obs2, sum_mod2, log_obs, log_mod, \
                    log_obs_mod, log_obs2, log_mod2, \
                    TP1, FN1, FP1, TN1, TP2, FN2, FP2, TN2
    else:
        return bias, se, log_bias, log_se, \
               sum_obs, sum_mod, sum_obs_mod, \
               sum_obs2, sum_mod2, log_obs, \
                log_mod, log_obs_mod, log_obs2, log_mod2,

def compute_avg_stats(ds):
    '''
    Compute average statistics based on file with
    all statistics for all frames. 
    '''
    N = ds.n_cond.values
    N_bias = N.copy()
    N_bias[~np.isfinite(ds.bias.values)] = np.nan
    N_rmse = N.copy()
    N_rmse[~np.isfinite(ds.se.values)] = np.nan
    N_sum = N.copy()
    N_sum[~np.isfinite(ds.sum_ec.values)] = np.nan
    N_sum = np.nansum(N_sum, axis=2)
    bias = np.nansum(ds.bias.values * ds.n_cond.values, axis=2) \
           / np.nansum(N_bias, axis=2)
    rel_bias = (np.nansum(ds.sum_rac.values, axis=2) -
                np.nansum(ds.sum_ec.values, axis=2)) \
               / np.nansum(ds.sum_ec.values, axis=2)
    rmse = np.sqrt(np.nansum(ds.se.values, axis=2) /
               np.nansum(N_rmse, axis=2))
    rel_rmse = rmse / np.nanmean(ds.sum_ec.values, axis=2)
    cov = (np.nansum(ds.sum_ec_rac.values, axis=2) / N_sum) \
          - (np.nansum(ds.sum_ec.values, axis=2)/N_sum) \
            * (np.nansum(ds.sum_rac.values, axis=2)/N_sum)
    var_ec = (np.nansum(ds.sum_ec2.values, axis=2)/N_sum) \
             - (np.nansum(ds.sum_ec.values, axis=2)/N_sum)**2
    var_rac = (np.nansum(ds.sum_rac2.values, axis=2)/N_sum) \
              - (np.nansum(ds.sum_rac.values, axis=2)/N_sum)**2
    r2 = (cov / np.sqrt(var_ec * var_rac))**2
    TP_total1 = np.nansum(ds.TPwc1.values, axis=2)
    FP_total1 = np.nansum(ds.FPwc1.values, axis=2)
    FN_total1 = np.nansum(ds.FNwc1.values, axis=2)
    TN_total1 = np.nansum(ds.TNwc1.values, axis=2)
    csi1 = TP_total1 / (TP_total1 + FP_total1 + FN_total1)
    pod1 = TP_total1 / (TP_total1 + FN_total1)
    far1 = FP_total1 / (TN_total1 + FP_total1)
    TP_total2 = np.nansum(ds.TPwc2.values, axis=2)
    FP_total2 = np.nansum(ds.FPwc2.values, axis=2)
    FN_total2 = np.nansum(ds.FNwc2.values, axis=2)
    TN_total2 = np.nansum(ds.TNwc2.values, axis=2)
    csi2 = TP_total2 / (TP_total2 + FP_total2 + FN_total2)
    pod2 = TP_total2 / (TP_total2 + FN_total2)
    far2 = FP_total2 / (TN_total2 + FP_total2)
    S_obs = ds.sum_ec_log.values   
    S_mod = ds.sum_rac_log.values
    S_obs2 = ds.sum_ec2_log.values
    S_mod2 = ds.sum_rac2_log.values
    S_obsmod = ds.sum_ec_rac_log.values
    N_total = np.nansum(N, axis=2)
    S_obs_total = np.nansum(S_obs, axis=2)
    S_mod_total = np.nansum(S_mod, axis=2)
    S_obs2_total = np.nansum(S_obs2, axis=2)
    S_mod2_total = np.nansum(S_mod2, axis=2)
    S_obsmod_total = np.nansum(S_obsmod, axis=2)
    cov_log = (N_total*S_obsmod_total - S_obs_total*S_mod_total) / N_total
    var_obs_log = (N_total*S_obs2_total - S_obs_total**2) / N_total
    var_mod_log = (N_total*S_mod2_total - S_mod_total**2) / N_total
    log_r2 = (cov_log / np.sqrt(var_obs_log * var_mod_log))**2
    log_bias = np.nansum(ds.log_bias.values * N, axis=2) / np.nansum(N, axis=2)
    log_rmse = np.sqrt(np.nansum(ds.log_se.values, axis=2) / np.nansum(N, axis=2))
    return log_bias, log_rmse, log_r2, pod1, far1, csi1, \
        pod2, far2, csi2, bias, rmse, r2, rel_bias, rel_rmse