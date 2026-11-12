"""Optional new-run diagnostics; published statistical tables are independent references."""
import numpy as np
from scipy import stats
from lorenzetti_repro.metrics import assess

def five_disjoint_subsets(e_true,e_fast,alpha_pred,seed=0,n_splits=5):
    """Five disjoint test subsets for NEW-run uncertainty diagnostics.

    The historical partition indices/seeds were not supplied, so resulting
    standard deviations are not claimed to reproduce the manuscript values.
    """
    e_true=np.asarray(e_true);e_fast=np.asarray(e_fast);a=np.asarray(alpha_pred)
    if not (len(e_true)==len(e_fast)==len(a)):raise ValueError('Length mismatch')
    idx=np.random.default_rng(seed).permutation(len(a))
    subsets=np.array_split(idx,n_splits)
    out=[]
    for k,s in enumerate(subsets,1):
        row=assess(e_true[s],e_fast[s],a[s]); row['subset']=k
        out.append(row)
    return out

def paired_residual_checks(e_true,e_fast,alpha_pred):
    """New-run descriptive paired tests, NOT historic p-value replication."""
    t=np.asarray(e_true,dtype=float);f=np.asarray(e_fast,dtype=float);a=np.asarray(alpha_pred,dtype=float)
    if not (len(t)==len(f)==len(a)):raise ValueError('Length mismatch')
    er0=t-f;er1=t-f*a; d=er1-er0
    n=len(d)
    if n<4:raise ValueError('Need at least four matched events')
    scale=np.std(d,ddof=1)
    z=float(np.mean(d)/(scale/np.sqrt(n))) if scale>0 else float('nan')
    wil=stats.wilcoxon(d,zero_method='wilcox',alternative='two-sided')
    s=er1+er0
    rho=float(np.corrcoef(d,s)[0,1])
    tpm=rho*np.sqrt((n-2)/(1-rho**2)) if np.isfinite(rho) and abs(rho)<1 else np.nan
    var_ratio=float(np.var(er1,ddof=1)/np.var(er0,ddof=1))
    return {'n_paired':int(n),'mean_paired_shift_native_energy_units':float(np.mean(d)),
            'z_mean_shift':z,'z_two_sided_p':float(2*stats.norm.sf(abs(z))),
            'wilcoxon_two_sided_p':float(wil.pvalue),'variance_ratio':var_ratio,
            'pitman_morgan_t':float(tpm),'pitman_morgan_two_sided_p':float(2*stats.t.sf(abs(tpm),df=n-2))}

def directional_paired_inference(e_true,e_fast,alpha_pred,alpha_level=0.05):
    """Directional paired location and dispersion tests, thesis §5.13.

    d=(Etrue-Ecalib)-(Etrue-Efast). The one-sided location alternative
    is mean(d)<0. The variance alternative is Var(calib)<Var(fast).
    These tests are diagnostic and must be read with RMSE/IQR magnitude.
    """
    et=np.asarray(e_true,float);ef=np.asarray(e_fast,float);ap=np.asarray(alpha_pred,float)
    if not (len(et)==len(ef)==len(ap)) or len(et)<4:
        raise ValueError('At least four matched events are necessary')
    if not all(np.isfinite(x).all() for x in (et,ef,ap)):
        raise ValueError('Nonfinite values in paired observations')
    fast=et-ef; calibrated=et-ef*ap;delta=calibrated-fast
    n=len(delta);dmean=float(delta.mean());sd=float(delta.std(ddof=1))
    t_stat=dmean/(sd/np.sqrt(n)) if sd>0 else np.nan
    t_p=float(stats.t.cdf(t_stat,n-1)) if np.isfinite(t_stat) else np.nan
    z_p=float(stats.norm.cdf(t_stat)) if np.isfinite(t_stat) else np.nan
    upper_mean=dmean+stats.t.ppf(1-alpha_level,n-1)*sd/np.sqrt(n)
    w=stats.wilcoxon(delta,alternative='less',zero_method='wilcox')
    s=calibrated+fast
    corr=float(np.corrcoef(delta,s)[0,1])
    t_pm=corr*np.sqrt((n-2)/(1-corr*corr)) if abs(corr)<1 else np.nan
    p_pm=float(stats.t.cdf(t_pm,n-2)) if np.isfinite(t_pm) else np.nan
    vr=float(np.var(calibrated,ddof=1)/np.var(fast,ddof=1))
    return dict(n=int(n),mean_paired_shift_MeV=dmean,
        t_statistic=t_stat,t_one_sided_p=t_p,z_one_sided_p=z_p,
        mean_shift_upper95_MeV=float(upper_mean),
        wilcoxon_one_sided_p=float(w.pvalue),
        variance_ratio=vr,pitman_morgan_t=float(t_pm),
        pitman_morgan_one_sided_p=p_pm,
        reject_mean_H0=bool(t_p<alpha_level),
        reject_variance_H0=bool(p_pm<alpha_level))


def paired_bootstrap_variance_ratio(e_true,e_fast,alpha_pred,n_resamples=1000,seed=0):
    """Paired bootstrap 95% percentile interval for variance ratio.

    Common event indices are resampled for calibrated and FastCalo arrays.
    Resampling is batch-wise to bound memory on a ~90k-event test partition.
    The historical report may use a different seed/BCa convention; compare
    interval methods explicitly rather than silently treating them as equal.
    """
    et=np.asarray(e_true,float);ef=np.asarray(e_fast,float);ap=np.asarray(alpha_pred,float)
    if not (len(et)==len(ef)==len(ap)):raise ValueError('mismatched lengths')
    if len(et)<10:raise ValueError('At least ten events required for variance bootstrap')
    e_cal=et-ef*ap;e_std=et-ef
    rng=np.random.default_rng(seed);values=np.empty(n_resamples)
    n=len(et)
    for i in range(n_resamples):
        idx=rng.integers(0,n,size=n)
        values[i]=np.var(e_cal[idx],ddof=1)/np.var(e_std[idx],ddof=1)
    return dict(lower_95=float(np.percentile(values,2.5)),upper_95=float(np.percentile(values,97.5)),
                upper_one_sided_95=float(np.percentile(values,95)),n_resamples=n_resamples)
