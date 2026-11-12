"""Evaluation metrics consistent with the thesis definitions (5.8–5.13)."""
import numpy as np

def iqr(x):
    return float(np.percentile(x,75)-np.percentile(x,25))

def assess(Etrue,Efast,alpha_pred):
    et=np.asarray(Etrue,dtype=float).reshape(-1)
    ef=np.asarray(Efast,dtype=float).reshape(-1)
    ap=np.asarray(alpha_pred,dtype=float).reshape(-1)
    if not (len(et)==len(ef)==len(ap)):raise ValueError('length mismatch')
    if not (np.isfinite(et).all() and np.isfinite(ef).all() and np.isfinite(ap).all()):raise ValueError('non-finite')
    ec=ef*ap
    residuals=et-ec
    fast_res=et-ef
    rmse=float(np.sqrt(np.mean(residuals**2)))
    baseline_rmse=float(np.sqrt(np.mean(fast_res**2)))
    qi=iqr(residuals);bf=iqr(fast_res)
    return {'n_test':len(et),'rmse':rmse,'iqr':qi,'d0':float(np.hypot(rmse,qi)),
            'baseline_rmse':baseline_rmse,'baseline_iqr':bf,'baseline_d0':float(np.hypot(baseline_rmse,bf)),
            'rmse_reduction_pct':100*(1-rmse/baseline_rmse),'iqr_reduction_pct':100*(1-qi/bf),
            'pair_mean_delta':float(np.mean(residuals-fast_res)),
            'variance_ratio':float(np.var(residuals,ddof=1)/np.var(fast_res,ddof=1))}

def energy_resolution(Etrue,Ecalib,bins):
    """Sample standard deviation of r=Ecalib/Etrue within each truth-energy interval."""
    a=np.asarray(Etrue,float);b=np.asarray(Ecalib,float);bins=np.asarray(bins,float)
    if np.any(a<=0):raise ValueError('Etrue must be positive for resolution')
    which=np.digitize(a,bins,right=False)-1
    out=[]
    for k,(lo,hi) in enumerate(zip(bins[:-1],bins[1:])):
        r=b[which==k]/a[which==k]
        out.append({'low':float(lo),'high':float(hi),'n':int(len(r)),
                    'sigma_E_over_E':float(np.std(r,ddof=1)) if len(r)>1 else None})
    return out

def ear_curve(Etrue,Emethod,delta_max=None,points=1000):
    """EAR_m(Delta lambda)=fraction(|Etrue-Em|<Delta lambda). NOT online trigger efficiency."""
    et=np.asarray(Etrue,dtype=float);em=np.asarray(Emethod,dtype=float)
    e=np.abs(et-em)
    if delta_max is None:delta_max=6*float(np.std(et-em))
    grid=np.linspace(0,float(delta_max),int(points))
    # Exact empirical strict inequality using a sorted residual array: O(N log N).
    ordered=np.sort(e)
    return grid,np.searchsorted(ordered,grid,side='left')/len(ordered)

def shared_ear(Etrue,Ecalib,Efast,points=1000):
    """Common scan range: 6*max(sigma_e_calib,sigma_e_fast); empirical ddof=1."""
    a=np.asarray(Etrue,float);b=np.asarray(Ecalib,float);c=np.asarray(Efast,float)
    maximum=6*max(np.std(a-b,ddof=1),np.std(a-c,ddof=1))
    x,y=ear_curve(a,b,maximum,points)
    _,z=ear_curve(a,c,maximum,points)
    return x,y,z

def delta_lambda_90(x,ear,q=0.90):
    hit=np.flatnonzero(np.asarray(ear)>=q)
    return float(x[hit[0]]) if len(hit) else None
