"""Null test for xtcav_settling_check_2026-10-19.py: the same early/late response at FAKE mid-step boundaries of the E300 runs
(pre = shots 20-59, boundary at shot 60) vs the real step boundaries."""
import numpy as np, importlib.util
spec=importlib.util.spec_from_file_location("sc","scripts/xtcav_settling_check_2026-10-19.py"); sc=importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
def run(x, st, mask, fake):
    out=[]; steps=np.unique(st)
    for s0,s1 in zip(steps[:-1],steps[1:]):
        A=np.flatnonzero(st==s0); B=np.flatnonzero(st==s1)
        if fake: A,B,k0 = B[(B-B[0]>=20)&(B-B[0]<60)], B, 60     # pre = shots 20-59 of the same step, "boundary" at shot 60
        else: k0=0
        kB=B-B[0]
        if mask is not None: A=A[mask[A]]; okB=mask[B]
        else: okB=np.ones(len(B),bool)
        test=B[(kB>=k0)&(kB<k0+30)&okB]; late=B[(kB>=k0+30)&(kB<k0+90)&okB]
        if len(A)<8 or len(late)<8 or len(test)<5: continue
        pre=np.nanmedian(x[A[-40:]]); post=np.nanmedian(x[late]); sig=sc.rsig(np.r_[x[A[-40:]]-pre, x[late]-post])
        if not np.isfinite(sig) or abs(post-pre) < 6*sig/np.sqrt(len(late)): continue
        r=(x[test]-pre)/(post-pre); k=kB[np.isin(B,test)]-k0
        out.append((np.nanmedian(r[k<5]) , np.nanmedian(r[k>=20])))
    return np.array(out)
for c in ["BLEN_LI14_888_BRAW","BPMS_LI14_801_X","BPMS_LI20_2445_X","BPMS_LI20_3156_X","KLYS_LI11_11_FB_FAST_PACT","KLYS_LI11_21_FB_FAST_PACT","dtotr2_com_x_mm|1","dtotr2_com_x_mm|-1","dtotr2_com_y_mm|1","dtotr2_com_y_mm|-1"]:
    res={}
    for fake in (False,True):
        R=[]
        for runn in ("E300_15671","E300_15673"):
            z=np.load(sc.ROOT/"daq2npz/out"/f"{runn}.npz",allow_pickle=True); names=[str(n) for n in z["states_scalar_feature_names"]]
            S,st,tc,gi=z["states_scalar"],z["step"],z["tcav_class"],z["good_image"]
            if "|" in c: a,cl=c.split("|"); x=S[:,names.index(a)].astype(float); m=(tc==int(cl))&gi
            else: x=S[:,names.index(c)].astype(float); m=None
            r=run(x,st,m,fake); 
            if len(r): R.append(r)
        R=np.concatenate(R) if R else np.zeros((0,2))
        res[fake]=R
    f=lambda R: f"n={len(R):3d} first0.5s {np.nanmedian(R[:,0]):.2f} 2-3s {np.nanmedian(R[:,1]):.2f}" if len(R) else "n=0"
    print(f"{c:30s} REAL: {f(res[False])} | FAKE(mid-step): {f(res[True])}")
