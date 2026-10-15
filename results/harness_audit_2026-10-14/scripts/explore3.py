import scipy.io, numpy as np, sys
BASE='/home/rwu4/photoinjector-rl/XTCAV_analysis'
r=sys.argv[1]
d=scipy.io.loadmat(f'{BASE}/{r}/{r}.mat',simplify_cells=True)['data_struct']
s=d['scalars']; st=np.asarray(d['pulseID']['steps'])
rows=[]
for L,v in s.items():
    if not isinstance(v,dict) or L.startswith('nonBSA_List_LaserS20') : continue
    for pv,a in v.items():
        a=np.asarray(a,float).ravel()
        if len(a)!=len(st) : continue
        a=np.where(np.abs(a)>9000,np.nan,a)
        med=np.array([np.nanmedian(a[st==k]) for k in np.unique(st)])
        rms=np.nanmedian([np.nanstd(a[st==k]) for k in np.unique(st)])
        if not np.isfinite(rms) or np.all(~np.isfinite(med)): continue
        span=np.nanmax(med)-np.nanmin(med)
        rows.append((span/(rms+1e-12),L,pv,med[0],med[len(med)//2],med[-1],rms))
rows.sort(key=lambda x:-x[0])
for x in rows[:70]: print('%8.1f %-24s %-32s %12.4g %12.4g %12.4g  shot-rms %.4g'%x)
