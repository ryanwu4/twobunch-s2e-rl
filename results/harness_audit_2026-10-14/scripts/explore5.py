import scipy.io, numpy as np, sys
BASE='/home/rwu4/photoinjector-rl/XTCAV_analysis'
r=sys.argv[1]
d=scipy.io.loadmat(f'{BASE}/{r}/{r}.mat',simplify_cells=True)['data_struct']
s=d['scalars']; st=np.asarray(d['pulseID']['steps'])
print(r, d['params']['comment'], 'Waits',d['params']['Waits'],'Tol',d['params']['Tolerance'], 'nshot',d['params']['n_shot'])
for L,pv in [('BSA_List_S11RF','KLYS_LI11_11_FB_FAST_PACT'),('BSA_List_S11RF','KLYS_LI11_21_FB_FAST_PACT'),('BSA_List_S11RF','KLYS_LI11_11_FB_FAST_AACT'),('BSA_List_S11','BLEN_LI11_359_BRAW'),('BSA_List_S14','BLEN_LI14_888_BRAW'),('BSA_List_S20','BLEN_LI20_3014_BRAW'),('BSA_List_S11','BPMS_LI11_333_X'),('BSA_List_S14','BPMS_LI14_801_X'),('BSA_List_S10','BPMS_IN10_731_X'),('BSA_List_S20','BPMS_LI20_2445_X'),('BSA_List_S20','TORO_LI20_2452_TMIT'),('BSA_List_S10','TORO_IN10_591_TMIT'),('BSA_List_S10RF','KLYS_LI10_41_FB_FAST_PACT')]:
    if pv not in s.get(L,{}): print('missing',pv); continue
    a=np.asarray(s[L][pv],float); a=np.where(np.abs(a)>9000,np.nan,a)
    med=np.array([np.nanmedian(a[st==k]) for k in np.unique(st)]); sd=np.array([np.nanstd(a[st==k]) for k in np.unique(st)])
    print('%-28s med steps: '%pv+' '.join('%.4g'%x for x in med[::4]), '| shot std med %.3g'%np.nanmedian(sd), 'nan frac %.2f'%np.isnan(a).mean())
