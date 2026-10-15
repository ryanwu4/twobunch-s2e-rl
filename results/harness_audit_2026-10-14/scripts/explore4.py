import scipy.io, numpy as np, sys
BASE='/home/rwu4/photoinjector-rl/XTCAV_analysis'
r=sys.argv[1]
d=scipy.io.loadmat(f'{BASE}/{r}/{r}.mat',simplify_cells=True)['data_struct']
s=d['scalars']; st=np.asarray(d['pulseID']['steps'])
def g(L,pv): return np.asarray(s[L][pv],float).ravel()
for L,pv in [('nonBSA_List_LINAC_SBST','LI12_SBST_1_PHAS'),('nonBSA_List_LINAC_SBST','LI14_SBST_1_PHAS'),('nonBSA_List_LINAC_SBST','LI12_SBST_1_PDES'),('nonBSA_List_S10RF','KLYS_LI10_31_PHAS')]:
    a=g(L,pv)
    for k in [1,2,3]:
        m=st==k; x=a[m]; ch=np.r_[0,np.where(np.diff(x)!=0)[0]+1]
        print(pv,'step',k,'n',m.sum(),'n changes',len(ch)-1,'change idx',ch[:20], 'vals',np.round(x[ch][:12],3))
# BSA rms>0 ranking
rows=[]
for L,v in s.items():
    if not isinstance(v,dict) or not L.startswith('BSA'): continue
    for pv,a in v.items():
        a=np.asarray(a,float).ravel(); a=np.where(np.abs(a)>9000,np.nan,a)
        med=np.array([np.nanmedian(a[st==k]) for k in np.unique(st)])
        rms=np.nanmedian([np.nanstd(a[st==k]) for k in np.unique(st)])
        if not (rms>0): continue
        rows.append(((np.nanmax(med)-np.nanmin(med))/rms,L,pv,med[0],med[16],med[-1],rms))
rows.sort(key=lambda x:-x[0])
for x in rows[:45]: print('%8.1f %-16s %-30s %12.4g %12.4g %12.4g  shot-rms %.4g'%x)
