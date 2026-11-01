import scipy.io, numpy as np, sys
BASE='/home/rwu4/photoinjector-rl/XTCAV_analysis'
r=sys.argv[1]
d=scipy.io.loadmat(f'{BASE}/{r}/{r}.mat',simplify_cells=True)['data_struct']
s=d['scalars']; p=d['pulseID']
t=np.asarray(p['SLAC_time'],float); st=np.asarray(p['steps']); pid=np.asarray(p['scalar_PID'])
print(r, 'N',len(t),'dt med',np.median(np.diff(t)), 'pid diff uniq',np.unique(np.diff(pid))[:10], np.unique(np.diff(pid),return_counts=True)[1][:10])
print('monotone t', np.all(np.diff(t)>=0))
for k in np.unique(st):
    m=st==k
    if k<4 or k>30:
        print(' step',k,'n',m.sum(),'dur %.1f s'%(t[m].max()-t[m].min()))
gaps=[t[st==k+1].min()-t[st==k].max() for k in np.unique(st)[:-1]]
durs=[t[st==k].max()-t[st==k].min() for k in np.unique(st)]
print('inter-step gap s: med %.2f min %.2f max %.2f'%(np.median(gaps),np.min(gaps),np.max(gaps)))
print('step durations s: med %.2f min %.2f max %.2f'%(np.median(durs),np.min(durs),np.max(durs)))
print('total span min', (t.max()-t.min())/60)
A=np.asarray(s['BSA_List_S20']['TCAV_LI20_2400_A'],float); P=np.asarray(s['BSA_List_S20']['TCAV_LI20_2400_P'],float)
on=A>10
print('TCAV on frac',on.mean(),'plus',(on&(abs(P-90)<1)).mean(),'neg',(on&(abs(P+90)<1)).mean(), 'off(A<5)',(A<5).mean(), 'A on med',np.median(A[on]))
# pattern in step 5
m=np.where(st==5)[0]
print('pattern step5:', ''.join('+' if (A[i]>10 and P[i]>0) else ('-' if A[i]>10 else 'o') for i in m))
print('pattern step6:', ''.join('+' if (A[i]>10 and P[i]>0) else ('-' if A[i]>10 else 'o') for i in np.where(st==6)[0]))
print('phase of on shots: quantiles', np.percentile(P[on],[1,5,25,50,75,95,99]))
