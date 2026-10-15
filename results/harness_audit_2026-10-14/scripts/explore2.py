import scipy.io, numpy as np, sys
BASE='/home/rwu4/photoinjector-rl/XTCAV_analysis'
r=sys.argv[1]
d=scipy.io.loadmat(f'{BASE}/{r}/{r}.mat',simplify_cells=True)['data_struct']
s=d['scalars']; p=d['pulseID']
st=np.asarray(p['steps'])
A=np.asarray(s['BSA_List_S20']['TCAV_LI20_2400_A'],float); P=np.asarray(s['BSA_List_S20']['TCAV_LI20_2400_P'],float)
def c(i):
    if A[i]<5: return 'o'
    if abs(A[i]-22.04)>0.25: return 'a'
    if abs(P[i]-90)<1: return '+'
    if abs(P[i]+90)<1: return '-'
    return 'x'
for k in [1,2,5,6,17,33]:
    m=np.where(st==k)[0]; print(k, ''.join(c(i) for i in m))
cl=np.array([c(i) for i in range(len(A))])
for ch in '+-xoa': print(ch, (cl==ch).mean())
ph=P[cl=='x']; print('x phase pct',np.percentile(ph,[5,25,50,75,95]))
# per step counts
cnt=[(np.sum((st==k)&(cl=='+')),np.sum((st==k)&(cl=='-')),np.sum((st==k)&(cl=='o')),np.sum((st==k)&(cl=='x'))) for k in np.unique(st)]
cnt=np.array(cnt); print('per-step +,-,o,x median',np.median(cnt,0),'min',cnt.min(0),'max',cnt.max(0))
# number of sign blocks per step
nb=[]
for k in np.unique(st):
    seq=[x for x in cl[st==k] if x in '+-']
    nb.append(1+sum(seq[i]!=seq[i-1] for i in range(1,len(seq))))
print('sign blocks per step', np.bincount(nb))
