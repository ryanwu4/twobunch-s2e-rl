"""Per-image raw statistics (max pixel, count of pixels at max, total counts) for DTOTR2 (read-only)."""
import h5py, numpy as np, sys, glob, json, os
BASE='/home/rwu4/photoinjector-rl/XTCAV_analysis'
OUT='/home/rwu4/photoinjector-rl/twobunch-s2e-rl/results/harness_audit_2026-10-14/data'
r=sys.argv[1]
rows=[]
for fn in sorted(glob.glob(f'{BASE}/{r}/images/DTOTR2/*_step*.h5')):
    step=int(fn.split('step')[-1][:2])
    with h5py.File(fn,'r') as f:
        D=f['entry/data/data']
        ts=f['entry/instrument/NDAttributes/NDArrayEpicsTSSec'][:]+1e-9*f['entry/instrument/NDAttributes/NDArrayEpicsTSnSec'][:]
        uid=f['entry/instrument/NDAttributes/NDArrayUniqueId'][:]
        for i in range(D.shape[0]):
            a=D[i]
            mx=int(a.max()); rows.append((step,i,float(ts[i]),int(uid[i]),mx,int((a>=mx).sum()),int((a>=4000).sum()),float(a.sum()),float(np.median(a))))
rows=np.array(rows)
np.save(f'{OUT}/{r}_dtotr2_imgstats.npy',rows)
mx=rows[:,4]
print(r,'n img',len(rows),'max pixel pct',np.percentile(mx,[1,50,99,100]),'n imgs with max>=4095',(mx>=4095).sum(),'>=65535',(mx>=65535).sum(), 'median bkg',np.median(rows[:,8]))
