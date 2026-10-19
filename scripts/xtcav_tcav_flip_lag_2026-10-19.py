"""TCAV +-V flips (2026-10-19): after the TCAV phase readback first locks within 1 deg of +-90, how many shots until the
DTOTR2 centroid reaches its settled scatter? Median |deviation| / settled robust sigma per shot (settled = 0.67)."""
import numpy as np, importlib.util
spec=importlib.util.spec_from_file_location("sc","scripts/xtcav_settling_check_2026-10-19.py"); sc=importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
# per flip: the first shot where the TCAV phase readback is within 1 deg of +-90 after having been > 10 deg away (sign change).
# For the next 25 shots: phase-readback deviation from the settled value, and the DTOTR2 x / y centroid deviation from that
# sign's settled median in this step (shots >= 15 after lock), normalized by the per-shot robust sigma of the settled shots.
K=25
rows={"phase":[], "comx":[], "comy":[], "amp":[]}
nflip=0
for run in sc.RUNS:
    f=sc.ROOT/"daq2npz/out"/f"{run}.npz"
    if not f.exists(): continue
    z=np.load(f,allow_pickle=True); names=[str(n) for n in z["states_scalar_feature_names"]]; S=z["states_scalar"]; st=z["step"]; gi=z["good_image"]
    P=S[:,names.index("TCAV_LI20_2400_P")].astype(float); A=S[:,names.index("TCAV_LI20_2400_A")].astype(float)
    X=S[:,names.index("dtotr2_com_x_mm")].astype(float); Y=S[:,names.index("dtotr2_com_y_mm")].astype(float)
    for i in range(1,len(P)-K):
        lock=abs(abs(P[i])-90)<1
        if not lock: continue
        # a flip just finished: some shot within the previous 12 had |P| far from 90 or the other sign, same step
        prev=np.arange(max(0,i-12),i)
        prev=prev[st[prev]==st[i]]
        if len(prev)==0 or not (np.abs(np.abs(P[prev])-90)>10).any(): continue
        if abs(abs(P[i-1])-90)<1: continue            # first locked shot only
        idx=np.arange(i,i+K); idx=idx[st[idx]==st[i]]
        sgn=np.sign(P[i]); same=idx[np.sign(P[idx])==sgn]
        ref=same[(same-i)>=15]
        ref=ref[gi[ref]]
        if len(ref)<5: continue
        nflip+=1
        for key,v in (("phase",P),("amp",A),("comx",X),("comy",Y)):
            m=np.full(K,np.nan); r0=np.nanmedian(v[ref]); s=sc.rsig(v[ref]-r0)
            if not np.isfinite(s) or s==0: continue
            ok=same if key in ("phase","amp") else same[gi[same]]
            m[ok-i]=np.abs(v[ok]-r0)/s
            rows[key].append(m)
print("flips analysed:",nflip)
for key in rows:
    R=np.array(rows[key]); med=np.nanmedian(R,0)
    print(f"{key:6s} median |dev|/sigma by shot after phase lock: "+" ".join(f"{x:.1f}" for x in med[:16]))
