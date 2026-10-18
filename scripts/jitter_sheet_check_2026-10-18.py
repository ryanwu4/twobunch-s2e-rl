"""Per-shot spacing jitter implied by the Yocky / Zack jitter sheet (lab-notebook refs, 2024 run / 2025), propagated through
the B2-cov surrogate gradients (frozen base sample) at a grid of L1 / L2 operating points; compare with the measured
per-image spacing scatter of the real XTCAV scans (online replay: robust sigma 5.7 / 15.0 / 13.3 um)."""
import torch, numpy as np, json
from twobunch_s2e_rl.surrogate.model import TwoBunchFlow
from twobunch_s2e_rl.datagen.sweep_params import resolve_sweep_set, SECTION_GAIN_EV
torch.manual_seed(0)
m=TwoBunchFlow.load_from_checkpoint("trained/xtcav23e_inj_covB2_2026-10-17/checkpoints/best-epoch=034-val_paired_delta_rms_um=5.4293.ckpt",map_location="cpu").eval()
keys=list(m.hparams.knob_keys); lo=torch.tensor(m.knob_low.numpy()); hi=torch.tensor(m.knob_high.numpy())
_,_,_,base=resolve_sweep_set("xtcav23e_l0b2_inj12")
b=torch.tensor([float(base[k]) for k in keys])
gf=float(base["GUNF:rf_field_scale"]); q=float(base["distgen:total_charge:value"])
# per-shot rms jitter in physical knob units: (2024 run, 2025 where measured) -- Yocky / Zack sheet
J24={"L1PhaseSet":0.43,"L1EnergyOffset":0.0065*SECTION_GAIN_EV["L1EnergyOffset"],"L2PhaseSet":0.4,
     "L2EnergyOffset":0.003*SECTION_GAIN_EV["L2EnergyOffset"],"L3EnergyOffset":0.003*SECTION_GAIN_EV["L3EnergyOffset"],
     "L0BPhaseSet":0.1,"L0BEnergyOffset":0.005*SECTION_GAIN_EV["L0BEnergyOffset"],"laser_timing_ps":0.05,
     "GUNF:rf_field_scale":0.0025*gf,"GUNF:theta0_deg":0.15,"distgen:total_charge:value":0.023*q,"L0AF_phase_offset_deg":0.1}
J25=dict(J24, **{"L1EnergyOffset":0.0025*SECTION_GAIN_EV["L1EnergyOffset"],"GUNF:rf_field_scale":0.001*gf,
                 "GUNF:theta0_deg":0.5,"distgen:total_charge:value":0.0274*q})
zd=torch.randn(4096,6); zw=torch.randn(4096,6)
def spacing(phys):
    kn=(phys-lo)/(hi-lo)
    o=m.observables(kn.unsqueeze(0), n=4096, z_drive=zd, z_witness=zw)
    return o["bunch_spacing"][0], o["drive_sigma_z"][0]
res=[]
for L1 in (-24.0,-22.0,-20.3):
    for L2 in (-40.5,-38.5,-36.5,-34.5):
        p=b.clone(); p[keys.index("L1PhaseSet")]=L1; p[keys.index("L2PhaseSet")]=L2
        p.requires_grad_(True)
        s,sz=spacing(p); g=torch.autograd.grad(s,p)[0]
        row={"L1":L1,"L2":L2,"spacing_um":float(s)*1e6}
        for tag,J in (("2024",J24),("2025",J25)):
            c={k:abs(float(g[keys.index(k)]))*J[k]*1e6 for k in J}
            row[f"sigma_{tag}_um"]=float(np.sqrt(sum(v*v for v in c.values())))
            row[f"top_{tag}"]=sorted(c.items(), key=lambda kv:-kv[1])[:4]
        res.append(row)
for r in res:
    print(f"L1 {r['L1']:6.1f} L2 {r['L2']:6.1f}: spacing {r['spacing_um']:6.1f} um | per-shot sigma 2024 {r['sigma_2024_um']:5.1f} um, 2025 {r['sigma_2025_um']:5.1f} um | top 2024: " + ", ".join(f"{k.split(':')[-1][:14]} {v:.1f}" for k,v in r['top_2024']))
json.dump(res, open("results/jitter_sheet_check_2026-10-18/res.json","w"), indent=1, default=float)
