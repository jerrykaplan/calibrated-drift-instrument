import torch, json
T=["honesty","sycophancy","harmlessness","power_seeking","helpfulness","confidence","corrigibility"]
rows=json.load(open("internal/ic_analysis_gate.json"))
for L in ("16","12","20"):
    d=torch.load(f"internal/directions_L{L}.pt"); D=torch.stack([d[t].float() for t in T])
    G=D@D.T
    print(f"\n=== Layer {L}: cosine between trait directions")
    print(" "*14+"".join(f"{t[:9]:>10s}" for t in T))
    for i,t in enumerate(T): print(f"{t[:13]:14s}"+"".join(f"{G[i,j].item():+10.2f}" for j in range(7)))
    ev,V=torch.linalg.eigh(G); ev=ev.flip(0); V=V.flip(1)
    print("share of total variance of the 7 directions on each principal axis:", " ".join(f"{e/ev.sum():.2f}" for e in ev))
    u=V[:,0]*(1 if V[3,0]>0 else -1)
    print("loadings on axis 1 (sign: power_seeking +):", " ".join(f"{t[:6]}={u[i].item():+.2f}" for i,t in enumerate(T)))
    for pair in ("goodmed:base","badmed:goodmed"):
        p=torch.tensor([next(r["diff_raw"] for r in rows if r["layer"]==L and r["pair"]==pair and r["trait"]==t) for t in T])
        c=torch.linalg.solve(G,p); s=c@D
        a1=(u@D); a1=a1/a1.norm(); along=(s@a1).item()
        norm=next(r["norm"] for r in rows if r["layer"]==L)
        print(f"{pair:15s} min-norm shift length {s.norm().item():.2f} (hidden-state length {norm:.1f}); fraction of squared length on axis 1: {along**2/s.norm().item()**2:.2f}; sign on axis 1: {along:+.2f}")
