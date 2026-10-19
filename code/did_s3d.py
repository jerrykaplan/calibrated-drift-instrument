"""EXPLORATORY (not registered): difference-in-differences between conditions, (arm - 0%)_c1 - (arm - 0%)_c2, B6, paired by probe and
sample, stem-level bootstrap (10,000, seed 20260928)."""
import io, contextlib, json
with contextlib.redirect_stdout(io.StringIO()):
    import analyze_s3d2 as A2
A = A2.A
def did(arm, c1, c2, dim="B6"):
    out = []
    for s in "AB":
        for pid, g1 in A.R[("goodmed", c1, s)].items():
            if g1["dim"] != dim: continue
            try: x1, g2, x2 = A.R[(arm, c1, s)][pid], A.R[("goodmed", c2, s)][pid], A.R[(arm, c2, s)][pid]
            except KeyError: continue
            v = [A.al(r) for r in (g1, x1, g2, x2)]
            if None in v: continue
            out.append((g1["family"], (int(v[1]) - int(v[0])) - (int(v[3]) - int(v[2]))))
    return A.boot(out)
for arm in ("mix75", "badmed"):
    for c1, c2 in (("fg256", "n0"), ("fd256", "n0"), ("r256", "fg256"), ("r256", "fd256"), ("r256", "n0"), ("fg256", "fd256")):
        print(f"{arm:7s} ({c1} - {c2}) {A.fmt(did(arm, c1, c2))}")
