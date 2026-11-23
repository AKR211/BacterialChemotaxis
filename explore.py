import os
import sys
import time
import numpy as np

from chemotactic_abp import (Params, Simulation, structure_factor, pair_correlation, cluster_sizes)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)


def run_case(params: Params, T, T_burn, sample_every=25, label="", species=None, verbose=True, snapshot_every=None):
    sim = Simulation(params)
    N, L = params.N, params.L
    ts, ts_snap = [], []
    P, A, PA, PB, Lc, S1, gmax = [], [], [], [], [], [], []
    rs, thetas, fields = [], [], []

    def obs(s):
        ts.append(s.t)
        if snapshot_every is not None and len(ts) % snapshot_every == 0:
            ts_snap.append(s.t)
            rs.append(s.wrapped().copy())
            thetas.append(s.theta.copy())
            fields.append(s.field_grid(100))
        P.append(s.polarization())
        A.append(s.chemotactic_alignment())
        if species is not None:
            PA.append(s.polarization(species == 0))
            PB.append(s.polarization(species == 1))
        cs = cluster_sizes(s.r, L, rc=0.5)
        Lc.append(cs[0] / N)
        k, S = structure_factor(s.r, L, mmax=8)
        S1.append(S[:3].mean())
        gmax.append(np.linalg.norm(s.grad, axis=1).mean())

    t0 = time.time()
    sim.run(T, sample_every=sample_every, observer=obs)
    if verbose:
        print(f"  {label}: {time.time()-t0:5.0f}s  "
              f"<P>={np.mean(P[len(P)//2:]):.3f}  <A_ch>={np.nanmean(A[len(A)//2:]):.3f}  "
              f"<largest cluster>={np.mean(Lc[len(Lc)//2:]):.3f}  "
              f"<S(k_low)>={np.mean(S1[len(S1)//2:]):.2f}", flush=True)
    res = dict(t=np.array(ts), P=np.array(P), A=np.array(A), Lc=np.array(Lc),
               S1=np.array(S1), gradmean=np.array(gmax),
               r=sim.wrapped(), theta=sim.theta, T_burn=T_burn, L=L, N=N)
    if species is not None:
        res.update(PA=np.array(PA), PB=np.array(PB), species=species)
    if snapshot_every is not None:
        res.update(t_snap=np.array(ts_snap), r_snap=np.array(rs),
                   theta_snap=np.array(thetas), field_snap=np.array(fields))
    res["field"] = sim.field_grid(100)
    return res, sim


def steady_mean(res, key):
    t = res["t"]
    sel = t >= res["T_burn"]
    return np.nanmean(res[key][sel])


def scan_g(gs=(0, 5, 15, 40, 100), T=80, T_burn=40, N=200, L=20.0, Pe=5.0, Dr=0.5):
    out = {}
    for g in gs:
        p = Params(N=N, L=L, Pe=Pe, g=float(g), Dr=Dr, a=0.25, dt=0.01,
                   dtype="complex64", seed=7)
        out[g], _ = run_case(p, T, T_burn, label=f"g={g:g}")
    np.savez_compressed(os.path.join(OUT, "scan_g.npz"),
                        gs=np.array(gs), **{f"{k}_{g}": v for g in gs
                                            for k, v in out[g].items()})
    return out


def compare(g=15.0, T=60, T_burn=30, N=200, L=20.0, Pe=5.0, Dr=0.5, seed=7):
    cases = {
        "exclude (model)":       dict(memory="dynamic", self_mode="exclude"),
        "include self-signal":   dict(memory="dynamic", self_mode="include"),
        "quasi-static":          dict(memory="quasistatic", self_mode="exclude"),
    }
    out = {}
    for name, kw in cases.items():
        p = Params(N=N, L=L, Pe=Pe, g=g, Dr=Dr, a=0.25, dt=0.01, dtype="complex64",
                   seed=seed, **kw)
        out[name], _ = run_case(p, T, T_burn, label=name)
    np.savez_compressed(os.path.join(OUT, "compare.npz"),
                        names=np.array(list(cases)),
                        **{f"{k}__{n}": v for n in cases for k, v in out[n].items()})
    return out


def dt_check(g=100.0, T=30, N=200, L=20.0, Pe=5.0, Dr=0.5, seed=7):
    for dt in (0.01, 0.005, 0.0025):
        p = Params(N=N, L=L, Pe=Pe, g=g, Dr=Dr, a=0.25, dt=dt, dtype="complex64", seed=seed)
        res, sim = run_case(p, T, T / 2, sample_every=int(round(0.25 / dt)), label=f"dt={dt}")
        cs = cluster_sizes(sim.r, L, 0.5)
        print(f"      clusters (size>=5) at t={T}: {int((cs>=5).sum())},"
              f" sizes {cs[cs>=5].tolist()}", flush=True)


def pair_corr_cross(r1, r2, L, nbins=25, rmax=5.0, same=False):
    d = r1[:, None, :] - r2[None, :, :]
    d -= L * np.rint(d / L)
    dist = np.sqrt((d ** 2).sum(-1))
    if same:
        dist = dist[np.triu_indices(len(r1), 1)]
        norm = 0.5 * len(r1) * (len(r1) / L ** 2)
    else:
        dist = dist.ravel()
        norm = len(r1) * len(r2) / L ** 2
    h, edges = np.histogram(dist, bins=nbins, range=(0, rmax))
    shell = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    return 0.5 * (edges[1:] + edges[:-1]), h / (norm * shell)


def two_species(T=80, T_burn=40, N=200, L=20.0, Pe=5.0, Dr=0.5, seed=7):
    nA = N // 2
    species = np.r_[np.zeros(nA, int), np.ones(N - nA, int)]
    cases = {
        # strong-secretor A, strong-responder B
            dict(gA=3.0, sA=1.0, gB=40.0, sB=0.2),
        # pursuit-evasion (B chases A):
            dict(gA=-15.0, sA=1.0, gB=15.0, sB=1.0),
    }
    out = {}
    for name, c in cases.items():
        g = np.where(species == 0, c["gA"], c["gB"])
        sec = np.where(species == 0, c["sA"], c["sB"])
        GAB, GBA = c["gA"] * c["sB"], c["gB"] * c["sA"]
        p = Params(N=N, L=L, Pe=Pe, g=g, secretion=sec, Dr=Dr, a=0.25, dt=0.01,
                   dtype="complex64", seed=seed)
        res, sim = run_case(p, T, T_burn, species=species, label=name, verbose=False)
        sel = res["t"] >= T_burn
        rA, rB = res["r"][species == 0], res["r"][species == 1]
        _, gAA = pair_corr_cross(rA, rA, L, same=True)
        _, gBB = pair_corr_cross(rB, rB, L, same=True)
        rr, gAB = pair_corr_cross(rA, rB, L)
        print(f"  {name}\n    G_A<-B = {GAB:+.1f}, G_B<-A = {GBA:+.1f}\n"
              f"    <P_A>={res['PA'][sel].mean():.3f} <P_B>={res['PB'][sel].mean():.3f} "
              f"<P>={res['P'][sel].mean():.3f}  (noise level {1/np.sqrt(N/2):.3f} per species)\n"
              f"    largest cluster = {res['Lc'][sel].mean():.2f},  S(k_low) = {res['S1'][sel].mean():.1f}\n"
              f"    g_AA(r<0.5)={gAA[:2].mean():.1f}  g_BB(r<0.5)={gBB[:2].mean():.1f}  "
              f"g_AB(r<0.5)={gAB[:2].mean():.1f}", flush=True)
        out[name] = res
    np.savez_compressed(os.path.join(OUT, "two_species.npz"), names=np.array(list(cases)),**{f"{k}__{n}": v for n in cases for k, v in out[n].items()})
    return out

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "scan_g"
    if cmd == "scan_g":
        scan_g()
    if cmd == "compare":
        compare()
    if cmd == "dt_check":
        dt_check()
    if cmd == "two_species":
        two_species()
    if cmd == "run":
        g = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
        res, sim = run_case(Params(N=200, L=20.0, Pe=5.0, g=g, Dr=0.5, a=0.25, dt=0.01, dtype="complex64", seed=7),T=60, T_burn=30, label=f"g={g:g}", snapshot_every=1, sample_every=5)
        np.savez_compressed(os.path.join(OUT, f"run_g{g:g}.npz"), **{f"{k}": v for k, v in res.items()})
    


# plotting
def plot_scan(npz_path, out_png, key_label="g"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(npz_path)
    gs = d["gs"]
    n = len(gs)
    fig = plt.figure(figsize=(3.0 * n, 8.2))
    gsp = fig.add_gridspec(3, n, height_ratios=[1.15, 1.15, 0.9])
    for j, g in enumerate(gs):
        r, th, L = d[f"r_{g}"], d[f"theta_{g}"], float(d[f"L_{g}"])
        ax = fig.add_subplot(gsp[0, j])
        ax.imshow(d[f"field_{g}"], origin="lower", extent=[0, L, 0, L], cmap="magma",
                  vmax=np.percentile(d[f"field_{g}"], 99.5))
        ax.quiver(r[:, 0], r[:, 1], np.cos(th), np.sin(th), color="white", scale=45,
                  width=0.004, alpha=0.9)
        ax.set_title(f"{key_label} = {g:g}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    ax1 = fig.add_subplot(gsp[1, :n // 2 + 1]); ax2 = fig.add_subplot(gsp[1, n // 2 + 1:])
    for g in gs:
        ax1.plot(d[f"t_{g}"], d[f"Lc_{g}"], label=f"{key_label}={g:g}")
        ax2.plot(d[f"t_{g}"], d[f"P_{g}"])
    ax1.set_ylabel("largest cluster fraction"); ax2.set_ylabel("polarization P")
    for ax in (ax1, ax2):
        ax.set_xlabel("t  (units of 1/mu)")
    ax1.legend(fontsize=8, ncol=2)
    ax2.axhline(1 / np.sqrt(float(d[f"N_{gs[0]}"])), color="k", ls=":", lw=1)
    ax2.text(0.02, 0.9, "dotted: 1/sqrt(N)", transform=ax2.transAxes, fontsize=8)
    ax3 = fig.add_subplot(gsp[2, :])
    x = np.arange(n)
    def sm(g, k):
        sel = d[f"t_{g}"] >= float(d[f"T_burn_{g}"]); return np.nanmean(d[f"{k}_{g}"][sel])
    ax3.bar(x - 0.27, [sm(g, "P") for g in gs], 0.27, label="P")
    ax3.bar(x, [sm(g, "A") for g in gs], 0.27, label="A_ch")
    ax3.bar(x + 0.27, [sm(g, "Lc") for g in gs], 0.27, label="largest cluster")
    ax3.set_xticks(x); ax3.set_xticklabels([f"{g:g}" for g in gs]); ax3.set_xlabel(key_label)
    ax3.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    print("wrote", out_png)


def plot_case(npz_path, out_png, fps=30, snapshot_every_obs=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import PIL

    with np.load(npz_path) as d:
        rs = np.asarray(d["r_snap"])
        thetas = np.asarray(d["theta_snap"])
        fields = np.asarray(d["field_snap"])
        L = float(d["L"])
        ts = np.asarray(d["t"])
        print(f"Loaded {len(rs)} snapshots from {npz_path}.")
        if "t_snap" in d.files:
            ts_snap = np.asarray(d["t_snap"])
        else:
            idx = np.arange(snapshot_every_obs - 1, len(ts), snapshot_every_obs)
            if len(idx) < len(rs):
                raise ValueError()
            ts_snap = ts[idx[:len(rs)]]

    if len(rs) == 0:
        raise ValueError("No snapshots found in the .npz file.")
    if not (len(rs) == len(thetas) == len(fields) == len(ts_snap)):
        raise ValueError(
            "Snapshot arrays have inconsistent lengths: "
            f"r={len(rs)}, theta={len(thetas)}, field={len(fields)}, "
            f"time={len(ts_snap)}."
        )

    finite_fields = fields[np.isfinite(fields)]
    if finite_fields.size == 0:
        raise ValueError("field_snap contains no finite values.")
    vmax = float(np.percentile(finite_fields, 99.5))
    if vmax <= 0:
        vmax = float(np.max(finite_fields)) if np.max(finite_fields) > 0 else 1.0

    files = []

    for i, (r, theta, field, t) in enumerate(zip(rs, thetas, fields, ts_snap)):
        r = np.mod(r, L)

        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
        ax.imshow(
            field,
            origin="lower",
            extent=(0, L, 0, L),
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
            interpolation="bilinear",
            aspect="equal",
        )
        ax.quiver(
            r[:, 0], r[:, 1], np.cos(theta), np.sin(theta),
            color="white",
            angles="xy",
            scale_units="xy",
            scale=2.0,
            width=0.0028,
            headwidth=3.5,
            headlength=4.5,
            alpha=0.9,
        )
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"t = {t:.2f}")
        ax.set_xticks([])
        ax.set_yticks([])

        filename = out_png.replace(".png", f"_{i}.png").replace("results/", "")
        files.append(f"./temp/{filename}")
        plt.savefig(f'./temp/{filename}', dpi=110)
        plt.clf()
        plt.close(fig)

    imgs = [PIL.Image.open(file) for file in files]
    imgs[0].save(fp=f"./{out_png.replace('.png', '.gif')}", format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps))




if __name__ == "__main__" and len(sys.argv) > 2:
    if sys.argv[1] == "plot":
        plot_scan(sys.argv[2], sys.argv[2].replace(".npz", ".png"))
    if sys.argv[1] == "plot_case":
        plot_case(sys.argv[2], sys.argv[2].replace(".npz", ".png"), fps=int(sys.argv[3]) if len(sys.argv) > 3 else 30)