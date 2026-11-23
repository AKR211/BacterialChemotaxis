"""
checks for chemotactic_abp

  1. Baseline ABPs (g = 0): orientation decorrelation, MSD, P ~ N^-1/2, S(k)=g(r)=1
  2. Field solver: stationary & moving source vs analytic Green-function integrals
  3. Self-exclusion: isolated particle feels zero torque; two-body gradient vs
     analytic; the 'include' control visibly fails the same tests
  4. Convergence in dt and in the spectral cutoff (deterministic trajectories)
  5. Dynamic field to quasi-static field in the stationary limit

"""
import sys
import numpy as np
from scipy.integrate import quad

from chemotactic_abp import (Params, Simulation, structure_factor,
                             pair_correlation)

results = []
want = lambda n: len(sys.argv) == 1 or str(n) in sys.argv[1:]


def check(name, value, tol, note=""):
    ok = bool(abs(value) <= tol)
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<58s} {value: .3e} (tol {tol:.0e}) {note}")

def c_steady(r, a, s=1.0):
    f = lambda tau: np.exp(-tau) * np.exp(-r ** 2 / (2 * (2 * tau + a ** 2))) \
        / (2 * np.pi * (2 * tau + a ** 2))
    return s * quad(f, 0, np.inf, epsabs=1e-13, epsrel=1e-12)[0]


def dc_steady(r, a, s=1.0):
    f = lambda tau: np.exp(-tau) * np.exp(-r ** 2 / (2 * (2 * tau + a ** 2))) \
        / (2 * np.pi * (2 * tau + a ** 2)) * (-r / (2 * tau + a ** 2))
    return s * quad(f, 0, np.inf, epsabs=1e-13, epsrel=1e-12)[0]


def c_transient_centre(t, a):
    f = lambda tau: np.exp(-tau) / (2 * np.pi * (2 * tau + a ** 2))
    return quad(f, 0, t, epsabs=1e-13, epsrel=1e-12)[0]


def c_line_source(pt, t, a, r0, v):
    def f(tau):
        s2 = 2 * tau + a ** 2
        rp = r0 + v * (t - tau)
        d2 = ((pt - rp) ** 2).sum()
        return np.exp(-tau) * np.exp(-d2 / (2 * s2)) / (2 * np.pi * s2)
    return quad(f, 0, t, epsabs=1e-13, epsrel=1e-11, limit=200)[0]



print("\n1. Baseline active Brownian particles (g = 0)")

def part1():
    Pe, Dr, Dt = 4.0, 0.5, 0.05
    NB = 1500
    sim = Simulation(Params(N=NB, L=20, Pe=Pe, g=0.0, Dr=Dr, Dt=Dt, dt=0.01,
                            self_mode="include", seed=11, dtype="complex64"))
    p0 = sim.orientations().copy()
    r_start = sim.r.copy()
    ts, corr, msd, Pvals, Sk = [], [], [], [], []


    def obs(s):
        ts.append(s.t)
        corr.append((s.orientations() * p0).sum(1).mean())
        msd.append(((s.r - r_start) ** 2).sum(1).mean())


    sim.run(6.0, sample_every=25, observer=obs)
    ts_a, corr_a, msd_a = map(np.array, (ts, corr, msd))
    check("<p(t).p(0)> vs exp(-Dr t): max abs dev", np.abs(corr_a - np.exp(-Dr * ts_a)).max(),
          4 / np.sqrt(NB), f"(statistical error ~{1/np.sqrt(NB):.3f} per point)")
    msd_th = 4 * Dt * ts_a + 2 * Pe ** 2 / Dr ** 2 * (Dr * ts_a - 1 + np.exp(-Dr * ts_a))
    check("MSD(t) vs ABP formula: max rel dev", np.abs(msd_a / msd_th - 1).max(), 0.08)

    for _ in range(12):
        sim.run(6.0, sample_every=10 ** 9)
        Pvals.append(sim.polarization())
        Sk.append(structure_factor(sim.r, 20, mmax=12)[1])
    check("<P^2> N - 1 (isotropic => 0)", np.mean(np.square(Pvals)) * NB - 1, 0.9,
          "(12 samples: weak test, stat. error ~0.3)")
    check("mean S(k) - 1 (ideal gas, 12 snapshots)", np.mean(Sk) - 1, 0.06)
    rr, gr = pair_correlation(sim.r[:1500], 20, nbins=20, rmax=5)
    check("mean g(r) - 1 (ideal gas)", gr.mean() - 1, 0.05)



if want(1):
    part1()


print("\n2. Field solver vs analytic Green-function solutions")

a = 0.25
for ka in (3.5, 5.0):
    sim = Simulation(Params(N=1, L=20, Pe=0, g=0, Dr=0, a=a, dt=0.01, ka=ka,
                            seed=0), r0=[[10., 10.]], theta0=[0.])
    sim.run(1.0)
    c0 = sim.field_at([[10., 10.]])[0]
    ref = c_transient_centre(1.0, a)
    check(f"transient c(0,t=1), ka={ka}: rel. error", c0 / ref - 1, 3e-3 if ka == 3.5 else 3e-4)
sim.run(14.0)
errs = []
for d in (0.5, 1.0, 2.0, 4.0):
    c = sim.field_at([[10. + d, 10.]])[0]
    errs.append(c / c_steady(d, a) - 1)
check("steady c(r), r=0.5..4: max rel. error (ka=5)", np.abs(errs).max(), 5e-4)

L, Pe_m, T = 40.0, 5.0, 2.5
sim = Simulation(Params(N=1, L=L, Pe=Pe_m, g=0, Dr=0, a=a, dt=0.005, ka=5.0, seed=0),
                 r0=[[5., 20.]], theta0=[0.])
sim.run(T)
r0, v = np.array([5., 20.]), np.array([Pe_m, 0.])
head = r0 + v * T
pts = [head, head - [1.0, 0], head - [4.0, 0], head - [10.0, 0],
       head - [4.0, 0] + [0, 0.5], head - [4.0, 0] + [0, 1.0], head + [1.0, 0]]
num = sim.field_at(np.array(pts))
ref = np.array([c_line_source(pt, T, a, r0, v) for pt in pts])
check("moving-source field, 7 points: max rel. error", np.abs(num / ref - 1).max(), 5e-3,
      "(dt = 0.005)")

print("\n3. Self-exclusion")

# (a) isolated particle: zero chemotactic torque, strong coupling & noise
tmax = {"exclude": 0.0, "include": 0.0}
trms = {}
for mode in ("exclude", "include"):
    sim = Simulation(Params(N=1, L=20, Pe=5.0, g=50.0, Dr=0.5, a=0.25, dt=0.01,
                            self_mode=mode, seed=3))
    tq = []
    for _ in range(1500):
        sim.step()
        tq.append(sim.last_torque[0])
    tq = np.array(tq[200:])
    tmax[mode], trms[mode] = np.abs(tq).max(), np.sqrt((tq ** 2).mean())
check("isolated particle, exclude: max |torque| (g=50)", tmax["exclude"], 1e-9)
print(f"        'include' control: rms torque = {trms['include']:.3f}  "
      f"(nonzero -> the test discriminates)")
results.append(trms["include"] > 1e-2)

# (b) two stationary particles: gradient vs analytic smoothed field
for d in (0.8, 1.5, 3.0):
    sim = Simulation(Params(N=2, L=20, Pe=0, g=0, Dr=0, a=a, dt=0.01, ka=5.0, seed=0),
                     r0=[[8., 10.], [8. + d, 10.]], theta0=[0., 0.])
    sim.run(15.0)
    ex, ey = sim._phases(sim.r)
    gr = sim._gradient(ex, ey)
    ref = -dc_steady(d, a)
    check(f"two-body gradient, d={d}: rel. error (particle 1)", gr[0, 0] / ref - 1, 1e-3)
    check(f"two-body gradient, d={d}: antisymmetry g1 + g2", (gr[0] + gr[1]).sum(), 1e-10)

# (b2) sign convention: g > 0 turns a particle TOWARDS its neighbour (torque = g|grad| sin(phi-theta))
for g_sign in (+3.0, -3.0):
    sim = Simulation(Params(N=2, L=20, Pe=0, g=g_sign, Dr=0, a=a, dt=0.01, ka=5.0, seed=0),
                     r0=[[8., 10.], [9.5, 10.]], theta0=[np.pi / 4, 0.0])
    sim.run(15.0)
    sim.theta[:] = [np.pi / 4, 0.0]
    sim.step()
    tq = sim.last_torque[0]
    gmag = np.linalg.norm(sim.grad[0])
    expected = g_sign * gmag * np.sin(0.0 - np.pi / 4)
    check(f"torque sign/size, g={g_sign:+.0f}: rel. error vs g|grad|sin(phi-theta)",
          tq / expected - 1, 1e-6, f"(torque {tq:+.3f}: " +
          ("turns toward neighbour)" if tq < 0 else "turns away)"))

# (c) moving particle 1 leaves a trail; stationary particle 2 at a distance.
def sensed_vs_exact(mode):
    sim = Simulation(Params(N=2, L=40, Pe=0.0, g=0.0, Dr=0, a=a, dt=0.01, ka=5.0,
                            self_mode=mode, seed=0),
                     r0=[[20., 20.], [23., 22.]], theta0=[0., 0.])
    sim.run(14.0)
    sim.p.Pe = 0.0
    errs = []
    s_vec = np.array([23., 22.])
    for n in range(300):
        th = 2 * np.pi * n / 300 * 0.5
        sim.r[0] = [20. + 1.5 * np.sin(th), 20. + 1.5 * (1 - np.cos(th))]

        sim._deposit_dyn(sim.r)
        sim.t += sim.p.dt
        if n % 30 == 29:
            ex, ey = sim._phases(sim.r)
            gr = sim._gradient(ex, ey)[0]
            dvec = s_vec - sim.r[0]
            dist = np.linalg.norm(dvec)
            ref = -dc_steady(dist, a) * dvec / dist
            errs.append(np.linalg.norm(gr - ref) / np.linalg.norm(ref))
    return max(errs)


e_ex, e_in = sensed_vs_exact("exclude"), sensed_vs_exact("include")
check("moving p1 / resting p2: |grad c_-1 - grad c_2|/|grad c_2|", e_ex, 5e-3)
print(f"        'include' control relative error = {e_in:.2f}  (own trail contaminates)")
results.append(e_in > 5e-2)

print("\n4. Convergence (deterministic 4-body dynamics, g=8, T=3)")

rng = np.random.default_rng(5)
r_init = rng.uniform(8, 12, (4, 2))
th_init = rng.uniform(0, 2 * np.pi, 4)


def final_state(dt, ka):
    s = Simulation(Params(N=4, L=20, Pe=3.0, g=8.0, Dr=0, Dt=0, a=0.25, dt=dt, ka=ka),
                   r0=r_init, theta0=th_init)
    s.run(3.0)
    return s.r.copy(), s.theta.copy()


ref_r, ref_th = final_state(0.00125, 5.0)
errs = []
for dt in (0.02, 0.01, 0.005, 0.0025):
    r_, th_ = final_state(dt, 5.0)
    errs.append(max(np.abs(r_ - ref_r).max(), np.abs(th_ - ref_th).max()))
print("        dt:    0.02      0.01      0.005     0.0025   (vs dt=0.00125)")
print("        error: " + "  ".join(f"{e:.2e}" for e in errs))
order = np.log2(errs[1] / errs[2])
check("error(dt=0.005) / error(dt=0.01) ~ 1/2 (1st order): |ratio-0.5|",
      errs[2] / errs[1] - 0.5, 0.2)
check("final error at dt = 0.0025 (positions in ell_c)", errs[3], 5e-3)
r35, th35 = final_state(0.00125, 3.5)
check("spectral cutoff ka=3.5 vs 5.0: max deviation", max(np.abs(r35 - ref_r).max(),
      np.abs(th35 - ref_th).max()), 2e-2, "(default ka)")

print("\n5. Dynamic field -> quasi-static in the stationary limit")

rr0 = [[5., 5.], [6., 5.5], [5.5, 7.]]
kw = dict(N=3, L=20, Pe=0, g=0, Dr=0, a=0.25, dt=0.01, seed=0)
sd = Simulation(Params(memory="dynamic", **kw), r0=rr0, theta0=[0, 0, 0])
sq = Simulation(Params(memory="quasistatic", **kw), r0=rr0, theta0=[0, 0, 0])
sd.run(18.0)
sq.step()
check("max |C_dynamic - C_quasistatic| / max|C|", np.abs(sd.C - sq.C).max() / np.abs(sq.C).max(),
      1e-6)

print("\n" + "=" * 70)
print(f"{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
