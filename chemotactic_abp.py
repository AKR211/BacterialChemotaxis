from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


# parameters
@dataclass
class Params:
    N: int = 200                  # number of particles
    L: float = 20.0               # box side
    Pe: float = 5.0               # v0
    g: object = 5.0               # chemotactic strength; scalar or (N,) array
    secretion: object = 1.0       # secretion rate; scalar or (N,) array
    Dr: float = 0.5               # rotational diffusion
    Dt: float = 0.0               # translational diffusion
    a: float = 0.25               # source width
    dt: float = 0.01              # time step
    ka: float = 3.5               # spectral cutoff k_max * a
    memory: str = "dynamic"       # 'dynamic' or 'quasistatic'
    self_mode: str = "exclude"    # 'exclude' (model) or 'include' (control)
    seed: object = None
    dtype: str = "complex128"     # complex64 is ~2x faster, ~1e-6 accuracy


# simulation
class Simulation:
    def __init__(self, params: Params, r0=None, theta0=None):
        p = params

        self.p = p

        assert p.memory in ("dynamic", "quasistatic")
        assert p.self_mode in ("exclude", "include")

        self.rng = np.random.default_rng(p.seed)

        N, L = p.N, p.L

        self.cdt = np.dtype(p.dtype)
        self.rdt = np.float64 if self.cdt == np.complex128 else np.float32

        # per-particle coupling constants (species)
        self.g = np.broadcast_to(np.asarray(p.g, float), (N,)).copy()
        self.s = np.broadcast_to(np.asarray(p.secretion, float), (N,)).copy()

        # state
        self.r = (self.rng.uniform(0, L, (N, 2)) if r0 is None else np.array(r0, float).reshape(N, 2))
        self.theta = (self.rng.uniform(0, 2 * np.pi, N) if theta0 is None else np.array(theta0, float).reshape(N))
        self.t = 0.0
        self.grad = np.zeros((N, 2))

        # Fourier modes (half plane, circular cutoff)
        dk = 2 * np.pi / L

        self.dk = dk

        M = int(np.ceil(p.ka / p.a / dk))

        self.M = M
        self.kx = dk * np.arange(0, M + 1)
        self.ky = dk * np.arange(-M, M + 1)

        KX, KY = np.meshgrid(self.kx, self.ky)
        k2 = KX ** 2 + KY ** 2
        
        self.k2 = k2

        mask = (k2 <= (M * dk) ** 2) & ((KX > 0) | ((KX == 0) & (KY > 0)))

        self.mask = mask

        Khat = np.exp(-0.5 * p.a ** 2 * k2)
        E = np.exp(-(k2 + 1.0) * p.dt)

        self.E = E.astype(self.rdt)
        self.coef_dyn = (mask * Khat * (1 - E) / (k2 + 1)).astype(self.rdt)
        self.coef_qs = (mask * Khat / (k2 + 1)).astype(self.rdt)
        self.ikx = (1j * self.kx).astype(self.cdt)
        self.iky = (1j * self.ky).astype(self.cdt)

        Ky, Kx = k2.shape
        self.C = np.zeros((Ky, Kx), self.cdt)
        self.mass = np.zeros(N) if p.memory == "dynamic" else self.s.copy()
        self.need_f = (p.memory == "dynamic" and p.self_mode == "exclude")
        if self.need_f:
            self.f = np.zeros((N, Ky, Kx), self.cdt)
            self._buf = np.empty_like(self.f)
        if p.memory == "quasistatic":
            self._deposit_qs(self.r)

    # Fourier helpers
    def _phases(self, r):
        rm = np.mod(r, self.p.L)
        ex = np.exp(1j * rm[:, 0:1] * self.kx[None, :]).astype(self.cdt)
        ey = np.exp(1j * rm[:, 1:2] * self.ky[None, :]).astype(self.cdt)
        return ex, ey

    def _deposit_dyn(self, r_src):
        ex, ey = self._phases(r_src)
        A = np.conj(ey) * self.s[:, None].astype(self.rdt)    # (N,Ky)
        B = np.conj(ex)                                       # (N,Kx)
        self.C *= self.E
        self.C += self.coef_dyn * (A.T @ B)
        if self.need_f:
            self.f *= self.E
            np.multiply(A[:, :, None], B[:, None, :], out=self._buf)
            self._buf *= self.coef_dyn
            self.f += self._buf

    def _deposit_qs(self, r_src):
        ex, ey = self._phases(r_src)
        A = np.conj(ey) * self.s[:, None].astype(self.rdt)
        B = np.conj(ex)
        self.C = self.coef_qs * (A.T @ B)

    def _gradient(self, ex, ey):
        L2 = self.p.L ** 2
        CT = self.C.T                                  # (Kx,Ky)
        u = ex @ CT                                    # (N,Ky)
        v = (ex * self.ikx) @ CT
        gx = (ey * v).sum(1).real
        gy = ((ey * self.iky) * u).sum(1).real
        if self.p.self_mode == "exclude" and self.need_f:
            M2 = np.stack([ex, ex * self.ikx], axis=2)         # (N,Kx,2)
            W = np.matmul(self.f, M2)                          # (N,Ky,2)
            gx = gx - (ey * W[:, :, 1]).sum(1).real
            gy = gy - ((ey * self.iky) * W[:, :, 0]).sum(1).real
        return (2.0 / L2) * np.stack([gx, gy], axis=1)


    # time stepping
    def step(self):
        p, dt, N = self.p, self.p.dt, self.p.N
        ex, ey = self._phases(self.r)
        grad = self._gradient(ex, ey)
        self.grad = grad

        th = self.theta
        c, s = np.cos(th), np.sin(th)
        torque = self.g * (-s * grad[:, 0] + c * grad[:, 1])
        self.last_torque = torque

        th_new = th + torque * dt + np.sqrt(2 * p.Dr * dt) * self.rng.standard_normal(N)
        dr = p.Pe * dt * np.stack([c, s], axis=1)
        if p.Dt > 0:
            dr = dr + np.sqrt(2 * p.Dt * dt) * self.rng.standard_normal((N, 2))
        r_new = self.r + dr

        if p.memory == "dynamic":
            self._deposit_dyn(0.5 * (self.r + r_new))
            e0 = np.exp(-dt)
            self.mass = self.mass * e0 + self.s * (1 - e0)
        else:
            self._deposit_qs(r_new)

        self.r, self.theta = r_new, th_new
        self.t += dt

    def run(self, T, sample_every=50, observer=None, progress=False):
        n = int(round(T / self.p.dt))
        for i in range(n):
            self.step()
            if observer is not None and (i + 1) % sample_every == 0:
                observer(self)
            if progress and (i + 1) % max(1, n // 10) == 0:
                print(f"  t = {self.t:8.2f}", flush=True)

    # field evaluation
    def field_at(self, pts, particle=None):
        pts = np.atleast_2d(np.asarray(pts, float))
        ex, ey = self._phases(pts)
        F = self.C if particle is None else self.f[particle]
        u = ex @ F.T
        m0 = self.mass.sum() if particle is None else self.mass[particle]
        return (2.0 / self.p.L ** 2) * (ey * u).sum(1).real + m0 / self.p.L ** 2

    def field_grid(self, n=200, particle=None):
        L = self.p.L
        xs = (np.arange(n) + 0.5) * L / n
        px = np.exp(1j * xs[:, None] * self.kx[None, :])
        py = np.exp(1j * xs[:, None] * self.ky[None, :])
        F = self.C if particle is None else self.f[particle]
        m0 = self.mass.sum() if particle is None else self.mass[particle]
        return (2.0 / L ** 2) * (py @ F @ px.T).real + m0 / L ** 2   # [y, x]

    # observables
    def orientations(self):
        return np.stack([np.cos(self.theta), np.sin(self.theta)], axis=1)

    def polarization(self, sel=None):
        p = self.orientations()
        if sel is not None:
            p = p[sel]
        return np.linalg.norm(p.mean(0))

    def chemotactic_alignment(self, gmin=1e-3):
        gn = np.linalg.norm(self.grad, axis=1)
        sel = gn > gmin
        if not sel.any():
            return np.nan
        p = self.orientations()
        return float((p[sel] * self.grad[sel]).sum() / gn[sel].sum())

    def wrapped(self):
        return np.mod(self.r, self.p.L)


# structural observables (functions of positions only)
def structure_factor(r, L, mmax=20):
    N = len(r)
    m = np.arange(-mmax, mmax + 1)
    rm = np.mod(r, L)
    ex = np.exp(-1j * 2 * np.pi / L * rm[:, 0:1] * m[None, :])
    ey = np.exp(-1j * 2 * np.pi / L * rm[:, 1:2] * m[None, :])
    rho = ey.T @ ex                                       # [my, mx]
    S2 = (np.abs(rho) ** 2) / N
    MX, MY = np.meshgrid(m, m)
    mag = np.sqrt(MX ** 2 + MY ** 2)
    shell = np.rint(mag).astype(int)
    keep = (shell >= 1) & (shell <= mmax)
    S = np.bincount(shell[keep], S2[keep], minlength=mmax + 1)[1:]
    cnt = np.bincount(shell[keep], minlength=mmax + 1)[1:]
    return 2 * np.pi / L * np.arange(1, mmax + 1), S / cnt


def pair_correlation(r, L, nbins=60, rmax=None):
    N = len(r)
    rmax = rmax or L / 2
    d = r[:, None, :] - r[None, :, :]
    d -= L * np.rint(d / L)
    dist = np.sqrt((d ** 2).sum(-1))[np.triu_indices(N, 1)]
    h, edges = np.histogram(dist, bins=nbins, range=(0, rmax))
    mid = 0.5 * (edges[1:] + edges[:-1])
    shell = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    g = 2 * h / (N * (N / L ** 2) * shell)
    return mid, g


def cluster_sizes(r, L, rc=0.5):
    N = len(r)
    tree = cKDTree(np.mod(r, L), boxsize=L)
    pairs = tree.query_pairs(rc, output_type="ndarray")
    if len(pairs) == 0:
        return np.ones(N, int)
    A = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(N, N))
    _, labels = connected_components(A, directed=False)
    return np.sort(np.bincount(labels))[::-1]
