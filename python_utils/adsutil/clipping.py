"""Second harmonic clipping contours (Canning, Tasker, Cripps, IEEE TMTT 2014, docs/).

All impedances are at the intrinsic drain (the load the current source sees) and are
normalized to R_L, the class B load line resistance (paper eq. 5). The current is the
fixed half-wave rectified cosine of the paper,
    I(t) = Imax * (1/pi + cos(t)/2 + 2/(3 pi) cos(2t) + 0*cos(3t) - ...),
so with R_L = 2 Vdc / Imax the normalized drain voltage is

    v(t) = V(t)/Vdc = 1 - Re[z1 e^{jt}] - (4 / 3 pi) Re[z2 e^{j2t}],    zn = Zn / R_L.

The 3rd harmonic current is zero for this waveform, so Z3 does not enter. Class B is
z1 = 1, z2 = 0; class J is z1 = 1 + j, z2 = -j 3 pi / 8, giving v = (1 - cos t)(1 + sin t).

The paper writes the fundamental voltage as v1 = v1r + j v1q with cos/sin coefficients.
In those terms v1r = -Re z1 and v1q = Im z1, and the label "v1 = 0.95 + j0.9" in its
figures is z1 = 0.95 + j0.9. Efficiency is eta = (pi / 4) Re z1, power scales with Re z1.

Two ways to get the boundary of the non-clipping 2f0 region, used to check each other:
  contour()  the paper's closed form, eqs. 7, 8, 10 and 14, sweeping delta.
  margin()   min over t of v(t). The contour is where it is exactly 0.
Every t gives a linear constraint on z2, so the safe region is convex.

    python_utils/adsenv.sh python -m adsutil.clipping          # self-check and figures
"""

from __future__ import annotations

import numpy as np

K2 = 4 / (3 * np.pi)     # 2f0 current over fundamental current, times 2 (see v(t) above)
THETA = np.linspace(0, 2 * np.pi, 1441)


def voltage(z1, z2, theta=THETA):
    """Normalized drain voltage v(t) over one period. z1, z2 broadcast against each other;
    theta runs along a new last axis."""
    z1 = np.asarray(z1, complex)[..., None]
    z2 = np.asarray(z2, complex)[..., None]
    e = np.exp(1j * theta)
    return 1 - (z1 * e).real - K2 * (z2 * e * e).real


def margin(z1, z2, theta=THETA, chunk: int = 4096):
    """min v(t). >= 0 means no clipping; 0 is the clipping contour.
    Works through the points in chunks so a large grid does not need points x len(theta)
    memory at once (a 541 x 541 grid with 1441 samples would be about 7 GB)."""
    z1, z2 = np.broadcast_arrays(np.asarray(z1, complex), np.asarray(z2, complex))
    shape = z1.shape
    z1, z2 = z1.ravel(), z2.ravel()
    out = np.empty(z1.size)
    for i in range(0, z1.size, chunk):
        out[i:i + chunk] = voltage(z1[i:i + chunk], z2[i:i + chunk], theta).min(axis=-1)
    return out.reshape(shape)


def contour(z1, n: int = 4001, beta_max: float = 1.0):
    """Paper's second harmonic clipping contour for a fixed fundamental z1 (eqs. 10, 14, 8).

    Sweeps delta over one period and returns z2 (normalized to R_L) for each delta whose
    beta lies in [-beta_max, beta_max], in sweep order. Points can lie outside the Smith
    chart (Re z2 < 0), as in the paper's Fig. 4-6.
    """
    v1r, v1q = -np.real(z1), np.imag(z1)
    d = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / n     # avoid the 0/0 at delta = 0
    s, c = np.sin(d), np.cos(d)
    # eq. 14; gamma and gamma + pi give the same waveform (beta changes sign), so no branch care
    num = v1r + v1r * c**2 + 2 * c - v1q * c * s
    den = 2 * s + v1r * s * c - 2 * v1q + v1q * c**2
    g = np.arctan2(-num, den)
    beta = (2 * c + 2 * v1r) / (2 * np.sin(g) - v1r * np.sin(d - g))         # eq. 10
    dc = 1 - beta / 2 * np.sin(g - d)
    v2r = -beta / 2 * np.sin(g + d) / dc                                      # eq. 8
    v2q = -beta / 2 * np.cos(g + d) / dc
    # back to impedance: the voltage at 2f0 is Re[(v2r - j v2q) e^{j2t}] = -K2 Re[z2 e^{j2t}]
    z2 = -(v2r - 1j * v2q) / K2
    keep = np.isfinite(z2) & (np.abs(beta) <= beta_max)
    return z2[keep]


def safe_region(z1, z2_grid, theta=THETA):
    """Boolean mask over a grid of z2: True where the waveform does not clip."""
    return margin(z1, z2_grid, theta) >= 0


# -- self-check and figures ---------------------------------------------------------

def _check():
    # classical points
    for name, z1, z2, want in [("class B", 1, 0, 0.0), ("class J", 1 + 1j, -1j * 3 * np.pi / 8, 0.0)]:
        print(f"{name}: margin = {margin(z1, z2):+.4f} (want {want})")
    tj = THETA
    vj = voltage(1 + 1j, -1j * 3 * np.pi / 8)
    print("class J waveform equals (1 - cos t)(1 + sin t):",
          np.allclose(vj, (1 - np.cos(tj)) * (1 + np.sin(tj))))
    # the closed-form contour must graze zero at every point
    for z1 in (1.0, 1 + 0.9j, 0.95 + 0.9j, 0.9 + 0.9j, 0.8 - 0.5j):
        z2 = contour(z1)
        m = margin(z1, z2)
        print(f"z1 = {z1:.2f}: {len(z2)} contour points, margin on contour "
              f"min {m.min():+.1e} max {m.max():+.1e}")


def _figures(out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plot_loads import COLORS, TEXT, gamma, smith_axes

    RL = 50.0        # the paper's Fig. 4 and 6 use R_L = 50 ohm on a 50 ohm chart
    gx = np.linspace(-1.6, 1.1, 301)
    G = gx[None, :] + 1j * gx[::-1, None]
    Z2 = 50 * (1 + G) / (1 - G)                # impedance grid in the Gamma plane

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), facecolor="white")
    cases = [("Fig. 4: z1 = 1 + j0", [1.0]),
             ("Fig. 6a: z1 = (1.00, 0.95, 0.90) + j0.9", [1 + 0.9j, 0.95 + 0.9j, 0.9 + 0.9j]),
             ("Fig. 6b: z1 = 0.95 + j0.9", [0.95 + 0.9j])]
    shades = ["#1baf7a", "#8fd3b8", "#c9eadc"]
    for ax, (title, z1s) in zip(axes, cases):
        smith_axes(ax, title)
        ax.set_xlim(-1.6, 1.1); ax.set_ylim(-1.35, 1.35)
        for z1, col in zip(z1s, shades):
            safe = safe_region(z1, Z2 / RL, theta=THETA[::4])     # coarse t is plenty for shading
            ax.contourf(G.real, G.imag, safe, levels=[0.5, 1.5], colors=[col], alpha=0.35, zorder=1)
            g = gamma(RL * contour(z1))
            ax.plot(g.real, g.imag, ".", ms=1.5, color=COLORS[3], zorder=2)
            g1 = gamma(RL * z1)
            ax.plot(g1.real, g1.imag, "x", color=COLORS[2], ms=8, mew=2, zorder=4)
        ax.text(-1.55, -1.3, "shaded: non-clipping 2f0 (margin >= 0)\ndots: closed-form contour, x: Z1",
                fontsize=8, color=TEXT)
    fig.tight_layout()
    out = out_dir / "clipping_paper_figs.png"
    fig.savefig(out, dpi=140)
    return out


if __name__ == "__main__":
    from . import PACKAGE_DIR
    _check()
    out_dir = PACKAGE_DIR.parent / "results"
    out_dir.mkdir(exist_ok=True)
    print(_figures(out_dir))
