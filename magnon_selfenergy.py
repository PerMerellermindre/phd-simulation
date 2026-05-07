"""
Self-energy calculation for magnon modes dressed by SC quasiparticles and cavity photons.

System: YIG ferromagnet (FM) + MgB2 superconductor (SC) in a microwave cavity.
Computes Re[Sigma(lambda_k, k)] as a function of magnon wavevector k, including
contributions from both SC quasiparticles and cavity photon modes.
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
import numba
from joblib import Parallel, delayed

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"


# Load parameters from config file
def load_params(path = "params.yaml"):
    """Load run parameters from a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)

prm = load_params()


# Universal constants
pi    = np.pi
hbar  = 1.055e-34
c     = 299792458.0
kB    = 1.38e-23
mu0   = 1.257e-6
eps0  = 8.854e-12
e     = 1.602e-19
me    = 9.109e-31
muB   = 9.27400968e-24

# Cavity
L   = prm["cav"]["L"]
Lx  = Ly = L
Lz  = prm["cav"]["Lz_factor"] * L
V   = Lx * Ly * Lz
omega0 = c * pi / Lz     # fundamental ell_z = 1 mode
Qz     = pi / Lz         # z-momentum for all ell_z = 1 modes

# FM (YIG)
g_FM  = prm["fm"]["g_factor"]
aFM   = prm["fm"]["lattice_constant"]
lxFM  = lyFM = L
NxFM  = int(lxFM / aFM)
NyFM  = int(lyFM / aFM)
chi   = prm["fm"]["chi"]
Da2   = 2.0 * prm["fm"]["spin_wave_stiffness_factor"] / aFM**2
Bext  = prm["fm"]["B_ext"]
muEta = g_FM * muB / Da2 * Bext

# SC (MgB2)
aSC   = prm["sc"]["lattice_constant"]
lxSC  = lySC = L
NxSC  = int(lxSC / aSC)
NySC  = int(lySC / aSC)

m_eff    = prm["sc"]["m_eff_ratio"] * me
t        = hbar**2 / (2.0 * m_eff * aSC**2)    # tight-binding hopping (Schlawin S15)
Tc       = prm["sc"]["Tc"]
Delta0   = 1.76 * kB * Tc
T        = prm["sc"]["T"]
# Delta    = 0.0            # set to 0 for normal-state test run
Delta  = Delta0 * np.tanh(1.74 * np.sqrt(Tc / T - 1.))   # full BCS gap

beta     = hbar / (kB * T)
betahbar = beta / (2.0 * hbar)

mu_el = prm["sc"]["mu_el"]
m_rat = prm["sc"]["m_ratio"]
C_tb  = prm["sc"]["C_tb"]
kF    = np.sqrt(-mu_el * m_rat / C_tb)

lambda_L = prm["sc"]["lambda_L"]
jC       = prm["sc"]["j_C"]
n_s      = m_eff / (e**2 * mu0) * (1.0 / lambda_L)**2
PC       = jC * m_eff / (e * n_s * hbar)      # critical superfluid COM momentum

loss     = prm["numerics"]["loss_MHz"] * 1e6 * hbar * 2.0 * pi   # linewidth [J]
n_jobs   = prm["numerics"]["n_jobs"]

kx_zoom         = prm["grid"]["kx_zoom"]
ky_zoom         = prm["grid"]["ky_zoom"]
ky_increase_res = prm["grid"]["ky_increase_res"]
kx_step         = prm["grid"]["kx_step"]
ky_step         = prm["grid"]["ky_step"]
px_step         = prm["grid"]["px_step"]
py_step         = prm["grid"]["py_step"]
cos_bounds      = prm["grid"]["cos_bounds"]
paSC_bounds     = prm["grid"]["paSC_bounds"]


# Dispersion relations
def omegaq(kX, kY):
    """Cavity photon frequency for in-plane momentum (kX, kY), ell_z = 1."""
    return omega0 * np.sqrt(1.0 + (c / omega0)**2 * (kX**2 + kY**2))

def gammak(kX, kY):
    """Nearest-neighbour structure factor on the FM square lattice."""
    return 0.5 * (np.cos(kX * aFM) + np.cos(kY * aFM))

def omegak(kX, kY):
    """Magnon frequency (Heisenberg exchange + Zeeman)."""
    return Da2 / hbar * ((1.0 - gammak(kX, kY)) + muEta)

def ksi(pX, pY):
    """SC tight-binding dispersion measured from the Fermi energy."""
    return 2.0 * t * (-(np.cos(pX * aSC) + np.cos(pY * aSC)) + np.cos(kF * aSC))

def ESC(pX, pY, M, PX, PY):
    """
    Bogoliubov quasiparticle energy.
    M = 0 lower branch, M = 1 upper. PX, PY: superfluid COM momentum.
    """
    ksi_p = ksi(pX + PX, pY + PY)
    ksi_m = ksi(-pX + PX, -pY + PY)
    return 0.5 * (ksi_p - ksi_m
                  + (-1.0)**M * np.sqrt(4.0 * Delta**2 + (ksi_p + ksi_m)**2))

def up(pX, pY, PX, PY):
    """Bogoliubov u coherence factor."""
    ksi_p = ksi(pX + PX, pY + PY)
    ksi_m = ksi(-pX + PX, -pY + PY)
    return np.sqrt(0.5 * (1.0 + (ksi_p + ksi_m)
                          / np.sqrt(4.0 * Delta**2 + (ksi_p + ksi_m)**2)))

def vp(pX, pY, PX, PY):
    """Bogoliubov v coherence factor."""
    ksi_p = ksi(pX + PX, pY + PY)
    ksi_m = ksi(-pX + PX, -pY + PY)
    return np.sqrt(0.5 * (1.0 - (ksi_p + ksi_m)
                          / np.sqrt(4.0 * Delta**2 + (ksi_p + ksi_m)**2)))


# Coupling matrix elements
def Ox(kX, kY):
    """x-component of the cavity polarisation rotation matrix (varsigma = 1)."""
    q2 = kX**2 + kY**2
    return Qz / np.sqrt(q2 + Qz**2) * kX / np.sqrt(q2)

def Oy(kX, kY):
    """y-component of the cavity polarisation rotation matrix (varsigma = 1)."""
    q2 = kX**2 + kY**2
    return Qz / np.sqrt(q2 + Qz**2) * kY / np.sqrt(q2)

def gFM(kX, kY):
    """
    Zeeman coupling between cavity photon (kX, kY) and FM magnon.
    Valid in the DC (P = 0) case.
    """
    q2 = kX**2 + kY**2
    return (g_FM * muB
            * np.sqrt((chi * hbar * NxFM * NyFM) / (2.0 * eps0 * V))
            * kY * 1j * np.sqrt(q2 / (omegaq(kX, kY) * (q2 + Qz**2))))

def gSC(kX, kY, pX, pY):
    """
    Paramagnetic coupling between cavity photon (kX, kY) and SC electron (pX, pY).
    Valid in the DC (P = 0) case.
    Note: at kX = 0 the Ox term vanishes and only Oy contributes.
    """
    return (2j * aSC * e * t / hbar
            * np.sqrt(hbar / (eps0 * V * omegaq(kX, kY)))
            * (np.sin((pX - kX / 2.0) * aSC) * Ox(kX, kY)
               + np.sin((pY - kY / 2.0) * aSC) * Oy(kX, kY)))


# Wavevector grids
kx_lim = int(NxFM / (2.0 * kx_zoom))
ky_lim = int(NyFM / (2.0 * ky_zoom))

kx_range = (np.arange(-int(kx_lim / kx_step) * kx_step,
                        int(kx_lim / kx_step) * kx_step + 1,
                        kx_step) * 2.0 * pi / lxFM + 1e-20)

ky_range = (np.arange(0,
                       int(ky_lim / ky_step) * ky_step * ky_increase_res + 1,
                       ky_step) * 2.0 * pi / lyFM / ky_increase_res + 1e-20)

ny_range = np.arange(0,
                     int(ky_lim / ky_step) * ky_step * ky_increase_res + 1,
                     ky_step)


# Fermi-surface sampling
p_unit = 2.0 * pi / lxSC
px_lim = int(NxSC / 2)

if np.cos(kF * aSC) + 1.0 + cos_bounds <= 1.0:
    px_range_fs = (np.arange(
        int(np.arccos(np.cos(kF * aSC) + 1.0 + cos_bounds) / aSC / p_unit / px_step) * px_step,
        int(px_lim / px_step) * px_step + 1,
        px_step) * 2.0 * pi / lxSC + 1.0)
else:
    px_range_fs = (np.arange(0, int(px_lim / px_step) * px_step + 1, px_step)
                   * 2.0 * pi / lxSC + 1.0)

# Build the Fermi-surface mask on a 2D grid and extract matching (px, py) pairs
px2d, py2d = np.meshgrid(px_range_fs, px_range_fs, indexing = "ij")
fs_mask    = np.abs(np.cos(px2d * aSC) + np.cos(py2d * aSC) - np.cos(kF * aSC)) < cos_bounds
pxy_list   = np.array([px2d[fs_mask], py2d[fs_mask]])

print(f"Fermi-surface points sampled: {pxy_list.shape[1]}\n")


# Cached coherence factors
u_p0    = up(pxy_list[0], pxy_list[1], 0.0, 0.0)
v_p0    = vp(pxy_list[0], pxy_list[1], 0.0, 0.0)
E_p0    = ESC(pxy_list[0], pxy_list[1], 0, 0.0, 0.0)
tanh_p0 = np.tanh(betahbar * E_p0)


# Numba JIT kernel for the p-space summation
@numba.njit(cache = True)
def _psum_kernel(px_arr, py_arr, ky,
                 t_val, aSC_val, kF_val, Delta_val, betahbar_val,
                 u_arr, v_arr, tanh_p_arr, E_p_arr,
                 g_pre_real, g_pre_imag,
                 Z_real, Z_imag):
    """
    JIT-compiled p-space sum for the SC self-energy integrand.
    Evaluates at kX = 0, PX = PY = 0.
    """
    n      = len(px_arr)
    result = 0.0 + 0.0j
    g_pre  = g_pre_real + 1j * g_pre_imag
    Z      = Z_real + 1j * Z_imag

    for i in range(n):
        px = px_arr[i]
        py = py_arr[i]

        # Bogoliubov energy of the ky-shifted state  E(p - k),  kX = 0
        ksi_s = 2.0 * t_val * (-(np.cos(px * aSC_val) + np.cos((py - ky) * aSC_val))
                                + np.cos(kF_val * aSC_val))
        E_s   = np.sqrt(Delta_val**2 + ksi_s**2)

        # Coherence factors for the shifted state (max guards against fp rounding)
        u_s = np.sqrt(max(0.0, 0.5 * (1.0 + ksi_s / E_s)))
        v_s = np.sqrt(max(0.0, 0.5 * (1.0 - ksi_s / E_s)))

        # Paramagnetic coupling at kX = 0  (Ox = 0 -> only Oy survives)
        gsc = g_pre * np.sin((py - ky / 2.0) * aSC_val)

        # Interference factor
        coherence_sq = (u_arr[i] * u_s + v_arr[i] * v_s)**2

        # Thermal weight and energy denominator (imaginary loss shifts pole off axis)
        tanh_diff = np.tanh(betahbar_val * E_s) - tanh_p_arr[i]
        denom     = E_p_arr[i] - E_s - Z

        result += 2.0 * abs(gsc)**2 * coherence_sq * tanh_diff / denom

    return result


# Per-ky worker function (parallelization with joblib)
def _ky_worker(ky, px_arr, py_arr, u_arr, v_arr, tanh_p_arr, E_p_arr):
    """
    Compute the SC self-energy contribution for a single ky value.
    Structured as a standalone function so joblib can serialise and dispatch it.
    """
    oq    = omegaq(0.0, ky)
    oy    = Qz / np.sqrt(ky**2 + Qz**2)                   # Oy at kX = 0, ky > 0
    g_pre = 2j * aSC * e * t / hbar * np.sqrt(hbar / (eps0 * V * oq)) * oy

    Z      = hbar * omegak(0.0, ky) + 1j * loss
    gFM_sq = np.abs(gFM(1e-20, ky))**2

    psum = _psum_kernel(px_arr, py_arr, ky,
                        t, aSC, kF, abs(Delta), betahbar,
                        u_arr, v_arr, tanh_p_arr, E_p_arr,
                        g_pre.real, g_pre.imag,
                        Z.real, Z.imag)

    oq_h = hbar * oq
    ok_h = hbar * omegak(0.0, ky)
    prefactor = 2.0 * oq_h**2 / (oq_h**2 - (ok_h + 1j * loss)**2)**2

    return np.real(100.0 * gFM_sq * prefactor * psum * px_step * py_step)


# Cavity-only and analytic approximate self-energies
def Qk_selfEnergy_noSC(kY):
    """Cavity-only magnon self-energy [J] (no SC quasiparticle loop)."""
    return np.real(-100.0 * np.abs(gFM(1e-20, kY))**2
                   * 2.0 / (hbar * omegaq(1e-20, kY) - hbar * omegak(1e-20, kY) - 1j * loss))

def Qk_selfEnergy_onlyk(kY):
    """Analytic estimate keeping only k-dependent prefactors (no p-sum)."""
    return (-100.0 * np.abs(gFM(1e-20, kY))**2
            * (4.0 / (hbar * omegaq(1e-20, kY))**3)
            * np.abs(Oy(1e-20, kY))**2)


#### Main computation: ####

# Warm Numba up on a tiny slice so the JIT compiles before the timed run
_dummy = _psum_kernel(pxy_list[0][:2], pxy_list[1][:2], ky_range[0],
                      t, aSC, kF, abs(Delta), betahbar,
                      u_p0[:2], v_p0[:2], tanh_p0[:2], E_p0[:2],
                      0.0, 1.0, 0.0, 1e-30)
print("Numba kernel compiled and warmed up.\n")

# u_p0, v_p0, E_p0, tanh_p0 are identical across all four quadrants because cos(-p) = cos(p), so ksi(+px, +py) = ksi(-px, -py) = ksi(-px, +py) = ...
quadrant_signs = [(+1, +1), (-1, -1), (-1, +1), (+1, -1)]
QSC_total      = np.zeros(ky_range.size)

for sx, sy in quadrant_signs:
    px_q = sx * pxy_list[0]
    py_q = sy * pxy_list[1]

    results = Parallel(n_jobs = n_jobs, verbose = 1)(
        delayed(_ky_worker)(ky, px_q, py_q, u_p0, v_p0, tanh_p0, E_p0)
        for ky in ky_range
    )
    QSC_total += np.array(results)
    print(f"Quadrant ({'+' if sx > 0 else '-'}px, {'+' if sy > 0 else '-'}py) done.\n")

Qonlyk = Qk_selfEnergy_onlyk(ky_range)
Qcav   = Qk_selfEnergy_noSC(ky_range) / (hbar * 2.0 * pi * 1e6)
QSC    = QSC_total / (hbar * 2.0 * pi * 1e6)


#### Plots: ####


k_idx         = np.argmax(np.abs(QSC))
test = (ESC(pxy_list[0], -pxy_list[1], 0, 0, 0)
        - ESC(pxy_list[0], -pxy_list[1] - ky_range[k_idx], 0, 0, 0))
idx_max = np.argmax(test)

energy_diff   = (ESC(pxy_list[0][idx_max], -pxy_list[1][idx_max], 0, 0, 0)
                 - ESC(pxy_list[0][idx_max], -pxy_list[1][idx_max] - ky_range, 0, 0, 0))

fig, ax = plt.subplots()
ax.scatter(pxy_list[0], -pxy_list[1], s = 1, label = "Fermi surface sample")
ax.scatter(pxy_list[0][idx_max], -pxy_list[1][idx_max],
           zorder = 5, label = "max BQ energy difference")
ax.set_xlabel(r"$p_x$")
ax.set_ylabel(r"$p_y$")
ax.set_title("Fermi-surface sample and point of steepest BQ energy difference")
ax.set_aspect("equal", adjustable = "box")
ax.legend()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
ax.plot(ny_range, hbar * omegak(0.0, ky_range), label = "magnon dispersion")
ax.plot(ny_range, energy_diff, label = "steepest BQ energy difference")
ax.set_xlabel(r"$m_y$")
ax.set_ylabel("Energy [J]")
ax.set_title("Magnon dispersion vs steepest BQ energy difference")
ax.legend()
plt.margins(x = 0)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
ax.plot(ny_range / ky_increase_res, QSC)
ax.set_xlabel(r"$m_y$")
ax.set_ylabel(r"$\Re\,\Sigma_\mathrm{SC}(\lambda_{\mathbf{k}},\mathbf{k})$ [MHz]")
plt.margins(x = 0)
plt.tight_layout()
plt.show()

# fig, ax = plt.subplots()
# ax.plot(ny_range / ky_increase_res,
#         Qonlyk / np.max(np.abs(Qonlyk)) * 5, label = "k-only estimate (scaled)")
# ax.plot(ny_range / ky_increase_res, -np.abs(QSC),
#         label = r"$-|\Re\,\Sigma_\mathrm{SC}|$")
# ax.set_xlabel(r"$m_y$")
# ax.set_ylabel("[MHz]")
# ax.legend()
# plt.margins(x = 0)
# plt.tight_layout()
# plt.show()

fig, ax = plt.subplots()
ax.plot(ny_range / ky_increase_res, Qcav, label = "cavity")
ax.plot(ny_range / ky_increase_res, Qcav + QSC, label = "cavity + SC")
ax.set_xlabel(r"$m_y$")
ax.set_ylabel(r"$\Re\,\Sigma$ [MHz]")
ax.set_title("Magnon energy correction")
ax.legend()
plt.margins(x = 0)
plt.tight_layout()
plt.show()

step     = 100
n_sample = pxy_list[0][::step].size
fig, ax  = plt.subplots()
ax.plot(ny_range / ky_increase_res,
        hbar * omegak(0.0, ky_range) / (hbar * 2.0 * pi * 1e9),
        color = "black", lw = 1.5, label = "magnon dispersion")
for i in range(n_sample):
    px_i  = pxy_list[0][::step][i]
    py_i  = pxy_list[1][::step][i]
    ediff = (ESC(px_i, -py_i, 0, 0, 0)
             - ESC(px_i, -py_i - ky_range, 0, 0, 0)) / (hbar * 2.0 * pi * 1e9)
    ax.plot(ny_range / ky_increase_res, ediff, lw = 5, alpha = 0.005, color = "steelblue")
ax.set_xlabel(r"$m_y$")
ax.set_ylabel("Energy [GHz]")
ax.set_title("Magnon dispersion overlaid with sampled BQ energy differences")
ax.set_ylim([0, 1e-23 / (hbar * 2.0 * pi * 1e9)])
plt.margins(x = 0)
plt.tight_layout()
plt.show()
