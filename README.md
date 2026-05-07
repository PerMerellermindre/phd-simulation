# Magnon Self-Energy from Coupled Superconductor–Cavity–Ferromagnet System

Numerical simulation of the magnon self-energy arising from the coupling of a ferromagnetic insulator and a superconductor inside an electromagnetic cavity. The code computes the corrections to the magnetic quasiparticle (magnon) energy spectra Re[Σ(λ_**k**, **k**)] as a function of the magnon wavevector **k** as well as a few other quantities, capturing contributions from both the superconductor quasiparticles (Cooper pairs) and cavity photons.

The field is theoretical quantum condensed matter physics; the methods are general-purpose scientific computing and are the main point of this repository.

---

## Numerical methods

### 1. Configuration management (`params.yaml`)

All physical and numerical parameters are stored in a YAML file and loaded at runtime, fully separating configuration from code. This makes parameter sweeps reproducible, enables scripted batch runs, and allows downstream tools in other languages to consume the same configuration without parsing Python.

### 2. Vectorised Brillouin-zone sampling

The Fermi-surface constraint

$$|\cos(p_x a) + \cos(p_y a) - \cos(k_F a)| < \epsilon$$

is evaluated over the full 2D momentum grid using `np.meshgrid` and a single NumPy boolean mask. This pattern prevents a computational bottleneck at scale due to explicit element-by-element iteration.

### 3. Pre-computation of loop-invariant quantities

Significant computation time is saved by identifying and pre-computing static quantites in loops. For instance, here the Bogoliubov coherence factors u(**p**), v(**p**) and the Cooper pair energies E(**p**) at zero momentum depend only on a static momentum grid, not on the outer loop variable **k**. They are computed once before the main loop and passed as read-only arrays into the kernel. Additionally, the even symmetry of the tight-binding dispersion (cos(−p) = cos(p)) means a single set of cached arrays serves all four Brillouin-zone quadrants.

### 4. JIT compilation of the inner kernel (Numba)

The **k**-space loop is compiled to native machine code with `@numba.njit`, saving a significant computation time over running regular Python code.

### 5. Coarse-grained parallelization (joblib)

The **k**-space loop is furthermore an obvious candidate for parallelized computation. It is distributed across all available CPU cores using `joblib.Parallel` with `n_jobs = −1`. The number of workers is controlled via `params.yaml`, making it straightforward to tune for shared cluster nodes.

### 6. Additional comment on two-level parallelism and oversubscription

Numba supports internal thread-level parallelism (`parallel=True`, `prange`). To avoid competing with joblib for resource requests (oversubscription), a design choice has been made such that joblib is assigned to parallelize the computation across the cores, while Numba runs serially but fast inside each worker.

---

## Dependencies

```
numpy
scipy
matplotlib
pyyaml
numba
joblib
```

Install with:

```bash
pip install numpy scipy matplotlib pyyaml numba joblib
```