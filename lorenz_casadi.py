import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import time

# 1. LORENZ-96 PARAMETERS
N_L   = 20       # number of Lorenz-96 variables
F_L   = 8.0      # forcing constant (chaotic regime)

# Reference state: unstable equilibrium x_i* = F
x_ref = F_L * np.ones(N_L)

# 2. OCP PARAMETERS
T_final = 2.0        # Lorenz time units (~3 Lyapunov times)
N       = 400        # collocation intervals
dt      = T_final / N

# Control bounds
u_max = 10.0         # u_i ∈ [−10, 10]

# Cost matrices
Q_diag = 1.0 * np.ones(N_L)     # Q = I
R_diag = 0.01 * np.ones(N_L)    # R = 0.01 I
S_diag = 5.0 * np.ones(N_L)     # S = 5 I

# Initial condition
perturbation = 0.5 * np.sin(2.0 * np.pi * np.arange(N_L) / N_L)
x0_np = F_L + perturbation       # x_i(0) = F + 0.5 sin(2π i/N)

# 3. CASADI SYMBOLIC DYNAMICS
x_sym = ca.MX.sym('x', N_L)
u_sym = ca.MX.sym('u', N_L)

# dx_i/dt = (x_{i+1} − x_{i-2}) x_{i-1} − x_i + F + u_i
x_ip1 = ca.vertcat(x_sym[1:], x_sym[0])                    # x_{i+1}  (shift -1)
x_im1 = ca.vertcat(x_sym[-1], x_sym[:-1])                   # x_{i-1}  (shift +1)
x_im2 = ca.vertcat(x_sym[-2:], x_sym[:-2])                  # x_{i-2}  (shift +2)
x_dot = (x_ip1 - x_im2) * x_im1 - x_sym + F_L + u_sym

f = ca.Function('f', [x_sym, u_sym], [x_dot], ['x', 'u'], ['xdot'])

# 4. RUNNING COST  (matched to RunningCost, NN lines 153-163)
#    L = (x−x*)ᵀ Q (x−x*)  +  uᵀ R u
Q_sym = ca.diag(ca.DM(Q_diag))
R_sym = ca.diag(ca.DM(R_diag))
x_ref_sym = ca.DM(x_ref)
e_sym = x_sym - x_ref_sym
L = ca.bilin(Q_sym, e_sym, e_sym) + ca.bilin(R_sym, u_sym, u_sym)
l_func = ca.Function('l', [x_sym, u_sym], [L])

# 5. RK4 INTEGRATOR STEP
def rk4_step(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + dt/2 * k1, u)
    k3 = f(x + dt/2 * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

# 6.  NLP VIA CASADI OPTI
opti = ca.Opti()

X = opti.variable(N_L, N + 1)    # states at each node  (20 × 401)
U = opti.variable(N_L, N)        # controls (piecewise constant, 20 × 400)

J = 0
for k in range(N):
    x_k = X[:, k]
    u_k = U[:, k]

    # Running cost via RK4 quadrature
    k1_x = f(x_k, u_k)
    x_mid1 = x_k + dt/2 * k1_x
    k2_x = f(x_mid1, u_k)
    x_mid2 = x_k + dt/2 * k2_x
    k3_x = f(x_mid2, u_k)
    x_end = x_k + dt * k3_x

    L1 = l_func(x_k, u_k)
    L2 = l_func(x_mid1, u_k)
    L3 = l_func(x_mid2, u_k)
    L4 = l_func(x_end, u_k)
    J += (dt / 6) * (L1 + 2*L2 + 2*L3 + L4)

# Terminal cost: (x(T)−x*)ᵀ S (x(T)−x*)
S_sym = ca.diag(ca.DM(S_diag))
e_T = X[:, -1] - x_ref_sym
J += ca.bilin(S_sym, e_T, e_T)

opti.minimize(J)

for k in range(N):
    x_next = rk4_step(f, X[:, k], U[:, k], dt)
    opti.subject_to(X[:, k + 1] == x_next)

# --- Initial condition ---
opti.subject_to(X[:, 0] == x0_np)

# --- Control bounds ---
for i in range(N_L):
    opti.subject_to(opti.bounded(-u_max, U[i, :], u_max))

# 7. WARM START: Uncontrolled trajectory → zero control
print("Generating warm-start trajectory (zero control)...")
x_init = np.zeros((N_L, N + 1))
u_init = np.zeros((N_L, N))
x_init[:, 0] = x0_np

for k in range(N):
    x_k = x_init[:, k]
    u_k = np.zeros(N_L)

    k1 = np.array(f(x_k, u_k)).flatten()
    k2 = np.array(f(x_k + dt/2*k1, u_k)).flatten()
    k3 = np.array(f(x_k + dt/2*k2, u_k)).flatten()
    k4 = np.array(f(x_k + dt*k3, u_k)).flatten()
    x_init[:, k + 1] = x_k + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

for k in range(N):
    u_init[:, k] = np.clip(-2.0 * (x_init[:, k] - x_ref), -u_max, u_max)

x_init[:, 0] = x0_np
for k in range(N):
    x_k = x_init[:, k]
    u_k = u_init[:, k]

    k1 = np.array(f(x_k, u_k)).flatten()
    k2 = np.array(f(x_k + dt/2*k1, u_k)).flatten()
    k3 = np.array(f(x_k + dt/2*k2, u_k)).flatten()
    k4 = np.array(f(x_k + dt*k3, u_k)).flatten()
    x_init[:, k + 1] = x_k + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

dev_init = np.linalg.norm(x_init[:, -1] - x_ref)
print(f"  Warm-start final ||x_T − x*|| = {dev_init:.4f}")

opti.set_initial(X, x_init)
opti.set_initial(U, u_init)

# 8. SOLVER
opts = {
    'ipopt.max_iter': 5000,
    'ipopt.tol': 1e-8,
    'ipopt.acceptable_tol': 1e-6,
    'ipopt.linear_solver': 'mumps',
    'ipopt.print_level': 5,
    'ipopt.sb': 'yes',
}
opti.solver('ipopt', opts)

print("\n" + "=" * 60)
print("  Solving OCP with IPOPT (direct transcription, RK4)")
print("  Parameters matched to LORENZ_NN.py")
print("=" * 60)
t0 = time.time()
try:
    sol = opti.solve()
    solve_time = time.time() - t0
    converged = True
    print(f"\n  ✓ IPOPT converged in {solve_time:.1f}s")
except RuntimeError as e:
    solve_time = time.time() - t0
    converged = False
    sol = opti.debug
    print(f"\n  ⚠ IPOPT failed ({solve_time:.1f}s): {e}")

# 9.  SOLUTION
X_sol = np.array(sol.value(X))
U_sol = np.array(sol.value(U))
J_sol = float(sol.value(J))
t_grid = np.linspace(0, T_final, N + 1)
t_ctrl = np.linspace(0, T_final - dt, N)

J_running = 0.0
for k in range(N):
    x_k = X_sol[:, k]
    u_k = U_sol[:, k]
    e_k = x_k - x_ref
    J_running += dt * (e_k @ np.diag(Q_diag) @ e_k + u_k @ np.diag(R_diag) @ u_k)

e_T = X_sol[:, -1] - x_ref
J_terminal = e_T @ np.diag(S_diag) @ e_T

dev_final = np.linalg.norm(X_sol[:, -1] - x_ref)
dev_init  = np.linalg.norm(x0_np - x_ref)

print("\n" + "=" * 80)
print("    LORENZ-96 (20D) — GROUND TRUTH (CasADi / IPOPT)")
print("=" * 80)
print(f"  Forcing F = {F_L},  N_states = {N_L},  Equilibrium: x* = F = {F_L}")
print(f"  Horizon: {T_final} time units,  N = {N},  dt = {dt:.5f}")
print(f"  Initial perturbation ||x₀ − x*|| = {dev_init:.6f}")
print("-" * 80)
print(f"  Final ||x_T − x*||:  {dev_final:.8f}")
print(f"  Final x_T (first 5): [{', '.join(f'{v:.6f}' for v in X_sol[:5, -1])}]")
print(f"  Reference  (first 5): [{', '.join(f'{v:.6f}' for v in x_ref[:5])}]")
print("-" * 80)
print(f"  J_running  = {J_running:.6f}")
print(f"  J_terminal = {J_terminal:.6f}")
print(f"  J* (IPOPT) = {J_sol:.6f}")
print("-" * 80)
print(f"  Solve time: {solve_time:.1f}s")
print("=" * 80)

# 10. SAVE BENCHMARK DATA
np.savez('lorenz96_ground_truth.npz',
         t_state=t_grid, t_ctrl=t_ctrl,
         X=X_sol, U=U_sol,
         J_total=J_sol, J_running=J_running, J_terminal=J_terminal,
         Q_diag=Q_diag, S_diag=S_diag, R_diag=R_diag,
         x0=x0_np, x_ref=x_ref,
         T_final=T_final, N_L=N_L, F_L=F_L, u_max=u_max)
print("\nBenchmark saved to lorenz96_ground_truth.npz")

# 11. PLOTS
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# --- Top-left: State deviation heatmap (Hovmöller) ---
ax = axes[0, 0]
dev_data = X_sol - x_ref.reshape(-1, 1)
vmax = np.max(np.abs(dev_data))
im = ax.imshow(
    dev_data,
    aspect='auto', origin='lower',
    extent=[0, T_final, 0.5, N_L + 0.5],
    cmap='RdBu_r', interpolation='bilinear',
    vmin=-vmax, vmax=vmax
)
ax.set_xlabel('Time');  ax.set_ylabel('State index $i$')
ax.set_title('State deviation - CasADI', fontsize=11)
fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

# --- Top-right: Selected state traces ---
ax = axes[0, 1]
for idx in [0, 4, 9, 14, 19]:
    ax.plot(t_grid, X_sol[idx, :], lw=1.5, label=f'$x_{{{idx+1}}}$')
ax.axhline(F_L, ls=':', color='k', alpha=0.6, label=f'$x^* = {F_L}$')
ax.set_xlabel('Time');  ax.set_ylabel('$x_i$')
ax.set_title('State traces - CasADI', fontsize=11)
ax.legend(fontsize=9, ncol=3);  ax.grid(True, alpha=0.3)

# --- Bottom-left: Control heatmap ---
ax = axes[1, 0]
vmax_u = np.max(np.abs(U_sol))
im = ax.imshow(
    U_sol,
    aspect='auto', origin='lower',
    extent=[0, T_final - dt, 0.5, N_L + 0.5],
    cmap='PiYG', interpolation='bilinear',
    vmin=-vmax_u, vmax=vmax_u
)
ax.set_xlabel('Time');  ax.set_ylabel('Control index $i$')
ax.set_title('Control dynamics - CasADI', fontsize=11)
fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

# --- Bottom-right: Selected control traces ---
ax = axes[1, 1]
for idx in [0, 4, 9, 14, 19]:
    ax.step(t_ctrl, U_sol[idx, :], lw=1.5, where='post', label=f'$u_{{{idx+1}}}$')
ax.axhline(0, ls=':', color='k', alpha=0.4)
ax.axhline(u_max, ls='--', color='grey', alpha=0.5, label=f'$\\pm u_{{max}}$')
ax.axhline(-u_max, ls='--', color='grey', alpha=0.5)
ax.set_xlabel('Time');  ax.set_ylabel('$u_i$')
ax.set_title('Control traces - CasADI', fontsize=11)
ax.legend(fontsize=9, ncol=3);  ax.grid(True, alpha=0.3)

plt.suptitle(
    f'Lorenz-96 optimal control — CasADI',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig('lorenz96_ground_truth.png', dpi=300, bbox_inches='tight')
print("Plot saved to lorenz96_ground_truth.png")
plt.show()