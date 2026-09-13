import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import solve_continuous_are

# 1. PHYSICAL PARAMETERS
m   = 1.0       # kg
g   = 9.81      # m/s²
Ixx = 0.01      # kg·m²
Iyy = 0.01
Izz = 0.02

# 2. OCP PARAMETERS
T_final = 5.0       # seconds
N       = 500       # collocation intervals
dt      = T_final / N

# Control bounds
dT_max  = 15.0      # δT ∈ [−15, 15] N
tau_max = 0.5       # τ_i ∈ [−0.5, 0.5] N·m

# Cost matrices
Q_diag = np.array([
    10.0, 10.0, 10.0,    # position
     1.0,  1.0,  1.0,    # velocity
     5.0,  5.0,  1.0,    # Euler angles (roll/pitch > yaw)
     0.1,  0.1,  0.1     # angular rates
])
R_diag = np.array([0.01, 1.0, 1.0, 1.0])
S_diag = 2.0 * Q_diag    # S = 2Q

# Initial condition
x0_np = np.array([
    1.0,  0.5,  2.0,     # position
    0.0,  0.0,  0.0,     # velocity
    0.1,  0.05, 0.0,     # Euler angles
    0.0,  0.0,  0.0      # angular rates
])

# 3. CASADI SYMBOLIC DYNAMICS
x_sym = ca.MX.sym('x', 12)
u_sym = ca.MX.sym('u', 4)

# Unpack state
vx, vy, vz       = x_sym[3], x_sym[4], x_sym[5]
phi, theta, psi   = x_sym[6], x_sym[7], x_sym[8]
p_b, q_b, r_b     = x_sym[9], x_sym[10], x_sym[11]

# Unpack control: [δT, τ_x, τ_y, τ_z]
delta_T = u_sym[0]
tau_x   = u_sym[1]
tau_y   = u_sym[2]
tau_z   = u_sym[3]

T_total = m * g + delta_T    # total thrust
cphi   = ca.cos(phi);   sphi   = ca.sin(phi)
ctheta = ca.cos(theta); stheta = ca.sin(theta)
cpsi   = ca.cos(psi);   spsi   = ca.sin(psi)

# --- Eq 1: ṗ = v ---
px_dot = vx
py_dot = vy
pz_dot = vz

# --- Eq 2: v̇ = [0,0,−g] + (1/m) R(η) [0,0,T] ---
# ZYX rotation: third column of R = Rz(ψ) Ry(θ) Rx(φ)
thrust_x = (cpsi * stheta * cphi + spsi * sphi) * T_total / m
thrust_y = (spsi * stheta * cphi - cpsi * sphi) * T_total / m
thrust_z = (ctheta * cphi) * T_total / m

vx_dot = thrust_x
vy_dot = thrust_y
vz_dot = -g + thrust_z

# --- Eq 3: η̇ = E(η) ε ---
tan_theta = ca.tan(theta)
sec_theta = 1.0 / ctheta

phi_dot   = p_b + sphi * tan_theta * q_b + cphi * tan_theta * r_b
theta_dot = cphi * q_b - sphi * r_b
psi_dot   = sphi * sec_theta * q_b + cphi * sec_theta * r_b

# --- Eq 4: ε̇ = I⁻¹ (τ − ε × Iε) ---
p_dot = (1.0 / Ixx) * (tau_x - (Izz - Iyy) * q_b * r_b)
q_dot = (1.0 / Iyy) * (tau_y - (Ixx - Izz) * p_b * r_b)
r_dot = (1.0 / Izz) * (tau_z - (Iyy - Ixx) * p_b * q_b)

# Full dynamics vector
x_dot = ca.vertcat(
    px_dot, py_dot, pz_dot,
    vx_dot, vy_dot, vz_dot,
    phi_dot, theta_dot, psi_dot,
    p_dot, q_dot, r_dot
)
f = ca.Function('f', [x_sym, u_sym], [x_dot], ['x', 'u'], ['xdot'])

# 4. RUNNING COST
#    L = xᵀQx + uᵀRu   (u = [δT, τ_x, τ_y, τ_z])
Q_sym = ca.diag(ca.DM(Q_diag))
R_sym = ca.diag(ca.DM(R_diag))

L = ca.bilin(Q_sym, x_sym, x_sym) + ca.bilin(R_sym, u_sym, u_sym)
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

X = opti.variable(12, N + 1)    # states at each node
U = opti.variable(4, N)         # controls (piecewise constant)

# --- Objective ---
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

# Terminal cost: xᵀ S x  where S = 2Q
S_sym = ca.diag(ca.DM(S_diag))
J += ca.bilin(S_sym, X[:, -1], X[:, -1])

opti.minimize(J)

# --- Dynamics constraints ---
for k in range(N):
    x_next = rk4_step(f, X[:, k], U[:, k], dt)
    opti.subject_to(X[:, k + 1] == x_next)

# --- Initial condition ---
opti.subject_to(X[:, 0] == x0_np)

# --- Control bounds ---
opti.subject_to(opti.bounded(-dT_max, U[0, :], dT_max))      # δT
opti.subject_to(opti.bounded(-tau_max, U[1, :], tau_max))     # τ_x
opti.subject_to(opti.bounded(-tau_max, U[2, :], tau_max))     # τ_y
opti.subject_to(opti.bounded(-tau_max, U[3, :], tau_max))     # τ_z

# --- Euler angle bounds ---
opti.subject_to(opti.bounded(-1.4, X[6, :], 1.4))    # φ
opti.subject_to(opti.bounded(-1.4, X[7, :], 1.4))    # θ

# 7. WARM START: LQR trajectory
# Numerical linearization at hover (x=0, u=0 since u is deviation)
x_hover = np.zeros(12)
u_hover = np.zeros(4)
eps_fd = 1e-7

A = np.zeros((12, 12))
for j in range(12):
    xp = x_hover.copy(); xp[j] += eps_fd
    xm = x_hover.copy(); xm[j] -= eps_fd
    A[:, j] = (np.array(f(xp, u_hover)).flatten() -
               np.array(f(xm, u_hover)).flatten()) / (2 * eps_fd)

B = np.zeros((12, 4))
for j in range(4):
    up = u_hover.copy(); up[j] += eps_fd
    um = u_hover.copy(); um[j] -= eps_fd
    B[:, j] = (np.array(f(x_hover, up)).flatten() -
               np.array(f(x_hover, um)).flatten()) / (2 * eps_fd)

P_lqr = solve_continuous_are(A, B, 2 * np.diag(Q_diag), 2 * np.diag(R_diag))
K_lqr = np.linalg.inv(np.diag(R_diag)) @ B.T @ P_lqr

print("Generating LQR warmstart trajectory...")
x_init = np.zeros((12, N + 1))
u_init = np.zeros((4, N))
x_init[:, 0] = x0_np

for k in range(N):
    x_k = x_init[:, k]
    u_k = -K_lqr @ x_k
    u_k[0] = np.clip(u_k[0], -dT_max, dT_max)
    u_k[1:4] = np.clip(u_k[1:4], -tau_max, tau_max)
    u_init[:, k] = u_k

    k1 = np.array(f(x_k, u_k)).flatten()
    k2 = np.array(f(x_k + dt/2*k1, u_k)).flatten()
    k3 = np.array(f(x_k + dt/2*k2, u_k)).flatten()
    k4 = np.array(f(x_k + dt*k3, u_k)).flatten()
    x_init[:, k + 1] = x_k + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

print(f"  LQR final pos: [{x_init[0,-1]:.4f}, {x_init[1,-1]:.4f}, {x_init[2,-1]:.4f}]")

opti.set_initial(X, x_init)
opti.set_initial(U, u_init)

# 8. SOLVE WITH IPOPT
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
print("  Parameters matched to ROBO_NN.py")
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

# 9. EXTRACT SOLUTION
X_sol = np.array(sol.value(X))
U_sol = np.array(sol.value(U))
J_sol = float(sol.value(J))
t_grid = np.linspace(0, T_final, N + 1)
t_ctrl = np.linspace(0, T_final - dt, N)

# Decompose cost (trapezoidal approximation for display)
J_running = 0.0
for k in range(N):
    x_k = X_sol[:, k]
    u_k = U_sol[:, k]
    J_running += dt * (x_k @ np.diag(Q_diag) @ x_k + u_k @ np.diag(R_diag) @ u_k)
xT = X_sol[:, -1]
J_terminal = xT @ np.diag(S_diag) @ xT

print("\n" + "=" * 80)
print("    QUADROTOR 12D — GROUND TRUTH (CasADi / IPOPT)")
print("=" * 80)
print(f"  Initial state: p₀ = {x0_np[:3].tolist()},  η₀ = {x0_np[6:9].tolist()}")
print(f"  Horizon: {T_final}s,  N = {N},  dt = {dt:.4f}s")
print("-" * 80)
print(f"  Final position:  [{xT[0]:.6f}, {xT[1]:.6f}, {xT[2]:.6f}] m")
print(f"  Final velocity:  [{xT[3]:.6f}, {xT[4]:.6f}, {xT[5]:.6f}] m/s")
print(f"  Final angles:    [{np.degrees(xT[6]):.4f}°, {np.degrees(xT[7]):.4f}°, {np.degrees(xT[8]):.4f}°]")
print(f"  Final ang rates: [{xT[9]:.6f}, {xT[10]:.6f}, {xT[11]:.6f}] rad/s")
print("-" * 80)
print(f"  J_run      = {J_running:.6f}")
print(f"  J_terminal = {J_terminal:.6f}")
print(f"  J* (IPOPT) = {J_sol:.6f}")
print("-" * 80)
print(f"  Solve time: {solve_time:.1f}s")
print("=" * 80)

# 10. SAVE BENCHMARK DATA
np.savez('quadrotor_ground_truth.npz',
         t_state=t_grid, t_ctrl=t_ctrl,
         X=X_sol, U=U_sol,
         J_total=J_sol, J_running=J_running, J_terminal=J_terminal,
         Q_diag=Q_diag, S_diag=S_diag, R_diag=R_diag,
         x0=x0_np, T_final=T_final,
         params=np.array([m, g, Ixx, Iyy, Izz, dT_max, tau_max]))
print("\nBenchmark saved to quadrotor_ground_truth.npz")

# 11. PLOTS  (4-panel layout matching CasADi convention)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Position
ax = axes[0, 0]
ax.plot(t_grid, X_sol[0], lw=2, label=r'$p_x$')
ax.plot(t_grid, X_sol[1], lw=2, label=r'$p_y$')
ax.plot(t_grid, X_sol[2], lw=2, label=r'$p_z$')
ax.axhline(0, ls=':', color='k', alpha=0.4)
ax.set_title('State dynamics - CasADI', fontsize=11)
ax.set_xlabel('Time [s]');  ax.set_ylabel('[m]')
ax.legend();  ax.grid(True, alpha=0.3)

# Euler Angles
ax = axes[0, 1]
ax.plot(t_grid, np.degrees(X_sol[6]), lw=2, label=r'$\phi$ (roll)')
ax.plot(t_grid, np.degrees(X_sol[7]), lw=2, label=r'$\theta$ (pitch)')
ax.plot(t_grid, np.degrees(X_sol[8]), lw=2, label=r'$\psi$ (yaw)')
ax.axhline(0, ls=':', color='k', alpha=0.4)
ax.set_title('State angles - CasADI', fontsize=11)
ax.set_xlabel('Time [s]');  ax.set_ylabel('[deg]')
ax.legend();  ax.grid(True, alpha=0.3)

# Thrust (as δT)
ax = axes[1, 0]
ax.step(t_ctrl, U_sol[0], lw=2, color='tab:red', where='post', label=r'$\delta T$')
ax.axhline(0, ls=':', color='k', alpha=0.7, label='Hover (δT=0)')
ax.axhline(dT_max, ls='--', color='grey', alpha=0.5, label=r'$\pm\delta T_{max}$')
ax.axhline(-dT_max, ls='--', color='grey', alpha=0.5)
ax.set_title('Control dynamics - CasADI', fontsize=11)
ax.set_xlabel('Time [s]');  ax.set_ylabel('[N]')
ax.legend();  ax.grid(True, alpha=0.3)

# Torques
ax = axes[1, 1]
ax.step(t_ctrl, U_sol[1], lw=2, where='post', label=r'$\tau_x$')
ax.step(t_ctrl, U_sol[2], lw=2, where='post', label=r'$\tau_y$')
ax.step(t_ctrl, U_sol[3], lw=2, where='post', label=r'$\tau_z$')
ax.axhline( tau_max, ls='--', color='grey', alpha=0.5)
ax.axhline(-tau_max, ls='--', color='grey', alpha=0.5, label=r'$\pm\tau_{max}$')
ax.set_title('Control torques - CasADI', fontsize=11)
ax.set_xlabel('Time [s]');  ax.set_ylabel('[N·m]')
ax.legend();  ax.grid(True, alpha=0.3)

plt.suptitle(f'Quadrotor optimal control — CasADI',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('quadrotor_ground_truth.png', dpi=300, bbox_inches='tight')
print("Plot saved to quadrotor_ground_truth.png")
plt.show()