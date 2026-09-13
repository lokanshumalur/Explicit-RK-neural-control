import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

# 1. PK-PD MODEL & OCP PARAMETERS
lambda_0 = 0.146
lambda_1 = 0.334
k_1 = 0.469
k_2 = 8.42e-4
Psi = 20.0

C_T_MAX_CONC = 173.4
U_MAX = k_2 * C_T_MAX_CONC

T = 27.0
dt = 0.001
N = int(T / dt)

Q = np.diag([1.2, 0.10, 0.10, 0.10])
S = 0.0075 * np.eye(4)
R_original_cost_on_c = 9.5e-4
R = R_original_cost_on_c / (k_2 ** 2)

t_treatment_start = 13.0
w_at_treatment_start = 0.5684
x0 = np.array([w_at_treatment_start, 0.0, 0.0, 0.0])

# 2. PMP IMPLEMENTATION
def u_star_pkpd(x, p):
    x1 = x[0]
    p1, p2 = p[0], p[1]
    u_free = (x1) / (2 * R) * (p1 - p2)
    return float(np.clip(u_free, 0.0, U_MAX))

def ode_pmp_pkpd(z):
    x, p = z[:4], z[4:]
    x1, x2, x3, x4 = x
    u = u_star_pkpd(x, p)

    w = max(np.sum(x), 1e-12)
    V = 1 + (lambda_0 / lambda_1 * w) ** Psi
    G = V ** (1 / Psi)
    growth_rate = (lambda_0 / G) * x1

    x_dot = np.zeros(4)
    x_dot[0] = growth_rate - u * x1
    x_dot[1] = u * x1 - k_1 * x2
    x_dot[2] = k_1 * (x2 - x3)
    x_dot[3] = k_1 * (x3 - x4)

    J_f = np.zeros((4, 4))
    dG_dw = (V ** ((1 - Psi) / Psi)) * ((lambda_0 / lambda_1 * w) ** (Psi - 1)) * (lambda_0 / lambda_1)
    df_growth_dw = -lambda_0 * x1 / (G ** 2) * dG_dw
    J_f[0, :] = df_growth_dw

    J_f[0, 0] += (lambda_0 / G) - u
    J_f[1, 0] = u
    J_f[1, 1] = -k_1
    J_f[2, 1] = k_1
    J_f[2, 2] = -k_1
    J_f[3, 2] = k_1
    J_f[3, 3] = -k_1

    dldx = 2.0 * (Q @ x)
    p_dot = -dldx - J_f.T @ p

    return np.hstack([x_dot, p_dot])

def Lagrangian(x, p):
    u = u_star_pkpd(x, p)
    return (x @ Q @ x) + R * u ** 2

def rk4_integrate(z0, N, dt):
    traj = np.zeros((len(z0), N + 1))
    traj[:, 0] = z0
    J_running = 0.0

    for i in range(N):
        z = traj[:, i]

        k1 = ode_pmp_pkpd(z)
        k2 = ode_pmp_pkpd(z + 0.5 * dt * k1)
        k3 = ode_pmp_pkpd(z + 0.5 * dt * k2)
        k4 = ode_pmp_pkpd(z + dt * k3)
        traj[:, i + 1] = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        L1 = Lagrangian(z[:4], z[4:])
        L2 = Lagrangian((z + 0.5 * dt * k1)[:4], (z + 0.5 * dt * k1)[4:])
        L3 = Lagrangian((z + 0.5 * dt * k2)[:4], (z + 0.5 * dt * k2)[4:])
        L4 = Lagrangian((z + dt * k3)[:4], (z + dt * k3)[4:])
        J_running += (dt / 6.0) * (L1 + 2 * L2 + 2 * L3 + L4)

    return traj, J_running

def shoot_residual_pkpd(p0_guess):
    z0 = np.hstack([x0, p0_guess])
    traj, _ = rk4_integrate(z0, N, dt)
    xT, pT = traj[:4, -1], traj[4:, -1]
    return pT - 2.0 * (S @ xT)

# 3. SHOOTING + INTEGRATION
print("Solving for optimal p(0)...")
p0_init = 2.0 * (S @ x0)
sol_root = root(shoot_residual_pkpd, p0_init, method='hybr',
                tol=1e-10, options={'maxfev': 10000})
if not sol_root.success:
    print("Warning: Shooting failed to converge.", sol_root.message)
p0_star = sol_root.x

z0_star = np.hstack([x0, p0_star])
traj, J_running = rk4_integrate(z0_star, N, dt)

x_traj, p_traj = traj[:4, :], traj[4:, :]
t = np.linspace(0, T, N + 1) + t_treatment_start
u_traj = np.array([u_star_pkpd(x_traj[:, i], p_traj[:, i]) for i in range(N + 1)])
w_traj = np.sum(x_traj, axis=0)

xT_final = x_traj[:, -1]
J_terminal = xT_final @ S @ xT_final
J_total = J_running + J_terminal

res_T = p_traj[:, -1] - 2.0 * (S @ x_traj[:, -1])

# 4. RESULTS & PLOTS
print("\n=== Key Summary (PMP Benchmark) ===")
print(f"Final tumor w(T): {w_traj[-1]:.4f} g")
print(f"AUC_w: {np.trapz(w_traj, np.linspace(0, T, N + 1)):.4f} g·day")
print(f"Total control effort ∫u dt: {np.trapz(u_traj, np.linspace(0, T, N + 1)):.2f} days⁻¹")
print(f"Peak u_max: {np.max(u_traj):.4f} days⁻¹ at day {t[np.argmax(u_traj)]:.2f}")
print(f"J_run={J_running:.4f}, J_term={J_terminal:.4f}, J_total={J_total:.4f}")
print("\n--- PMP Diagnostics ---")
print("‖p(T) - 2 S x(T)‖₂ =", np.linalg.norm(res_T))
print("Shooting residual norm =", np.linalg.norm(sol_root.fun))
print("--- End Diagnostics ---")

# Tumor Trajectory Plot
fig1, ax1 = plt.subplots(figsize=(10, 7))
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

ax1.plot(t, x_traj[0, :], color=color_cycle[0], linewidth=3, alpha=0.9,
         label=r"$\mathit{x₁}\ \mathit{(proliferating\ cells)}$")
ax1.plot(t, w_traj, color=color_cycle[1], linestyle='--', linewidth=3, alpha=0.9,
         label=r"$\mathit{m}\ \mathit{(total\ tumor)}$")
ax1.set_xlabel('Time (days)', fontsize=16)
ax1.set_ylabel('Tumor Mass', fontsize=16)
ax1.set_title('Tumor dynamics: Pontryagin', fontsize=18, fontweight='bold', pad=20)
ax1.legend(fontsize=12, loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 5])
plt.tight_layout()
plt.show()

# Optimal Control Plot
c_traj = u_traj / k_2

fig2, ax2 = plt.subplots(figsize=(10, 7))
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

ax2.plot(t, c_traj, color=color_cycle[2], linewidth=3, alpha=0.9, label=r"$\mathit{u(t)}$")
ax2.axhline(C_T_MAX_CONC, ls=":", color="k", linewidth=2, alpha=0.7, label=r"$\mathit{Cₜ\ cap}$")
ax2.set_xlabel('Time (days)', fontsize=16)
ax2.set_ylabel('Drug Concentration', fontsize=16)
ax2.set_title('Control policy: Pontryagin', fontsize=18, fontweight='bold', pad=20)
ax2.legend(fontsize=12, loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 180])
plt.tight_layout()
plt.show()
