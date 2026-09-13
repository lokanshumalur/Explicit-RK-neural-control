import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci_int
import random

# 0. TENSORFLOW-METAL / GPU & PRECISION SETUP
RANDOM_SEED = 222
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

if tf.keras.backend.floatx() != 'float64':
    tf.keras.backend.set_floatx('float64')

DTYPE_TF = tf.keras.backend.floatx()
DTYPE_NP = np.float64
print(f"INFO: TensorFlow precision set to {DTYPE_TF}.")

try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("INFO: GPU found. Configuring memory growth.")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("WARNING: No GPU found. Running on CPU.")
except Exception as e:
    print(f"ERROR: GPU setup failed. Running on CPU. Details: {e}")

# 1. BUTCHER TABLEAUS
BUTCHER_TABLEAUS = {
    "euler": {
        "A": np.array([[0.0]], dtype=np.float64),
        "b": np.array([1.0], dtype=np.float64),
        "c": np.array([0.0], dtype=np.float64)
    },
    "midpoint": {
        "A": np.array([[0.0, 0.0],
                       [0.5, 0.0]], dtype=np.float64),
        "b": np.array([0.0, 1.0], dtype=np.float64),
        "c": np.array([0.0, 0.5], dtype=np.float64)
    },
    "heun": {
        "A": np.array([[0.0, 0.0],
                       [1.0, 0.0]], dtype=np.float64),
        "b": np.array([0.5, 0.5], dtype=np.float64),
        "c": np.array([0.0, 1.0], dtype=np.float64)
    },
    "ralston": {
        "A": np.array([[0.0, 0.0],
                       [2 / 3, 0.0]], dtype=np.float64),
        "b": np.array([0.25, 0.75], dtype=np.float64),
        "c": np.array([0.0, 2 / 3], dtype=np.float64)
    },
    "rk4": {
        "A": np.array([[0, 0, 0, 0],
                       [0.5, 0, 0, 0],
                       [0, 0.5, 0, 0],
                       [0, 0, 1, 0]], dtype=np.float64),
        "b": np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6], dtype=np.float64),
        "c": np.array([0, 0.5, 0.5, 1], dtype=np.float64)
    }
}

# 2. MLP
class MLP(tf.keras.Model):
    def __init__(self, input_dim, hidden_layers, activations, output_dim=None, u_max=None):
        super(MLP, self).__init__()
        self.output_dim = input_dim if output_dim is None else output_dim
        self.input_dim = input_dim

        if len(hidden_layers) != len(activations):
            raise ValueError("hiddenlayers must match activations")

        self.u_max = tf.constant(u_max, dtype=tf.float64) if u_max is not None and output_dim == 1 else None

        self.hidden_layers = []
        for units, act in zip(hidden_layers, activations):
            self.hidden_layers.append(
                tf.keras.layers.Dense(
                    units,
                    activation=getattr(tf.nn, act),
                    dtype=tf.float64,
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
                    bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
                )
            )
        self.output_layer = tf.keras.layers.Dense(
            self.output_dim,
            activation=None,
            dtype=tf.float64,
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
            bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
        )

    def call(self, state_t_x):
        x = state_t_x
        for layer in self.hidden_layers:
            x = layer(x)
        raw_output = self.output_layer(x)

        if self.u_max is not None:
            return self.u_max * tf.nn.sigmoid(raw_output)
        return raw_output

# 3. PK-PD Simeoni Dynamics
class PKPDDynamics:
    def __init__(self, lambda0=0.146, lambda1=0.334, k1=0.469, k2=8.42e-4, Psi=20.0):
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.k1 = k1
        self.k2 = k2
        self.Psi = Psi

    def __call__(self, x, u, t, g=None):
        x1, x2, x3, x4 = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
        w = tf.reduce_sum(x, axis=1, keepdims=True)

        V = 1.0 + (self.lambda0 / self.lambda1 * w) ** self.Psi
        G = V ** (1.0 / self.Psi)
        growth_rate = (self.lambda0 / G) * x1

        x1_dot = growth_rate - self.k2 * u * x1
        x2_dot = self.k2 * u * x1 - self.k1 * x2
        x3_dot = self.k1 * (x2 - x3)
        x4_dot = self.k1 * (x3 - x4)

        return tf.concat([x1_dot, x2_dot, x3_dot, x4_dot], axis=1)

# 4. Costs
class RunningCost:
    def __init__(self, Q, R):
        self.Q = tf.constant(Q, dtype=tf.float64)
        self.R = tf.constant(R, dtype=tf.float64)

    def __call__(self, x, u, t=None):
        u_col = tf.reshape(u, [-1, 1])
        state_cost = tf.reduce_sum((x @ self.Q) * x, axis=1, keepdims=True)
        control_cost = tf.reduce_sum((u_col @ self.R) * u_col, axis=1, keepdims=True)
        return state_cost + control_cost

class TerminalCost:
    def __init__(self, S):
        self.S = tf.constant(S, dtype=tf.float64)

    def __call__(self, x, u=None):
        return tf.reduce_sum((x @ self.S) * x, axis=1, keepdims=True)

# 5. Trajectory Simulator
class TrajectorySimulator:
    def __init__(self, dynamics_fn, model, running_cost_fn, terminal_cost_fn, butcher_tableau, u_max):
        self.dynamics_fn = dynamics_fn
        self.model = model
        self.running_cost = running_cost_fn
        self.terminal_cost = terminal_cost_fn
        self.butcher_tableau = butcher_tableau

    def simulate_trajectory(self, x0, dt, T_final, track_trajectory=False):
        steps = int(T_final / dt)
        x = tf.convert_to_tensor(x0, dtype=tf.float64)
        total_cost = tf.constant([[0.0]], dtype=tf.float64)

        trajectory_x = [] if track_trajectory else None
        trajectory_u = [] if track_trajectory else None

        A, b, c = tf.constant(self.butcher_tableau["A"], dtype=DTYPE_TF), \
            tf.constant(self.butcher_tableau["b"], dtype=DTYPE_TF), \
            tf.constant(self.butcher_tableau["c"], dtype=DTYPE_TF)
        s = A.shape[0]

        dt_f = tf.constant(dt, dtype=DTYPE_TF)
        T_final_f = tf.constant(T_final, dtype=DTYPE_TF)

        for step in tf.range(steps):
            t_current = tf.cast(step, dtype=DTYPE_TF) * dt_f
            t_current = tf.reshape(t_current, [1, 1])

            k_values, l_values = [], []
            for i in range(s):
                x_k_arg = x
                for j in range(i):
                    x_k_arg += dt_f * A[i, j] * k_values[j]

                t_k_arg = t_current + dt_f * c[i]
                tau_k = t_k_arg / T_final_f

                state_k = tf.concat([tau_k, x_k_arg], axis=1)
                u_k = self.model(state_k)

                k_i = self.dynamics_fn(x_k_arg, u_k, t_k_arg, None)
                k_values.append(k_i)
                l_values.append(self.running_cost(x_k_arg, u_k, tau_k))

                if i == 0 and track_trajectory:
                    trajectory_x.append(x)
                    trajectory_u.append(u_k)

            x_next = x
            for i in range(s):
                x_next += dt_f * b[i] * k_values[i]
            x = x_next

            step_cost = tf.constant([[0.0]], dtype=tf.float64)
            for i in range(s):
                step_cost += b[i] * l_values[i]
            total_cost += dt_f * step_cost

        total_cost += self.terminal_cost(x)

        if track_trajectory:
            trajectory_x.append(x)

            t_final_tf = tf.reshape(T_final_f, [1, 1])
            tau_final = t_final_tf / T_final_f
            state_final = tf.concat([tau_final, x], axis=1)
            u_final = self.model(state_final)
            trajectory_u.append(u_final)

            final_trajectory_x = np.concatenate([t.numpy() for t in trajectory_x], axis=0)
            final_trajectory_u = np.concatenate([u.numpy() for u in trajectory_u], axis=0)
        else:
            final_trajectory_x = None
            final_trajectory_u = None

        return x, tf.squeeze(total_cost), final_trajectory_x, final_trajectory_u

# 6. Optimal Controller
class OptimalController:
    def __init__(self, dynamics_fn, running_cost_fn, terminal_cost_fn, initial_state,
                 hidden_units, activations, input_dim, output_dim, dt, T_final, butcher_tableau, u_max,
                 t_treatment_start, optimizer=None):

        self.dynamics_fn = dynamics_fn
        self.running_cost = running_cost_fn
        self.terminal_cost = terminal_cost_fn

        self.model = MLP(input_dim=input_dim, hidden_layers=hidden_units,
                         activations=activations, output_dim=output_dim, u_max=u_max)

        self.simulator = TrajectorySimulator(
            dynamics_fn,
            self.model,
            running_cost_fn,
            terminal_cost_fn,
            BUTCHER_TABLEAUS[butcher_tableau],
            u_max
        )

        self.x0 = tf.convert_to_tensor(initial_state, dtype=tf.float64)
        self.dt = dt
        self.T_final = T_final
        self.t_treatment_start = t_treatment_start

        if optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
        else:
            self.optimizer = optimizer

        self.loss_history = []
        self.trajectory_x = None
        self.trajectory_u = None
        self.final_J_disc = None
        self.u_max = u_max

    @tf.function
    def training_step(self):
        with tf.GradientTape() as tape:
            _, loss_val, _, _ = self.simulator.simulate_trajectory(self.x0, self.dt, self.T_final,
                                                                   track_trajectory=False)
        grads = tape.gradient(loss_val, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss_val

    def train(self, epochs=500):
        print("\n=== Training started ===")
        for epoch in range(1, epochs + 1):
            loss_val = self.training_step()
            self.loss_history.append(loss_val.numpy())
            if epoch % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch}/{epochs}, loss={loss_val.numpy():.6f}")

        _, final_J, self.trajectory_x, self.trajectory_u = self.simulator.simulate_trajectory(self.x0, self.dt,
                                                                                              self.T_final,
                                                                                              track_trajectory=True)
        self.final_J_disc = final_J.numpy()

        print(f"Final J_disc = {self.final_J_disc:.6f} (Solver: {self.simulator.butcher_tableau['b'].size}-stage RK)")
        return self.loss_history

    def plot_trajectory(self, method_name: str):
        """Plot tumor dynamics with ICML-friendly styling."""
        if self.trajectory_x is None:
            raise RuntimeError("Trajectory not calculated. Run .train() first.")

        t_vals = np.linspace(0, self.T_final, self.trajectory_x.shape[0]) + self.t_treatment_start
        m_vals = self.trajectory_x.sum(axis=1)

        fig, ax = plt.subplots(figsize=(10, 7))
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        ax.plot(
            t_vals,
            self.trajectory_x[:, 0],
            color=color_cycle[0],
            linewidth=3,
            alpha=0.9,
            label=r"$\mathit{x₁}\ \mathit{(proliferating\ cells)}$",
        )

        ax.plot(
            t_vals,
            m_vals,
            color=color_cycle[1],
            linestyle="--",
            linewidth=3,
            alpha=0.9,
            label=r"$\mathit{m}\ \mathit{(total\ tumor)}$",
        )

        ax.set_title(f"Tumor dynamics: {method_name}", fontsize=18, fontweight="bold", pad=20)
        ax.set_xlabel("Time (days)", fontsize=16)
        ax.set_ylabel("Tumor Mass", fontsize=16)
        ax.legend(fontsize=12, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 5])
        plt.tight_layout()
        plt.show()

    def plot_control(self, method_name: str):
        """Plot control policy with styling consistent across methods."""
        if self.trajectory_u is None:
            raise RuntimeError("Control trajectory not calculated. Run .train() first.")

        t_vals = np.linspace(0, self.T_final, self.trajectory_u.shape[0]) + self.t_treatment_start
        U = self.trajectory_u

        fig, ax = plt.subplots(figsize=(10, 7))
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        ax.plot(
            t_vals,
            U.flatten(),
            color=color_cycle[2],
            linewidth=3,
            alpha=0.9,
            label=r"$\mathit{u(t)}$",
        )

        ax.axhline(
            self.model.u_max.numpy().item(),
            ls=":",
            color="k",
            linewidth=2,
            alpha=0.7,
            label=r"$\mathit{Cₜ\ cap}$",
        )

        ax.set_title(f"Control policy: {method_name}", fontsize=18, fontweight="bold", pad=20)
        ax.set_xlabel("Time (days)", fontsize=16)
        ax.set_ylabel("Drug Concentration", fontsize=16)
        ax.legend(fontsize=12, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 180])
        plt.tight_layout()
        plt.show()

# 7. SciPy High-Fidelity Evaluation
def scipy_evaluate_policy(model, dynamics_fn, running_cost, terminal_cost, x0_np, T_final, t_treatment_start,
                          solver_name):
    """Evaluates a trained policy using SciPy RK45 solver."""

    def augmented_ode_func(t, Y):
        x_np = Y[:-1]

        tau = tf.constant([[t / T_final]], dtype=DTYPE_TF)
        x_tf = tf.convert_to_tensor(x_np.reshape(1, -1), dtype=DTYPE_TF)
        state_t_x = tf.concat([tau, x_tf], axis=1)

        u_tf = model(state_t_x)
        x_dot_tf = dynamics_fn(x_tf, u_tf, tf.constant([[t]], dtype=DTYPE_TF), None)
        x_dot_np = x_dot_tf.numpy().flatten()

        L_tf = running_cost(x_tf, u_tf, tau)
        L_np = L_tf.numpy().item()

        return np.append(x_dot_np, L_np)

    Y0 = np.append(x0_np.flatten(), 0.0)

    print(f"\n--- Evaluating Policy trained by {solver_name} with SciPy RK45 ---")

    sol = sci_int.solve_ivp(
        augmented_ode_func,
        [0, T_final],
        Y0,
        method='RK45',
        rtol=1e-6,
        atol=1e-9,
        dense_output=True
    )

    x_final_np = sol.y[:-1, -1]
    J_running_final = sol.y[-1, -1]

    x_final_tf = tf.convert_to_tensor(x_final_np.reshape(1, -1), dtype=DTYPE_TF)
    J_terminal = terminal_cost(x_final_tf).numpy().item()

    J_total = J_running_final + J_terminal

    return J_total, x_final_np, sol

# 8. EXECUTION
if __name__ == "__main__":
    hidden_units = [5, 3]
    activations = ["silu", "silu"]
    input_dim, output_dim = 1 + 4, 1

    Q_val = np.diag([1.2, 0.10, 0.10, 0.10])
    R_val = np.array([[9.5e-4]])
    S_val = 0.0075 * np.eye(4)
    running_cost = RunningCost(Q=Q_val, R=R_val)
    terminal_cost = TerminalCost(S=S_val)
    x0_pkpd = [[0.5684, 0.0, 0.0, 0.0]]
    c_t = 173.4
    t_treatment_start = 13
    T_final = 27
    dynamics = PKPDDynamics()

    dt_common = 0.1
    epochs_common = 25000

    # Train Euler Policy
    print("\n\n<<< STARTING EULER TRAINING (dt=0.1) >>>")
    euler_controller = OptimalController(
        dynamics_fn=dynamics,
        running_cost_fn=running_cost,
        terminal_cost_fn=terminal_cost,
        initial_state=x0_pkpd,
        hidden_units=hidden_units,
        activations=activations,
        input_dim=input_dim,
        output_dim=output_dim,
        dt=dt_common,
        T_final=T_final,
        butcher_tableau="euler",
        u_max=c_t,
        t_treatment_start=t_treatment_start
    )
    euler_controller.train(epochs=epochs_common)

    J_disc_euler = euler_controller.final_J_disc
    W_final_disc_euler = np.sum(euler_controller.trajectory_x[-1])
    u_avg_euler = np.mean(euler_controller.trajectory_u[:-1])
    x1_avg_euler = np.mean(euler_controller.trajectory_x[:-1, 0])

    # Train Midpoint Policy
    print("\n\n<<< STARTING MIDPOINT TRAINING (dt=0.1) >>>")
    midpoint_controller = OptimalController(
        dynamics_fn=dynamics,
        running_cost_fn=running_cost,
        terminal_cost_fn=terminal_cost,
        initial_state=x0_pkpd,
        hidden_units=hidden_units,
        activations=activations,
        input_dim=input_dim,
        output_dim=output_dim,
        dt=dt_common,
        T_final=T_final,
        butcher_tableau="midpoint",
        u_max=c_t,
        t_treatment_start=t_treatment_start
    )
    midpoint_controller.train(epochs=epochs_common)

    J_disc_midpoint = midpoint_controller.final_J_disc
    W_final_disc_midpoint = np.sum(midpoint_controller.trajectory_x[-1])
    u_avg_midpoint = np.mean(midpoint_controller.trajectory_u[:-1])
    x1_avg_midpoint = np.mean(midpoint_controller.trajectory_x[:-1, 0])

    # Train RK4 Policy
    print("\n\n<<< STARTING RK4 TRAINING (dt=0.1) >>>")
    rk4_controller = OptimalController(
        dynamics_fn=dynamics,
        running_cost_fn=running_cost,
        terminal_cost_fn=terminal_cost,
        initial_state=x0_pkpd,
        hidden_units=hidden_units,
        activations=activations,
        input_dim=input_dim,
        output_dim=output_dim,
        dt=dt_common,
        T_final=T_final,
        butcher_tableau="rk4",
        u_max=c_t,
        t_treatment_start=t_treatment_start
    )
    rk4_controller.train(epochs=epochs_common)

    J_disc_rk4 = rk4_controller.final_J_disc
    W_final_disc_rk4 = np.sum(rk4_controller.trajectory_x[-1])
    u_avg_rk4 = np.mean(rk4_controller.trajectory_u[:-1])
    x1_avg_rk4 = np.mean(rk4_controller.trajectory_x[:-1, 0])

    # High-Fidelity Evaluation
    initial_tumor_mass = np.sum(np.array(x0_pkpd))

    J_true_euler, x_final_euler, sol_euler = scipy_evaluate_policy(
        euler_controller.model,
        euler_controller.dynamics_fn,
        euler_controller.running_cost,
        euler_controller.terminal_cost,
        euler_controller.x0.numpy(),
        T_final,
        t_treatment_start,
        solver_name="Euler Training Policy"
    )
    W_final_euler = np.sum(x_final_euler)

    J_true_midpoint, x_final_midpoint, sol_midpoint = scipy_evaluate_policy(
        midpoint_controller.model,
        midpoint_controller.dynamics_fn,
        midpoint_controller.running_cost,
        midpoint_controller.terminal_cost,
        midpoint_controller.x0.numpy(),
        T_final,
        t_treatment_start,
        solver_name="Midpoint Training Policy"
    )
    W_final_midpoint = np.sum(x_final_midpoint)

    J_true_rk4, x_final_rk4, sol_rk4 = scipy_evaluate_policy(
        rk4_controller.model,
        rk4_controller.dynamics_fn,
        rk4_controller.running_cost,
        rk4_controller.terminal_cost,
        rk4_controller.x0.numpy(),
        T_final,
        t_treatment_start,
        solver_name="RK4 Training Policy"
    )
    W_final_rk4 = np.sum(x_final_rk4)

    # Final Results
    print("\n\n" + "=" * 100)
    print("           FINAL POLICY EVALUATION & DISCRETE METRICS")
    print("=" * 100)
    print(f"INITIAL TOTAL TUMOR MASS (w_0): {initial_tumor_mass:.6f}")
    print(f"Integration time step (dt): {dt_common}")
    print(f"Training Epochs: {epochs_common}")
    print("-" * 100)
    print(f"{'Metric':<40}{'EULER':<20}{'MIDPOINT':<20}{'RK4':<20}")
    print("-" * 100)

    print(f"{'Cost Functional (J^h)':<40}{J_disc_euler:<20.6f}{J_disc_midpoint:<20.6f}{J_disc_rk4:<20.6f}")
    print(f"{'Final Tumor Mass (G)':<40}{W_final_disc_euler:<20.6f}{W_final_disc_midpoint:<20.6f}{W_final_disc_rk4:<20.6f}")
    print(f"{'Drug Applied (u(t)) (Avg.)':<40}{u_avg_euler:<20.6f}{u_avg_midpoint:<20.6f}{u_avg_rk4:<20.6f}")
    print(f"{'Proliferating Cell Count (x1) (Avg.)':<40}{x1_avg_euler:<20.6f}{x1_avg_midpoint:<20.6f}{x1_avg_rk4:<20.6f}")

    print("-" * 100)
    print(f"{'J_True (SciPy RK45)':<40}{J_true_euler:<20.6f}{J_true_midpoint:<20.6f}{J_true_rk4:<20.6f}")
    print(f"{'Final Tumor Mass (w_f) (SciPy RK45)':<40}{W_final_euler:<20.6f}{W_final_midpoint:<20.6f}{W_final_rk4:<20.6f}")
    print("=" * 100)

    euler_controller.plot_trajectory("Euler")
    euler_controller.plot_control("Euler")
    midpoint_controller.plot_trajectory("Midpoint")
    midpoint_controller.plot_control("Midpoint")
    rk4_controller.plot_trajectory("RK4")
    rk4_controller.plot_control("RK4")

    # 9. TRAJECTORY DEVIATION PLOT (Integrator − SciPy RK45)
    fig, ax = plt.subplots(figsize=(10, 7))
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for ctrl, sol, label, color, ls in [
        (euler_controller, sol_euler, "Euler", color_cycle[0], "-"),
        (midpoint_controller, sol_midpoint, "Midpoint", color_cycle[1], "--"),
        (rk4_controller, sol_rk4, "RK4", color_cycle[2], "-."),
    ]:
        # Discrete trajectory time grid (relative to treatment start)
        n_pts = ctrl.trajectory_x.shape[0]
        t_grid = np.linspace(0, T_final, n_pts)

        # Discrete total tumor mass at each grid point
        m_disc = ctrl.trajectory_x.sum(axis=1)

        # SciPy (analytical) total tumor mass interpolated to the same grid
        # sol.sol is the dense output interpolant; state dims are rows 0..3
        y_analytical = sol.sol(t_grid)          # shape (5, n_pts) — 4 states + 1 cost
        m_analytical = y_analytical[:4, :].sum(axis=0)

        # Deviation
        deviation = m_disc - m_analytical

        ax.plot(
            t_grid + t_treatment_start,
            deviation,
            color=color,
            linestyle=ls,
            linewidth=3,
            alpha=0.9,
            label=rf"$\mathit{{{label}}} - \mathit{{RK45}}$",
        )

    ax.axhline(0, ls=":", color="k", linewidth=1.5, alpha=0.5)
    ax.set_title("Trajectory deviation from reference solution", fontsize=18, fontweight="bold", pad=20)
    ax.set_xlabel("Time (days)", fontsize=16)
    ax.set_ylabel("Total tumor mass deviation", fontsize=16)
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()