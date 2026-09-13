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

        if u_max is not None:
            u_max_np = np.atleast_1d(np.asarray(u_max, dtype=np.float64))
            if u_max_np.size == 1 and (output_dim is not None and output_dim == 1):
                self.u_max = tf.constant(u_max_np.item(), dtype=tf.float64)
                self._bound_mode = "sigmoid"
            else:
                self.u_max = tf.constant(u_max_np.reshape(1, -1), dtype=tf.float64)
                self._bound_mode = "tanh"
        else:
            self.u_max = None
            self._bound_mode = None

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

        if self._bound_mode == "sigmoid":
            return self.u_max * tf.nn.sigmoid(raw_output)
        elif self._bound_mode == "tanh":
            return self.u_max * tf.nn.tanh(raw_output)
        return raw_output


# 3. LORENZ-96 CHAOTIC DYNAMICS (20D)
class Lorenz96Dynamics:
    def __init__(self, N=20, F=8.0):
        self.N = N
        self.F = F

    def __call__(self, x, u, t, g=None):
        """
        x : [batch, N]   physical state
        u : [batch, N]   control forcing
        Returns: [batch, N]  dx/dt
        """
        x_ip1 = tf.roll(x, shift=-1, axis=1)    # x_{i+1}
        x_im1 = tf.roll(x, shift=1, axis=1)     # x_{i-1}
        x_im2 = tf.roll(x, shift=2, axis=1)     # x_{i-2}

        # Eq. 5: dx_i/dt = (x_{i+1} − x_{i-2}) x_{i-1} − x_i + F + u_i
        dx = (x_ip1 - x_im2) * x_im1 - x + self.F + u

        return dx


# 4. Costs
class RunningCost:
    def __init__(self, Q, R, x_ref=None):
        self.Q = tf.constant(Q, dtype=tf.float64)
        self.R = tf.constant(R, dtype=tf.float64)
        self.x_ref = tf.constant(x_ref, dtype=tf.float64) if x_ref is not None else None

    def __call__(self, x, u, t=None):
        e = (x - self.x_ref) if self.x_ref is not None else x
        state_cost = tf.reduce_sum((e @ self.Q) * e, axis=1, keepdims=True)
        control_cost = tf.reduce_sum((u @ self.R) * u, axis=1, keepdims=True)
        return state_cost + control_cost

class TerminalCost:
    def __init__(self, S, x_ref=None):
        self.S = tf.constant(S, dtype=tf.float64)
        self.x_ref = tf.constant(x_ref, dtype=tf.float64) if x_ref is not None else None

    def __call__(self, x, u=None):
        e = (x - self.x_ref) if self.x_ref is not None else x
        return tf.reduce_sum((e @ self.S) * e, axis=1, keepdims=True)


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

    def plot_4panel(self, method_name: str, x_ref=None, F_L=8.0):
        if self.trajectory_x is None or self.trajectory_u is None:
            raise RuntimeError("Trajectory not calculated. Run .train() first.")

        X_sol = self.trajectory_x.T                # shape (N_L, N+1)
        U_sol = self.trajectory_u.T                # shape (N_L, N+1)
        N_L   = X_sol.shape[0]

        if x_ref is None:
            x_ref_flat = F_L * np.ones(N_L)
        else:
            x_ref_flat = np.asarray(x_ref).flatten()

        u_max_val = float(np.asarray(self.u_max).flat[0])

        t_grid = np.linspace(0, self.T_final, X_sol.shape[1])
        t_ctrl = np.linspace(0, self.T_final, U_sol.shape[1])

        trace_idx = [int(round(i * (N_L - 1) / 4)) for i in range(5)]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # ── Top-left: State deviation Hovmöller ──
        ax = axes[0, 0]
        dev_data = X_sol - x_ref_flat.reshape(-1, 1)
        vmax = np.max(np.abs(dev_data))
        im = ax.imshow(
            dev_data,
            aspect='auto', origin='lower',
            extent=[0, self.T_final, 0.5, N_L + 0.5],
            cmap='RdBu_r', interpolation='bilinear',
            vmin=-vmax, vmax=vmax
        )
        ax.set_xlabel('Time');  ax.set_ylabel('State index $i$')
        ax.set_title(f'State deviation - {method_name}', fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

        # ── Top-right: Selected state traces ──
        ax = axes[0, 1]
        for idx in trace_idx:
            ax.plot(t_grid, X_sol[idx, :], lw=1.5, label=f'$x_{{{idx+1}}}$')
        ax.axhline(F_L, ls=':', color='k', alpha=0.6, label=f'$x^* = {F_L}$')
        ax.set_xlabel('Time');  ax.set_ylabel('$x_i$')
        ax.set_title(f'State traces - {method_name}', fontsize=11)
        ax.legend(fontsize=9, ncol=3);  ax.grid(True, alpha=0.3)

        # ── Bottom-left: Control Hovmöller ──
        ax = axes[1, 0]
        vmax_u = np.max(np.abs(U_sol))
        im = ax.imshow(
            U_sol,
            aspect='auto', origin='lower',
            extent=[0, self.T_final, 0.5, N_L + 0.5],
            cmap='PiYG', interpolation='bilinear',
            vmin=-vmax_u, vmax=vmax_u
        )
        ax.set_xlabel('Time');  ax.set_ylabel('Control index $i$')
        ax.set_title(f'Control dynamics - {method_name}', fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

        # ── Bottom-right: Selected control traces ──
        ax = axes[1, 1]
        for idx in trace_idx:
            ax.plot(t_ctrl, U_sol[idx, :], lw=1.5, label=f'$u_{{{idx+1}}}$')
        ax.axhline(0, ls=':', color='k', alpha=0.4)
        ax.axhline( u_max_val, ls='--', color='grey', alpha=0.5, label=r'$\pm u_{max}$')
        ax.axhline(-u_max_val, ls='--', color='grey', alpha=0.5)
        ax.set_xlabel('Time');  ax.set_ylabel('$u_i$')
        ax.set_title(f'Control traces - {method_name}', fontsize=11)
        ax.legend(fontsize=9, ncol=3);  ax.grid(True, alpha=0.3)

        plt.suptitle(
            f'Lorenz-96 optimal control — Neural Network ({method_name})  '
            f'($J^h$ = {self.final_J_disc:.4f})',
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout()
        plt.savefig(f'lorenz96_nn_{method_name.lower()}.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved to lorenz96_nn_{method_name.lower()}.png")
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
        atol=1e-9
    )

    x_final_np = sol.y[:-1, -1]
    J_running_final = sol.y[-1, -1]

    x_final_tf = tf.convert_to_tensor(x_final_np.reshape(1, -1), dtype=DTYPE_TF)
    J_terminal = terminal_cost(x_final_tf).numpy().item()

    J_total = J_running_final + J_terminal

    return J_total, x_final_np


# 8. EXECUTION — Lorenz-96 Chaotic Stabilization Benchmark (20D)
if __name__ == "__main__":

    # ── Lorenz-96 parameters ──
    N_LORENZ = 20
    F_LORENZ = 8.0    # chaotic regime

    dynamics = Lorenz96Dynamics(N=N_LORENZ, F=F_LORENZ)

    # ── Reference state: the unstable equilibrium x_i* = F ──
    x_ref = F_LORENZ * np.ones((1, N_LORENZ))

    # ── Network architecture ──
    STATE_DIM   = N_LORENZ   # 20
    CONTROL_DIM = N_LORENZ   # 20
    input_dim   = 1 + STATE_DIM   # 21
    output_dim  = CONTROL_DIM     # 20

    hidden_units = [256, 128]
    activations  = ["silu", "silu"]

    u_max = 10.0 * np.ones(CONTROL_DIM)

    # ── Cost matrices ──
    # Q: uniform penalty on deviation from equilibrium
    Q_val = 1.0 * np.eye(STATE_DIM)

    # R: control regularization
    R_val = 0.01 * np.eye(CONTROL_DIM)

    # S: terminal cost (heavier to pin final state near equilibrium)
    S_val = 5.0 * np.eye(STATE_DIM)

    running_cost  = RunningCost(Q=Q_val, R=R_val, x_ref=x_ref)
    terminal_cost = TerminalCost(S=S_val, x_ref=x_ref)

    # x_i(0) = F + perturbation
    perturbation = 0.5 * np.sin(2.0 * np.pi * np.arange(N_LORENZ) / N_LORENZ)
    x0_lorenz = [(F_LORENZ + perturbation).tolist()]

    # ── Integration parameters ──
    T_final   = 2.0     # Lorenz time units (~3 Lyapunov times at F=8)
    dt_common = 0.01    # 200 steps
    epochs_common = 25000

    t_treatment_start = 0   # no offset

    # ── Train Euler Policy ──
    print("\n\n<<< STARTING EULER TRAINING (dt=0.01) >>>")
    euler_controller = OptimalController(
        dynamics_fn=dynamics,
        running_cost_fn=running_cost,
        terminal_cost_fn=terminal_cost,
        initial_state=x0_lorenz,
        hidden_units=hidden_units,
        activations=activations,
        input_dim=input_dim,
        output_dim=output_dim,
        dt=dt_common,
        T_final=T_final,
        butcher_tableau="euler",
        u_max=u_max,
        t_treatment_start=t_treatment_start
    )
    euler_controller.train(epochs=epochs_common)

    # ── Train Midpoint Policy ──
    print("\n\n<<< STARTING MIDPOINT TRAINING (dt=0.01) >>>")
    midpoint_controller = OptimalController(
        dynamics_fn=dynamics,
        running_cost_fn=running_cost,
        terminal_cost_fn=terminal_cost,
        initial_state=x0_lorenz,
        hidden_units=hidden_units,
        activations=activations,
        input_dim=input_dim,
        output_dim=output_dim,
        dt=dt_common,
        T_final=T_final,
        butcher_tableau="midpoint",
        u_max=u_max,
        t_treatment_start=t_treatment_start
    )
    midpoint_controller.train(epochs=epochs_common)

    # ── Train RK4 Policy ──
    print("\n\n<<< STARTING RK4 TRAINING (dt=0.01) >>>")
    rk4_controller = OptimalController(
        dynamics_fn=dynamics,
        running_cost_fn=running_cost,
        terminal_cost_fn=terminal_cost,
        initial_state=x0_lorenz,
        hidden_units=hidden_units,
        activations=activations,
        input_dim=input_dim,
        output_dim=output_dim,
        dt=dt_common,
        T_final=T_final,
        butcher_tableau="rk4",
        u_max=u_max,
        t_treatment_start=t_treatment_start
    )
    rk4_controller.train(epochs=epochs_common)

    # ── Metrics ──
    J_disc_euler    = euler_controller.final_J_disc
    J_disc_midpoint = midpoint_controller.final_J_disc
    J_disc_rk4      = rk4_controller.final_J_disc

    dev_final_euler    = np.linalg.norm(euler_controller.trajectory_x[-1] - x_ref.flatten())
    dev_final_midpoint = np.linalg.norm(midpoint_controller.trajectory_x[-1] - x_ref.flatten())
    dev_final_rk4      = np.linalg.norm(rk4_controller.trajectory_x[-1] - x_ref.flatten())

    # ── High-Fidelity SciPy Evaluation (skip if training diverged) ──
    if np.isfinite(J_disc_euler):
        J_true_euler, x_final_euler = scipy_evaluate_policy(
            euler_controller.model, dynamics, running_cost, terminal_cost,
            euler_controller.x0.numpy(), T_final, t_treatment_start,
            solver_name="Euler Training Policy"
        )
        dev_scipy_euler = np.linalg.norm(x_final_euler - x_ref.flatten())
    else:
        print("\n--- Skipping SciPy eval for Euler (training diverged to NaN) ---")
        J_true_euler = float('nan')
        dev_scipy_euler = float('nan')

    if np.isfinite(J_disc_midpoint):
        J_true_midpoint, x_final_midpoint = scipy_evaluate_policy(
            midpoint_controller.model, dynamics, running_cost, terminal_cost,
            midpoint_controller.x0.numpy(), T_final, t_treatment_start,
            solver_name="Midpoint Training Policy"
        )
        dev_scipy_midpoint = np.linalg.norm(x_final_midpoint - x_ref.flatten())
    else:
        print("\n--- Skipping SciPy eval for Midpoint (training diverged to NaN) ---")
        J_true_midpoint = float('nan')
        dev_scipy_midpoint = float('nan')

    if np.isfinite(J_disc_rk4):
        J_true_rk4, x_final_rk4 = scipy_evaluate_policy(
            rk4_controller.model, dynamics, running_cost, terminal_cost,
            rk4_controller.x0.numpy(), T_final, t_treatment_start,
            solver_name="RK4 Training Policy"
        )
        dev_scipy_rk4 = np.linalg.norm(x_final_rk4 - x_ref.flatten())
    else:
        print("\n--- Skipping SciPy eval for RK4 (training diverged to NaN) ---")
        J_true_rk4 = float('nan')
        dev_scipy_rk4 = float('nan')

    # ── Final Results ──
    print("\n\n" + "=" * 100)
    print("     LORENZ-96 (20D) CHAOTIC STABILIZATION — FINAL POLICY EVALUATION")
    print("=" * 100)
    print(f"Forcing F = {F_LORENZ}  |  Equilibrium: x_i* = F = {F_LORENZ} (unstable, chaotic)")
    print(f"Initial perturbation ||x_0 − x*|| = {np.linalg.norm(perturbation):.6f}")
    print(f"Integration step (dt): {dt_common}   |   Horizon: {T_final} time units")
    print(f"Training Epochs: {epochs_common}")
    print("-" * 100)
    print(f"{'Metric':<45}{'EULER':<22}{'MIDPOINT':<22}{'RK4':<20}")
    print("-" * 100)
    print(f"{'Cost Functional (J^h)':<45}{J_disc_euler:<22.6f}{J_disc_midpoint:<22.6f}{J_disc_rk4:<20.6f}")
    print(f"{'Final ||x_f − x*|| (disc.)':<45}{dev_final_euler:<22.6f}{dev_final_midpoint:<22.6f}{dev_final_rk4:<20.6f}")
    print("-" * 100)
    print(f"{'J_True (SciPy RK45)':<45}{J_true_euler:<22.6f}{J_true_midpoint:<22.6f}{J_true_rk4:<20.6f}")
    print(f"{'Final ||x_f − x*|| (SciPy)':<45}{dev_scipy_euler:<22.6f}{dev_scipy_midpoint:<22.6f}{dev_scipy_rk4:<20.6f}")
    print("=" * 100)

    if np.isfinite(J_disc_euler):
        euler_controller.plot_4panel("Euler", x_ref=x_ref, F_L=F_LORENZ)
    else:
        print("Skipping Euler plots (training diverged).")

    if np.isfinite(J_disc_midpoint):
        midpoint_controller.plot_4panel("Midpoint", x_ref=x_ref, F_L=F_LORENZ)
    else:
        print("Skipping Midpoint plots (training diverged).")

    if np.isfinite(J_disc_rk4):
        rk4_controller.plot_4panel("RK4", x_ref=x_ref, F_L=F_LORENZ)
    else:
        print("Skipping RK4 plots (training diverged).")

    # 9. TRAJECTORY DEVIATION PLOT (Integrator − CasADi Ground Truth)
    gt = np.load('lorenz96_ground_truth.npz')
    X_gt = gt['X']           # shape (20, N_gt+1)
    t_gt = gt['t_state']     # shape (N_gt+1,)
    x_ref_flat = x_ref.flatten()

    fig, ax = plt.subplots(figsize=(10, 7))
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for ctrl, label, color, ls in [
        (euler_controller,    "Euler",    color_cycle[0], "-"),
        (midpoint_controller, "Midpoint", color_cycle[1], "--"),
        (rk4_controller,      "RK4",     color_cycle[2], "-."),
    ]:
        if not np.isfinite(ctrl.final_J_disc):
            continue

        n_pts = ctrl.trajectory_x.shape[0]
        t_nn = np.linspace(0, T_final, n_pts)

        # ||x(t) - x*|| from the discrete NN trajectory
        dev_nn = np.linalg.norm(ctrl.trajectory_x - x_ref_flat, axis=1)

        # Interpolate CasADi ground truth ||x(t) - x*|| onto the NN time grid
        dev_gt_all = np.linalg.norm(X_gt.T - x_ref_flat, axis=1)  # shape (N_gt+1,)
        dev_gt = np.interp(t_nn, t_gt, dev_gt_all)

        deviation = dev_nn - dev_gt

        ax.plot(
            t_nn,
            deviation,
            color=color,
            linestyle=ls,
            linewidth=3,
            alpha=0.9,
            label=rf"$\mathit{{{label}}} - \mathit{{J^*}}$",
        )

    ax.axhline(0, ls=":", color="k", linewidth=1.5, alpha=0.5)
    ax.set_title("Trajectory deviation from reference solution", fontsize=18, fontweight="bold", pad=20)
    ax.set_xlabel("Time (Lorenz units)", fontsize=16)
    ax.set_ylabel("Total deviation (m)", fontsize=16)
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()