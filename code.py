import math
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

R_GAS = 8.314462618  # J/(mol·K) or Pa·m³/(mol·K)

# Substance Database
SUBSTANCES = {
    "Carbon Dioxide (CO2)": {
        "Tc": 304.13, "Pc": 7.3773e6, "omega": 0.224, "MW": 44.01,
        "cp": (19.8, 0.07344, -5.602e-5, 1.715e-8)
    },
    "Methane (CH4)": {
        "Tc": 190.56, "Pc": 4.5992e6, "omega": 0.011, "MW": 16.04,
        "cp": (19.25, 0.0521, 1.197e-5, -1.132e-8)
    },
    "Ethane (C2H6)": {
        "Tc": 305.32, "Pc": 4.872e6, "omega": 0.099, "MW": 30.07,
        "cp": (5.409, 0.1781, -6.938e-5, 8.713e-9)
    },
    "Nitrogen (N2)": {
        "Tc": 126.20, "Pc": 3.396e6, "omega": 0.037, "MW": 28.01,
        "cp": (28.98, -1.57e-3, 8.08e-6, -2.87e-9)
    },
    "Water (H2O)": {
        "Tc": 647.10, "Pc": 22.064e6, "omega": 0.344, "MW": 18.02,
        "cp": (32.24, 1.92e-3, 1.055e-5, -3.59e-9)
    }
}


class CubicEOSCore:
    """Core mathematical engine for cubic EOS, fugacity, and departures."""

    def __init__(self, Tc: float, Pc: float, omega: float, model: str = "PR"):
        self.Tc, self.Pc, self.omega, self.model = Tc, Pc, omega, model.upper()
        if self.model == "VDW":
            self.u, self.w = 0.0, 0.0
            self.Omega_a, self.Omega_b = 27.0 / 64.0, 1.0 / 8.0
        elif self.model in ["RK", "SRK"]:
            self.u, self.w = 1.0, 0.0
            self.Omega_a, self.Omega_b = 0.42748, 0.08664
        elif self.model == "PR":
            self.u, self.w = 2.0, -1.0
            self.Omega_a, self.Omega_b = 0.45724, 0.07780

        self.b = self.Omega_b * (R_GAS * self.Tc) / self.Pc
        self.a_c = self.Omega_a * ((R_GAS * self.Tc) ** 2) / self.Pc

    def _m_factor(self) -> float:
        if self.model == "SRK":
            return 0.480 + 1.574 * self.omega - 0.176 * (self.omega ** 2)
        elif self.model == "PR":
            return (0.37464 + 1.54226 * self.omega - 0.26992 * (self.omega ** 2)) if self.omega <= 0.49 else \
                   (0.379642 + 1.48503 * self.omega - 0.164423 * (self.omega ** 2) + 0.016666 * (self.omega ** 3))
        return 0.0

    def _alpha_and_deriv(self, T: float):
        Tr = T / self.Tc
        sqrt_Tr = math.sqrt(Tr)
        if self.model == "VDW":
            return 1.0, 0.0
        elif self.model == "RK":
            return 1.0 / sqrt_Tr, -0.5 / sqrt_Tr
        elif self.model in ["SRK", "PR"]:
            m = self._m_factor()
            alpha = (1.0 + m * (1.0 - sqrt_Tr)) ** 2
            T_dalpha_dT = -m * sqrt_Tr * (1.0 + m * (1.0 - sqrt_Tr))
            return alpha, T_dalpha_dT
        return 1.0, 0.0

    def get_AB_kappa(self, T: float, P: float):
        alpha, T_dalpha = self._alpha_and_deriv(T)
        a = self.a_c * alpha
        A = (a * P) / ((R_GAS * T) ** 2)
        B = (self.b * P) / (R_GAS * T)
        kappa = T_dalpha / alpha if alpha > 0 else 0.0
        return A, B, kappa, a

    @staticmethod
    def solve_cubic(a2: float, a1: float, a0: float):
        Q = (3.0 * a1 - a2 ** 2) / 9.0
        R = (9.0 * a2 * a1 - 27.0 * a0 - 2.0 * a2 ** 3) / 54.0
        D = Q ** 3 + R ** 2
        roots = []
        if D > 0:
            S = (R + math.sqrt(D)) ** (1.0 / 3.0) if (R + math.sqrt(D)) >= 0 else -(-R - math.sqrt(D)) ** (1.0 / 3.0)
            T_term = (R - math.sqrt(D)) ** (1.0 / 3.0) if (R - math.sqrt(D)) >= 0 else -(-R + math.sqrt(D)) ** (1.0 / 3.0)
            roots.append(-a2 / 3.0 + (S + T_term))
        else:
            theta = math.acos(max(-1.0, min(1.0, R / math.sqrt(-Q ** 3))))
            roots.append(2.0 * math.sqrt(-Q) * math.cos(theta / 3.0) - a2 / 3.0)
            roots.append(2.0 * math.sqrt(-Q) * math.cos((theta + 2.0 * math.pi) / 3.0) - a2 / 3.0)
            roots.append(2.0 * math.sqrt(-Q) * math.cos((theta + 4.0 * math.pi) / 3.0) - a2 / 3.0)
        return sorted([r for r in roots if r > 0])

    def solve_z(self, T: float, P: float):
        A, B, _, _ = self.get_AB_kappa(T, P)
        a2 = -(1.0 + B - self.u * B)
        a1 = A + self.w * (B ** 2) - self.u * B - self.u * (B ** 2)
        a0 = -(A * B + self.w * (B ** 2) + self.w * (B ** 3))
        return [z for z in self.solve_cubic(a2, a1, a0) if z > B]

    def properties(self, T: float, P: float, z: float):
        A, B, kappa, _ = self.get_AB_kappa(T, P)
        # Fugacity coefficient ln(phi)
        if self.model == "VDW":
            ln_phi = z - 1.0 - math.log(z - B) - (A / z)
            H_res = R_GAS * T * (z - 1.0 - (A / z))
            S_res = R_GAS * math.log(z - B)
        elif self.model in ["RK", "SRK"]:
            ratio = math.log(1.0 + B / z)
            ln_phi = z - 1.0 - math.log(z - B) - (A / B) * ratio
            term = -1.5 if self.model == "RK" else (kappa - 1.0)
            s_term = -0.5 if self.model == "RK" else kappa
            H_res = R_GAS * T * (z - 1.0 + (A / B) * term * ratio)
            S_res = R_GAS * (math.log(z - B) + (A / B) * s_term * ratio)
        elif self.model == "PR":
            d = math.sqrt(2.0)
            ratio = (1.0 / (2.0 * d)) * math.log((z + (1.0 + d) * B) / (z + (1.0 - d) * B))
            ln_phi = z - 1.0 - math.log(z - B) - (A / B) * ratio
            H_res = R_GAS * T * (z - 1.0 + (A / B) * (kappa - 1.0) * ratio)
            S_res = R_GAS * (math.log(z - B) + (A / B) * kappa * ratio)

        phi = math.exp(ln_phi)
        return {
            "Z": z, "Vm": (z * R_GAS * T) / P, "phi": phi,
            "f": phi * P, "H_res": H_res, "S_res": S_res
        }


class PVTApp(tk.Tk):
    """Tkinter graphical desktop interface for PVT & Fugacity calculations."""

    def __init__(self):
        super().__init__()
        self.title("PVT & Fugacity Real Gas Simulator (Cubic EOS)")
        self.geometry("1200x820")
        self.configure(bg="#f4f6f9")

        self.setup_ui()
        self.recalculate()

    def setup_ui(self):
        # Top Header
        header = tk.Frame(self, bg="#1e293b", height=50)
        header.pack(fill=tk.X)
        tk.Label(
            header, text="Cubic Equation of State PVT & Fugacity Explorer",
            fg="white", bg="#1e293b", font=("Helvetica", 14, "bold")
        ).pack(side=tk.LEFT, padx=20, pady=10)

        # Main Layout Container
        main_paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left Control & Output Frame
        left_frame = ttk.Frame(main_paned, width=380, padding=10)
        main_paned.add(left_frame, weight=1)

        # Right Chart Frame
        right_frame = ttk.Frame(main_paned, padding=10)
        main_paned.add(right_frame, weight=3)

        # Inputs Group
        input_box = ttk.LabelFrame(left_frame, text="Input Parameters", padding=10)
        input_box.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(input_box, text="Substance:").grid(row=0, column=0, sticky="w", pady=4)
        self.substance_var = tk.StringVar(value="Carbon Dioxide (CO2)")
        self.substance_cb = ttk.Combobox(
            input_box, textvariable=self.substance_var,
            values=list(SUBSTANCES.keys()), state="readonly", width=22
        )
        self.substance_cb.grid(row=0, column=1, pady=4)
        self.substance_cb.bind("<<ComboboxSelected>>", lambda e: self.recalculate())

        ttk.Label(input_box, text="Equation of State:").grid(row=1, column=0, sticky="w", pady=4)
        self.model_var = tk.StringVar(value="PR")
        self.model_cb = ttk.Combobox(
            input_box, textvariable=self.model_var,
            values=["PR", "SRK", "RK", "VDW"], state="readonly", width=22
        )
        self.model_cb.grid(row=1, column=1, pady=4)
        self.model_cb.bind("<<ComboboxSelected>>", lambda e: self.recalculate())

        ttk.Label(input_box, text="Temperature T [K]:").grid(row=2, column=0, sticky="w", pady=4)
        self.t_var = tk.DoubleVar(value=320.0)
        self.t_entry = ttk.Entry(input_box, textvariable=self.t_var, width=12)
        self.t_entry.grid(row=2, column=1, sticky="w", pady=4)

        ttk.Label(input_box, text="Pressure P [bar]:").grid(row=3, column=0, sticky="w", pady=4)
        self.p_var = tk.DoubleVar(value=50.0)
        self.p_entry = ttk.Entry(input_box, textvariable=self.p_var, width=12)
        self.p_entry.grid(row=3, column=1, sticky="w", pady=4)

        btn_calc = ttk.Button(input_box, text="Calculate & Plot", command=self.recalculate)
        btn_calc.grid(row=4, column=0, columnspan=2, pady=10, sticky="ew")

        # Results Group
        results_box = ttk.LabelFrame(left_frame, text="Calculated Thermodynamic State", padding=10)
        results_box.pack(fill=tk.BOTH, expand=True)

        self.res_text = tk.Text(results_box, wrap=tk.WORD, font=("Courier", 10), bg="#ffffff")
        self.res_text.pack(fill=tk.BOTH, expand=True)

        # Matplotlib Figures
        self.fig = Figure(figsize=(7, 7), dpi=100)
        self.ax_pv = self.fig.add_subplot(211)
        self.ax_fug = self.fig.add_subplot(212)
        self.fig.tight_layout(pad=3.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, right_frame)
        toolbar.update()

    def recalculate(self):
        try:
            T = float(self.t_var.get())
            P_bar = float(self.p_var.get())
            P = P_bar * 1e5
            sub_key = self.substance_var.get()
            model = self.model_var.get()
            sub_data = SUBSTANCES[sub_key]
        except ValueError:
            messagebox.showerror("Input Error", "Please provide valid numeric values for T and P.")
            return

        core = CubicEOSCore(sub_data["Tc"], sub_data["Pc"], sub_data["omega"], model=model)
        roots = core.solve_z(T, P)

        if not roots:
            messagebox.showwarning("Solver", "No physically admissible roots (Z > B) found.")
            return

        # Prepare Text Summary
        self.res_text.delete("1.0", tk.END)
        self.res_text.insert(tk.END, f"Fluid : {sub_key.split('(')[0].strip()}\n")
        self.res_text.insert(tk.END, f"Model : {model} | T={T:.1f} K | P={P_bar:.1f} bar\n")
        self.res_text.insert(tk.END, f"Tc    : {sub_data['Tc']:.1f} K | Pc={sub_data['Pc']/1e5:.1f} bar\n")
        self.res_text.insert(tk.END, "=" * 38 + "\n")

        if len(roots) == 1:
            p = core.properties(T, P, roots[0])
            self.res_text.insert(tk.END, "[Single Phase / Supercritical]\n")
            self.res_text.insert(tk.END, f"Z        : {p['Z']:.4f}\n")
            self.res_text.insert(tk.END, f"Vm       : {p['Vm']*1e6:.2f} cm³/mol\n")
            self.res_text.insert(tk.END, f"phi (φ)  : {p['phi']:.4f}\n")
            self.res_text.insert(tk.END, f"Fugacity : {p['f']/1e5:.3f} bar\n")
            self.res_text.insert(tk.END, f"H_res    : {p['H_res']/1e3:.2f} kJ/mol\n")
            self.res_text.insert(tk.END, f"S_res    : {p['S_res']:.2f} J/(mol·K)\n")
        else:
            liq = core.properties(T, P, roots[0])
            vap = core.properties(T, P, roots[-1])
            self.res_text.insert(tk.END, "[Two-Phase Subcritical Region]\n\n")
            self.res_text.insert(tk.END, "Liquid Phase (lowest root):\n")
            self.res_text.insert(tk.END, f"  Z_L    : {liq['Z']:.4f}\n")
            self.res_text.insert(tk.END, f"  Vm_L   : {liq['Vm']*1e6:.2f} cm³/mol\n")
            self.res_text.insert(tk.END, f"  phi_L  : {liq['phi']:.4f}\n")
            self.res_text.insert(tk.END, f"  f_L    : {liq['f']/1e5:.3f} bar\n")
            self.res_text.insert(tk.END, f"  H_res  : {liq['H_res']/1e3:.2f} kJ/mol\n\n")
            self.res_text.insert(tk.END, "Vapor Phase (highest root):\n")
            self.res_text.insert(tk.END, f"  Z_V    : {vap['Z']:.4f}\n")
            self.res_text.insert(tk.END, f"  Vm_V   : {vap['Vm']*1e6:.2f} cm³/mol\n")
            self.res_text.insert(tk.END, f"  phi_V  : {vap['phi']:.4f}\n")
            self.res_text.insert(tk.END, f"  f_V    : {vap['f']/1e5:.3f} bar\n")
            self.res_text.insert(tk.END, f"  H_res  : {vap['H_res']/1e3:.2f} kJ/mol\n")
            self.res_text.insert(tk.END, f"f_L / f_V: {liq['f']/vap['f']:.4f}\n")

        # Generate Graphs
        self.plot_graphs(core, T, P, roots)

    def plot_graphs(self, core: CubicEOSCore, T: float, P: float, roots: list):
        # 1. P-V Isotherm Plot
        self.ax_pv.clear()
        v_min = core.b * 1.05
        v_max = (R_GAS * T) / 1e5 * 2.0  # approximate volume at ~1 bar
        v_points = []
        p_points = []

        # Logarithmically spaced volumes
        steps = 250
        for i in range(steps):
            vm = math.exp(math.log(v_min) + i * (math.log(v_max) - math.log(v_min)) / (steps - 1))
            alpha, _ = core._alpha_and_deriv(T)
            a = core.a_c * alpha
            denom = vm**2 + core.u * core.b * vm + core.w * (core.b**2)
            if denom > 0:
                p_calc = (R_GAS * T) / (vm - core.b) - a / denom
                if 0 < p_calc < 2.5 * core.Pc:
                    v_points.append(vm * 1e6)  # cm³/mol
                    p_points.append(p_calc / 1e5)  # bar

        self.ax_pv.plot(v_points, p_points, color="#1f77b4", lw=2, label=f"Isotherm at {T:.1f} K")
        self.ax_pv.set_xscale("log")
        self.ax_pv.set_xlabel("Molar Volume Vm [cm³/mol]")
        self.ax_pv.set_ylabel("Pressure [bar]")
        self.ax_pv.set_title(f"P-V Isotherm ({core.model} EOS)")
        self.ax_pv.grid(True, which="both", ls="--", alpha=0.5)

        # Plot current operating point(s)
        for r in roots:
            vm_r = (r * R_GAS * T) / P * 1e6
            self.ax_pv.scatter(vm_r, P / 1e5, color="#d62728", s=60, zorder=5)
        self.ax_pv.axhline(P / 1e5, color="#d62728", ls=":", alpha=0.6, label=f"P = {P/1e5:.1f} bar")
        self.ax_pv.legend(loc="upper right", fontsize=8)

        # 2. Fugacity vs Pressure Plot
        self.ax_fug.clear()
        p_eval = [p_step * 1e5 for p_step in range(1, int(1.8 * core.Pc / 1e5), 2)]
        f_eval = []
        p_valid = []

        for p_val in p_eval:
            r_eval = core.solve_z(T, p_val)
            if r_eval:
                # evaluate for vapor root (highest z)
                z_vap = r_eval[-1]
                p_props = core.properties(T, p_val, z_vap)
                p_valid.append(p_val / 1e5)
                f_eval.append(p_props["f"] / 1e5)

        self.ax_fug.plot(p_valid, f_eval, color="#2ca02c", lw=2, label="EOS Real Gas Fugacity (f)")
        self.ax_fug.plot(p_valid, p_valid, color="#7f7f7f", ls="--", label="Ideal Gas (f = P)")
        self.ax_fug.set_xlabel("System Pressure P [bar]")
        self.ax_fug.set_ylabel("Pure Component Fugacity f [bar]")
        self.ax_fug.set_title("Pure Gas Fugacity vs Pressure")
        self.ax_fug.grid(True, ls="--", alpha=0.5)
        self.ax_fug.legend(loc="upper left", fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = PVTApp()
    app.mainloop()