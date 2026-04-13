import numpy as np
from scipy.optimize import fsolve, minimize

class ParallelThermoelectricSensitivity:
    def __init__(self, params):
        # Bloco de alumínio
        self.Lx, self.Ly, self.Lz = params['dim_bloco']
        self.rho_al = params.get('rho_al', 2700.0)
        self.cp_al = params.get('cp_al', 900.0)
        self.m_al = self.rho_al * (self.Lx * self.Ly * self.Lz)
        self.C_al = self.m_al * self.cp_al

        # Isolamento
        self.L_ins = params['L_ins']
        self.k_ins = params['k_ins']
        self.h_int = params.get('h_int', 10.0)
        self.h_ext = params.get('h_ext', 10.0)
        self.T_amb = params['T_amb']
        self._update_geometry()

        # TECs
        self.n_TEC = params['n_TEC']
        self.alpha0 = params['alpha0']
        self.R0 = params['R0']
        self.K0 = params['K0']
        self.beta_alpha = params.get('beta_alpha', 0.0)
        self.beta_R = params.get('beta_R', 0.0)
        self.beta_K = params.get('beta_K', 0.0)

        # Dissipador
        self.cooling_type = params['cooling_type']
        if self.cooling_type == 'air':
            self.R_hs = params['R_hs']
        elif self.cooling_type == 'water':
            self.m_dot_w = params['m_dot_w']
            self.cp_w = params.get('cp_w', 4180.0)
            self.T_w_in = params['T_w_in']
            self.R_block = params.get('R_block', 0.02)
        else:
            raise ValueError("cooling_type deve ser 'air' ou 'water'")

    def _update_geometry(self):
        Lx_ext = self.Lx + 2*self.L_ins
        Ly_ext = self.Ly + 2*self.L_ins
        Lz_ext = self.Lz + 2*self.L_ins
        self.A_ext = 2*(Lx_ext*Ly_ext + Lx_ext*Lz_ext + Ly_ext*Lz_ext)
        self.U = 1.0 / (1.0/self.h_ext + self.L_ins/self.k_ins + 1.0/self.h_int)

    def alpha(self, Tm):
        return self.alpha0 * (1 + self.beta_alpha * (Tm - 25.0))
    def dalpha_dTm(self, Tm):
        return self.alpha0 * self.beta_alpha

    def R(self, Tm):
        return self.R0 * (1 + self.beta_R * (Tm - 25.0))
    def dR_dTm(self, Tm):
        return self.R0 * self.beta_R

    def K(self, Tm):
        return self.K0 * (1 + self.beta_K * (Tm - 25.0))
    def dK_dTm(self, Tm):
        return self.K0 * self.beta_K

    def Qc_single(self, Tc, Th, I, alpha, R, K):
        Tc_K = Tc + 273.15
        Th_K = Th + 273.15
        return alpha * I * Tc_K - 0.5 * I**2 * R - K * (Th_K - Tc_K)

    def Qh_single(self, Tc, Th, I, alpha, R, K):
        Tc_K = Tc + 273.15
        Th_K = Th + 273.15
        return alpha * I * Th_K + 0.5 * I**2 * R - K * (Th_K - Tc_K)

    def Pel_single(self, Tc, Th, I, alpha, R, K):
        Tc_K = Tc + 273.15
        Th_K = Th + 273.15
        return I**2 * R + alpha * I * (Th_K - Tc_K)

    def total_performance(self, Tc, Th, I):
        Tm = (Tc + Th) / 2.0
        alpha = self.alpha(Tm)
        R = self.R(Tm)
        K = self.K(Tm)
        Qc_tot = 0.0
        Qh_tot = 0.0
        Pel_tot = 0.0
        for _ in range(self.n_TEC):
            Qc_tot += self.Qc_single(Tc, Th, I, alpha, R, K)
            Qh_tot += self.Qh_single(Tc, Th, I, alpha, R, K)
            Pel_tot += self.Pel_single(Tc, Th, I, alpha, R, K)
        COP = Qc_tot / Pel_tot if Pel_tot > 0 else 0.0
        return Qc_tot, Qh_tot, Pel_tot, COP, alpha, R, K

    def steady_state_residuals(self, x, I):
        Tc, Th = x
        Qc, Qh, _, _, _, _, _ = self.total_performance(Tc, Th, I)
        loss = self.U * self.A_ext * (self.T_amb - Tc)
        if self.cooling_type == 'air':
            Th_calc = self.T_amb + Qh * self.R_hs
        else:
            delta_Tw = Qh / (self.m_dot_w * self.cp_w)
            Th_calc = self.T_w_in + delta_Tw + Qh * self.R_block
        return np.array([Qc - loss, Th - Th_calc]).flatten()

    def steady_state(self, I, guess=None):
        # Converte array para escalar, se necessário
        if isinstance(I, (np.ndarray, list, tuple)):
            I = I[0]
        I = float(I)
        if guess is None:
            guess = [self.T_amb - 15.0, self.T_amb + 10.0]
        sol = fsolve(self.steady_state_residuals, guess, args=(I,), xtol=1e-6)
        Tc, Th = sol
        Qc, Qh, Pel, COP, alpha, R, K = self.total_performance(Tc, Th, I)
        return {'Tc': Tc, 'Th': Th, 'Qc': Qc, 'Qh': Qh, 'Pel': Pel, 'COP': COP,
                'alpha': alpha, 'R': R, 'K': K}

    def sensitivity_dTc_dI(self, I):
        if isinstance(I, (np.ndarray, list, tuple)):
            I = I[0]
        I = float(I)
        ss = self.steady_state(I)
        Tc, Th = ss['Tc'], ss['Th']
        Tm = (Tc+Th)/2.0
        alpha, R, K = ss['alpha'], ss['R'], ss['K']
        dalpha = self.dalpha_dTm(Tm)
        dR = self.dR_dTm(Tm)
        dK = self.dK_dTm(Tm)

        dalpha_dTc = dalpha * 0.5
        dalpha_dTh = dalpha * 0.5
        dR_dTc = dR * 0.5
        dR_dTh = dR * 0.5
        dK_dTc = dK * 0.5
        dK_dTh = dK * 0.5

        Tc_K = Tc + 273.15
        Th_K = Th + 273.15
        n = self.n_TEC

        dQc_dalpha = I * Tc_K
        dQc_dR = -0.5 * I**2
        dQc_dK = -(Th_K - Tc_K)
        dQc_dTc = alpha * I + K
        dQc_dTh = -K

        dQh_dalpha = I * Th_K
        dQh_dR = 0.5 * I**2
        dQh_dK = -(Th_K - Tc_K)
        dQh_dTc = K
        dQh_dTh = alpha * I - K

        dQc_dTc_tot = n * (dQc_dTc + dQc_dalpha*dalpha_dTc + dQc_dR*dR_dTc + dQc_dK*dK_dTc)
        dQc_dTh_tot = n * (dQc_dTh + dQc_dalpha*dalpha_dTh + dQc_dR*dR_dTh + dQc_dK*dK_dTh)
        dQc_dI_tot = n * (alpha * Tc_K - I * R)

        dQh_dTc_tot = n * (dQh_dTc + dQh_dalpha*dalpha_dTc + dQh_dR*dR_dTc + dQh_dK*dK_dTc)
        dQh_dTh_tot = n * (dQh_dTh + dQh_dalpha*dalpha_dTh + dQh_dR*dR_dTh + dQh_dK*dK_dTh)
        dQh_dI_tot = n * (alpha * Th_K + I * R)

        dLoss_dTc = -self.U * self.A_ext

        if self.cooling_type == 'air':
            dG_dTc = -self.R_hs * dQh_dTc_tot
            dG_dTh = 1.0 - self.R_hs * dQh_dTh_tot
            dG_dI = -self.R_hs * dQh_dI_tot
        else:
            dTh_dQh = 1.0/(self.m_dot_w * self.cp_w) + self.R_block
            dG_dTc = -dTh_dQh * dQh_dTc_tot
            dG_dTh = 1.0 - dTh_dQh * dQh_dTh_tot
            dG_dI = -dTh_dQh * dQh_dI_tot

        dF_dTc = dQc_dTc_tot - dLoss_dTc
        dF_dTh = dQc_dTh_tot
        dF_dI = dQc_dI_tot

        A = np.array([[dF_dTc, dF_dTh], [dG_dTc, dG_dTh]])
        b = np.array([-dF_dI, -dG_dI])
        try:
            dTc_dI, dTh_dI = np.linalg.solve(A, b)
        except:
            dTc_dI = dTh_dI = np.nan
        return dTc_dI, dTh_dI

    def sensitivity_dQc_dI(self, I):
        if isinstance(I, (np.ndarray, list, tuple)):
            I = I[0]
        I = float(I)
        dTc, dTh = self.sensitivity_dTc_dI(I)
        ss = self.steady_state(I)
        Tc, Th = ss['Tc'], ss['Th']
        Tm = (Tc+Th)/2.0
        alpha, R, K = ss['alpha'], ss['R'], ss['K']
        dalpha = self.dalpha_dTm(Tm)
        dR = self.dR_dTm(Tm)
        dK = self.dK_dTm(Tm)

        dalpha_dI = dalpha * 0.5 * (dTc + dTh)
        dR_dI = dR * 0.5 * (dTc + dTh)
        dK_dI = dK * 0.5 * (dTc + dTh)

        Tc_K = Tc + 273.15
        Th_K = Th + 273.15
        n = self.n_TEC
        dQc = n * (alpha * Tc_K + alpha * I * dTc - I * R - 0.5*I**2 * dR_dI
                   - K * (dTh - dTc) - (Th_K - Tc_K) * dK_dI + dalpha_dI * I * Tc_K)
        return dQc

    def sensitivity_dCOP_dI(self, I):
        if isinstance(I, (np.ndarray, list, tuple)):
            I = I[0]
        I = float(I)
        ss = self.steady_state(I)
        Qc = ss['Qc']; Pel = ss['Pel']
        dQc = self.sensitivity_dQc_dI(I)

        Tc, Th = ss['Tc'], ss['Th']
        Tm = (Tc+Th)/2.0
        alpha, R, K = ss['alpha'], ss['R'], ss['K']
        dalpha = self.dalpha_dTm(Tm); dR = self.dR_dTm(Tm); dK = self.dK_dTm(Tm)
        dTc, dTh = self.sensitivity_dTc_dI(I)
        dalpha_dI = dalpha * 0.5 * (dTc + dTh)
        dR_dI = dR * 0.5 * (dTc + dTh)
        dK_dI = dK * 0.5 * (dTc + dTh)
        Tc_K = Tc + 273.15
        Th_K = Th + 273.15
        n = self.n_TEC
        dPel_dI = n * (2*I*R + I**2 * dR_dI + alpha * (Th_K - Tc_K) +
                       dalpha_dI * I * (Th_K - Tc_K) + alpha * I * (dTh - dTc))
        dCOP = (dQc * Pel - Qc * dPel_dI) / (Pel**2) if Pel > 0 else np.nan
        return dCOP

    def optimize_current_gradient(self, I_initial=3.0):
        def obj(I):
            # I é array ou escalar; converte para escalar
            if isinstance(I, (np.ndarray, list, tuple)):
                I_val = I[0]
            else:
                I_val = I
            return -self.steady_state(I_val)['COP']
        def grad(I):
            if isinstance(I, (np.ndarray, list, tuple)):
                I_val = I[0]
            else:
                I_val = I
            return -self.sensitivity_dCOP_dI(I_val)
        res = minimize(obj, I_initial, method='L-BFGS-B', jac=grad, bounds=[(1.0, 7.0)])
        I_opt = res.x[0]
        return I_opt, self.steady_state(I_opt)

    def sensitivity_dTc_dLins(self, I, delta=1e-4):
        if isinstance(I, (np.ndarray, list, tuple)):
            I = I[0]
        I = float(I)
        L_orig = self.L_ins
        A_orig, U_orig = self.A_ext, self.U

        def Tc_at_L(L):
            self.L_ins = L
            self._update_geometry()
            try:
                return self.steady_state(I)['Tc']
            except:
                return np.nan

        Tc_plus = Tc_at_L(L_orig + delta)
        Tc_minus = Tc_at_L(L_orig - delta)
        self.L_ins = L_orig
        self._update_geometry()
        if np.isnan(Tc_plus) or np.isnan(Tc_minus):
            return np.nan
        return (Tc_plus - Tc_minus) / (2*delta)


if __name__ == "__main__":
    params = {
        'dim_bloco': (0.20, 0.20, 0.20),
        'L_ins': 0.05,
        'k_ins': 0.035,
        'h_int': 10.0,
        'h_ext': 10.0,
        'T_amb': 36.0,
        'n_TEC': 2,
        'alpha0': 0.050,
        'R0': 2.0,
        'K0': 0.8,
        'beta_alpha': -0.0015,
        'beta_R': 0.004,
        'beta_K': 0.0015,
        'cooling_type': 'air',
        'R_hs': 0.1
    }
    sys = ParallelThermoelectricSensitivity(params)

    I_test = 3.82
    print(f"Estado estacionário para I = {I_test} A:")
    ss = sys.steady_state(I_test)
    for k, v in ss.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    dTc, dTh = sys.sensitivity_dTc_dI(I_test)
    print(f"\nSensibilidade dTc/dI = {dTc:.4f} °C/A")
    print(f"dTh/dI = {dTh:.4f} °C/A")
    print(f"dQc/dI = {sys.sensitivity_dQc_dI(I_test):.4f} W/A")
    print(f"dCOP/dI = {sys.sensitivity_dCOP_dI(I_test):.4f} 1/A")

    I_opt, res_opt = sys.optimize_current_gradient()
    print(f"\nCorrente ótima (máximo COP) via gradiente: I = {I_opt:.3f} A")
    for k, v in res_opt.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    dTc_dL = sys.sensitivity_dTc_dLins(I_opt, delta=1e-3)
    print(f"\nSensibilidade dTc/dL_ins = {dTc_dL:.2f} °C/m (≈ {dTc_dL*1000:.2f} °C/mm)")
