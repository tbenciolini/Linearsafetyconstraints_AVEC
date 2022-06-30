import numpy as np
from scipy import optimize as opt

class SMPC_controller(object):

    def __init__(self, linearization, build_pred_matrices, x_ref, dos_model, lin_safety_constr_coeff,
                    prm, dmin, dmax, linc_vals):
        self.linearization = linearization # linearization method
        self.build_pred_matrices = build_pred_matrices # method for prediction
        self.x_ref = x_ref # reference state
        self.dos_model = dos_model  # model for prediction of obstacles
        self.lin_safety_constr_coeff = lin_safety_constr_coeff # generation safety linear constraints
        self.linc_vals = linc_vals # parameters for linear constraints generation
        self.n = prm.n # state dimension
        self.m = prm.m # input dimension
        self.T = prm.T # sampling time
        self.N = prm.N # prediction horizon
        self.lr = prm.lr # parameter bicycle model
        self.lf = prm.lf # parameter bicycle model
        self.u_guess = np.zeros((self.m*self.N)) # guess solution optimal control problem
        self.u_last = np.zeros((self.m)) # last applied input
        self.Q = prm.Q  # state weight
        self.R = prm.R # input weight
        self.S = prm.S # input rate weight
        self.u_min = np.tile(prm.u_min, (1, self.N)).T # input lower bound
        self.u_max = np.tile(prm.u_max, (1, self.N)).T # input upper bound
        self.du_max = np.tile(prm.du_max, (1, self.N-1)).T # maximum input rate
        self.dmin = np.tile(dmin, (1, self.N)).T # minimum lateral displacement
        self.dmax = np.tile(dmax, (1, self.N)).T # maximum lateral displacement
        self.vmin = np.tile(prm.vmin, (1, self.N)).T # minimum speed
        self.vmax = np.tile(prm.vmax, (1, self.N)).T # maximum speed
        self.set_bounds()

    """
        set bounds for input sequence
    """
    def set_bounds(self):
        self.bounds = []
        for _ in range(self.N):
            self.bounds.append((self.u_min[0, 0], self.u_max[0, 0]))
            self.bounds.append((self.u_min[1, 0], self.u_max[1, 0]))

    """
        update bounds for first input based on last input applied
    """
    def update_bounds_online(self):
        for j in range(self.m):
            self.bounds[j] = (max(self.u_min[j, 0], self.u_last[j]-self.du_max[j, 0]),
                                min(self.u_max[j, 0], self.u_last[j]+self.du_max[j, 0]))

    """
        numerically evaluate cost function during numerical optimization
    """
    def costf(self, U):
        U0 = U.reshape(-1,1)
        x = self.pmtr['AA'].dot(self.x0)+self.pmtr['BB'].dot(U0)+self.pmtr['CC'].dot(self.pmtr['cterm']) # states only from 1 to N (no step 0)
        u_last = self.u_last
        cost = 0
        for h in range(self.N):
            dx = x[h*self.n:(h+1)*self.n, :]-self.x_ref[h*self.n:(h+1)*self.n, :]   # is actually dx(k+1)
            u0 = U0[h*self.m:(h+1)*self.m, :]
            du = u0-u_last
            cost += (dx.T.dot(self.Q)).dot(dx)+(u0.T.dot(self.R)).dot(u0)+(du.T.dot(self.S)).dot(du)
            u_last = u0
        return cost[0][0]

    """
        numerically evaluate constraints function during numerical optimization
    """
    def cstrf(self, U):
        U0 = U.reshape(-1,1)
        x = self.pmtr['AA'].dot(self.x0)+self.pmtr['BB'].dot(U0)+self.pmtr['CC'].dot(self.pmtr['cterm']) # states only from 1 to N (no step 0)
        cineq = [x[1:x.shape[0]:4]-self.dmin]
        cineq.append(self.dmax-x[1:x.shape[0]:4])
        cineq.append(x[3:x.shape[0]:4]-self.vmin)
        cineq.append(self.vmax-x[3:x.shape[0]:4])
        cineq.append(self.du_max-np.abs(U0[2:]-U0[:-2]))
        for qi in self.q:
            cineq.append(np.multiply(x[0:-1:self.n], qi['s'])+np.multiply(x[1:-1:self.n], qi['d'])+qi['t'])
        return np.vstack(cineq)[:, 0]

    """
        predict future trajectory of DOs
    """
    def predict_dos_traj(self, dos_obj, cartesian2frenet, prm):
        self.dos_traj = []
        for obj in dos_obj:
            traj = self.dos_model(obj[0], self.N, prm.A_do, prm.B_do, obj[1], obj[0])
            self.dos_traj.append((cartesian2frenet([obj[0]]), cartesian2frenet(traj), obj[2], obj[3]))
            # gain K is assumed known (the exact one is used), the current state is used as reference

    """
        computes coefficients of positional constraints
    """
    def set_pos_cstr_coeffs(self):
        self.q = []
        for traj in self.dos_traj:
            # breakpoint()
            # qi = self.lin_safety_constr_coeff(self.x0, traj[0], traj[1], traj[2], traj[3])
            qi = self.lin_safety_constr_coeff(self.x0, traj[0], traj[1], traj[2], traj[3], self.linc_vals)
            if len(qi['s'])>0:
                self.q.append(qi)

    """
        run SMPC step
    """
    def run_step(self, x0, k):
        self.x0 = x0
        bmtr = self.linearization(x0, self.T, self.lf, self.lr, k) # linear discrete bicycle model matrices
        self.pmtr = self.build_pred_matrices(bmtr['A'], bmtr['B'], bmtr['fc'], x0, self.N, self.T) # matrices for compact prediction
        self.set_pos_cstr_coeffs()
        self.update_bounds_online()
        cstr = [{'type': 'ineq', 'fun': lambda U: self.cstrf(U)}]

        # solve optimal control problem
        try:
            res = opt.minimize(self.costf, self.u_guess, args=(), method='SLSQP',
                                bounds=self.bounds, constraints=cstr)
        except:
            # if initial guess violates constraints, solve with null guess
            res = opt.minimize(self.costf, 0*self.u_guess, args=(), method='SLSQP',
                                bounds=self.bounds, constraints=cstr)
        U = res.x.reshape(-1, 1)
        self.u_guess = np.concatenate((U[2:], 0*U[:2]), axis=0)
        self.u_last = U[:2]
        return self.u_last[:, 0]