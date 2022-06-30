import numpy as np
import scipy
class Road:
    
    w_lane = 3.5
    xy_max = 150
    num_lanes = 2
    s_end_straight = xy_max-num_lanes*w_lane
    R_turn = (2*num_lanes-1)*w_lane
    k_turn = 1/R_turn
    s_end_turn = s_end_straight+R_turn*np.pi/2
    x_corner = (0.5-num_lanes)*w_lane
    y_corner = (num_lanes-0.5)*w_lane
    ref_dict = {'r': [], 't': [], 'n': [], 'k': []}
    dmin = -w_lane/2
    dmax = (num_lanes-0.5)*w_lane
    dos = []

    """
        transform cartesian coordinates into Frenet coordinates (for AV)
    """
    def cartesian2frenet(self, car_p):
        fre_p = []
        for p in car_p:
            diff = np.linalg.norm(self.ref_dict['r']-p[:2], axis=0)
            idx = np.argmin(diff)
            dv = p[:2].T-self.ref_dict['r'][:, idx]
            sgn = np.sign(self.ref_dict['t'][idx][0, 0]*dv[0, 1]-self.ref_dict['t'][idx][1, 0]*dv[0, 0])
            phi = p[2, 0]-np.arctan2(self.ref_dict['t'][idx][1, 0], self.ref_dict['t'][idx][0, 0])
            fre_p.append(np.array([[self.ref_dict['s'][idx]],
                                    [sgn*diff[idx]],
                                    [phi],
                                    [p[3, 0]],
                                    [self.ref_dict['k'][idx]]]))
        return fre_p

    """
        transform cartesian coordinates into Frenet coordinates (for DOs)
    """
    def cartesian2frenet_dos(self, car_p):
        fre_p = []
        for p in car_p:
            diff = np.linalg.norm(self.ref_dict['r']-p[0:-1:2], axis=0)
            idx = np.argmin(diff)
            dv = p[0:-1:2].T-self.ref_dict['r'][:, idx]
            sgn = np.sign(self.ref_dict['t'][idx][0, 0]*dv[0, 1]-self.ref_dict['t'][idx][1, 0]*dv[0, 0])
            fre_p.append(np.array([[self.ref_dict['s'][idx]],
                                    [sgn*diff[idx]]]))
        return np.reshape(np.stack(fre_p), (len(fre_p), -1))

    """
        transform Frenet coordinates into cartesian coordinates (sequence)
    """
    def frenet2cartesian(self, fre_p):
        car_p = []
        for p in fre_p:
            idx = np.argmin(np.abs(self.ref_dict['s']-p[0]))
            car_p.append(self.ref_dict['r'][:, idx]+p[1]*self.ref_dict['n'][idx].T)
        return car_p

    """
        return cartesian coordinates of a point at a given (Frenet) position s along the road (with zero displacement) 
    """
    def frenet2cartesian_s(self, s):
        # s=0, d=0 is x=-xy_min, y=-(num_lanes-0.5)*w_lane; then proceed straight on right-most lane and turn left
        if s<=self.s_end_straight:
            r = np.array([[-self.xy_max+s], [-(self.num_lanes-0.5)*self.w_lane]])
            t = np.array([[1], [0]])
            k = 0
        elif s<=self.s_end_turn:
            theta = (s-self.s_end_straight)/self.R_turn
            ctheta = np.cos(theta)
            stheta = np.sin(theta)
            r = np.array([[self.x_corner+self.R_turn*stheta], [self.y_corner-self.R_turn*ctheta]])
            t = np.array([[ctheta], [stheta]])
            k = -self.k_turn
        else:
            r = np.array([[(self.num_lanes-0.5)*self.w_lane], [self.y_corner+s-self.s_end_turn]])
            t = np.array([[0], [1]])
            k = 0
        n = np.array([[-t[1, 0]], [t[0, 0]]])
        return {'r': r, 't': t, 'n': n, 'k': k}


    """
        generate reference transformation from Frenet to cartesian frames
    """
    def generate_reference(self, s_min=0, s_max=300, n_points=6001):
        s_sequence = np.linspace(s_min, s_max, n_points)
        for s in s_sequence:
            dict_curr = self.frenet2cartesian_s(s)
            for key in self.ref_dict.keys(): self.ref_dict[key].append(dict_curr[key])
        self.ref_dict['s'] = s_sequence
        self.ref_dict['r'] = np.squeeze(np.array(self.ref_dict['r'])).T

    """
        plot road limits
    """
    def plot_road(self, ax, n_points=40):
        lim_v = np.linspace(self.w_lane*self.num_lanes, self.xy_max, n_points)
        c_v = 0*lim_v+1
        for h in range(4):
            s1 = (-1)**(h//2)
            s2 = (-1)**(h%2)
            ax.plot(s1*lim_v, 0*c_v, "k")
            ax.plot(0*c_v, s2*lim_v, "k")
            for j in range(self.num_lanes):
                if j==self.num_lanes-1: str = "k"
                else: str = "k--"
                ax.plot(s1*lim_v, s2*(j+1)*self.w_lane*c_v, str)
                ax.plot(s1*(j+1)*self.w_lane*c_v, s2*lim_v, str)

    """
        plot linear safety constraints
    """
    def plot_constraint(self, ax, q, x0_av, sd_int_vec):
        for qi in q:
            if qi['d'][0, 0]!=0:
                s_vec = x0_av[0]+sd_int_vec
                d_vec = -(qi['s'][0, 0]*s_vec+qi['t'][0, 0])/qi['d'][0, 0]
            else:
                d_vec = x0_av[1]+sd_int_vec
                s_vec = -(qi['d'][0, 0]*d_vec+qi['t'][0, 0])/qi['s'][0, 0]
            fre_p = np.array([s_vec, d_vec]).T.tolist()
            car_p = np.array(self.frenet2cartesian(fre_p))
            ax.plot(car_p[:, 0, 0], car_p[:, 0, 1], 'g', linewidth=2)
        
    """
        generate DOs given the specifications
    """
    def generate_dos(self, A, B, K_do, Sigma, do_specs=[{'type': 'vh', 'state': np.array([[5*w_lane, -10, 0.6*w_lane, 0]]).T,
                                'ref': np.array([[0, -12, 0.5*w_lane, 0]]).T}], beta=None, N=0, prop_covariance=None,
                                len_dos=None, wid_dos=None):
        self.A_do = A
        self.B_do = B
        for spec in do_specs:
            typ = spec['type']
            if len(typ)>2 or (typ[0]!='v' and typ[0]!='c' and typ[0]!='p') or (typ[1]!='h' and typ[1]!='v'):
                continue

            self.dos.append({'type': typ[0], 'K': K_do[typ], 'state': spec['state'], 'ref': spec['ref'],
                                'Sigma_sqrt': scipy.linalg.sqrtm(Sigma[typ]),'traj': [spec['state']]})
            if beta!=None:
                P = prop_covariance(N, A+B.dot(K_do[typ]), B, Sigma[typ])
                kappa = -np.log(1-beta)
                len_unc = 0.5*len_dos[typ[0]]*1.6+np.stack([np.sqrt(pp[0, 0]*kappa) for pp in P])
                wid_unc = 0.5*wid_dos[typ[0]]*1.6+np.stack([np.sqrt(pp[2, 2]*kappa) for pp in P])
                # len_unc = 1.3*(0.5*(len_dos['v']+len_dos[typ[0]])+np.stack([np.sqrt(pp[0, 0]*kappa) for pp in P]))
                # wid_unc = 1.3*(0.5*(wid_dos['v']+wid_dos[typ[0]])+np.stack([np.sqrt(pp[2, 2]*kappa) for pp in P]))
                self.dos[-1]['S'] = {'fl': (len_unc, wid_unc),
                                        'fr': (len_unc, -wid_unc),
                                        'bl': (-len_unc, wid_unc),
                                        'br': (-len_unc, -wid_unc)}

    """
        get measurements of DO and information needed for prediction
    """
    def get_dos_info(self):
        dos_obj = []
        for d in self.dos:
            dos_obj.append((d['state'], d['K'], d['type'], d['S']))
        return dos_obj

    """
        update position of DOs
    """
    def dos_step(self):
        for d in self.dos:
            d['state'] = (self.A_do+self.B_do.dot(d['K'])).dot(d['state'])+self.B_do.dot(-d['K'].dot(d['ref'])
                            +d['Sigma_sqrt'].dot(np.random.randn(self.B_do.shape[1], 1)))
            d['traj'].append(d['state'])
