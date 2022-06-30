import numpy as np
import cmath
from scipy.integrate import odeint
from matplotlib import pyplot as plt, patches

"""
    numerically evaluate matrices of linearized discretized bicycle model in Frenet coordinates
"""
def curved_bicycle_matrices(x0, T, lf, lr, k, tol=1e-5):
    d = x0[1].item()
    v = x0[3].item()
    c0 = np.cos(x0[2].item())
    s0 = np.sin(x0[2].item())
    alpha1 = lr/(lr+lf)
    
    w0 = 1-k*d
    w1 = 1/w0
    l = v*k*w1

    z0 = cmath.sqrt(1-5*c0**2)
    z1 = s0+z0
    z2 = s0-z0
    e1 = np.exp(l*z1*T/2)
    e2 = np.exp(l*z2*T/2)
    z3 = e1-e2
    z4 = z1*e2-z2*e1
    z5 = z1*e1-z2*e2

    if np.abs(z0)<tol:
        z6 = l*T
        z7 = 2
        z8 = 2*(1+l*s0*T)
        z9 = l*T**2/2
        z10 = 2*T
        z11 = 2*T+l*s0*T**2
    else:
        z6 = z3/z0
        z7 = z4/z0
        z8 = z5/z0
        if np.abs(l)<tol:
            z9 = 0
            z10 = 2*T
            z11 = 2*T+l*s0*T**2
        else:
            if np.abs(z1*z2)<tol:
                if np.abs(z1)<tol:
                    z9 = (T-2*(e2-1)/l/z2)/z0
                    z10 = -z2/z0*T
                else:
                    z9 = (2*(e1-1)/l/z1-T)/z0
                    z10 = z1/z0*T
            else:
                z9 = 2*((e1-1)/z1-(e2-1)/z2)/z0/l
                z10 = 2*((e2-1)*z1/z2-(e1-1)*z2/z1)/z0/l
            z11 = 2*z6/l

    if np.abs(l)<tol:
        a13 = -v*s0*T
        a14 = T*c0*w1
        a23 = v*c0*T
        a24 = T*s0
        a34 = -k*c0*T*w1
        b11 = c0*T**2/2*w1
        b21 = s0*T**2/2
        b31 = -k*T**2*c0/2*w1
        b12 = -(1+v*T/2/lr)*v*alpha1*s0*T*w1
        b22 = (1+v*T/2/lr)*v*T*c0*alpha1
        b32 = v*alpha1*T/lr
    else:
        if np.abs(c0)<tol:
            a14 = 0
            a24 = s0*T
            a34 = 0
            b11 = 0
            b21 = s0*T**2/2
            b31 = 0
        else:
            a14 = (s0*(1-z8/2)+z6)/(v*k*c0)
            a24 = (-1+s0*c0**2*z6+z7/2)/l/c0**2
            a34 = (s0*(z8/2-1)-z6)/v/c0
            b11 = (s0*(T-z11/2)+z9)/(v*k*c0)
            b21 = (-T+s0*c0**2*z9+z10/2)/l/c0**2
            b31 = (s0*(z11/2-T)-z9)/v/c0
        a13 = (1-z8/2)/k
        a23 = w0*c0*z6/k
        w2 = (w0/k/lr+s0)
        b12 = (-T*s0+c0**2*z9+(T-z11/2)*w2)*v*alpha1*w1
        b22 = (z10/2+z9*w2)*v*c0*alpha1
        b32 = (z11*w2/2-z9*c0**2)*l*alpha1
    

    A = np.array([[1, c0*w1*z6, a13, a14], 
                    [0, z7/2, a23, a24],
                    [0, -k*c0*z6*w1, z8/2, a34],
                    [0, 0, 0, 1]])

    B = np.array([[b11, b12],
                    [b21, b22],
                    [b31, b32],
                    [T, 0]])
      
    q1 = 1/(1-k*x0[1].item())
    q2 = np.cos(x0[2].item())
    q3 = x0[3].item()*q2*q1
    fc = np.array([[q3, x0[3].item()*np.sin(x0[2].item()), -k*q3, 0]]).T
    return {"A": A.real, "B": B.real, "fc": fc.real}

"""
    prediction matrices for compact state prediction
"""
def build_pred_matrices(A, B, fc, x0, N, T):
    AA = np.tile(A, (N, 1))
    BB = np.zeros((N*B.shape[0], N*B.shape[1]))
    BB[:B.shape[0],:B.shape[1]] = B
    CC = np.zeros((N*A.shape[0], N*A.shape[1]))
    CC[:A.shape[0],:A.shape[1]] = np.eye(A.shape[0])
    for k in range(1,N,1):
        AA[k*A.shape[0]:(k+1)*A.shape[0], :] = A.dot(AA[(k-1)*A.shape[0]:k*A.shape[0], :])
        BB[k*B.shape[0]:(k+1)*B.shape[0], B.shape[1]:] = BB[(k-1)*B.shape[0]:k*B.shape[0], :-B.shape[1]]
        BB[k*B.shape[0]:(k+1)*B.shape[0], :B.shape[1]] = A.dot(
                                                            BB[(k-1)*B.shape[0]:k*B.shape[0], :B.shape[1]])
        CC[k*A.shape[0]:(k+1)*A.shape[0], A.shape[0]:] = CC[(k-1)*A.shape[0]:k*A.shape[0], :-A.shape[1]]
        CC[k*A.shape[0]:(k+1)*A.shape[0], :A.shape[0]] = A.dot(
                                                           CC[(k-1)*A.shape[0]:k*A.shape[0], :A.shape[1]])
    cterm = np.tile(x0+T*fc-A.dot(x0), (N, 1))
    return {"AA": AA, "BB": BB, "CC": CC, "cterm": cterm}

"""
    nonlinear bicycle model for real state update in cartesian coordinates
"""
def nl_bicycle_model(x, t, u, prm):
    al_ = np.arctan(prm.lr/(prm.lr+prm.lf)*np.tan(u[1]))
    theta = al_+x[2]
    return np.array([x[3]*np.cos(theta),
                    x[3]*np.sin(theta),
                    x[3]*np.sin(al_)/prm.lr,
                    u[0]])

"""
    nonlinear method of AV dynamics
"""
def av_nonlin_model(x0, u, T, prm, n_steps=100):
    y = odeint(nl_bicycle_model, x0, np.linspace(0, T, n_steps), args=(u, prm))
    return y[-1, :]

"""
    prediction method of DOs
"""
def dos_model(x0, N, A, B, K, x_ref):
    A_cl = A+B.dot(K)
    u_cl = -B.dot(K.dot(x_ref))
    x = [x0]
    for j in range(N):
        x.append(A_cl.dot(x[-1])+u_cl)
    return x[1:]

"""
    propagation of DO covariance
"""
def dos_propagate_covariance(N, Phi, D, Sigma_w, P0=None):
    if P0==None:
        P0 = 0*Phi
    P = [P0]
    for j in range(N):
        P.append(Phi.dot(P[-1]).dot(Phi.T)+D.dot(Sigma_w).dot(D.T))
    return P[1:]

"""
    plot sequence of simulation frames
"""
def plot_dynamic_sequence(x_log, x_fre_log, q_log, dos_info, road, timesteps, frame_time=5):
    _, ax = plt.subplots()
    plt.suptitle('Dynamic scene', fontsize=12)
    road.plot_road(ax)
    ax.set_xlabel('x [m]', size=10)
    ax.set_ylabel('y [m]', size=10)
    plt.show(block=False)
    sd_lint_vec = np.linspace(-2, 3, num=10)
    for j in timesteps:
        for obj in dos_info:
            rect = patches.Rectangle((obj[0][j, 0]-obj[1]*np.cos(obj[2]+obj[0][j, 2]), obj[0][j, 1]-obj[1]*np.sin(obj[2]+obj[0][j, 2])),
                                        obj[3], obj[4], edgecolor='k', facecolor=obj[5], linewidth=2, angle=np.rad2deg(obj[0][j, 2]))
            ax.add_patch(rect)
        if j>0:
            road.plot_constraint(ax, q_log[j], x_fre_log[j-1, :], sd_lint_vec)
        ax.plot(x_log[j, 0], x_log[j, 1], 'or')
        lim_fact = 4.5
        ax.set_xlim([x_log[j, 0]-lim_fact*road.w_lane, x_log[j, 0]+lim_fact*road.w_lane])
        ax.set_ylim([max(x_log[j, 1]-lim_fact*road.w_lane, x_log[j, 1]-lim_fact*road.w_lane*2/3),
                        max(x_log[j, 1]+lim_fact*road.w_lane, x_log[j, 1]-lim_fact*road.w_lane*4/3)])
        plt.show(block=False)
        plt.pause(frame_time)
        del ax.lines[-1]
        del ax.patches[-len(dos_info):]
        if j>0 and len(q_log[j])>0:
            del ax.lines[-len(q_log[j]):]
    plt.close()


# simulation parameters
T = 0.1 # sampling time
N = 8 # prediction horizon length
frame_time = 0.08 # time for frame visualization
N_iter = 300 # number iteration steps

# AV parameters
x_av = np.array([[-105, -5.25, 0, 9]]).T # initial state in cartesian coordinates [x, y, yaw, v]
x_ref = np.tile(np.array([[0, 0, 0, 10]]).T, (N, 1))
n = 4 # state dimension
m = 2 # input dimension
u_min = np.array([-9, -0.4]) # input lower bounds (acceleration, steering angle)
u_max = np.array([5, 0.4]) # input upper bounds (acceleration, steering angle)
du_max = np.array([9, 0.4]) # maximum input rate (acceleration, steering angle)
vmin = 0 # minimum speed
vmax = 50/3.6 # maximal speed
lf = 1.9 # length from front axle
lr = 1.9 # length from rear axle
Q = np.diag([0, 0.1, 5, 1]) # state weight
R = np.diag([1, 0.1]) # input weight
S = np.diag([1, 0.1]) # input rate weight

# DOs parameters
len_o = {'v': 4.5,# length of obstacles (per type: v=vehicle, c=cyclist, p=pedestrian)
        'c': 2,
        'p': 1}
len_s = {'v': 4.5, # length of obstacle shapes (per type: v=vehicle, c=cyclist, p=pedestrian)
        'c': 2.5,
        'p': 2.5}
wid_o = {'v': 1.8, # width of obstacles (per type: v=vehicle, c=cyclist, p=pedestrian)
        'c': 1,
        'p': 1}
wid_s = {'v': 1.8, # width of obstacle shapes (per type: v=vehicle, c=cyclist, p=pedestrian)
        'c': 2.5,
        'p': 2.5}
gamma = {key: np.arctan2(wid_o[key], len_o[key]) for key in len_o.keys()}
diag = {key: np.sqrt(len_o[key]**2+wid_o[key]**2)/2 for key in len_o.keys()}
A_do = np.array([[1, T, 0, 0], # DO model A matrix
                    [0, 1, 0, 0],
                    [0, 0, 1, T],
                    [0, 0, 0, 1]])
B_do = np.array([[0.5*T**2, 0], # DO model A matrix
                    [T, 0],
                    [0, 0.5*T**2],
                    [0, T]])
K_do = {'vh': np.array([[0, -0.68, 0, 0], [0, 0, -0.7, -1.32]]), # DO feedback gains (h=horizontal, v=vertical)
        'vv': np.array([[-0.7, -1.32, 0, 0], [0, 0, 0, -0.68]]),
        'ch': np.array([[0, -0.53, 0, 0], [0, 0, -0.52, -1.14]]),
        'cv': np.array([[-0.52, -1.14, 0, 0], [0, 0, 0, -0.53]]),
        'ph': np.array([[0, -0.31, 0, 0], [0, 0, -0.30, -0.84]]),
        'pv': np.array([[-0.30, -0.84, 0, 0], [0, 0, 0, -0.31]])}
Sigma = {'vh': np.diag([0.44, 0.09]), # DO additive noise covariance matrices (h=horizontal, v=vertical)
                'vv': np.diag([0.09, 0.44]),
                'ch': np.diag([0.2, 0.09]),
                'cv': np.diag([0.09, 0.2]),
                'ph': np.diag([0.05, 0.05]),
                'pv': np.diag([0.05, 0.05])}
do_specs=[{'type': 'vh', 'state': np.array([[80.5, -7, 1.75, 0]]).T, 'ref': np.array([[0, -7, 1.75, 0]]).T},
            {'type': 'ph', 'state': np.array([[33.25, -0.8, 14, 0]]).T, 'ref': np.array([[0, -2, 14, 0]]).T},
            {'type': 'vh', 'state': np.array([[-91, 5, -4.55, 0]]).T, 'ref': np.array([[0, 5, -5.25, 0]]).T},
            {'type': 'cv', 'state': np.array([[6.3, 0, -25, 1.5]]).T, 'ref': np.array([[6.3, 0, 0, 4]]).T}]


# linear safety constraints parameters
lin_c_params = {'N': N,
                'ds_close': 10,
                'ds_far': 17,
                'ds_far_back': 8,
                'ds_overtaken': len_o['v']*1.2,
                'a_min': u_min[0],
                't_vec': np.reshape(T*np.arange(1, N+1, 1), (-1, 1)),
                't2_vec': np.reshape(T*np.arange(1, N+1, 1), (-1, 1))**2,
                'av_dim': np.array([[len_o['v']], [wid_o['v']]]),
                'Deltadmax': 0.15}