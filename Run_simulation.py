import time
import numpy as np
import User_params as prm
from Road_file import Road
from Controller import SMPC_controller
from Linear_safety_constraints_cases import get_coefficient_for_cases
np.set_printoptions(precision=4, suppress=True)

# create road and controller objects and load parameters
road = Road()
road.generate_reference()
road.generate_dos(prm.A_do, prm.B_do, prm.K_do, prm.Sigma, do_specs=prm.do_specs, beta=0.95, N=prm.N,
                    prop_covariance=prm.dos_propagate_covariance, len_dos=prm.len_s, wid_dos=prm.wid_s)
lin_c_params = prm.lin_c_params
dmin = road.dmin+prm.wid_o['v']/2
dmax = road.dmax-prm.wid_o['v']/2
lin_c_params['wlane'] = road.w_lane
lin_c_params['s_int_lb'] = road.s_end_straight
lin_c_params['s_int_ub'] = road.s_end_turn
lin_c_params['dmax'] = dmax
x_av = prm.x_av
x_ref = prm.x_ref
controller = SMPC_controller(prm.curved_bicycle_matrices, prm.build_pred_matrices, x_ref, prm.dos_model,
                                get_coefficient_for_cases, prm, dmin, dmax, lin_c_params)

# logging variables
x_log = [x_av] # cartesia coordinates AV state
x_fre_log = [] # frenet coordinates AV state
q_log = []
comp_time_log = []
timesteps = range(prm.N_iter)

# simulation loop
for j in timesteps:

    # AV state in frenet frame
    x_av_fre = road.cartesian2frenet([x_av])[0] # [s, d, phi, k]
    # measurement DOs
    dos_obj = road.get_dos_info()

    # SMPC step
    t0 = time.time()
    controller.predict_dos_traj(dos_obj, road.cartesian2frenet_dos, prm)
    u = controller.run_step(x_av_fre[:-1], x_av_fre[-1, 0])
    comp_time_log.append(time.time()-t0)

    # update state AV (nonlinear model)
    x_av = np.reshape(prm.av_nonlin_model(x_av[:, 0], u, prm.T, prm), (-1, 1))
    print(j, 'x = ', x_av[:, 0].T, 'u = ', u.T)

    # update state DOs
    road.dos_step()

    # log variables
    x_log.append(x_av)
    x_fre_log.append(x_av_fre)
    q_log.append(controller.q)

x_log = np.array(x_log)
x_fre_log = np.array(x_fre_log)
print('Average computation time: ', np.average(comp_time_log))
dos_info = [(x_log[:, :3], prm.diag['v'], prm.gamma['v'], prm.len_o['v'], prm.wid_o['v'], 'b')]
for d in road.dos:
    typ = d['type']
    traj = np.array(d['traj'])
    traj = np.concatenate((traj[:, 0], traj[:, 2], np.arctan2(traj[:, 3], traj[:, 1])), axis=1)
    dos_info.append((traj, prm.diag[typ], prm.gamma[typ], prm.len_o[typ], prm.wid_o[typ], 'r'))

prm.plot_dynamic_sequence(x_log, x_fre_log, q_log, dos_info, road, timesteps, frame_time=prm.frame_time)


