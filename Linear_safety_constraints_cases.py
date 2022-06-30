import numpy as np

"""
    return coefficents for linear safety constraints
"""
def get_coefficient_for_cases(x0_av, x0_do, x_do, type_do, S_do, lin_c_params):
    # x0_av:    initial position AV
    # x0_do:    initial position DO
    # x_do:     predicted positions DO
    # type_do:  type of DO
    # S_do:     safety shape of DO

    case = identify_case(x0_av, x0_do, x_do, type_do, lin_c_params)
    if case=='stop_before_intersection':
        qs = -np.ones((len(x_do), 1))
        qd = np.zeros((len(x_do), 1))
        qt = np.maximum((lin_c_params['s_int_lb']+1)*np.ones((len(x_do), 1)),
                            x0_av[0, 0]+x0_av[3, 0]*lin_c_params['t_vec']+0.5*lin_c_params['a_min']*lin_c_params['t2_vec'])
    elif case=='vertical_back_of_V':
        qs = -np.ones((len(x_do), 1))
        qd = np.zeros((len(x_do), 1))
        qt = np.reshape(-0.5*lin_c_params['av_dim'][0]+x_do[:, 0]+S_do['br'][0], (-1, 1))
    elif case=='vertical_front_of_V':
        qs = np.ones((len(x_do), 1))
        qd = np.zeros((len(x_do), 1))
        qt = -np.reshape(0.5*lin_c_params['av_dim'][0]+x_do[:, 0]+S_do['fl'][0], (-1, 1))
    elif case=='inclined_bl':
        p0 = np.array([[x0_av[0, 0]+0.5*lin_c_params['av_dim'][0]],
                        [x0_av[1, 0]-0.5*lin_c_params['av_dim'][1]]])
        p1 = (x_do[:, 0]+S_do['bl'][0], x_do[:, 1]+S_do['bl'][1])
        qs = np.reshape(np.minimum(0, p0[1]-p1[1]), (-1, 1))
        qd = np.reshape(np.maximum(0, p1[0]-p0[0]), (-1, 1))
        qt = -qs*x0_av[0, 0]-qd*x0_av[1, 0]
    elif case=='horizontal_left_of_V':
        d0 = (np.round(x0_do[0, 1]/lin_c_params['wlane'])+0.5)*lin_c_params['wlane'] # right border of left lane
        qs = np.zeros((len(x_do), 1))
        qd = np.ones((len(x_do), 1))
        qt = -min(x0_av[1, 0]+lin_c_params['Deltadmax'], d0+lin_c_params['av_dim'][1]/2)*np.ones((len(x_do), 1))
    elif case=='horizontal_right_of_V':
        d0 = 0.5*lin_c_params['wlane'] # right border of current lane (right of V)
        qs = np.zeros((len(x_do), 1))
        qd = -np.ones((len(x_do), 1))
        qt = max(x0_av[1, 0]-lin_c_params['Deltadmax'], d0-lin_c_params['av_dim'][1]/2)*np.ones((len(x_do), 1))
    elif case=='vertical_back_of_P':
        qs = -np.ones((len(x_do), 1))
        qd = np.zeros((len(x_do), 1))
        qt = np.reshape(-0.5*lin_c_params['av_dim'][0]+x_do[:, 0]+S_do['bl'][0], (-1, 1))
    elif case=='horizontal_left_of_C':
        qs = np.zeros((len(x_do), 1))
        qd = np.ones((len(x_do), 1))
        qt = np.reshape(-(0.5*lin_c_params['av_dim'][1]+x_do[:, 1]+S_do['bl'][1]), (-1, 1))
    else:
        qs = []
        qd = []
        qt = []

    return {'s': qs, 'd': qd, 't': qt}

"""
    return base case id for linear safety constraints generation
"""
def identify_case(x0_av, x0_do, x_do, type_do, lin_c_params):
    # x0_av:        initial position AV
    # x0_do:        initial position DO
    # x_do:         predicted positions of DO
    # type_do:      type of DO
    # lin_c_params:     dictionary of parameters

    ds0 = x0_do[0, 0]-x0_av[0, 0]
    if ds0>-lin_c_params['ds_far_back'] and ds0<lin_c_params['ds_far']:
        if type_do=='v':
            if ds0>0 and np.logical_and(x_do[:, 0]>lin_c_params['s_int_lb'], x_do[:, 0]<lin_c_params['s_int_ub']).any():
                # intersection
                if x_do[-1, 1]>=x_do[0, 1]:
                    # V will intersect path from the right-->AV gives right of way
                    return 'stop_before_intersection'
                # else no constraint, AV has right of way
            else:
                dlane = np.round(x0_do[0, 1]/lin_c_params['wlane'])-np.round(x0_av[1, 0]/lin_c_params['wlane'])
                if dlane!=0:
                    # parallel lane
                    if dlane==1 or dlane==-1:
                        dd0 = x0_do[0, 1]-x0_av[1, 0]
                        if ds0>-lin_c_params['ds_overtaken']:
                            if dd0>-1.5*lin_c_params['wlane'] and dd0<1.5*lin_c_params['wlane']:
                                if dd0>0:
                                    return 'horizontal_right_of_V'
                                else:
                                    return 'horizontal_left_of_V'
                        else:
                            return 'vertical_front_of_V'
                else:
                    # same lane
                    if (np.round(x0_do[0, 1]/lin_c_params['wlane'])+1)*lin_c_params['wlane']<lin_c_params['dmax']:
                        # left lane is available
                        if ds0>lin_c_params['ds_close']:
                            return 'inclined_bl'
                        elif ds0>-lin_c_params['ds_overtaken']:
                            return 'horizontal_left_of_V'
                        else:
                            return 'vertical_front_of_V'
                    else:
                        # left lane is not free
                        return 'vertical_back_of_V'
        elif type_do=='c':
            if ds0>lin_c_params['ds_close']:
                return 'inclined_bl'
            elif ds0>-lin_c_params['ds_overtaken']:
                return 'horizontal_left_of_C'
        elif type_do=='p':
            if ds0>0 and np.logical_and(x_do[:, 1]>-0.8*lin_c_params['wlane'], x_do[:, 1]<1.5*lin_c_params['wlane']).any():
                return 'vertical_back_of_P'
    return 'no_constraint'