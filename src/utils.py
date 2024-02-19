def grid_projection_1D(x_float,
                       x_origin = 0.,
                       grid_step = 2.):

    x = x_float - x_origin
    r_x = x % grid_step
    t_x = x - r_x

    if r_x < grid_step / 2.:
        return x_origin + t_x
    else:
        return x_origin + t_x + grid_step


def one_vs_rest_label(rhy_grp,
                      one_targets = ['sinus']):
    if rhy_grp in one_targets:
        return 1
    else:
        return 0
