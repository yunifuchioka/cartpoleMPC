"""
Runs a simulation of the MPC controlled cartpole, and displays results.
Simulates MPC computation delay by defining a control period (inverse control frequency),
and subdividing the simulation into corresponding control intervals, over which a single
value for MPC control output is used
In addition, if the particular MPC evaluation fails to converge within the control period,
then the previous control output is repeated
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpc import MPC
plt.style.use('seaborn')

m1 = 1.0 #cart mass
m2 = 0.1 #pendulum mass
l = 0.5 #pendulum mass
g = 9.8 #gravity

T = 60.0 # simulation time
Tc = 0.1 #control period

# MPC parameters
tf=2.0
N=5
Q=[100,150,5,5]
Qf=[10,50,5,5]
R=5
verbose=0

save_anim = False
anim_file_name = 'cartpoleMPC'

#toggles whether to make the MPC track a smooth cubic spline trajectory, or to track a
#piecewise constant pose without preview
trajectory_tracking = False

x_init = np.array([0.0, 0.0, 0.0, 0.0])

q_des_mat = np.array([ \
    np.linspace(0.0, T, 15),
    3*np.random.rand(15) - 1.5,
    #np.pi*np.random.randint(2, size=15),
    np.pi * np.array([(i+1)%2 for i in range(15)]),
    #np.repeat(np.pi, 15),
    ])

def q_des_func(t):
    t_idx = [i for i in range(len(q_des_mat[0])) if q_des_mat[0,i]<=t][-1]
    return np.array([ \
        q_des_mat[1, t_idx],
        q_des_mat[2, t_idx]])

t_des = np.linspace(0, T, 100)
q_des = np.array([q_des_func(i) for i in t_des]).T
q_des_spline = CubicSpline(t_des, q_des, axis=1)

'''
#uncomment this to see what the desired spline trajectory looks like
t_fine = np.linspace(0, T, 1000)
y_fine = q_des_spline(t_fine)
plt.plot(t_des, q_des.T, 'o')
plt.plot(t_fine, y_fine.T)
plt.show()
'''

controller = MPC()
controller.set_parameters(tf=tf, N=N, Q=Q, Qf=Qf,R=R,verbose=verbose)

def qddot(x,u):
    q1ddot = (l*m2*np.sin(x[1])*x[3]**2 + u + \
        m2*g*np.cos(x[1])*np.sin(x[1]))/ \
        (m1 + m2*(1 - np.cos(x[1])**2))
    q2ddot = -1*(l*m2*np.cos(x[1])*np.sin(x[1])*x[3]**2 \
        + u*np.cos(x[1]) + (m1 + m2)*g*np.sin(x[1])) / \
        (l*m1 + l*m2*(1 - np.cos(x[1])**2))
    return np.array([q1ddot, q2ddot])

def plot_traj(t_sim, x_sim):
    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].plot(t_sim, x_sim[0,:], 'o')
    ax[0].set_title('Position')
    ax[1].plot(t_sim, x_sim[1,:], 'o')
    ax[1].set_title('Angle')

def animate_traj(t_sim, x_sim, sol_x_memory, tc, anim_frames, repeat=False):
    T = t_sim[-1] #final time
    traj_len = sol_x_memory.shape[2] #length of the trajectory predicted by MPC

    t_interp = np.linspace(0, T, anim_frames)
    x_interp = np.array([np.interp(t_interp, t_sim, x_sim[j]) \
        for j in range(max(x_init.shape))])

    anim_fig = plt.figure(figsize=(2*10, 0.75*10))
    ax = plt.axes(xlim=(-2, 2), ylim=(-0.75, 0.75))
    lines = [plt.plot([], [])[0] for _ in range(2 + traj_len)]

    def animate(i):
        hist, _ = np.histogram(t_interp[i], bins=tc)
        tc_idx = hist.argmax()

        if sol_x_memory[tc_idx,0,0] is not None:
            for traj_idx in range(traj_len):
                # draw cartpole trajectory predicted by MPC
                cart_pos_traj = np.array([sol_x_memory[tc_idx,0,traj_idx], 0])
                pend_pos_traj = cart_pos_traj + l*np.array([np.cos(sol_x_memory[tc_idx,1,traj_idx]-np.pi/2), 
                    np.sin(sol_x_memory[tc_idx,1,traj_idx]-np.pi/2)])
                lines[traj_idx].set_data(\
                    np.array([cart_pos_traj[0], pend_pos_traj[0]]),
                    np.array([cart_pos_traj[1], pend_pos_traj[1]]))
                lines[traj_idx].set_color(color='g')
                lines[traj_idx].set_alpha(0.2*(traj_len-traj_idx)/traj_len)

        # draw desired cartpole pose
        if trajectory_tracking:
            cart_pos_des = np.array([q_des_spline(t_interp[i])[0], 0])
            pend_pos_des = cart_pos_des + l*np.array([np.cos(q_des_spline(t_interp[i])[1]-np.pi/2), 
                np.sin(q_des_spline(t_interp[i])[1]-np.pi/2)])
        else:
            cart_pos_des = np.array([q_des_func(t_interp[i])[0], 0])
            pend_pos_des = cart_pos_des + l*np.array([np.cos(q_des_func(t_interp[i])[1]-np.pi/2), 
                np.sin(q_des_func(t_interp[i])[1]-np.pi/2)])
        lines[-2].set_data(\
            np.array([cart_pos_des[0], pend_pos_des[0]]),
            np.array([cart_pos_des[1], pend_pos_des[1]]))
        lines[-2].set_alpha(0.5)
        lines[-2].set_color(color='r')

        # draw cartpole pose
        cart_pos = np.array([x_interp[0,i], 0])
        pend_pos = cart_pos + l*np.array([np.cos(x_interp[1,i]-np.pi/2), 
            np.sin(x_interp[1,i]-np.pi/2)])
        lines[-1].set_data( \
            np.array([cart_pos[0], pend_pos[0]]), 
            np.array([cart_pos[1], pend_pos[1]]))
        lines[-1].set_color(color='b')

        for line in lines:
            line.set_linewidth(5)
            line.set_marker('o')
            line.set_markeredgewidth(7)

        return lines

    anim = animation.FuncAnimation(anim_fig, animate, frames=anim_frames, 
        interval=T*1000/anim_frames, repeat=repeat)

    if save_anim:
        Writer = animation.writers['ffmpeg']
        #writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1000)
        anim.save(anim_file_name + '.mp4', writer=writer)

    plt.show()

# variables holding simulation results
t_sim = np.array([0])
x_sim = x_init[:, np.newaxis]
sol_x_memory = None #for storing trajectories generated by the MPC

# time intervals for each control period
tc = np.linspace(0, T, int(T/Tc))

u_prev = 0
q_des_prev = q_des_func(0)

# simulation loop
for i in range(max(tc.shape)-1):

    if not trajectory_tracking:
    # reset MPC warm start solution if the target state changed
        if np.linalg.norm(q_des_func(t_sim[-1]) - q_des_prev) > 0.01:
            controller.reset_sol_prev()
            print('Resetting guess solution...')

    t_idx = [i for i in range(len(t_des)) \
        if t_des[i]<=t_sim[-1] and t_sim[-1]<t_des[i+1]][0]
    t_des_t = np.insert(t_des[t_idx+1:]-t_sim[-1], 0, 0.0)
    q_des_t = np.hstack((q_des_spline(t_sim[-1])[:,None], q_des[:, t_idx+1:]))


    # evaluate the MPC for the current physical and desired state
    if trajectory_tracking:
        sol_t, sol_x, sol_u, toc = controller.get_trajectory(x_init=x_sim[:,-1].tolist(), 
            q_des=q_des_t.tolist(), t_des=t_des_t.tolist())
    else:
        sol_t, sol_x, sol_u, toc = controller.get_trajectory(x_init=x_sim[:,-1].tolist(), 
            q_des=q_des_func(t_sim[-1]).tolist(), t_des=None)
    
    print('Progress: {}/{}. MPC Computation time: {}'.format(i,max(tc.shape)-1, toc))

    # determine whether the MPC was successful, and update the MPC trajectory memeory accordingly
    # Note: assume that the MPC will be successful at the first timestep. Otherwise
    # sol_x_memory will not be initialized properly
    if sol_u is None:
        print('MPC failure. Trajectory optimization failed to converge...')
        sol_x_memory = np.append(sol_x_memory, 
            [np.full((sol_x_memory.shape[1],sol_x_memory.shape[2]), None)], axis=0)
    elif toc >= Tc:
        print('MPC failure. Computation took too long...')
        sol_x_memory = np.append(sol_x_memory, 
            [np.full((sol_x_memory.shape[1],sol_x_memory.shape[2]), None)], axis=0)
    else:
        # save trajectory predicted by MPC for animation purposes
        if sol_x_memory is None:
            sol_x_memory = np.array([sol_x])
        else:
            sol_x_memory = np.append(sol_x_memory, np.array([sol_x]), axis=0)

    # integrate for the current control interval
    def xdot(t, x):
        u = controller.get_control(t-tc[i])
        if u is None:
            # if the previous call to controller.get_trajectory was not able to find a solution
            u = 0.0
        return np.concatenate((x[2:], qddot(x,u)))
    sol = solve_ivp(xdot, [tc[i], tc[i+1]], x_sim[:,-1])
    
    t_sim = np.hstack((t_sim, np.delete(sol.t, 0)))
    x_sim = np.hstack((x_sim, np.delete(sol.y, 0, axis=1)))

    q_des_prev = q_des_func(t_sim[-1])

#plot_traj(t_sim, x_sim)
animate_traj(t_sim, x_sim, sol_x_memory, tc, anim_frames=int(t_sim[-1]*10), repeat=True)