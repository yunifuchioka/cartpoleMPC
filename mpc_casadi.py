import numpy as np
import casadi as ca
from scipy.interpolate import CubicSpline
import time

class MPC:
    """
    Model predictive controller for the cartpole. Supports finite horizion trajectory
    optimization for constant target pose stabilization or target trajectory tracking, and
    computation of inputs according to the resulting trajectory

    Implementation using the casadi library as the optimization framework
    """

    def __init__(self, 
            m1=1.0, m2=0.1, l=0.5, g=9.81, u_max=10.0, d_max=2.0,
            tf=2.0, N=5, Q=[100,150,5,5], Qf=[10,50,5,5], R=5, verbose=0):
        """
        Initializes a casadi optimization model that solves a trajectory optimization
        problem using direct hermite-simpson collocation in separated form, where the
        objective is finite horizon target state tracking using LQR costs.
        The mathematical expressions for collocation are based on Matthew Kelly 2017, 
        "An Introduction to Trajectory Optimization: How to Do Your Own Direct Collocation"

        Note: In hermite simpson collocation in separated form, a finite element consists 
        of three collocation points: left endpoint, midpoint, and right endpoint.
        Each finite element is indexed using the odd indices of the collocation points,
        as these correspond to the midpoints of the finite elements
        """

        self.nx = 4 #dimension of state vector
        self.nu = 1 #dimension of input vector
        self.m1 = m1 #cart mass
        self.m2 = m2 #pendulum mass
        self.l = l #pendulum length
        self.g = g #gravity
        self.u_max = u_max #maximum actuator force
        self.d_max = d_max #extent of rail that cart travels on

        self.tf = tf #optimization horizon
        self.N = N #number of finite elements (number of collocation points = 2*N+1)
        self.Q = Q #state error cost weight
        self.Qf = Qf #final state error cost weight
        self.R = R #input regulation cost weight

        # integer from 0 to 12 indicating how much IPOPT output to print
        self.verbose = verbose

        # initialize optimization solution memory
        self.reset_sol_prev()

        # casadi optimization model
        self.opti = ca.Opti()

        # parameters
        self.x_init = self.opti.parameter(self.nx) # initial state
        self.x_des = self.opti.parameter(self.nx, 2*self.N+1) # desired states

        # decision variables
        self.X = self.opti.variable(self.nx, 2*self.N+1) # state at each collocation point
        self.U = self.opti.variable(self.nu, 2*self.N+1) # input at each collocation point

        # objective
        # simpson quadrature coefficients, to be used to compute integrals
        simp = np.empty((1,2*N+1))
        simp[0,::2] = 2
        simp[0,1::2] = 4
        simp[0,0], simp[0,-1] = 1, 1

        # cost = finite horizon LQR cost computed with simpson quadrature
        J = 0.0
        J += ca.dot(simp, self.U[:,:]*self.U[:,:])
        for j in range(self.nx):
            if j == 1:
                # angle error must be computed using trig functions to account for
                # periodicity
                J += self.Q[j] * ca.dot(simp,(ca.cos(self.X[j,:])-ca.cos(self.x_des[j,:])) \
                    *(ca.cos(self.X[j,:])-ca.cos(self.x_des[j,:])))
                J += self.Q[j] * ca.dot(simp,(ca.sin(self.X[j,:])-ca.sin(self.x_des[j,:])) \
                    *(ca.sin(self.X[j,:])-ca.sin(self.x_des[j,:])))
                J += self.Qf[j] * (ca.cos(self.X[j,-1])-ca.cos(self.x_des[j,-1])) \
                    *(ca.cos(self.X[j,-1])-ca.cos(self.x_des[j,-1]))
                J += self.Qf[j] * (ca.sin(self.X[j,-1])-ca.sin(self.x_des[j,-1])) \
                    *(ca.sin(self.X[j,-1])-ca.sin(self.x_des[j,-1]))
            else:
                J += self.Q[j] * ca.dot(simp, (self.X[j,:]-self.x_des[j,:]) \
                    *(self.X[j,:]-self.x_des[j,:]))
                J += self.Qf[j] * (self.X[j,-1]-self.x_des[j,-1]) \
                    *(self.X[j,-1]-self.x_des[j,-1])

        self.opti.minimize(J)

        # position bound constraint
        self.opti.subject_to(self.opti.bounded( \
            np.full(self.X[0,:].shape, -self.d_max), 
            self.X[0,:], 
            np.full(self.X[0,:].shape, self.d_max)
            ))

        # control bound constraint
        self.opti.subject_to(self.opti.bounded( \
            np.full(self.U[:,:].shape, -self.u_max), 
            self.U[:,:], 
            np.full(self.U[:,:].shape, self.u_max)
            ))

        # initial condition constraint
        self.opti.subject_to(self.X[:,0] == self.x_init)

        # dynamics
        # symbolic variables used to derive equations of motion
        x = ca.SX.sym('x', self.nx) #state
        u = ca.SX.sym('u', self.nu) #control
        m1, m2, l, g = self.m1, self.m2, self.l, self.g
        # equations of motion taken from Russ Tedrake underactuated robotics ch.3
        # this method of deriving equations of motion from matrix operations is more
        # scalable to larger robotic systems than the scalar equations written in 
        # mpc_pyomo.py, and only made possible through the powerful symbolic framework of 
        # casadi 
        M = ca.SX(np.array([ \
            [m1 + m2, m2*l*ca.cos(x[1])],
            [m2*l*ca.cos(x[1]), m2*l**2]
            ]))
        C = ca.SX(np.array([ \
            [0.0, -m2*l*x[3]*ca.sin(x[1])],
            [0.0, 0.0]
            ]))
        tau_g = ca.SX(np.array([ \
            [0.0],
            [-m2*g*l*ca.sin(x[1])]
            ]))
        B = ca.SX(np.array([ \
            [1.0],
            [0.0]
            ]))
        xdot = ca.vertcat( \
            x[2:], # d/dt(q) = qdot
            ca.solve(M, -C@x[2:]+tau_g+B@u) # M@qddot = (-C@qdot+tau_g+B@u)
            )
        f = ca.Function('f', [x,u], [xdot]) # xdot = f(x,u)

        for i in range(2*self.N+1):
            if i%2 != 0:
                # for each finite element:
                x_left, x_mid, x_right = self.X[:,i-1], self.X[:,i], self.X[:,i+1]
                u_left, u_mid, u_right = self.U[:,i-1], self.U[:,i], self.U[:,i+1]
                f_left, f_mid, f_right = f(x_left,u_left), f(x_mid,u_mid), f(x_right,u_right)

                # interpolation constraints
                self.opti.subject_to( \
                    x_mid == (x_left+x_right)/2.0 + self.tf/self.N*(f_left-f_right)/8.0)

                # collocation constraints
                self.opti.subject_to( \
                    self.tf/self.N*(f_left+4*f_mid+f_right)/6.0 == x_right-x_left)
    
    def reset_sol_prev(self):
        """
        reset the optimization solution memory. These values are used to warm start the 
        optimization at the next timestep, and to compute the control signal in
        self.get_control()
        """
        self.sol_t_prev = None
        self.sol_x_prev = None
        self.sol_u_prev = None

    def get_trajectory(self, x_init, q_des, t_des=None):
        """
        Solve the finite horizion trajectory optimization problem.
        Arguments:
        x_init - current state
        q_des - desired pose (constant with time) or pose trajectory (interpolation points)
        t_des - timepoints corresponding to the pose trajectory. Should be set to None
            if the desired pose is constant
        Returns:
        sol_t, sol_x, sol_u - collocated trajectory returned by the optimization.
            sol_x is a cubic Hermite spline, whereas sol_u is a piecewise quadratic spline.
            Returns None if the optimization failed to converge
        toc - the amount of time it took to evaluate this function
        """
        
        # convert list to np array. Necessary for backwards compatibility with mpc_pyomo.py
        x_init = np.array(x_init)
        q_des = np.array(q_des)
        if t_des is not None:
            t_des = np.array(t_des)

        # keep track of how long this function takes to evaluate
        tic = time.time()

        if t_des is None:
            # if the desired pose is specified as a fixed target, then create a matrix
            # of identical values for all time
            x_des = np.vstack(( \
                np.full((2, 2*self.N+1), q_des[:,None]),
                np.zeros((2, 2*self.N+1))
                ))
        else:
            # if the desired pose is specified as a trajectory, construct a cubic spline
            # from the desired trajectory interpolation points, and evaluate them at the
            # collocation points
            spline = CubicSpline(t_des, q_des, axis=1)
            t_colloc = np.linspace(0, self.tf, 2*self.N+1)
            x_des = np.vstack((spline(t_colloc), spline(t_colloc,1)))

        # set numerical values to opti parameters
        self.opti.set_value(self.x_init, x_init)
        self.opti.set_value(self.x_des, x_des)

        # decision variable initial guess
        self.opti.set_initial(self.X, np.linspace(x_init, x_des[:,-1], 2*self.N+1).T)
        self.opti.set_initial(self.U, np.zeros(self.U.shape)) #zero control

        # warmstart optimization with the trajectory stored in the optimization solution
        # memory, which should be populated if the previous optimization was successful
        if self.sol_x_prev is not None and self.sol_u_prev is not None:
            self.opti.set_initial(self.X, self.sol_x_prev)
            self.opti.set_initial(self.U, self.sol_u_prev)
        else:
            # linear interpolation between current and final desired state, and zero control
            self.opti.set_initial(self.X, np.linspace(x_init, x_des[:,-1], 2*self.N+1).T)
            self.opti.set_initial(self.U, np.zeros(self.U.shape)) #zero control

        # solve NLP
        if self.verbose <= 0:
            p_opts = {'print_time' : False}
            s_opts = {'suppress_all_output' : 'yes'}
        else:
            p_opts = {}
            s_opts = {'print_level': verbose}
        self.opti.solver('ipopt', p_opts, s_opts)
        try:
            sol = self.opti.solve()
        except RuntimeError:
            # IPOPT returned infeasible
            sol_t, sol_x, sol_u = None, None, None
        else:
            # IPOPT successful, extract solution from optimization
            sol_t = np.linspace(0, self.tf, 2*self.N+1)
            sol_x = sol.value(self.X)
            sol_u = sol.value(self.U)

        # save the resulting trajectory in the optimization solution memory
        self.sol_t_prev, self.sol_x_prev, self.sol_u_prev = sol_t, sol_x, sol_u

        # evaluate how long this function took to evaluate
        toc = time.time() - tic

        return sol_t, sol_x, sol_u, toc

    def get_control(self, t):
        """
        Compute the control signal that should be used according to the last trajectory
        that was obtained from self.get_trajectory()
        Arguments:
        t - the physical time that has elapsed since the previous call to
            self.get_trajectory()
        Returns:
        u - the control signal that should be used. Obtained from evaluating the qudratic
            control spline trajectory at the time t. Returns None if there is are no
            values stored in the optimization solution memory
        """
        if any(sol_prev is None for sol_prev in [self.sol_t_prev, self.sol_u_prev]):
            # optimization solution memory is empty, so return None
            # this will happen if this function is erroneously called immediately after
            # self.reset_sol_prev(), or if the previous call to self.get_trajectory() was 
            # unsuccessful
            return None
        else:
            # evaluate the quadratic spline at time t
            if t >= self.sol_t_prev[-1]:
                # if t is out of bounds of the computed trajectory, then return the last
                # control value
                return self.sol_u_prev[-1]
            else:
                # get the index of the finite element that t belongs to
                fe_bins = [self.sol_t_prev[i] \
                    for i in range(len(self.sol_t_prev)) if i%2==0]
                fe_idx = [i for i in range(len(fe_bins)) if fe_bins[i]<=t][-1]

                #equation (4.10) in Kelly 2017
                hk = fe_bins[1] - fe_bins[0] #finite element length
                tau = t - self.sol_t_prev[2*fe_idx] #time since left endpoint
                uk = self.sol_u_prev[2*fe_idx] #control at left endpoint
                uk_half = self.sol_u_prev[2*fe_idx+1] #control at midpoint
                uk_next = self.sol_u_prev[2*fe_idx+2] #control at right endpoint
                return 2.0/hk**2 * (tau - hk/2.0) * (tau-hk) * uk \
                    - 4.0/hk**2 * (tau) * (tau - hk) * uk_half \
                    + 2.0/hk**2 * (tau) * (tau - hk/2.0) * uk_next