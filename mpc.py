from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition
import time

# Note: Pyomo implements its own math operations (+, *, sin, etc.), so care must be taken 
# when importing math libraries to not overload these operations
from math import pi
from scipy.interpolate import CubicSpline

class MPC:
    """
    Model predictive controller for the cartpole. Supports finite horizion trajectory
    optimization for constant target pose stabilization or target trajectory tracking, and
    computation of inputs according to the resulting trajectory

    Note: all inputs and outputs of this class are native python objects and not numpy
    arrays. This includes trajectories, which are represented with spline interpolation
    points
    """

    def __init__(self, 
            m1=1.0, m2=0.1, l=0.5, g=9.81, u_max=10.0, d_max=2.0,
            tf=2.0, N=5, Q=[100,150,5,5], Qf=[10,50,5,5], R=5, verbose=0):
        """
        Initializes a pyomo optimization model that solves a trajectory optimization
        problem using direct hermite-simpson collocation in separated form, where the
        objective is finite horizon target state tracking using LQR costs.
        The mathematical expressions for collocation are based on Matthew Kelly 2017, 
        "An Introduction to Trajectory Optimization: How to Do Your Own Direct Collocation"

        Note: In hermite simpson collocation in separated form, a finite element consists 
        of three colllocation points: left endpoint, midpoint, and right endpoint.
        Each finite element is indexed using the odd indices of the collocation points,
        as these correspond to the midpoints of the finite elements
        """
        self.n = 4 #dimension of state vector
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

        # toggles how much optimizer progress to print
        # 0 - print message only if optimzation failed to converge
        # 1 - print basic optimization results
        # 2 - print detailed IPOPT output, including algorithm iterations
        self.verbose = verbose

        # initialize optimization solution memory
        self.reset_sol_prev()

        # pyomo optimization model
        self.m = AbstractModel()

        # pyomo sets, which act as indices for parameters and decision variables
        self.m.t = RangeSet(0, 2*self.N) #index for time
        self.m.state = RangeSet(0, self.n-1) #index for state elements
        
        # model parameters and indices. Numerical values for these parameters are set at 
        # runtime
        self.m.x_init = Param(self.m.state, within=Reals) #initial state
        self.m.x_des = Param(self.m.t, self.m.state, within=Reals) #desired state
        
        # guess solutions for decision variables. These parameters will be given numerical
        # values corresponding to the previous optimization solution when 
        # self.get_trajectory() is called if the optimization solution memory is populated,
        # otherwise it defaults to a linear interpolation of state and zero control and 
        # dynamics
        def _linear_guess(m,i,j):
            return i/(2.0*self.N)*(m.x_des[2*self.N, j] - m.x_init[j]) + m.x_init[j]
        self.m.x_guess = Param(self.m.t, self.m.state, within=Reals, 
            initialize=_linear_guess)
        self.m.u_guess = Param(self.m.t, within=Reals, initialize=0)
        self.m.f_guess = Param(self.m.t, self.m.state, within=Reals, initialize=0)
        def _x_guess(m,i,j):
            return m.x_guess[i,j]
        def _u_guess(m,i):
            return m.u_guess[i]
        def _f_guess(m,i,j):
            return m.f_guess[i,j]

        # decision variables
        self.m.x = Var(self.m.t, self.m.state, initialize=_x_guess) #state
        self.m.u = Var(self.m.t, initialize=_u_guess) #control
        self.m.f = Var(self.m.t, self.m.state, initialize=_f_guess) #dynamics
        
        # objective
        def _obj(m):
            # simpson quadrature coefficient vector
            simp = [4*(i%2)+2*((i+1)%2) for i in range(2*self.N+1)]
            simp[0]=1
            simp[-1]=1

            # cost = finite horizon LQR cost computed with simpson quadrature
            J = 0
            J += self.R * quicksum(simp[i]*m.u[i]**2 for i in m.t)
            for j in m.state:
                if j == 1:
                    # angle error must be computed using trig functions to account for
                    # periodicity
                    J += self.Q[j] * quicksum(simp[i]*(cos(m.x_des[i, j]) - cos(m.x[i,j]))**2 \
                        for i in m.t)
                    J += self.Q[j] * quicksum(simp[i]*(sin(m.x_des[i, j]) - sin(m.x[i,j]))**2 \
                        for i in m.t)
                    J += self.Qf[j] * (cos(m.x_des[2*self.N,j]) - cos(m.x[2*self.N,j]))**2
                    J += self.Qf[j] * (sin(m.x_des[2*self.N,j]) - sin(m.x[2*self.N,j]))**2
                else:
                    J += self.Q[j] * quicksum(simp[i]*(m.x_des[i, j] - m.x[i,j])**2 \
                        for i in m.t)
                    J += self.Qf[j] * (m.x_des[2*self.N,j] - m.x[2*self.N,j])**2
            return J
        self.m.obj= Objective(rule=_obj, sense=minimize)
        
        # path constraints
        def _pos_bound(m, i):
            # -d_max <= position <= d_max
            return (-self.d_max, m.x[i,0], self.d_max)
        self.m.pos_bound = Constraint(self.m.t, rule=_pos_bound)
        def _control_bound(m, i):
            # -u_max <= control <= u_max
            return (-self.u_max, m.u[i], self.u_max)
        self.m.control_bound = Constraint(self.m.t, rule=_control_bound)
        
        # initial condition constraints
        def _init_cond(m):
            yield m.x[0,0] == m.x_init[0]
            yield m.x[0,1] == m.x_init[1]
            yield m.x[0,2] == m.x_init[2]
            yield m.x[0,3] == m.x_init[3]
        self.m.init_cond = ConstraintList(rule=_init_cond)
        
        # dynamics constraints
        def _dxdt(m, i, j):
            # i is time index, j is state index
            if j == 0:
                return m.f[i,0] == m.x[i,2]
            elif j == 1:
                return m.f[i,1] == m.x[i,3]
            elif j == 2:
                m1, m2, l, g = self.m1, self.m2, self.l, self.g
                # equation (6.1) in Kelly 2017
                return m.f[i,2] == (l*m2*sin(m.x[i,1])*m.x[i,3]**2 + m.u[i] + \
                    m2*g*cos(m.x[i,1])*sin(m.x[i,1]))/ \
                    (m1 + m2*(1 - cos(m.x[i,1])**2))
            elif j == 3:
                m1, m2, l, g = self.m1, self.m2, self.l, self.g
                # equation (6.2) in Kelly 2017
                return m.f[i,3] == -1*(l*m2*cos(m.x[i,1])*sin(m.x[i,1])*m.x[i,3]**2 \
                    + m.u[i]*cos(m.x[i,1]) + (m1 + m2)*g*sin(m.x[i,1])) / \
                    (l*m1 + l*m2*(1 - cos(m.x[i,1])**2))
        self.m.dxdt = Constraint(self.m.t, self.m.state, rule=_dxdt)
        
        # interpolation constraints
        def _interp(m, i, j):
            if i % 2 == 0:
                # only odd time indices correspond to the midpoint of finite elements
                return Constraint.Skip
            else:
                # equation (6.11) in Kelly 2017
                return m.x[i,j] == (m.x[i-1,j] + m.x[i+1,j])/2.0 \
                    + self.tf/self.N*(m.f[i-1,j]-m.f[i+1,j])/8.0
        self.m.interp = Constraint(self.m.t, self.m.state, rule=_interp)

        # collocation constraints
        def _colloc(m, i, j):
            if i % 2 == 0:
                # only odd time indices correspond to the midpoint of finite elements
                return Constraint.Skip
            else:
                # equation (6.12) in Kelly 2017
                return self.tf/self.N*(m.f[i-1,j]+4*m.f[i,j]+m.f[i+1,j])/6.0 \
                    == m.x[i+1,j] - m.x[i-1,j]
        self.m.colloc = Constraint(self.m.t, self.m.state, rule=_colloc)

    def reset_sol_prev(self):
        """
        reset the optimization solution memory. These values are used to warm start the 
        optimization at the next timestep, and to compute the control signal in
        self.get_control()
        """
        self.sol_t_prev = None
        self.sol_x_prev = None
        self.sol_u_prev = None
        self.sol_f_prev = None

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

        # keep track of how long this function takes to evaluate
        tic = time.time()

        # x_des - matrix of desired state at each collocation point
        if t_des is None:
            # if the desired pose is specified as a fixed target, then create a matrix
            # of identical values for all time
            x_des = [\
                [q_des[0] for i in range(2*self.N+1)],
                [q_des[1] for i in range(2*self.N+1)],
                [0 for i in range(2*self.N+1)],
                [0 for i in range(2*self.N+1)]]
        else:
            # if the desired pose is specified as a trajectory, construct a cubic spline
            # from the desired trajectory interpolation points, and evaluate them at the
            # collocation points
            spline = CubicSpline(t_des, q_des, axis=1)
            t_colloc = [i*self.tf/(2.0*self.N) for i in range(2*self.N+1)]
            x_des = spline(t_colloc).tolist() + spline(t_colloc,1).tolist()

        # prepare numerical data to feed to the optimization model
        data = {None: {
            'x_init' : {j : x_init[j] for j in range(self.n)},
            'x_des' : {(i,j) : x_des[j][i] \
                for i in range(2*self.N+1) for j in range(self.n)}
            }}
        # warmstart optimization with the trajectory stored in the optimization solution
        # memory, which should be populated if the previous optimization was successful
        if all(sol_prev is not None for sol_prev in
            [self.sol_x_prev, self.sol_u_prev, self.sol_f_prev]):
            data[None]['x_guess'] = {(i,j) : self.sol_x_prev[j][i] \
                    for i in range(2*self.N+1) for j in range(self.n)}
            data[None]['u_guess'] = {i : self.sol_u_prev[i] for i in range(2*self.N+1)}
            data[None]['f_guess'] = {(i,j) : self.sol_f_prev[j][i] \
                    for i in range(2*self.N+1) for j in range(self.n)}
        
        # create instance of optimization model populated with numerical data and solve
        mi = self.m.create_instance(data)
        solver = SolverFactory('ipopt')
        result = solver.solve(mi, tee=self.verbose>1) #tee=True makes IPOPT print progress
        if self.verbose == 1:
            print(result)

        # extract solution from optimization if successful
        if (result.solver.status == SolverStatus.ok) \
            and (result.solver.termination_condition == TerminationCondition.optimal):
            sol_t = [i*self.tf/(2*self.N) for i in mi.t]
            sol_x = [[value(mi.x[i,j]) for i in mi.t] for j in mi.state]
            sol_u = [value(mi.u[i]) for i in mi.t]
            sol_f = [[value(mi.x[i,j]) for i in mi.t] for j in mi.state]
        else:
            sol_t, sol_x, sol_u, sol_f = None, None, None, None

        # save the resulting trajectory in the optimization solution memory
        self.sol_t_prev, self.sol_x_prev, self.sol_u_prev, self.sol_f_prev = \
            sol_t, sol_x, sol_u, sol_f

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