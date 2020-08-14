from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition
#careful not to import anything else from math-that would override pyomo intrinsic functions
from math import pi
from scipy.interpolate import CubicSpline
import time

class MPC:
    """
    Model predictive controller for the cartpole. Performs trajectory optimization over
    a finite interval and returns the resulting trajectory, of which the first control
    value should be used in a simulation

    All inputs and outputs of this class are python lists, and not numpy arrays
    """

    def __init__(self):
        """
        Initializes an pyomo optimization model that solves a trajectory optimization
        problem using direct hermite-simpson collocation in separated form, where the
        objective is finite horizon target state tracking using LQR costs.

        A pyomo AbstractModel class is used here over a ConcreteModel, since this supports
        separation of model definition and data input at evaluation time, which is
        desirable for a MPC that needs to solve the same trajectory optimization problem
        multiple times with different numerical data

        Note: In hermite simpson collocation in separated form, a finite element consists 
        of three time discretization points: left endpoint, midpoint, and right endpoint.
        Each finite element is indexed using the odd indices of the time discretization,
        as these correspond to the midpoints of the finite elements
        """
        self.n = 4 #dimension of state vector
        self.m1 = 1.0 #cart mass
        self.m2 = 0.1 #pendulum mass
        self.l = 0.5 #pendulum mass
        self.g = 9.8 #gravity
        self.u_max = 10.0 #maximum actuator force
        self.d_max = 2.0 #extent of rail that cart travels on

        self.set_parameters()

        # pyomo optimization model
        self.m = AbstractModel()

        # numerical values for these parameters are set when the MPC is called
        self.m.tf = Param(within=Reals) #horizon length in seconds
        self.m.N = Param(within=Integers) #number of finite elements
        
        self.m.t = RangeSet(0, 2*self.m.N) #index for time. Values: k = 0, 0+1/2, 1, 1+1/2,..., N
        self.m.state = RangeSet(0, self.n-1) #index for state variables
        
        self.m.x_init = Param(self.m.state, within=Reals) #initial state
        self.m.x_des = Param(self.m.t, self.m.state, within=Reals) #desired state

        self.m.Q = Param(self.m.state, within=Reals) #state tracking cost weights
        self.m.Qf = Param(self.m.state, within=Reals) #final state cost weights
        self.m.R = Param(within=Reals) #input regularization cost weight
        
        # guess solutions for decision variables. Numerical values can be specified during
        # runtime, otherwise it defaults to linear state interpolation and zero control
        # and dynamics
        def _linear_guess(m,i,j):
            return i/(2.0*m.N)*(m.x_des[2*m.N, j] - m.x_init[j])
        self.m.x_guess = Param(self.m.t, self.m.state, within=Reals, initialize=_linear_guess) #state
        self.m.u_guess = Param(self.m.t, within=Reals, initialize=0) #control
        self.m.f_guess = Param(self.m.t, self.m.state, within=Reals, initialize=0) #dynamics
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
        # cost = finite horizon LQR cost computed with simpson quadrature
        def _obj(m):
            # simpson quadrature coefficient vector
            simp = [4*(i%2)+2*((i+1)%2) for i in range(2*m.N+1)]
            simp[0]=1
            simp[-1]=1

            J = 0
            # integral of input squared
            J += m.R * quicksum(simp[i]*m.u[i]**2 for i in m.t)

            '''
            # integral of error squared
            J += quicksum(m.Q[j] * quicksum(simp[i]*(m.x_des[j] - m.x[i,j])**2 \
                for i in m.t) for j in m.state)
            # final error
            J += quicksum(m.Qf[j] * (m.x_des[j] - m.x[2*m.N,j])**2 for j in m.state)
            '''
            for j in m.state:
                if j == 1:
                    J += m.Q[j] * quicksum(simp[i]*(cos(m.x_des[i, j]) - cos(m.x[i,j]))**2 \
                        for i in m.t)
                    J += m.Q[j] * quicksum(simp[i]*(sin(m.x_des[i, j]) - sin(m.x[i,j]))**2 \
                        for i in m.t)
                    J += m.Qf[j] * (cos(m.x_des[2*m.N,j]) - cos(m.x[2*m.N,j]))**2
                    J += m.Qf[j] * (sin(m.x_des[2*m.N,j]) - sin(m.x[2*m.N,j]))**2
                else:
                    J += m.Q[j] * quicksum(simp[i]*(m.x_des[i, j] - m.x[i,j])**2 \
                        for i in m.t)
                    J += m.Qf[j] * (m.x_des[2*m.N,j] - m.x[2*m.N,j])**2

            return J
        self.m.obj= Objective(rule=_obj, sense=minimize)
        
        # path constraints
        def _pos_bound(m, i):
            return (-self.d_max, m.x[i,0], self.d_max)
        self.m.pos_bound = Constraint(self.m.t, rule=_pos_bound)
        def _control_bound(m, i):
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
                # eq (6.1) in Matthew Kelly tutorial
                return m.f[i,2] == (l*m2*sin(m.x[i,1])*m.x[i,3]**2 + m.u[i] + \
                    m2*g*cos(m.x[i,1])*sin(m.x[i,1]))/ \
                    (m1 + m2*(1 - cos(m.x[i,1])**2))
            elif j == 3:
                m1, m2, l, g = self.m1, self.m2, self.l, self.g
                # eq (6.2) in Matthew Kelly tutorial
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
                # equation (6.11) in Matthew Kelly tutorial
                h = m.tf/m.N
                return m.x[i,j] == (m.x[i-1,j] + m.x[i+1,j])/2.0 + h*(m.f[i-1,j]-m.f[i+1,j])/8.0
        self.m.interp = Constraint(self.m.t, self.m.state, rule=_interp)

        # collocation constraints
        def _colloc(m, i, j):
            if i % 2 == 0:
                # only odd time indices correspond to the midpoint of finite elements
                return Constraint.Skip
            else:
                # equation (6.12) in Matthew Kelly tutorial
                h = m.tf/m.N
                return h*(m.f[i-1,j]+4*m.f[i,j]+m.f[i+1,j])/6.0 == m.x[i+1,j] - m.x[i-1,j]
        self.m.colloc = Constraint(self.m.t, self.m.state, rule=_colloc)

    def set_parameters(self, tf=2.0, N=5, Q=[10,5,0.5,0.5], Qf=[50,10,5,5], 
            R=0.5, verbose=0):
        # these parameters were designed to be modifiable at runtime, but they likely
        # do not need to be modified often
        self.tf = tf #optimization horizon
        self.N = N #number of finite elements
        self.Q = Q #target tracking cost weight
        self.Qf = Qf #final target tracking cost weight
        self.R = R #input regulation cost weight
        self.verbose = verbose #toggles how much optimizer progress to print

        #if collocation parameters are changed, then the previous solution cannot be used to
        #warmstart the optimization problem under the new collocation
        self.reset_sol_prev()

    def reset_sol_prev(self):
        # reset the previous solution to the optimization, which is used for warm starting
        self.sol_t_prev = None
        self.sol_x_prev = None
        self.sol_u_prev = None
        self.sol_f_prev = None

    def get_trajectory(self, x_init, q_des, t_des=None):
        """
        Given current state and desired pose or pose trajectory, solve the trajectory
        optimization problem and return the resulting trajectory, and the amount of
        time taken to obtain it
        Additionally records the results in an internal variable so it could be used to
        warm start the solution to the next call to this function
        If trajectory tracking is desired, then t_des and q_des should be python lists
        corresponding to the desired pose trajectory. Otherwise if a fixed pose is desired,
        then t_des should be None and q_des should be a single value
        """
        tic = time.time() #keep track of how long this function takes to evaluate

        if t_des is None:
            x_des = [\
                [q_des[0] for i in range(2*self.N+1)],
                [q_des[1] for i in range(2*self.N+1)],
                [0 for i in range(2*self.N+1)],
                [0 for i in range(2*self.N+1)]]
        else:
            spline = CubicSpline(t_des, q_des, axis=1)
            t_colloc = [i*self.tf/(2.0*self.N) for i in range(2*self.N+1)]
            x_des = spline(t_colloc).tolist() + spline(t_colloc,1).tolist()

        # prepare numerical data to feed to the optimization model
        if all(sol_prev is not None for sol_prev in
            [self.sol_x_prev, self.sol_u_prev, self.sol_f_prev]):
            # warmstart optimization with solution from previous solution
            data = {None: {
                'x_init' : {j : x_init[j] for j in range(self.n)},
                'x_des' : {(i,j) : x_des[j][i] \
                    for i in range(2*self.N+1) for j in range(self.n)},
                'x_guess' : {(i,j) : self.sol_x_prev[j][i] \
                    for i in range(2*self.N+1) for j in range(self.n)},
                'u_guess' : {i : self.sol_u_prev[i] for i in range(2*self.N+1)},
                'f_guess' : {(i,j) : self.sol_f_prev[j][i] \
                    for i in range(2*self.N+1) for j in range(self.n)},
                'tf' : {None: self.tf},
                'N' : {None: self.N},
                'Q' : {j : self.Q[j] for j in range(self.n)},
                'Qf' : {j : self.Qf[j] for j in range(self.n)},
                'R' : {None: self.R},
            }}
        else:
            # use default decision variable initialization, which is defined by
            # the initialize argument within self.x_guess, self.u_guess, and self.f_guess
            data = {None: {
                'x_init' : {j : x_init[j] for j in range(self.n)},
                'x_des' : {(i,j) : x_des[j][i] \
                    for i in range(2*self.N+1) for j in range(self.n)},
                'tf' : {None: self.tf},
                'N' : {None: self.N},
                'Q' : {j : self.Q[j] for j in range(self.n)},
                'Qf' : {j : self.Qf[j] for j in range(self.n)},
                'R' : {None: self.R},
            }}
        
        # create instance of optimization model populated with numerical data and solve
        mi = self.m.create_instance(data)
        solver = SolverFactory('ipopt')
        result = solver.solve(mi, tee=self.verbose>1) #tee=True makes IPOPT print progress
        if self.verbose == 1:
            print(result)

        # extract solution from optimization if successful
        if (result.solver.status == SolverStatus.ok) \
            and (result.solver.termination_condition == TerminationCondition.optimal):
            sol_t = [i*value(mi.tf)/(2*value(mi.N)) for i in mi.t]
            sol_x = [[value(mi.x[i,j]) for i in mi.t] for j in mi.state]
            sol_u = [value(mi.u[i]) for i in mi.t]
            sol_f = [[value(mi.x[i,j]) for i in mi.t] for j in mi.state]
        else:
            sol_t, sol_x, sol_u, sol_f = None, None, None, None

        # remember this solution so it could be used to warmstart the next optimization
        self.sol_t_prev, self.sol_x_prev, self.sol_u_prev, self.sol_f_prev = \
            sol_t, sol_x, sol_u, sol_f

        toc = time.time() - tic #keep track of how long this function takes to evaluate

        return sol_t, sol_x, sol_u, toc

    def get_control(self, t):
        """
        Given the amount of time that has elapsed since the previous call to self.get_trajectory(),
        returns the control input that should be used, based on the trajectory generated by the
        previous call to self.get_trajectory()
        """
        if any(sol_prev is None for sol_prev in [self.sol_t_prev, self.sol_u_prev]):
            #if there is no trajectory optimization solution in memory, then return None
            #this will happen if this function is called immediately after self.reset_sol_prev(),
            #or if the previous call to self.get_trajectory was unsuccessful
            return None
        else:
            #calculate the piecewise quadratic interpolation at the current time
            if t >= self.sol_t_prev[-1]:
                #if t is out of bounds of the computed trajectory, then return the last control
                return self.sol_u_prev[-1]
            else:
                # get the index of the finite element that t belongs to
                fe_bounds = [self.sol_t_prev[i] for i in range(len(self.sol_t_prev)) if i%2==0]
                fe_idx = [i for i in range(len(fe_bounds)) \
                    if fe_bounds[i]<=t and t<fe_bounds[i+1]][0]

                #equation 4.10 in Matthew Kelly tutorial
                hk = fe_bounds[1] - fe_bounds[0] #finite element length
                tau = t - self.sol_t_prev[2*fe_idx] #time since left endpoint
                uk = self.sol_u_prev[2*fe_idx] #left endpoint of finite element
                uk_half = self.sol_u_prev[2*fe_idx+1] #midpoint of finite element
                uk_next = self.sol_u_prev[2*fe_idx+2] #right endpoint of finite element
                return 2.0/hk**2 * (tau - hk/2.0) * (tau-hk) * uk \
                    - 4.0/hk**2 * (tau) * (tau - hk) * uk_half \
                    + 2.0/hk**2 * (tau) * (tau - hk/2.0) * uk_next