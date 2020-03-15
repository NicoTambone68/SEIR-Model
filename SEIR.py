"""
The MIT License (MIT)
Copyright (c) 2020 Nicolò Tambone
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy
from math import sin, cos, log, ceil
from matplotlib import pyplot, rcParams, dates as mdates
#from matplotlib import rcParams
#from matplotlib import dates as mdates


"""
    This class implements the base S.E.I.R model
    Calculations are made with Runge-Kutta's 2nd order algorithm
    
"""
class SEIR_Model:
    def __init__(self, **kwargs):
        self.sigma  = float(kwargs.get('sigma', 26))
        self.beta   = float(kwargs.get('beta', 0.0029))
        self.gamma  = float(kwargs.get('gamma', 26))
        self.N      = int(kwargs.get('N', 10**5))
        self.dt     = float(kwargs.get('dt', 0.038))
        self.S      = int(kwargs.get('S', 8972))
        self.E      = int(kwargs.get('E', 50))        
        self.I      = int(kwargs.get('I', 0))
        self.R      = int(kwargs.get('R', 90978))
        self.Srk    = self.S
        self.Erk    = self.E        
        self.Irk    = self.I
        self.Rrk    = self.R
        self.start  = int(kwargs.get('start', 0))
        self.end    = int(kwargs.get('end', 100))

        self.init_data_structures()
        

        
    def get_parameters(self):
        """ Returns the current parameters for debug purpose.
        """
        return ("sigma=%.2f; beta=%.4f; gamma=%.2f; N=%i; S=%i; E=%i; I=%i; R=%i; p=%.2f; dt=%.5f" % \
                ( self.sigma, self.beta, self.gamma, self.N, self.S, self.E, self.I, self.R, self.p, self.dt ))

    

    
    def f(self, u):
        """Returns the right-hand side of the S.E.I.R. system of equations.
    
        Parameters
        ----------
        u : array of float
            array containing the solution at time n.
        
        Returns
        -------
        dudt : array of float
            array containing the RHS given u.
        """
        
        S = u[0]
        E = u[1]
        I = u[2]
        R = u[3]
        N = numpy.sum(u)
        return numpy.array([  -(self.beta*I*S)/N,
                               (self.beta*I*S)/N - self.sigma*E,
                                    self.sigma*E - self.gamma*I,
                                                   self.gamma*I
                           ])
    
    
    
    

    def rk2_step(self, u, f, dt):
        """Returns the solution at the next time-step using 2nd-order Runge-Kutta.
    
        Parameters
        ----------
        u : array of float
            solution at the previous time-step.
        f : function
            function to compute the right hand-side of the system of equation.
        dt : float
            time-increment.
    
        Returns
        -------
        u_n_plus_1 : array of float
        solution at the next time step.
        """
        u_star = u + 0.5*dt*f(u)
        return u + dt*f(u_star)        



    def update_sivr(self, u):
        """ Update the current values for the parameters S, E, I, R
        """
        self.Srk, self.Erk, self.Irk, self.Rrk = u[0], u[1], u[2], u[3]
          
            
    def init_data_structures(self):
        """ Initialize the discretization vector and other data structures
        """
        
        time_span = int(self.end - self.start)
        
        # calculate the number of time-steps based on time-span and dt
        time_steps = int((time_span/self.dt)) + 1 # + 1
        
        # initialize the discretization vector
        self.discretization_vector = numpy.linspace(0.0, time_span, time_steps)
        
        # initialize the array containing the solution for each time-step (runge-kutta)
        self.rkutt_data_vector = numpy.empty((time_steps, 4))        
        
        # Initialize the Reproduction number vector (runge-kutta)
        self.repro_num_rkutt = numpy.empty(time_steps)    
            
            
    def run_simulation(self, **kwargs):
        """ Perform the computations over the given time span.
            Uses both Euler's method and Runge-Kutta's method.
        """
        self.p          = float(kwargs.get('p', 0.0))  
        
        # calculate the number of time-steps
        time_steps = int((self.end - self.start)/self.dt) + 1

        # calculate start time-step
        start = int(self.start/self.dt) 
       
        # fill 1st element with initial values
        self.rkutt_data_vector[start] = numpy.array([self.Srk, self.Erk, self.Irk, self.Rrk])

        # Time loop over the time-steps
        for n in range(start, time_steps - 1):
            # computation using the runge-kutta method
            self.rkutt_data_vector[n+1] = self.rk2_step(self.rkutt_data_vector[n], self.f, self.dt)
            
            # update the current value of the S I V R variables according to the runge-kutta's method
            self.update_sivr(self.rkutt_data_vector[n+1])

        # Calculate the reproduction number    
        for n in range(start, time_steps):
            # computation using the runge-kutta's method dataset
            N = numpy.sum(self.rkutt_data_vector[n,:])
            self.repro_num_rkutt[n] = (self.beta*self.rkutt_data_vector[n,0])/(self.gamma*N)
            
       
# this class encapsulates the SEIR model to ease the simulation stage
#
class Simulation:

    def __init__(self, **kwargs):
        self.R0          = float(kwargs.get('R0', 2.1))
        self.S           = int(kwargs.get('S', 10**5))
        self.E           = int(kwargs.get('E', 50))
        self.I           = int(kwargs.get('I', 0))
        self.R           = int(kwargs.get('R', 90978))
        self.itime       = int(kwargs.get('days_incubation', 14))
        self.rtime       = int(kwargs.get('days_recovery', 21))
        self.etime       = int(kwargs.get('days_elapsed', 100))

        self.sigma       = 1/self.itime
        self.gamma       = 1/self.rtime
        self.beta        = self.R0*self.gamma

#        self.N           = int(kwargs.get('N', 10**5))
#        self.p           = float(kwargs.get('p', 0.95))
#        self.omega       = float(kwargs.get('omega', 0.0005))        
        self.dt          = float(kwargs.get('dt', 0.001))
        self.start       = int(kwargs.get('start', 0))
        self.end         = int(kwargs.get('end', self.etime))
        self.rkutt_data_vector  = []


    def get_parameters(self):
        """ Return the current parameters for debug purpose
        """
        return ("SEIR params.: sigma %.4f; beta %.4f; gamma %.4f; S %i; E %i; I %i; R %i; dt %.4f;\n Method: 2nd order Runge-Kutta." % \
                ( self.sigma, self.beta, self.gamma, self.S, self.E, self.I, self.R, self.dt ))

    
    def build_discretization_vector(self):
        """ Set the discretization vector as date strings of 1 minute steps
        """
        dt = numpy.datetime64('2020-01-01') 
        
        self.discretization_vector = []
        
        for x in self.rkutt_data_vector:
            numpy.datetime_as_string(dt, unit='h')
            self.discretization_vector.append(dt)
            dt += numpy.timedelta64(1, 'h')
            
        
        
    def run_simulation(self, **kwargs):
        """ Template for the first batch of simulations of the assignement, 
        which requires varying the p and dt parameters

        Parameters
        ----------
        dt : float
            time-increment.
        
        Returns           

        Plots the requested graphs

        """
        # setting up the model for the simulation , N=self.N, p=self.p
        m = SEIR_Model(sigma=self.sigma, beta=self.beta, gamma=self.gamma
                                 , S=self.S, I=self.I, E=self.E, R=self.R
                                 , dt=self.dt
                                 , start=self.start, end=self.end)
        
        m.run_simulation()
              
        self.rkutt_data_vector = m.rkutt_data_vector
        
        self.discretization_vector = m.discretization_vector
        
        self.repro_num_rkutt = m.repro_num_rkutt

        # Array for infected individuals requiring I.C.
        self.infected_ic = numpy.apply_along_axis((lambda a : a * 0.1), 0, self.rkutt_data_vector[:, 2])
        

""" Plot the charts    
    def plot_simulation(self):

 
        
        fig, ax = pyplot.subplots(2, 1, figsize=(18.5, 18.5))
        #fig.set_size_inches(18.5, 10.5)
        
        # plot S
        ax[0].plot(self.discretization_vector, self.rkutt_data_vector[:, 0], 'm-', lw=1, label="Sensitive");
        
        # plot E
        ax[0].plot(self.discretization_vector, self.rkutt_data_vector[:, 1], 'b-', lw=1, label="Exposed");        
        
        # plot I
        ax[0].plot(self.discretization_vector, self.rkutt_data_vector[:, 2], 'r-', lw=2, label="Infected");
        
        
        # plot R
        ax[0].plot(self.discretization_vector, self.rkutt_data_vector[:, 3], 'g-', lw=1, label="Recovered");

        ax[0].legend(loc="best", fontsize=16)
        
        ax[0].set_xlim(self.start, self.end)
        ax[0].set_ylabel('Population', fontsize=16)
        ax[0].set_xlabel('Days\n'+ self.get_parameters(), fontsize=16)
                
        ax[0].grid(True)
        
        # Plot the reproduction number
        ax[1].plot(self.discretization_vector, self.repro_num_rkutt, 'b-', lw=2, label="Effective Reproduction Number R(t)");
        ax[1].set_xlim(self.start, self.end)
        ax[1].set_ylabel('Effective Reproduction Number R(t)', fontsize=16)
        ax[1].set_xlabel('Days', fontsize=16)
        ax[1].legend(loc="best", fontsize=16)
        ax[1].grid(True)
        
        pyplot.show()
"""


class Plotter:
    def __init__(self, **kwargs):
        self.sim = (kwargs.get('Simulation'))
        self.sim2 = (kwargs.get('Simulation_2'))


    def get_parameters(self):
        """ Return the current parameters for debug purpose
        """
        return ("SEIR params.: sigma %.4f; beta %.4f; gamma %.4f; S %i; E %i; I %i; R %i; dt %.4f;\n Method: 2nd order Runge-Kutta." % \
                ( self.sim.sigma, self.sim.beta, self.sim.gamma, self.sim.S, self.sim.E, self.sim.I, self.sim.R, self.sim.dt ))

    def get_parameters2(self):
        """ Return the current parameters for debug purpose
        """
        return ("Days - SEIR params.(A): sigma %.4f; beta %.4f; gamma %.4f; S %i; E %i; I %i; R %i; dt %.4f;\n \
                             (B): sigma %.4f; beta %.4f; gamma %.4f; S %i; E %i; I %i; R %i; dt %.4f;\n Method: 2nd order Runge-Kutta algorithm" % \
                ( self.sim.sigma, self.sim.beta, self.sim.gamma, self.sim.S, self.sim.E, self.sim.I, self.sim.R, self.sim.dt, 
                  self.sim2.sigma, self.sim2.beta, self.sim2.gamma, self.sim2.S, self.sim2.E, self.sim2.I, self.sim2.R, self.sim.dt ))


    def plot(self):
        #self.sim1.plot_simulation()
        fig, ax = pyplot.subplots(2, 1, figsize=(18.5, 18.5))
        #fig.set_size_inches(18.5, 10.5)
        
        # plot S
        ax[0].plot(self.sim.discretization_vector, self.sim.rkutt_data_vector[:, 0], 'm-', lw=1, label="Sensitive")
        
        # plot E
        ax[0].plot(self.sim.discretization_vector, self.sim.rkutt_data_vector[:, 1], 'b-', lw=1, label="Exposed")        
        
        # plot I
        ax[0].plot(self.sim.discretization_vector, self.sim.rkutt_data_vector[:, 2], 'r-', lw=2, label="Infected")
        
        
        # plot R
        ax[0].plot(self.sim.discretization_vector, self.sim.rkutt_data_vector[:, 3], 'g-', lw=1, label="Recovered")

        ax[0].legend(loc="best", fontsize=16)
        
        ax[0].set_xlim(self.sim.start, self.sim.end)
        ax[0].set_ylabel('Population', fontsize=16)
        ax[0].set_xlabel('Days\n'+ self.sim.get_parameters(), fontsize=16)
                
        ax[0].grid(True)
        
        # Plot the reproduction number
        ax[1].plot(self.sim.discretization_vector, self.sim.repro_num_rkutt, 'b-', lw=2, label="Effective Reproduction Number R(t)")
        ax[1].set_xlim(self.sim.start, self.sim.end)
        ax[1].set_ylabel('Effective Reproduction Number R(t)', fontsize=16)
        ax[1].set_xlabel('Days', fontsize=16)
        ax[1].legend(loc="best", fontsize=16)
        ax[1].grid(True)
        
        pyplot.show()


    def plot_comparison_rt(self):
        #self.sim1.plot_simulation()
        fig, ax = pyplot.subplots(2, 1, figsize=(18.5, 18.5))
        #fig.set_size_inches(18.5, 10.5)

        ax[0].set_xlim(self.sim.start, self.sim.end)

        # plot S
        #ax[0].plot(self.sim.discretization_vector, self.sim.rkutt_data_vector[:, 0], 'm-', lw=1, label="Sensitive")
        
        # plot E
        ax[0].plot(self.sim.discretization_vector, self.sim.rkutt_data_vector[:, 1], 'b-', lw=1, label="Exposed A")        

        # plot E2
        ax[0].plot(self.sim.discretization_vector, self.sim2.rkutt_data_vector[:, 1], 'b--', lw=1, label="Exposed B")                
        
        # plot I
        ax[0].plot(self.sim.discretization_vector, self.sim.rkutt_data_vector[:, 2], 'r-', lw=2, label="Infected A")
        
        # plot I2
        ax[0].plot(self.sim.discretization_vector, self.sim2.rkutt_data_vector[:, 2], 'r--', lw=2, label="Infected B")

        
        ## delete me ax[0].hlines(20, self.sim.start, self.sim.end, colors='g', linestyles='solid', label='')

        ax[0].hlines(20, ax[0].get_xlim()[0], ax[0].get_xlim()[1], colors="g",  linestyles='solid', zorder=100, label='IC Availability')

        # plot R
        #ax[0].plot(self.sim.discretization_vector, self.sim.rkutt_data_vector[:, 3], 'g-', lw=1, label="Recovered")

        # maxY1 = numpy.amax(self.sim.rkutt_data_vector[:, 2])
        # maxY2 = numpy.amax(self.sim2.rkutt_data_vector[:, 2])

        ax[0].legend(loc="best", fontsize=16)
        

        ax[0].set_ylabel('Population', fontsize=16)
        # + self.get_parameters2()
        ax[0].set_xlabel('Days\n', fontsize=16)
                
        ax[0].grid(True)

        # Plot the reproduction number
        ax[1].plot(self.sim.discretization_vector, self.sim.repro_num_rkutt, 'b-', lw=2, label="Effective Reproduction Number R(t) (A)")
        ax[1].plot(self.sim.discretization_vector, self.sim2.repro_num_rkutt, 'b--', lw=2, label="Effective Reproduction Number R(t) (B)")        
        ax[1].set_xlim(self.sim.start, self.sim.end)
        ax[1].set_ylabel('Effective Reproduction Number R(t)', fontsize=16)
        ax[1].set_xlabel('Days', fontsize=16)
        ax[1].legend(loc="best", fontsize=16)
        ax[1].grid(True)        

        pyplot.show()        


    def plot_comparison(self):
        
        fig, ax = pyplot.subplots(1, 1, figsize=(18.5, 10.5))

        ax.set_xlim(self.sim.start, self.sim.end)

        # plot S
        #ax[0].plot(self.sim.discretization_vector, self.sim.rkutt_data_vector[:, 0], 'm-', lw=1, label="Sensitive")
        
        # plot E
        ax.plot(self.sim.discretization_vector, self.sim.rkutt_data_vector[:, 1], 'b-', lw=1, label="Exposed A")        

        # plot E2
        ax.plot(self.sim.discretization_vector, self.sim2.rkutt_data_vector[:, 1], 'b--', lw=1, label="Exposed B")                
        
        # plot I
        ax.plot(self.sim.discretization_vector, self.sim.rkutt_data_vector[:, 2], 'm-', lw=1, label="Infected A")

        # plot I requiring I.C.
        ax.plot(self.sim.discretization_vector, self.sim.infected_ic, 'r-', lw=2, label='Infected A requiring IC')
        
        # plot I2
        ax.plot(self.sim.discretization_vector, self.sim2.rkutt_data_vector[:, 2], 'm--', lw=1, label="Infected B")
        
        # plot I2 requiring I.C.
        ax.plot(self.sim.discretization_vector, self.sim2.infected_ic, 'r--', lw=2, label='Infected B requiring IC')
        
        ax.hlines(20, ax.get_xlim()[0], ax.get_xlim()[1], colors="lime",  linestyles='solid', zorder=100, label='IC Availability')

        ax.legend(loc="best", fontsize=16)

        ax.set_ylabel('Population', fontsize=16)
        # + self.get_parameters2()
        ax.set_xlabel(self.get_parameters2(), fontsize=16)
                
        ax.grid(True)

        pyplot.show()  

