import matplotlib.pyplot as plt
import numpy as np
from thermosnooker.balls import Container, Ball


class Simulation:


    def next_collision(self):
        raise NotImplementedError('next_collision() needs to be implemented in derived classes')
    def setup_figure(self):
        raise NotImplementedError('setup_figure() needs to be implemented in derived classes')
    def run(self, num_collisions, animate=False, pause_time=0.001):
        if animate:
            fig, axes = self.setup_figure()
        for _ in range(num_collisions):
            self.next_collision()
            if animate:
                plt.pause(pause_time)
        if animate:
            plt.show()


class SingleBallSimulation(Simulation):
    """
    Class to create simulation figure and carry out collisions for a single ball in a container

    Attributes:
        __container (Container) : container object created in Container class
        __ball (Ball) : ball object created in Ball class

    Methods:
        container() : returns container
        ball() : returns ball
        setup_figure() : creates figure with ball and container
        next_collision() : updates velocity and position of ball
    """
    def __init__(self, container, ball):
        """
        Initialises ball and container object

        Args:
            container(Container) : container object
            ball(Ball) : ball object
        """
        if not isinstance(container, Container):
            raise TypeError("First object is not a container")
        if not isinstance(ball, Ball) or isinstance(ball, Container):
            raise TypeError("First object is not a ball")
        
        self.__container = container
        self.__ball = ball
    
    def container(self):
        return self.__container
    
    def ball(self):
        return self.__ball
    
    def setup_figure(self):
        """
        Creates figure with ball and container

        Returns:
            tuple[Figure,Axes]
        """
        rad = self.container().radius()
        fig = plt.figure()
        ax = plt.axes(xlim=(-rad, rad), ylim=(-rad, rad))
        ax.add_artist(self.container().patch())
        ax.add_patch(self.ball().patch())
        return fig, ax
    
    def next_collision(self):
        """
        Updates velocity and position of ball and container
        """
        time = self.container().time_to_collision(self.ball())
        self.__container.move(time)
        self.__ball.move(time)
        self.container().collide(self.ball())
        self.__ball.nudge() # Fix floating point error


class MultiBallSimulation(Simulation):
    """
    Class to create simulation of multiple balls within a container
    Calculates state variables

    Attributes:
        __container(Container) : creates container object
        __balls(list) : list of ball objects in concentric rings
        __mass(float) : ball mass
        __time(float) : keeps track of elapsed time in simulation
    
    Methods:
        container() : returns container
        balls() : returns list of balls
        time() : returns total time
        setup_figure() : creates figure with balls and container
        next_collision() : finds the collision combination of lowest time
        changes position of all balls and velocity of objects colliding
        kinetic_energy() : calculates total kinetic energy of the system
        momentum() : calculates total vectoral momentum
    """
    def __init__(self, c_radius=10., b_radius=1., b_speed=10., b_mass=1., 
                rmax=8., nrings=3, multi=6):
        """
        Initialises container and balls in concentric circles

        Args:
            c_radius(float) : radius of the container
            b_radius(float) : radius of the balls
            b_speed(float) : initial speed of the balls
            b_mass(float) : mass of the balls
            rmax(float) : radius of outermost ring of balls
            nrings(int) : number of rings of balls
            multi(int) : the multiplying factor between number of balls
            in each ring
        """
        if not isinstance(c_radius, (int, float)) or c_radius <= 0:
            raise ValueError("Container radius must be a float greater than zero")
        if not isinstance(b_radius, (int, float)) or b_radius <= 0 or b_radius > c_radius:
            raise ValueError("Ball radius must be a float greater than zero and less than container")
        if not isinstance(b_speed, (int, float)) or b_speed < 0:
            raise ValueError("Ball speed must be a float greater than zero")
        if not isinstance(b_mass, (int, float)) or b_mass <= 0:
            raise ValueError("Ball mass must be a float greater than zero")
        if not isinstance(rmax, (int, float)) or rmax <= 0 or rmax > c_radius:
            raise ValueError("Max radius must be a float greater than zero and less than container")
        if not isinstance(nrings, (int, float)) or nrings < 0:
            raise ValueError("Number of rings must be a integer greater than zero")
        if not isinstance(multi, (int, float)) or multi < 0:
            raise ValueError("Multi must be a integer greater than zero")
        
        self.__container = Container(radius=c_radius)
        self.__balls = []
        self.__mass = b_mass
        self.__time = 0.0

        def rtrings(rmax, nrings, multi):
            for i in range(nrings + 1): # i is the index of ring
                num = multi*i # Number of points in the ith ring
                if i == 0:
                    yield [0., 0.]
                else:
                    for j in range(num):
                        yield [i*rmax/nrings, j*2*np.pi/num]

        coords = rtrings(rmax, nrings, multi)
        x_values = []
        y_values = []
        for n in coords:
            x_values.append(n[0]*np.cos(n[1]))
            y_values.append(n[0]*np.sin(n[1]))
        
        for i in range(len(x_values)):
            angle = np.random.uniform(0, 2 * np.pi)
            vel = b_speed * np.array([np.cos(angle), np.sin(angle)])
            self.__balls.append(Ball([x_values[i], y_values[i]], vel,
                                     b_radius, b_mass))
        
    def container(self):
        return self.__container
    
    def balls(self):
        return self.__balls
    
    def time(self):
        return self.__time
    
    def setup_figure(self):
        """
        Creates figure with balls and container

        Returns:
            tuple[Figure,Axes]
        """
        rad = self.container().radius()
        fig = plt.figure()
        ax = plt.axes(xlim=(-rad, rad), ylim=(-rad, rad))
        ax.add_artist(self.container().patch())
        for i in range(len(self.balls())):
            ax.add_patch(self.balls()[i].patch())
        return fig, ax
    
    def next_collision(self):
        """
        Finds the collision combination of lowest time and then
        changes position of all balls and velocities of objects colliding
        """
        dt = np.inf
        balls = self.balls()
        ball_ind = [0, "container"] # Which ball and other ball/container has smallest time
        
        for i in range(len(balls)): # Loops over first ball
            cont_time = self.container().time_to_collision(balls[i])
            ballj_ind = "container"
            if cont_time is not None:
                time = cont_time
            else:
                time = np.inf

            for j in range(i+1, len(balls)): # Loops over other balls
                t = balls[i].time_to_collision(balls[j])
                if t is not None:
                    if t < time:
                        time = t
                        ballj_ind = j
            if time < dt:
                dt = time
                ball_ind = [i, ballj_ind]
        
        for ball in balls:
            ball.move(dt)
        self.__container.move(dt)

        i, j = ball_ind
        if j == "container":
            self.container().collide(balls[i])
            # Fix floating point error, cant use move or fails time test
            self.__balls[i].nudge()
        else:
            balls[i].collide(balls[j])
            self.__balls[i].nudge()
            self.__balls[j].nudge()

        self.__time += dt
    
    def kinetic_energy(self):
        """
        Returns:
            ke_tot(float) : total kinetic energy
        """
        ke_tot = 10000000.0*np.linalg.norm(self.container().vel())**2/2 # KE of container
        for ball in self.balls():
            ke_tot += self.__mass*np.linalg.norm(ball.vel())**2/2
        return ke_tot
    
    def momentum(self):
        """
        Returns:
            mom_tot(np.array) : total vectorised momentum
        """
        mom_tot = 10000000.0*self.container().vel() # Momentum of container
        for ball in self.balls():
            mom_tot += self.__mass*ball.vel()
        return mom_tot
    
    def pressure(self):
        """
        Returns:
            float : pressure calculates from total rate of change in momentum
        """
        cont = self.container()
        if self.time() == 0:
            raise ZeroDivisionError("Divide by zero error in calculation of pressure from zero time")
        else:
            return cont.dp_tot()/(cont.surface_area()*self.time())
    
    def t_equipartition(self):
        """
        Returns:
            float : equipartition temperature
        """
        return self.kinetic_energy()/(len(self.balls())*1.380649e-23)
    
    def t_ideal(self):
        """
        Returns:
            float : temperature calculated from ideal gas law and other state variables
        """
        return self.pressure()*self.container().volume()/(len(self.balls())*1.380649e-23)
    
    def speeds(self):
        """
        Returns:
            list : all the speeds of the balls at that instant in time
        """
        speeds_list = []
        for ball in self.balls():
            speeds_list.append(np.linalg.norm(ball.vel()))
        return speeds_list


class BrownianSimulation(MultiBallSimulation):
    """
    Demonstrates Brownian motion by simulating lots of small balls and one big ball in a container

    Attributes:
        __container(Container) : creates container object
        __balls(list) : list of ball objects in concentric rings
        __mass(float) : ball mass
        __time(float) : keeps track of elapsed time in simulation
        __bb(Ball) : ball object with larger mass and radius than others
        __bb_pos(list) : the position of the ball after each collision of any ball 
        in the container
    
    Methods:
        container() : returns container
        balls() : returns list of balls
        time() : returns total time
        setup_figure() : creates figure with balls and container
        next_collision() : finds the collision combination of lowest time
        and appends the position of the big ball
        changes position of all balls and velocity of objects colliding
        kinetic_energy() : calculates total kinetic energy of the system
        momentum() : calculates total vectoral momentum
        bb_positions() : returns list of big ball positions
    """
    def __init__(self, c_radius=10., b_radius=1., b_speed=10., b_mass=1.,
                  rmax=8., nrings=3, multi=6, bb_radius=2, bb_mass=10):
        """
        Initialises container and balls in concentric circles with big ball at center

        Args:
            c_radius(float) : radius of the container
            b_radius(float) : radius of the balls
            b_speed(float) : initial speed of the balls
            b_mass(float) : mass of the balls
            rmax(float) : radius of outermost ring of balls
            nrings(int) : number of rings of balls
            multi(int) : the multiplying factor between number of balls
            in each ring
            bb_radius(float) : radius of the big ball
            bb_mass(float) : mass of the big ball
        """
        if not isinstance(bb_radius, (int, float)) or bb_radius <= 0 or bb_radius > c_radius:
            raise ValueError("Ball radius must be a float greater than zero and less than container")
        if not isinstance(bb_mass, (int, float)) or bb_mass <= 0:
            raise ValueError("Ball mass must be a float greater than zero")
        
        MultiBallSimulation.__init__(self, c_radius, b_radius, b_speed, b_mass,
                                      rmax, nrings, multi)
        self.__bb = Ball(pos=[0., 0.], vel=[0., 0.], radius=bb_radius, mass=bb_mass)
        getattr(self, "_MultiBallSimulation__balls")[0] = self.__bb
        self.__bb_pos = []
    
    def next_collision(self):
        self.__bb_pos.append(np.copy(self.__bb.pos()))
        MultiBallSimulation.next_collision(self)

    def bb_positions(self):
        return self.__bb_pos