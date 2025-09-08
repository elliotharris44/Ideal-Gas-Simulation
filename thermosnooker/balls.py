import numpy as np
from matplotlib.patches import Circle


class Ball:
    """
    Class to model the elastic collision of two balls or a ball and a container in 2D
    
    Attributes:
        __pos (np.array) : 2D position of center of the ball
        __vel (np.array) : 2D velocity of ball
        __radius (float) : radius of the ball
        __mass (float) : mass of the ball
        __patch (matplotlib.patches.Circle) : graphical representation of the ball,
        function of its radius and position
        _is_container (bool) : keeps track of whether object is a container

    Methods:
        pos() : returns position
        vel() : returns the velocity
        radius() : returns the radius
        mass() : returns the mass
        patch() : returns the patch object
        is_container() : returns object type boolian
        set_vel(vel) : changes velocity to vel
        move(dt) : changes position to position after time dt
        nudge(dt) : moves ball in direction of velocity for an infinitesimal time
        Avoids floating point errors
        time_to_collision(other) : time to collision with other ball or container
        collide(other) : finds velocity after collision, aswell as momentum
    """

    def __init__(self, pos=[0., 0.], vel=[1., 0.], radius=1.0, mass=1.0, container=False):
        """
        Initialises a Ball object

        Args:
            pos(list): central 2D position of the ball
            vel(list): 2D velocity of the ball
            radius(float): radius of the ball
            mass(float): mass of the ball
        """
        if not isinstance(pos, (list, np.ndarray)):
            raise TypeError("Position needs to be list or np.array")
        if len(pos) != 2:
            raise ValueError("Position must be 2D")
        if not isinstance(vel, (list, np.ndarray)):
            raise TypeError("Velocity needs to be list or np.array")
        if len(vel) != 2:
            raise ValueError("Velocity must be 2D")
        if radius <= 0 or mass <= 0:
            raise ValueError("Radius and Mass must be positive")
        
        self.__pos = np.array(pos, dtype=float)
        self.__vel = np.array(vel, dtype=float)
        self.__radius = float(radius)
        self.__mass = float(mass)
        self.__patch = Circle(tuple(pos), radius)
        self._is_container = container
    
    def pos(self):
        return self.__pos
    
    def vel(self):
        return self.__vel
    
    def radius(self):
        return self.__radius
    
    def mass(self):
        return self.__mass
    
    def patch(self):
        return self.__patch
    
    def is_container(self):
        return self._is_container

    def set_vel(self, vel):
        """
        Changes velocity of ball to input

        Args:
            vel(np.array)
        
        Returns:
            Adjusted ball object
        """
        if not isinstance(vel, (list, np.ndarray)):
            raise TypeError("Velocity needs to be list or np.array")
        if len(vel) != 2:
            raise ValueError("Velocity must be 2D")
        self.__vel = np.array(vel)
        return self
    
    def move(self, dt):
        """
        Moves ball in direction of velocity for time of dt

        Args:
            dt(float) : time ball moves for
        
        Returns:
            Adjusted ball object
        """
        self.__pos += self.__vel*dt
        self.__patch.center += self.__vel*dt
        return self
    
    def nudge(self, dt=1e-10):
        """
        Moves ball in direction of velocity for an infinitesimal time
        Avoids floating point errors

        Args:
            dt(float) : time ball moves for
        
        Returns:
            Adjusted ball object
        """
        self.__pos += self.__vel*dt
        self.__patch.center += self.__vel*dt
    
    def time_to_collision(self, other):
        """
        Calculates the time for a collision between two balls or a ball and a container

        Args:
            other(Ball or Container) : object that self collides with
        
        Returns:
            float : Smallest time of collision or np.inf/None if no collision
        """
        rel_vel = self.vel() - other.vel()
        rel_pos = self.pos() - other.pos()
        if self.is_container() or other.is_container():
            # Taking negative combination for radius since collision between ball and container
            radius_tot = self.radius() - other.radius()
        else:
            # Taking positive combination for radius since collision between two balls
            radius_tot = self.radius() + other.radius()

        vel_mag = np.dot(rel_vel, rel_vel)
        pos_mag = np.dot(rel_pos, rel_pos)
        pos_vel = np.dot(rel_pos, rel_vel)
        discriminant = pos_vel**2-vel_mag*(pos_mag-radius_tot**2)

        if discriminant < 0 or vel_mag < 1e-8: # No solutions
            return None
        else:
            dt1 = (-pos_vel+np.sqrt(discriminant))/vel_mag
            dt2 = (-pos_vel-np.sqrt(discriminant))/vel_mag
            if dt1 < 0 and dt2 < 0: # Negative time solutions
                return np.inf
            elif dt1 < 0 and dt2 > 1e-8:
                return dt2
            elif dt2 < 0 and dt1 > 1e-8:
                return dt1
            elif min(dt1, dt2) > 1e-8:
                return min(dt1, dt2)
            else:
                return np.inf
            
    def collide(self, other):
        """
        Calculates the velocities of the two objects after collision and momentum change

        Args:
            other(Ball or Container) : object that self collides with
        
        Returns:
            tuple : (updated self object, updated other object, dp(float) : momentum change)
        """
        # Dealing with zero error
        offset = self.__pos-other.__pos
        if np.linalg.norm(offset) < 1e-12:
            offset = np.random.randn(2)
            offset = offset*1e-10/np.linalg.norm(offset)

        # Calculating final velocities
        m1 = self.mass()
        m2 = other.mass()
        vel_com = (m1*self.vel()+m2*other.vel())/(m1+m2)
        v1_com = self.vel() - vel_com
        v2_com = other.vel() - vel_com
        # Normal vector between centers at collision
        normal = (offset)/np.linalg.norm(offset)
        dot_v1_v2 = np.dot(v1_com-v2_com, normal)
        v1_com_fin = v1_com - 2*m2*dot_v1_v2*normal/(m1+m2)
        v2_com_fin = v2_com + 2*m1*dot_v1_v2*normal/(m1+m2)

        # Calculating momentum change of self object
        if self.is_container() or other.is_container():
            dp = m1*np.linalg.norm(v1_com_fin + vel_com - self.__vel)
        else:
            dp = 0
        self.__vel = v1_com_fin + vel_com
        other.__vel = v2_com_fin + vel_com
        return self, other, dp

class Container(Ball):
    """
    Class to initialise the container and keep track of collisions with it

    Attributes:
        __pos (array) : 2D position of center of the container, should always be at origin
        __vel (array) : 2D velocity of ball, should always be close to zero vector
        __radius (float) : radius of the container
        __mass (float) : mass of the ball, should be very large preventing movement
        _patch (matplotlib.patches.Circle) : graphical representation of the container,
        function of it's radius and position
        _is_container (bool) : keeps track of whether object is a container
        self.__dp_tot (float) : keeps track of the total momentum change
    
    Methods:
        Method inherited from Ball:
        pos() : returns position
        vel() : returns the velocity
        radius() : returns the radius
        mass() : returns the mass
        patch() : returns the patch object
        is_container() : returns object type boolian
        set_vel(vel) : changes velocity to vel
        move(dt) : changes position to position after time dt
        time_to_collision(other) : time to next collision of ball with container
        collide(other) : velocity of container remains unchanged, alters ball velocity
        volume() : calculates volume of the container, area in 2D
        surface_area() : calculates surface area of container, perimeter in 2D
        dp_tot() : keeps track of total momentum change of container
    """

    def __init__(self, pos=[0., 0.], vel=[0., 0.], radius=10.0, mass=10000000.0):
        """
        Initialises a Container object

        Args:
            pos(list): central 2D position of the container
            vel(list): 2D velocity of the container
            radius(float): radius of the container
            mass(float): mass of the container
        """
        Ball.__init__(self, pos, vel, radius, mass, container=True)
        self.__dp_tot = 0
        self._patch = Circle((0, 0), radius=radius, fill=False)

    def collide(self, other):
        """
        Calculates the velocities of the two objects after collision and momentum change

        Args:
            other(Ball) : object that self collides with
        
        Returns:
            Objects with updated velocity
        """
        self_new, other_new, dp = Ball.collide(self, other)
        self.__dp_tot += dp
        return self_new, other_new

    def volume(self):
        """
        Returns:
            float : Volume of the container
        """
        return np.pi*self.radius()**2
    
    def surface_area(self):
        """
        Returns:
            float : Surface area of the container
        """
        return 2*np.pi*self.radius()
    
    def dp_tot(self):
        return self.__dp_tot
    
    def patch(self):
        return self._patch