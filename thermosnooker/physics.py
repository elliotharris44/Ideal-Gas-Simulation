import numpy as np

def maxwell(speed, kbt, mass=1.):
    """
    Args:
        speed(float or np.array) : speed of the ball
        kbt(float) : temperature of system multiplied by boltzmann's constant
        mass(float) : mass of the balls
    Returns:
        maxwellian PDF(float or np.array)
    """
    return mass*speed*np.exp(-mass*speed**2/(2*kbt))/kbt

def t_ratio_func(b_radius, a):
    """
    Modelling t_ration function as a negative quadratic passing through (0,1)

    Args:
        b_radius(float or np.array) : radius of the ball
        a(float) : constant to be determined
    Returns:
        negative quadratic(float or np.array)
    """
    return 1-a*b_radius**2

def rayleigh(radius, sig):
    """
    Modelling pair distance histogram function as a rayleigh
    probability distribution function

    Args:
        radius(float or np.array) : radius of the ball
        sig(float) : constant to be determined, standard deviation
    Returns:
        rayleigh PDF(float or np.array)
    """
    return radius*np.exp(-radius**2/(2*sig**2))/sig**2

def linear(x, m, c):
    """
    Args:
        x(float or np.array) : data to be fitted
        m(float) : gradient of the line
        c(float) : y-intercept of the line
    Returns:
        straight line(float or np.array)
    """
    return m*x + c
