"""Analysis Module."""
import matplotlib.pyplot as plt
import numpy as np
import scipy

from thermosnooker.balls import Ball, Container
from thermosnooker.simulations import SingleBallSimulation, MultiBallSimulation, BrownianSimulation
from thermosnooker.physics import maxwell, t_ratio_func, rayleigh, linear
from thermosnooker.errors import values_task13, values_task14

def task9():
    """
    Task 9.

    In this function, you should test your animation. To do this, create a container
    and ball as directed in the project brief. Create a SingleBallSimulation object from these
    and try running your animation. Ensure that this function returns the balls final position and
    velocity.

    Returns:
        tuple[NDAarray[np.float64], NDAarray[np.float64]]: The balls final position and velocity
    """
    c = Container(radius=10.)
    b = Ball(pos=[-5, 0], vel=[1, 0.], radius=1., mass=1.)
    sbs = SingleBallSimulation(container=c, ball=b)

    sbs.run(num_collisions=20, animate=True, pause_time=0.5)
    return b.pos(), b.vel()


def task10(b_radius=1, multi=6, nrings=3, num_collisions=500):
    """
    Task 10.

    In this function we shall test your MultiBallSimulation. Create an instance of this class using
    the default values described in the project brief and run the animation for 500 collisions.

    Watch the resulting animation carefully and make sure you aren't seeing errors like balls sticking
    together or escaping the container.
    """
    mbs = MultiBallSimulation(b_radius=b_radius, nrings=nrings, multi=multi)
    mbs.run(num_collisions=num_collisions, animate=True)

def task11():
    """
    Task 11.

    In this function we shall be quantitatively checking that the balls aren't escaping or sticking.
    To do this, create the two histograms as directed in the project script. Ensure that these two
    histogram figures are returned.

    Returns:
        tuple[Figure, Figure]: The histograms (distance from centre, inter-ball spacing).
    """
    mbs = MultiBallSimulation(b_radius=0.1, nrings=5, multi=8)
    mbs.run(num_collisions=500)

    ball_dist = []
    ball_pairs_dist = []
    balls = mbs.balls()
    number = len(balls)
    for i in range(number):
        ball_dist.append(np.linalg.norm(balls[i].pos()))
        for j in range(i+1, number):
            ball_pairs_dist.append(np.linalg.norm(balls[i].pos()-balls[j].pos()))
    
    bins1 = np.arange(0, max(ball_dist) + 2, 1)
    counts1, bin_edges = np.histogram(ball_dist, bins=bins1)
    errors1 = np.sqrt(counts1)
    bin_centers1 = (bin_edges[:-1] + bin_edges[1:])/2
    popt1, pcov = scipy.optimize.curve_fit(linear, bin_centers1, counts1)
    fig1, ax1 = plt.subplots()

    ax1.hist(ball_dist, bins=bins1, label="Binned Data")
    ax1.errorbar(bin_centers1, counts1, yerr=errors1, fmt='.', capsize=3, label='Simulated Data with errors')
    ax1.plot(bin_centers1, linear(bin_centers1, *popt1), label="Linear Fit", color="red")
    ax1.set_title(f"Histogram showing distribution of {number} 0.1m balls within a container", fontsize=10)
    ax1.set_xlabel("Distance from the centre of the container (m)", fontsize=12)
    ax1.set_ylabel("Number of balls", fontsize=12)
    ax1.legend()

    num_pairs = number*(number-1)/2
    def scaled_rayleigh(radius, sig):
        return rayleigh(radius, sig)*num_pairs
    bins2 = np.arange(0, max(ball_pairs_dist)+2, 1)
    counts2, bin_edges = np.histogram(ball_pairs_dist, bins=bins2)
    errors2 = np.sqrt(counts2)
    bin_centers2 = (bin_edges[:-1] + bin_edges[1:])/2
    popt2, pcov = scipy.optimize.curve_fit(scaled_rayleigh, bin_centers2, counts2)
    fig2, ax2 = plt.subplots()

    ax2.hist(ball_pairs_dist, bins=bins2, label="Binned Data")
    ax2.errorbar(bin_centers2, counts2, yerr=errors2, fmt='.', capsize=3, label='Simulated Data with errors')
    ax2.plot(bin_centers2, scaled_rayleigh(bin_centers2, *popt2), label="Rayleigh Fit", color="red")
    ax2.set_title(f"Histogram showing distribution of {number} 0.1m balls within a container", fontsize=10)
    ax2.set_xlabel("Distance between each pair of balls (m)", fontsize=12)
    ax2.set_ylabel("Number of balls", fontsize=12)
    ax2.legend()
    return fig1, fig2


def task12(num_collisions=500):
    """
    Task 12.

    In this function we shall check that the fundamental quantities of energy and momentum are conserved.
    Additionally we shall investigate the pressure evolution of the system. Ensure that the 4 figures
    outlined in the project script are returned.

    Returns:
        tuple[Figure, Figure, Figure, Figure]: matplotlib Figures of the KE, momentum_x, momentum_y ratios
                                               as well as pressure evolution.
    """
    mbs = MultiBallSimulation(b_radius=1)
    ke_ini = mbs.kinetic_energy()
    mom_inix = mbs.momentum()[0]
    mom_iniy = mbs.momentum()[1]
    ratio_ke = []
    ratio_momx = []
    ratio_momy = []
    pressure = []
    time = []

    for i in range(num_collisions):
        mbs.next_collision()
        time.append(mbs.time())
        ratio_ke.append(mbs.kinetic_energy()/ke_ini)
        ratio_momx.append(mbs.momentum()[0]/mom_inix) # Not constant
        ratio_momy.append(mbs.momentum()[1]/mom_iniy)
        pressure.append(mbs.pressure())
    
    fig1, ax1 = plt.subplots()
    ax1.plot(time, np.array(ratio_ke)-1)
    ax1.set_title("Graph showing conservation of kinetic energy", fontsize=12)
    ax1.set_ylabel("Ratio of KE compared to initial (deviation from 1)", fontsize=10)
    ax1.set_xlabel("Time (s)", fontsize=12)

    fig2, ax2 = plt.subplots()
    ax2.plot(time, np.array(ratio_momx)-1)
    ax2.set_title("Graph showing conservation of x-momentum", fontsize=12)
    ax2.set_ylabel("mom_x[t]/mom_x[0] (deviation from 1)", fontsize=10)
    ax2.set_xlabel("Time (s)", fontsize=12)

    fig3, ax3 = plt.subplots()
    ax3.plot(time, np.array(ratio_momy)-1)
    ax3.set_title("Graph showing conservation of y-momentum", fontsize=12)
    ax3.set_ylabel("mom_y[t]/mom_y[0] (deviation from 1)", fontsize=10)
    ax3.set_xlabel("Time (s)", fontsize=12)

    fig4, ax4 = plt.subplots()
    ax4.plot(time, pressure)
    ax4.set_title("Graph showing pressure-time evolution", fontsize=12)
    ax4.set_ylabel("Pressure (Pa)", fontsize=12)
    ax4.set_xlabel("Time (s)", fontsize=12)
    return fig1, fig2, fig3, fig4


def task13(data_points=10):
    """
    Task 13.

    In this function we investigate how well our simulation reproduces the distributions of the IGL.
    Create the 3 figures directed by the project script, namely:
    1) PT plot
    2) PV plot
    3) PN plot
    Ensure that this function returns the three matplotlib figures.

    Returns:
        tuple[Figure, Figure, Figure]: The 3 requested figures: (PT, PV, PN)
    """
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    counter = 0
    v = np.pi*10**2 # Default volume, 37 default balls
    b_radius_list = [0.1, 0.5, 1]
    speed_list = np.linspace(0.1, 300, data_points)
    radius_list = np.linspace(10, 20, data_points)
    multi_list = np.arange(1, 9) # Multi=9 causes overlapping

    temp = [[], [], []]
    volume = [[], [], []]
    number = [[], [], []]

    pressure_temp = [[], [], []]
    pressure_volume = [[], [], []]
    pressure_number = [[], [], []]

    pressure_temp_ideal = []
    pressure_number_ideal = []
    for b_radius in b_radius_list:
        for speed in speed_list:
            mbs = MultiBallSimulation(b_radius=b_radius, b_speed=speed)
            mbs.run(150)
            temp[counter].append(mbs.t_equipartition())
            pressure_temp[counter].append(mbs.pressure())
            if counter == 0:
                pressure_temp_ideal.append(37*1.380649e-23*mbs.t_equipartition()/v)
        
        for radius in radius_list:
            mbs = MultiBallSimulation(c_radius=radius, b_radius=b_radius)
            mbs.run(150)
            volume[counter].append(mbs.container().volume())
            pressure_volume[counter].append(mbs.pressure())
        
        for multi in multi_list:
            mbs = MultiBallSimulation(b_radius=b_radius, multi=int(multi))
            mbs.run(300)
            number[counter].append(len(mbs.balls()))
            pressure_number[counter].append(mbs.pressure())
            if counter == 0:
                pressure_number_ideal.append(len(mbs.balls())*1.380649e-23*mbs.t_equipartition()/v)

        counter += 1
    pt1, stdt1, pt2, stdt2, pt3, stdt3, pv1, stdv1, pv2, stdv2, \
    pv3, stdv3, pn1, stdn1, pn2, stdn2, pn3, stdn3 = values_task13()
    # If using mean values, comment out if want to use generated values
    pressure_temp = [pt1, pt2, pt3]
    pressure_volume = [pv1, pv2, pv3]
    pressure_number = [pn1, pn2, pn3]

    std_temp = [stdt1, stdt2, stdt3]
    std_volume = [stdv1, stdv2, stdv3]
    std_number = [stdn1, stdn2, stdn3]

    for i in range(3):
        ax1.errorbar(temp[i], pressure_temp[i], yerr=std_temp[i], fmt=".", label=f"Radius = {b_radius_list[i]}m", capsize=2)
        ax2.errorbar(volume[i], pressure_volume[i], yerr=std_volume[i], fmt=".", label=f"Radius = {b_radius_list[i]}m", capsize=2)
        ax3.errorbar(number[i], pressure_number[i], yerr=std_number[i], fmt=".", label=f"Radius = {b_radius_list[i]}m", capsize=2)

    radius_list1 = np.linspace(10, 20, 100)
    volume_ideal = []
    pressure_volume_ideal = []

    for radius in radius_list1:
        mbs = MultiBallSimulation(c_radius=radius)
        vol = mbs.container().volume()
        volume_ideal.append(vol)
        pressure_volume_ideal.append(37*1.380649e-23*mbs.t_equipartition()/vol)
    
    ax1.plot(temp[0], pressure_temp_ideal, label="Ideal Gas Law Line")
    ax1.set_title("Pressure vs Temperature", fontsize=12)
    ax1.set_xlabel("Temperature (K)", fontsize=12)
    ax1.set_ylabel("Pressure (Pa)", fontsize=12)
    ax1.legend()

    ax2.plot(volume_ideal, pressure_volume_ideal, label="Ideal Gas Law Line")
    ax2.set_title("Pressure vs Volume", fontsize=12)
    ax2.set_xlabel("Volume (m^3)", fontsize=12)
    ax2.set_ylabel("Pressure (Pa)", fontsize=12)
    ax2.legend()

    ax3.plot(number[0], pressure_number_ideal, label="Ideal Gas Law Line")
    ax3.set_title("Pressure vs Number", fontsize=12)
    ax3.set_xlabel("Number", fontsize=12)
    ax3.set_ylabel("Pressure (Pa)", fontsize=12)
    ax3.legend()
    return fig1, fig2, fig3


def task14(data_points=20):
    """
    Task 14.

    In this function we shall be looking at the divergence of our simulation from the IGL. We shall
    quantify the ball radii dependence of this divergence by plotting the temperature ratio defined in
    the project brief.

    Returns:
        Figure: The temperature ratio figure.
    """
    b_radius_list = np.linspace(0.01, 1, data_points)
    t_ratio = []
    for radius in b_radius_list:
        mbs = MultiBallSimulation(b_radius=radius)
        for i in range(200): # 500 used in figure on summary report
            mbs.next_collision() # Task16 test fails if run used
        t_ratio.append(mbs.t_equipartition()/mbs.t_ideal())
    
    # Obtained from running code 5 times
    mean, std = values_task14()
    
    fig, ax = plt.subplots()
    # With error bars but not using current run
    # ax.errorbar(b_radius_list, mean, yerr=std, fmt=".", capsize=2)
    # Using current run without error bars
    ax.plot(b_radius_list, t_ratio, "+")
    ax.set_title("Ratio of t_equipartition to t_ideal vs Radius of ball", fontsize=12)
    ax.set_xlabel("Radius of ball (m)", fontsize=12)
    ax.set_ylabel("Temperature ratio", fontsize=12)
    return fig

def task15(num_collisions=500):
    """
    Task 15.

    In this function we shall plot a histogram to investigate how the speeds of the balls evolve from the initial
    value. We shall then compare this to the Maxwell-Boltzmann distribution. Ensure that this function returns
    the created histogram.

    Returns:
        Figure: The speed histogram.
    """
    fig, ax = plt.subplots()
    initial_speed = [10., 20., 30.]
    colors = ["red", "blue", "green"]
    for i, speed in enumerate(initial_speed):
        mbs = MultiBallSimulation(b_radius=0.5, b_speed=speed, multi=8)
        mbs.run(num_collisions)
        
        mbs_speeds = mbs.speeds()
        kbt = mbs.t_equipartition()*1.380649e-23
        speed_list = np.linspace(0, max(mbs_speeds), 100)

        bin_width = 3
        maxwell_scaled = len(mbs.balls())*bin_width*maxwell(speed_list, kbt)

        bins = np.arange(0, max(mbs_speeds)+bin_width, bin_width)
        counts, bin_edges = np.histogram(mbs_speeds, bins=bins)
        errors = np.sqrt(counts)
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
        ax.errorbar(bin_centers, counts, yerr=errors, fmt=".", color=colors[i], capsize=2, label=f"{int(speed)}m/s Simulated Data")
        ax.hist(mbs_speeds, bins=bins, alpha=0.5, color=colors[i])
        ax.plot(speed_list, maxwell_scaled, color=colors[i], label=f"{int(speed)}m/s Fit")
    ax.set_title("Maxwell speed distribution with 49 0.5m radius balls", fontsize=12)
    ax.set_xlabel("Speed (m/s)", fontsize=12)
    ax.set_ylabel("Number", fontsize=12)
    ax.legend()
    return fig


def task16(data_points=20):
    """
    Task 16.

    In this function we shall also be looking at the divergence of our simulation from the IGL. We shall
    quantify the ball radii dependence of this divergence by plotting the temperature ratio
    and volume fraction defined in the project brief. We shall fit this temperature ratio before
    plotting the VDW b parameters radii dependence.

    Returns:
        tuple[Figure, Figure]: The ratio figure and b parameter figure.
    """
    # From 5 runs of task14
    mean, std = values_task14()

    mbs = MultiBallSimulation()
    b_radius_list = np.linspace(0.01, 1, data_points)
    V_c = mbs.container().volume() # Default container radius 10
    V_b = np.pi*b_radius_list**2
    N = len(mbs.balls()) # Default number of balls 37, 3 rings, multi=6
    volume_fraction = (V_c-2*N*V_b)/V_c

    fig1 = task14(data_points)
    ax1 = fig1.axes[0]
    lines = ax1.get_lines()
    t_ratio = lines[0].get_ydata()
    ax1.plot(b_radius_list, volume_fraction, label="Volume Fraction", color="blue")
    ax1.errorbar(b_radius_list, t_ratio, yerr=std, fmt=".", capsize=2, label="t_ratio Simulated Data", color="orange")

    popt, pcov = scipy.optimize.curve_fit(t_ratio_func, b_radius_list, t_ratio)
    t_ratio_fit = t_ratio_func(b_radius_list, popt[0])
    ax1.plot(b_radius_list, t_ratio_fit, label="Temperature Ratio Fit", color="red")
    ax1.set_title("Ratio of t_equipartition to t_ideal and Volume Fraction vs Radius of ball", fontsize=10)
    ax1.set_xlabel("Radius of the ball (m)", fontsize=12)
    ax1.set_ylabel("Ratio (temperature or volume)", fontsize=12)
    ax1.legend()

    wdw_b = V_c*6.02214076e23*(1-t_ratio_fit)/N
    wdw_b_err = V_c *6.02214076e23*np.array(std)/N
    fig2, ax2 = plt.subplots()
    ax2.errorbar(b_radius_list, wdw_b, yerr=wdw_b_err, fmt='.', capsize=2, label='WDW b Simulated Data')
    ax2.plot(b_radius_list, 2*6.02214076e23*np.pi*b_radius_list**2, label="Approximation b=2*N_A*V_ball")
    ax2.set_title("WDV parameter b vs Radius of ball", fontsize=12)
    ax2.set_xlabel("Radius of the ball (m)", fontsize=12)
    ax2.set_ylabel("WDV parameter b", fontsize=12)
    ax2.legend()
    return fig1, fig2


def task17(num_collisions=500):
    """
    Task 17.

    In this function we shall run a Brownian motion simulation and plot the resulting trajectory of the 'big' ball.
    """
    bs = BrownianSimulation(b_radius=0.1)
    # Removes balls within big ball
    balls = getattr(bs, "_MultiBallSimulation__balls") # Gets around mangled variables
    new_balls = [balls[0]]
    bb_radius = balls[0].radius()
    for i in range(1, len(balls)):
        if np.linalg.norm(balls[i].pos()) >= (balls[i].radius()+bb_radius):
            new_balls.append(balls[i])
    setattr(bs, "_MultiBallSimulation__balls", new_balls)

    bs.setup_figure()
    for i in range(num_collisions):
        bs.next_collision()
        plt.pause(0.01)
    pos = bs.bb_positions()
    x_values, y_values = zip(*pos)
    plt.plot(x_values, y_values, color="red", label="Path of the big ball")
    plt.legend()
    plt.show()
    

if __name__ == "__main__":

    # Run task 9 function
    # BALL_POS, BALL_VEL = task9()

    # Run task 10 function
    # task10()

    # Run task 11 function
    # FIG11_BALLCENTRE, FIG11_INTERBALL = task11()

    # Run task 12 function
    FIG12_KE, FIG12_MOMX, FIG12_MOMY, FIG12_PT = task12()

    # Run task 13 function
    # IG13_PT, FIG13_PV, FIG13_PN = task13()

    # Run task 14 function
    # FIG14 = task14()

    # Run task 15 function
    # FIG15 = task15()

    # Run task 16 function
    # FIG16_RATIO, FIG16_BPARAM = task16()

    # Run task 17 function
    # task17()

    plt.show()