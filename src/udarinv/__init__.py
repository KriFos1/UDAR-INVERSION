"""Sequential 2D/3D look-around inversion of UDAR data.

At every logging point the maximum a posteriori model of ln Rh and the latent
ln(Rv/Rh) is found on a graded window around the bit, its Laplace posterior is
evaluated in closed form, and the cellwise mean and standard deviation in the
well plane are carried to the next point in closed form. Lengths are in feet
unless a name says otherwise; the solver interface converts to SI.
"""

__version__ = '0.1.0'
