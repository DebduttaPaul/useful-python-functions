from __future__ import division
from matplotlib import cm
from astropy.io import ascii
from astropy.table import Table
import numpy	as np
import debduttaS_functions as mf
import matplotlib.pyplot   as plt
plt.rc('axes', linewidth = 2)
plt.rc('font', family = 'serif', serif = 'cm10')
plt.rc('text', usetex = True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

P = np.pi # Dear old pi!
padding	 = 8 # The padding of the axes labels.
size_font = 18 # The fontsize in the images.

path_read = './data/DPHclean/random_DPHs_simulation/'
path_save = './plots/DPHclean/DPH_simulations/clubbed/'

BKG	=	ascii.read( path_read + 'BKG_duration.txt', format = 'fixed_width' )
GRB	=	ascii.read( path_read + 'GRB_duration.txt', format = 'fixed_width' )
BKG_x = BKG['allowable'] ; BKG_y = BKG['flagged percentage']
GRB_x = GRB['allowable'] ; GRB_y = GRB['flagged percentage']

plt.xlabel( r'$ M_{ \rm{sum} } / N_{ \rm{points} } $ ', fontsize = size_font )
plt.ylabel( r' $ \mathrm{ Flagged \;\; percentage } $', fontsize = size_font )
plt.plot( BKG_x, BKG_y, linestyle = '-' , color = 'k', lw = 2, label = r'$ \rm{ average \; counts } $' )
plt.plot( GRB_x, GRB_y, linestyle = '--', color = 'k', lw = 2, label = r'$ \rm{   high  \; counts } $' )
plt.legend()
plt.savefig( path_save + 'flagged_percentage--total.png' )
plt.savefig( path_save + 'flagged_percentage--total.pdf' )
plt.clf()
plt.close()
