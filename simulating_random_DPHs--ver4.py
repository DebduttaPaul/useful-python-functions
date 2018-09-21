"""

Author:			Debdutta Paul
Last updated:	11th June, 2018

"""


from __future__ import division
from astropy.io import ascii
from astropy.table import Table
import os
import time as t
import numpy as np
import debduttaS_functions as mf
import detecting_clusters__ver2 as dc
import matplotlib.pyplot as plt
plt.rc('axes', linewidth=2)
plt.rc('font', family='serif', serif='cm10')
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

start_time	=	t.time()

P = np.pi # Dear old pi!
padding	 = 8 # The padding of the axes labels.
size_font = 18 # The fontsize in the images.

################################################################################################################################################################################
#	General note: to test the cluster-detection algorithm parameters via simulations of random DPHs, including double-events.
#	N. B. Threshold is again fixed to its maximum allowed value, based on the results of the previos exercise.
################################################################################################################################################################################






################################################################################################################################################################################
#	To set the various CZTI specifics.

time_res	=	20 * 1e-6 # Temporal resolution of CZTI, in second.
L_x = 64; L_y = L_x # Dimensions of the CZTI quadrants in x & y directions.
Q_n	=	4	# Number of quadrants.
number_of_pixels_along_a_dimension_of_a_module	=	16
number_of_pixels_per_module		=	256
number_of_modules_per_quadrant	=	4**2
number_of_pixels_per_quadrant	=	number_of_pixels_per_module * number_of_modules_per_quadrant
total_number_of_pixels			=	number_of_pixels_per_quadrant * Q_n


################################################################################################################################################################################







################################################################################################################################################################################
#	To define the parameters to be used and tested.

t_bin		=	0.1	#	in sec

threshold	=	0.70
ID			=	3


total_time	=	90 * 60	#	in sec, roughly one orbit
mean_countrate_sglevents	=	90	# per sec, average countrate
mean_countrate_dblevents	=	60	# per sec, average countrate
path_save = os.getcwd() + '/plots/DPHclean/DPH_simulations/90,_60/zoom/'
#~ 
#~ total_time	=	20		#	in sec, roughly one bright GRB
#~ mean_countrate_sglevents	=	1500	# per sec, the brightest during the bright GRB
#~ mean_countrate_dblevents	=	600		# per sec, the brightest during the bright GRB
#~ path_save = os.getcwd() + '/plots/DPHclean/DPH_simulations/1500,_600/zoom/'

################################################################################################################################################################################





################################################################################################################################################################################
#	To define useful functions.


def choose_neighbouring_pixel( x1, y1 ):
	
	
	'''
	
	
	Parameters
	-----------
	x1:	
	y1:	
	
	Returns
	-----------
	x2:		
	y2:		
	
	
	'''
	
	
	
	##	First, the corner pixels, which have only 3 neighbouring pixels. Choosing one of them randomly...
	
	
	#	The bottom-left corner pixel
	if x1 == 15 and y1 == 0:
		choose	=	np.random.randint(3)
		if choose	==	0:	x2	=	15	;	y2	=	1
		elif choose	==	1:	x2	=	14	;	y2	=	1
		else:				x2	=	14	;	y2	=	0
	
	#	The bottom-right corner pixel
	elif x1 == 15 and y1 == 15:
		choose	=	np.random.randint(3)
		if choose	==	0:	x2	=	15	;	y2	=	14
		elif choose	==	1:	x2	=	14	;	y2	=	15
		else:				x2	=	14	;	y2	=	14
	
	#	The top-right corner pixel
	elif x1 == 0 and y1 == 15:
		choose	=	np.random.randint(3)
		if choose	==	0:	x2	=	1	;	y2	=	14
		elif choose	==	1:	x2	=	1	;	y2	=	15
		else:				x2	=	0	;	y2	=	14
	
	#	The top-left corner pixel
	elif x1 == 0 and y1 == 0:
		choose	=	np.random.randint(3)
		if choose	==	0:	x2	=	1	;	y2	=	0
		elif choose	==	1:	x2	=	1	;	y2	=	1
		else:				x2	=	0	;	y2	=	1
	
	
	
	
	##	Now, the pixels with boundary at one side, which have 5 neighbouring pixels...
	
	
	#	The bottom boundary pixels
	elif x1 == 15 and 0 < y1 < 15:
		
		pixels_to_choose	=	[ (15, y1-1), (14, y1-1), (14, y1), (14, y1+1), (15, y1+1) ]
		choose	=	np.random.randint( 5 )
		chosen_pixel	=	pixels_to_choose[ choose ]
		x2	=	chosen_pixel[0]	;	y2	=	chosen_pixel[1]
		
	#	The right boundary pixels
	elif 0 < x1 < 15 and y1 == 15:
		
		pixels_to_choose	=	[ (x1+1, 15), (x1+1, 14), (x1, 14), (x1-1, 14), (x1-1, 15) ]
		choose	=	np.random.randint( 5 )
		chosen_pixel	=	pixels_to_choose[ choose ]
		x2	=	chosen_pixel[0]	;	y2	=	chosen_pixel[1]
	
	#	The top boundary pixels
	elif x1 == 0 and 0 < y1 < 15:
		
		pixels_to_choose	=	[ (0, y1-1), (1, y1-1), (1, y1), (1, y1+1), (0, y1+1) ]
		choose	=	np.random.randint( 5 )
		chosen_pixel	=	pixels_to_choose[ choose ]
		x2	=	chosen_pixel[0]	;	y2	=	chosen_pixel[1]
	
	#	The left boundary pixels
	elif 0 < x1 < 15 and y1 == 0:
		
		pixels_to_choose	=	[ (x1+1, 0), (x1+1, 1), (x1, 1), (x1-1, 1), (x1-1, 0) ]
		choose	=	np.random.randint( 5 )
		chosen_pixel	=	pixels_to_choose[ choose ]
		x2	=	chosen_pixel[0]	;	y2	=	chosen_pixel[1]
	
	
	##	Now, any other pixel in the middle, which have 8 neighbouring pixels...
	else:
		pixels_to_choose	=	[ (x1+1, y1-1), (x1+1, y1), (x1+1, y1+1), (x1, y1+1), (x1-1, y1+1), (x1-1, y1), (x1-1, y1-1), (x1, y1-1) ]
		choose	=	np.random.randint( 8 )
		chosen_pixel	=	pixels_to_choose[ choose ]
		x2	=	chosen_pixel[0]	;	y2	=	chosen_pixel[1]
	
	
	return x2, y2



def place_module_in_quadrant( m, n, module_level_DPH ):
	
	
	'''
	
	
	Parameters
	-----------
	m				:	The first index of the module in the 4X4 module-space inside a quadrant.
	n				:	The corresponding second index.
	module_level_DPH:	The DPH of the module (resolved pixel-wise), which is to be placed at the above co-ordinates within the quadrant.
	
	Returns
	-----------
	Q_level_DPH:	The quadrant level DPH, with the module placed at the part given by its co-ordinates in the 4X4 module-space inside a quadrant.
	
	
	'''
	
	
	
	Q_level_DPH	=	np.zeros( (L_x, L_y) )
	
	number_of_modules_per_quadrant	=	16
	Q_level_DPH[ m * number_of_modules_per_quadrant	:	(m+1) * number_of_modules_per_quadrant,	n * number_of_modules_per_quadrant :	(n+1) * number_of_modules_per_quadrant ]	=	module_level_DPH
	
		
	return Q_level_DPH



def calculate_derivative( x, y ):
	
	
	'''
	
	
	Parameters
	-----------
	x:	1-D array, giving x co-ordiantes.
	y:	1-D array, giving corresponding y co-ordinates, y = f(x).
	
	Returns
	-----------
	val:	1-D array (lenght-wise one shorter than 'x' & 'y'), giving the points along the x-axis where the derivative is calculated.
	der:	1-D array, giving crude estimate of dy/dx, along 'val'.
	
	
	'''
	
	
	
	diff	=	y[1:] - y[:-1]
	delta	=	x[1:] - x[:-1]
	
	val		=	( x[1:] + x[:-1] ) / 2
	der		=	diff / delta
	
	
	return val, der



################################################################################################################################################################################







################################################################################################################################################################################
#	To do the simulation and make statistical estimates.

#	To calculate the numbers of events to be simulated.
mean_counts_tbin_sglevents			=	int( mean_countrate_sglevents * t_bin		)	#	counts in t_bin, for single events
mean_counts_tbin_dblevents_primary	=	int( mean_countrate_dblevents * t_bin / 2	)	#	counts in t_bin, for double events [CONFIRM WITH MITHUN]
how_many_DPHs	=	int( total_time / t_bin )	#	total number of DPHs need to be simulated


#~ allowable_array	=	np.arange( 0.5, 10.5, 0.5 )	#	Large
allowable_array	=	np.arange( 1.5,  3.5, 0.1 )	#	Zoom
flags_dict	=	{}
for allowable in allowable_array:
	flags_dict[ '{:.1f}'.format(allowable) ]	=	[]
#	To do the simulation of DPHs including single as well as double events at the known mean countrates, and store the outputs of the cluster detection algorithm on these DPHs.
for j in range( how_many_DPHs ):
	#	Looping over the total number of random DPHs to be simulated.
	
	
	
	
	
	##	Single events part...
	
	#	To initialize the DPH to be filled with random single events.
	DPH_sgl	=	np.zeros( (L_x, L_y) )
	
	#	To randomly select the number of single events to be placed, from a Poisson distribution with mean = mean_counts_tbin_sglevents.
	single_counts	=	np.random.poisson( mean_counts_tbin_sglevents )
	
	#	To randomly select pixels, 'single_counts' in number.
	detx_random	=	np.random.randint( 0, L_x, single_counts )
	dety_random	=	np.random.randint( 0, L_y, single_counts )
	
	#	To assign these pixels with one count.
	DPH_sgl[ detx_random, dety_random ]	=	1
	
	
	
	
	
	##	Double events part...
	
	#	To initialize the DPH to be filled by random double-events.
	DPH_dbl	=	np.zeros( (L_x, L_y) )
	
	#	To randomly select the number of double-event pairs to be placed, from a Poisson distribution with mean = mean_counts_tbin_dblevents_primary.
	doubleprimary_counts	=	np.random.poisson( mean_counts_tbin_dblevents_primary )
	
	#	To randomly select those detector modules that register double events.
	selected_detectormodules_x	=	np.random.randint( 0, 4, doubleprimary_counts )	#	here 4 is the number of modules along a dimension, in a quadrant, making 4**2 = 16 modules in the quadrant;
	selected_detectormodules_y	=	np.random.randint( 0, 4, doubleprimary_counts )	#	these numbers will be used to stitch the chosen modules and their DPHs into a quadrant level-DPH, which is defined below.
	
	#	To place the double events in blank modules, and then place them in the quadrant-level DPH and then to stitch them together.
	for k in range( doubleprimary_counts ):
				
		#	To initialize the DPH at the module-level.
		selected_module_DPH	=	np.zeros( (number_of_pixels_along_a_dimension_of_a_module, number_of_pixels_along_a_dimension_of_a_module) )
		
		#	To randomly choose the primary pixel.
		detx_primary_random	=	np.random.randint( 0, number_of_pixels_along_a_dimension_of_a_module )
		dety_primary_random	=	np.random.randint( 0, number_of_pixels_along_a_dimension_of_a_module )
		
		#	To randomly choose the secondary pixel, given the primary.
		detx_secondary_random, dety_secondary_random	=	choose_neighbouring_pixel( detx_primary_random, dety_primary_random )
		
		#	To define the DPH with the chosen pixels.
		selected_module_DPH[ detx_primary_random, dety_primary_random ]		= 1	#	setting the primary event
		selected_module_DPH[ detx_secondary_random, dety_secondary_random ]	= 1	#	setting the secondary event. The module is now ready to be placed inside the quadrant.
		
		#	To place the module in the full quadrant, ...
		Q_DPH	=	place_module_in_quadrant( selected_detectormodules_x[k], selected_detectormodules_y[k], selected_module_DPH )
		#	...and add it to the total DPH stitiching the other modules being placed one at a time...
		DPH_dbl	+=	Q_DPH
	
	
	
	##	Adding the single and double events together...
	DPH	=	DPH_sgl + DPH_dbl
	
	#	To check whether the DPH gets  flagged for the different values of the used parameter.
	for allowable in allowable_array: 
		#	To check for the chosen parameter whether the DPH shows clustering or not.
		flag, number_of_points, sum_of_measures, number_of_hot_pairs, cluster_image, bad_detx, bad_dety		=	dc.flag_the_dph( DPH, threshold, allowable, ID )
		flags_dict[ '{0:.1f}'.format(allowable) ].append( flag )


print round( ( t.time() - start_time ) / 60, 3 ), 'mins'

#	To calculate the percentage of DPHs getting flagged, for the parameter `Npoints'.
percentageflagged_array	=	[]
for allowable in allowable_array:
	flags_array	=	np.array( flags_dict[ '{0:.1f}'.format(allowable) ] )
	percentage	=	100 * ( len( np.where( flags_array == 1 )[0] ) / how_many_DPHs )
	percentageflagged_array.append( percentage )
percentageflagged_array	=	np.array( percentageflagged_array )
#	To calculate the gradient of this curve.
valx, ders	=	calculate_derivative( allowable_array, percentageflagged_array )

ascii.write( Table( [allowable_array, percentageflagged_array], names = ['allowable', 'flagged percentage'] ), path_save+'BKG_duration.txt', format = 'fixed_width' )
#~ ascii.write( Table( [allowable_array, percentageflagged_array], names = ['allowable', 'flagged percentage'] ), path_save+'GRB_duration.txt', format = 'fixed_width' )


#	To plot the percentage flagged, and its derivative, as a function of the cut-off.
plt.xlabel( r'$ M_{ \rm{sum} } / N_{ \rm{points} } $ ', fontsize = size_font )
plt.ylabel( r' $ \mathrm{ Flagged \;\; percentage } $', fontsize = size_font )
plt.plot( allowable_array, percentageflagged_array, linewidth = 2, color = 'k' )
plt.savefig( path_save + 'flagged_percentage.png' )
plt.savefig( path_save + 'flagged_percentage.pdf' )
plt.clf()
plt.close()

plt.xlabel( r'$ M_{ \rm{sum} } / N_{ \rm{points} } $ ', fontsize = size_font )
plt.ylabel( r' $ \mathrm{ Derivative \; of \; flagged \;\; percentage } $', fontsize = size_font )
plt.plot( valx, ders, linewidth = 2, color = 'k' )
plt.savefig( path_save + 'gradient_of_flagged_percentage.png' )
plt.savefig( path_save + 'gradient_of_flagged_percentage.pdf' )
plt.clf()
plt.close()


print round( ( t.time() - start_time ) / 60, 3 ), 'mins'
