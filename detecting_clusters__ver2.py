"""

Author:			Debdutta Paul
Last updated:	24th August, 2016

"""


from __future__ import division
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table
import warnings
import numpy as np
import debduttaS_functions as mf
import matplotlib.pyplot as plt
#~ plt.rc('axes', linewidth=2)
#~ plt.rc('font', family='serif', serif='cm10')
#~ plt.rc('text', usetex=True)
#~ plt.rcParams['text.latex.preamble'] = [r'\boldmath']

#	Ignoring silly warnings.
warnings.filterwarnings("ignore")


P = np.pi # Dear old pi!
padding	 = 8 # The padding of the axes labels.
size_font = 18 # The fontsize in the images.

################################################################################################################################################################################
#	General note: to identify structures in DPH.
################################################################################################################################################################################


#######################################################################################################################################################
#	To define the various CZTI specifications.


time_res = 20 * 1e-6 # Temporal resolution of CZT detectors, in second.

Q_n	=	4	# Number of Quadrants.
number_of_pixels_1D_per_Module	=	16
number_of_pixels_2D_per_Module	=	number_of_pixels_1D_per_Module ** 2
number_of_DMs_1D_per_Quadrant	=	4
number_of_DMs_2D_per_Quadrant	=	number_of_DMs_1D_per_Quadrant ** 2
number_of_pixels_1D_per_Quadrant=	number_of_pixels_1D_per_Module * number_of_DMs_1D_per_Quadrant
number_of_pixels_2D_per_Quadrant=	number_of_pixels_2D_per_Module * number_of_DMs_2D_per_Quadrant
total_number_of_pixels			=	number_of_pixels_2D_per_Quadrant * Q_n

L_x = number_of_pixels_1D_per_Quadrant ; L_y = L_x


#######################################################################################################################################################




def reverse_mapping( m, n ):
	
	
	'''
	
	
	Parameters
	-----------
	m:		Python first index for image.
	n:		Python second index for image.
	
	Returns
	-----------
	detx:	Corresponding detx.
	dety:	Corresponding dety.
	
	
	'''
	
	
	
	detx	=	n
	dety	=	(L_y-1) - m
	
	return detx, dety



def detecting_hotpairs( image, threshold ):	
	
	
	'''
	
	
	Parameters
	-----------
	image:		A 2-D numpy array giving the image.
	threshold:	The lower limit of measure, for potential 'hot pixels'.
	
	Returns
	-----------
	pairs:		A list of arrays, each array being the coordinates of the pair of neighbouring pixels with non-zero counts in the image.
	measures:	Similarly, a list of arrays, giving the measure of correlation between the above pairs.
	
	
	'''
	
	
	
	coords		=	np.where( image != 0 )
	amplitudes	=	image[ np.where( image != 0 ) ]
	n			=	amplitudes.size
	
	pairs		=	[]
	measures	=	[]
	for j in range( n ):
		#	Looping over the points with non-zero counts, of dimension 'n'.
		
		p1	=	[ coords[0][j], coords[1][j] ]
		amp_1	=	amplitudes[j]
		
		for k in range( j+1, n, 1 ):
			#	Such that the same pairs are not considered twice. This ensures nC2 pairs are selected.
			p2	=	[ coords[0][k], coords[1][k] ]
			D	=	mf.distance( p1, p2 )
			amp_2	=	amplitudes[k]
			
			measure	=	( amp_1 * amp_2 ) / D			
			if measure > threshold:		# if D < 2:
				##	This is the criterion for the pair to be considered a 'hot pair'.
				
				##	To check if the pair is an isolated double event, in which case it will not be considered.
				if D > np.sqrt(2) or ( amp_1 != 1 ) or ( amp_2 != 1 ):
					##	The first condition is to check if the events are in non-neighbouring pixels, while the rest are to check if either pixel registers more than one count,
					##	in which cases the pixels will be definetely considered 'hot'.
						pairs.append( [p1, p2] )
						measures.append( measure )
				
				else:
					##	In the case the pixels are neighbouring and both register only one count, the neighbourhood is considered.
					##	Only if there is at least one more pixel in the neighbourhood that registers at least one event, the pair is considered hot.
					
					#	To extract the image near the neighbourhood of the pair, the neighbourhood is defined.
					x_min	=	np.min( [ p1[0], p2[0] ] ) - 1
					x_max	=	np.max( [ p1[0], p2[0] ] ) + 1
					y_min	=	np.min( [ p1[1], p2[1] ] ) - 1
					y_max	=	np.max( [ p1[1], p2[1] ] ) + 1
					if x_min < 0:	x_min	=	0
					if y_min < 0:	y_min	=	0
					if x_max > L_x:	x_max	=	L_x
					if y_max > L_y:	y_max	=	L_y
					
					#	The neighbourhood is extracted.
					image_neighbourhood	=	image[ x_min : x_max+1, y_min : y_max+1 ]
					
					#	To check if there are any pixels in the neighbourhood registering non-zero events.
					sum_of_neighbourhood	=	np.sum( image_neighbourhood )
					if sum_of_neighbourhood > 2:
						
						#~ print image_neighbourhood, '\n'
						
						pairs.append( [p1, p2] )
						measures.append( measure )
	
	return pairs, measures



def flag_the_dph( image, threshold, allowable, ID ):
	
	
	'''
	
	
	Parameters
	-----------
	image:		A 2-D array giving the image.
	threshold:	The parameter 'threshold' that is used for calling the function `detecting_hotpairs'.
	allowable:	Minimum number of points in the dph that can be correlated randomly.
	ID:			What criterion is used to flag the data: Npoints if 0; Msum if 1; Npairs if 2.
	
	Returns
	-----------
	flag:					1 if structure or clump is detected in the image, 0 otherwise.
	number_of_points:		The number of non-identical points contributing to 'hot' pairs.
	sum_of_measures:		The sum of the 'hotness' in the DPH (i.e. the sum of measures of hotness for each detected pair).
	number_of_hot_pairs:	The total number of hot pairs that are detected.
	cluster_image:			A 2-D array giving the image, with only pixels selected as 'hot', highlighted.
	bad_detx:				A 1-D array containing the detx values of the pixels contributing to the structure.
	bad_dety:				A 1-D array containing the dety values of the pixels contributing to the structure.
	
	
	'''
	
	
	
	here		=	detecting_hotpairs( image, threshold )
	pairs		=	np.array( here[0] )
	measures	=	np.array( here[1] )
	
	number_of_hot_pairs	=	len( measures )	#	 Number of hot pairs that have been detected.
	sum_of_measures		=	np.sum( measures )
	
	cluster_image	=	np.zeros( ( L_x, L_y ) )
	
	bad_detx	=	np.array( [] )
	bad_dety	=	np.array( [] )
	
	for j in range( number_of_hot_pairs ):
		for k in range( 2 ):
			
			coords_x	=	pairs[j][k][0]
			coords_y	=	pairs[j][k][1]
			
			if cluster_image[ coords_x, coords_y ]	== 0:
				
				detx, dety	=	reverse_mapping( coords_x, coords_y )
				bad_detx	=	np.append( bad_detx, detx )
				bad_dety	=	np.append( bad_dety, dety )
				
				cluster_image[ coords_x, coords_y ]	=	image[ coords_x, coords_y ]
	
	flag	=	0
	number_of_points	=	( np.where( cluster_image != 0 )[0] ).size	#	Number of non-identical points contributing to hot pairs.
	
	if ID == 0:
		if number_of_points		> allowable:	flag	=	1
	if ID == 1:
		if sum_of_measures		> allowable:	flag	=	1
	if ID == 2:
		if number_of_hot_pairs	> allowable:	flag	=	1
	#~ if ID	==	3:
		#~ if ( number_of_points > 16 ) and ( sum_of_measures > 8 ) and ( number_of_hot_pairs > 8 ):	flag	=	1
	if ID	==	3:
		if sum_of_measures / number_of_points > allowable:	flag	=	1
	
	
	return flag, number_of_points, sum_of_measures, number_of_hot_pairs, cluster_image, bad_detx, bad_dety
