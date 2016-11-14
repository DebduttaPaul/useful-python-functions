"""

This python script contains a list of python functions that are useful in data analysis.
Each function has explanations in the beginning to understand the basic functionality.

Some functions are used in carrying out general operations in others, hence all have been included here.
The ones that might be useful for working with CZTI data are:

01) rebin
02) my_range
03) mkdir_p
04) fourier_transform
05) my_histogram
06) create_dph
07) redefine_boundaries
08) Q2D
09) Q2any
10) D2Q
11)	Q2full
12) modid_pixid_to_detx_dety
13) renormalization
14) reduced_chisquared
15) create_lightcurves
16) create_lightcurves_with_livetime_correction
17) detect_peaks
18) GRB_band
19) GRB_powerlaw
20) subtract_continuum
21)	fit_a_poisson
22)	fit_a_gaussian
23) flag_pixels
24)	stats_over_array
25) create_full_DPH
26) veto_channel2energy
27) bin_secondary_along_primary_axis
28) my_histogram_according_to_given_integers

Instructions to use this script
--------------------------------
This script needs to be kept in the same directory in which the main python script is being executed, and then imported in the header, thus:
"import debduttaS_functions". Alternatively, a simpler name may be adopted during the import, e.g. "df" via
"import debduttaS_functions as df". Then all the functions can be used with the module name "df", e.g.
if the function "detect_peaks" is to be used, it can be called thus: "df.detect_peaks(x, y, x_search, significance)".



Author:			Debdutta Paul
Last updated:	13th September, 2016

"""


from __future__ import division
import os, warnings, time
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
from astropy.table import Table
from scipy.misc import factorial as fact
from scipy import interpolate
from scipy.optimize import curve_fit

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

#~ print '--------------------------------------------------------------------\n\n'
#~ print 'Number of Quadrants:			' ,			Q_n
#~ print 'Number of pixels 1D per Module:		' , number_of_pixels_1D_per_Module
#~ print 'Number of pixels 2D per Module:		' , number_of_pixels_2D_per_Module
#~ print 'Number of Modules 1D per Quadrant:	' , number_of_DMs_1D_per_Quadrant
#~ print 'Number of Modules 2D per Quadrant:	' , number_of_DMs_2D_per_Quadrant
#~ print 'Number of pixels 1D per Quadrant:	' , number_of_pixels_1D_per_Quadrant
#~ print 'Number of pixels 2D per Quadrant:	' , number_of_pixels_2D_per_Quadrant
#~ print 'Number of CZT pixels, 2D Total:		' , total_number_of_pixels
#~ print '\n\n--------------------------------------------------------------------\n\n\n\n\n\n\n\n\n'


################################################################################################################################################################################




def window( low, high, primary, secondary ):
	
	
	'''
	
	
	Parameters
	-----------
	low:		Lower limit.
	high:		Upper limit.
	primary:	Array on which the limits are being set.
	secondary:	Another array, which has same length as "primary" and its corresponding data.
				If this array is not required, an empty array can be supplied.
	
	Returns
	-----------
	chopped_primary:	The elements in the "primary" within and including the lower and upper limits.
	chopped_secondary:	The corresponding elements in the "secondary"; or empty, if an empty array was supplied.
	
	Comments
	-----------
	Assumes that "primary" is arranged in ascending order.
	
	
	'''
	
	
	
	lower = np.where( primary >= low  )[0][0]
	upper = np.where( primary <= high )[0][-1] + 1
	
	chopped_primary = primary[lower:upper]
	
	if len(secondary)==0:	chopped_secondary = []
	else:	chopped_secondary = secondary[lower:upper]
	
	
	return	chopped_primary, chopped_secondary
	





def delete( bigger, smaller ):
	
	
	'''
	
	
	Parameters
	-----------
	bigger:		An array of length greater than that of 'smaller'.
	smaller:	An array containing a set of elements which exactly equal a subset of 'bigger'.
	
	Returns
	-----------
	deleted:	The overlying region removed from "bigger", i.e. "bigger" - "smaller" .
	
	Assumptions
	-----------
	The data in the overlying region are exactly the same.
	
	
	'''
	
	
	index = []
	
	for i in range( len(smaller) ):
		index.append( np.where(bigger == smaller[i])[0][0] )
	
	deleted = np.delete(bigger, index)
	
	
	return deleted
	





def in_array( array, value, margin ):
	
	
	'''
	
	
	Parameters 
	-----------
	array:		An array in which "value" is to be searched for.
	value:		The float variable that is to be looked for in "array" .
	margin:		The accuracy to which the match has to be be looked for.
	
	Returns
	-----------
	B:			Boolean variable that is 1 if "value" was found in "array" and 0 if it was not.
	i:			The first index of "array" in which "value" was found if B = 1; garbage large number (10^20) if B = 0.
	
	
	'''
	
	
	
	B	=	0
	l	=	len(array)
	
	ind = []
	
	for j in range(l):
		
		if np.abs( array[j] - value ) < margin:
			B = 1
			ind.append(j)
	
	if len(ind) > 0:
		i = ind[0]
	else:
		i = 1e20
	
	
	return B, i
	





def nearest( array, value ):
	
	
	'''
	
	
	Parameters
	-----------
	array:	An array, which is to be searched over.
	value:	The value that is being searched in "array".
	
	Returns
	-----------
	index:	The index in "array" corresponding to which element is the closest to "value".
	
	
	Assumptions
	-----------
	"value" lies close to at least one element in "array".
	
	
	'''
	
	
	diff	=	np.abs( array - value )
	minimum	=	np.min( diff )
	index	=	np.where( diff == minimum )[0][0]
	
	return index
	





def std_via_mad( array ):
	
	
	'''
	
	
	Parameters
	-----------
	array:		Array of values of which STD (standard deviation) is to be calculated via MAD (median absolute deviation).
	
	Returns
	-----------
	std:		Robust standard deviation of "array".
	
	
	'''
	
	
	
	med		=	np.median(array)
	mad	=	np.median( np.abs(array - med) )
	std	=	1.4826 * mad
	
	return std






def rebin( array, intervals ):
		
	
	'''
	
	
	Parameters
	-----------
	array:		The array which needs to divided into equal "intervals".
	intervals:	The number of intervals required.
	
	Returns
	-----------
	limits:		Array of arrays, each containing upper and lower limits of the rebinned array.
	
	
	'''
	
	
	
	array_lo = np.min(array)
	array_hi = np.max(array)
	
	A = np.linspace(array_lo, array_hi, intervals + 1 )
	
	limits = []
	
	for i in range( 0, intervals ):
		limits.append(  np.array( [ A[i], A[i+1] ] )  )
	
	limits = np.array(limits)
	
	
	return limits
	





def my_range( array ):
	
	
	'''
	
	
	Parameters
	-----------
	array:		The array whose range is to be found.
	
	Returns
	-----------
	range_of_array:		The required range.
	
	
	'''
	
	
	
	range_of_array = np.max(array) - np.min(array)	
	
	return range_of_array





def mkdir_p( my_path ):
	
	
    '''
	
	
	Parameters
	-----------
	my_path:	The path in which the directory is to be made.
	
	Comments
    -----------
	Creates a directory (equivalent to using "mkdir -p" on the command line).
	
	
	'''
	
	
    from errno import EEXIST
    from os import makedirs, path
	
    try:
        makedirs(my_path)
    except OSError as exc: # Python > 2.5
        
        if exc.errno == EEXIST and path.isdir(my_path):
            pass
        else: raise
        return





def poisson( k, mean, coeff ):
	
	
	'''
	
	
	Parameters 
	----------
	k:				The real number which is to be mapped to a poisson distribution function (pdf).
	mean:			The lambda parameter (mean) of the pdf.
	coeff:			The coefficient of the pdf.
	
	Returns
	----------
	poiss_k:		The mapped real number.
	
	
	'''
	
	
	poiss_k = coeff * ( mean**k / fact(k) )
	
	return poiss_k




def gaussian( x, mu, sigma, coeff ):
	
	
	'''
	
	
	Parameters 
	----------
	x:			The real number which is to be mapped to a gaussian distribution (gd).
	mu:			The mean of the gd.
	sigma:		The standard deviation of the gd.
	coeff:		The coefficient of the gd.
	
	Returns
	----------
	gauss_k:	The mapped real number.
	
	
	'''
	
	
	gauss_x	=	coeff * np.exp( -0.5 * ( (x-mu)/sigma )**2 )
	
	return gauss_x





def fourier_transform( x, y ):
	
	
	'''
	
	
	Parameters
	-----------
	x:	Array consisting of data along the x-axis.
	y:	Array consisting of data along the y-axis.
	
	Returns
	-----------
	nu:	The reciprocal space of "x".
	c:	The corresponding fourier co-efficients (complex numbers).
		
	Caveats
	-----------
	For the case when "x" is non-uniformly gridded, the Fourier transform may not be accurate. A model dataset is created with uniform binning, the bin-size being the mean of the intervals. 
	
	
	
	'''
	
	
	step	= np.mean( x[1:] - x[:-1] )
	
	c_raw	= np.fft.fft(y)
	c		= np.fft.fftshift(c_raw)
	
	nu_raw	=	np.fft.fftfreq( len(x), d = step )
	nu		=	np.fft.fftshift( nu_raw )
	
	l = len(nu)/2 + 1
	
	nu		=	nu[l:]
	c		=	 c[l:]
	
	
	return nu, c





def mid_array( input_array ):
	
	
	'''
	
	
	Parameters
	-----------
	input_array:	The array for which the midpoint between consecutive elements will be delivered.
	
	Returns
	-----------
	output_array:	The required array.
	
	Comments
	-----------
	This function is to provide the array of midpoints for the cases when the np.histogram returns 2 arrays of unequal length
	The length of edges-array is 1 greater than the values-array, so to match the values with the edge this function is used.
	
	
	'''
	
	
	output_array = ( input_array[:-1] + input_array[1:] ) / 2
	
	
	return output_array





def distance( p1, p2 ):
	
	
	'''
	
	
	Parameters
	----------
	p1:	An array or list containing the x & y co-ordinates, respectively, of the first point on a plane.
	p2:	Similarly for the second point.
	
	Returns
	---------
	d: The distance between the two points.
	
	
	'''
	
	
	d = ( p1[0] - p2[0] )**2 + ( p1[1] - p2[1] )**2
	d = np.sqrt(d)
	
	return d






def radian_to_degree( angle_in_radian ):
	
	
	'''
	
	
	Parameters
	-----------
	angle_in_radian:	The real number to be converted into degree.	
	
	Returns
	-----------
	angle_in_degree:	The same in degree.
	
	
	'''
	
	
	import numpy as np
	P	=	np.pi
	
	angle_in_degree	=	angle_in_radian * 180/P
	
	return angle_in_degree







def degree_to_radian (angle_in_degree ):
	
	
	'''
	
	
	Parameters
	-----------
	angle_in_degree:	The angle, in degree, to be converted into a real number.
	
	Returns
	-----------
	angle_in_radian:	The required real number.
	
	
	'''
	
	
	import numpy as np
	P	=	np.pi
	
	angle_in_radian	=	angle_in_degree * P/180
	
	return angle_in_radian







def my_histogram( array, bin_size ):
	
	
	'''
	
	
	Parameters
	-----------
	array:		Array whose histogram is required.
	bin_size:	Bin size for creating the histogram, in same units as above array.
	
	Returns
	-----------
	x:			Array of rebinned elements.
	y:			Array of same length, giving the number of events in the new bins.
	
	
	'''
	
	
	array	=	np.sort( array )
	a_range	=	my_range( array )
	
	number_of_bins	=	int( a_range / bin_size )
	
	rem	=	a_range % bin_size
	
	if rem != 0:
		stop 		=	number_of_bins*bin_size + array[0]
		array, []	=	window( array[0], stop, array, [] )
		array[-1]	=	stop
	
	bin_edges	=	np.arange( array[0], array[-1]+bin_size, bin_size )
	x	=	mid_array( bin_edges )
				
	hist	=	np.histogram( array, bin_edges )
	
	y = hist[0]
	
	
	return x, y








def create_dph(t, x, y, h):
	
	
	'''
	
	
	Parameters
	-----------
	t:			Array of time instances over which image information is supplied.
	x:			Array of corresponding detector X-coordinates.
	y:			Array of corresponding detector Y-coordinates.
	h:			Temporal resolution for creating DPHs.
	
	Returns
	-----------
	  time_binned:		Time array, with binning according to "h".
	counts_binned:		Counts array, corresponding to each element in "time_binned".
	 image_binned:		Array of 2-D arrays, corresponding to each element in "time_binned", each 2-D array being a DPH of the full quadrant resolved at single pixels.
	
	Comments
	-----------
	Currently limited to the CZTI conventions. Anywhere CZTI-specific conventions are used, it is explicitly mentioned.
	
	
	'''
	
	
	T = my_range(t)
	
	N = int( T // h )
	
	if N!= 0:
		
		t_lo = np.min(t)
		t_hi = np.max(t)
			
		time_binned		=	np.zeros( N )
		
		#~ print 'time_binned.shape:	', time_binned.shape, '\n\n'
		
		time_binned[0] = t_lo + h/2
		
		counts_binned	=	[]
		image_binned	=	[]
		
		# To store the time intervals, counts and the images.
		for j in range( N ):
			
			if j != 0: time_binned[j]	=	time_binned[j-1] + h
			
			# To select the observation time for the boundaries in the binned time array.
			lt	=	time_binned[j] - h/2
			rt	=	time_binned[j] + h/2
			
			# To select the observations within these boundaries.
			t_chop, x_chop = window( lt, rt, t, x )
			t_chop, y_chop = window( lt, rt, t, y )
				
			# To extract the length of the observations. This corresponds to the counts in these intervals.
			counts = len(t_chop)
			counts_binned.append( counts )
			
			image_chop = np.zeros( ( L_x, L_y ) )
			
			for k in range( counts ):
				
				#	To account for the convention of python-indices and CZTI indices, x & y should be flipped.		
				image_chop[ (L_y-1)-y_chop[k] , x_chop[k] ] += 1
			
			image_binned.append(  image_chop  )
			
		counts_binned	=	np.array( counts_binned )
		image_binned	=	np.array(  image_binned )
	
	
	else:
		
		time_binned	=	np.array( [np.mean(t)] )
					
		# To extract the length of the observations. This corresponds to the counts in these intervals.
		counts	=	len(t)
		counts_binned	=	np.array( [counts] )
		
		image = np.zeros( ( L_x, L_y ) )
		
		for k in range( counts ):
			
			#	To account for the convention of python-indices and CZTI indices, x & y should be flipped.		
			image[ (L_y-1)-y[k] , x[k] ] += 1
		
		image_binned	=	np.array( [image] )
		
	
	
	return time_binned, counts_binned, image_binned





def redefine_boundaries( time_array, start_time, stop_time ):
	
	
	'''
	
	
	Parameters
	-----------
	time_array:		Array whose starting value needs to be checked.
	start_time:		The starting value required in modified time array.
	stop_time:		The last value required in modified array.
	
	Returns
	-----------
	time_array:		Array with redefined starting value, if the first element > "start_time" and stopping value, if last element < "stop_time".
		
	
	'''
	
	
	
	if time_array[0] > start_time:
		time_array[0] = start_time
		
	
	if time_array[-1] < stop_time:
		time_array[-1]	=	stop_time
		
	
	return time_array







def Q2D( Q ):
	
	
	'''
	
	
	Parameters
	-----------
	Q:		dph of a full quadrant, resolved at single pixels.
	
	Returns
	-----------
	D:		dph of the same quadrant, resolved at single detector-modules.
	
	
	'''
	
	
	L	=	np.shape( Q )[0]
	h	=	16
	
	d	=	int( L/h )
	D	=	np.zeros(  ( d, d )  )
		
	for i in range( d ):
		for j in range( d ):
			D[i, j]	=	np.sum( Q[ i*h : (i+1)*h, j*h : (j+1)*h ] )
	
	return D






def Q2any( Q, n ):
	
	
	'''
	
	
	Parameters
	-----------
	Q:		DPH of a full quadrant, resolved at single pixels.
	n:		number of squares into which a quadrant is wished to be sub-divided.
	
	Returns
	-----------
	D:		DPH of the same quadrant, divided into "n" parts, such that the counts from all pixels in that part have been summed,
			i.e. resolved at single such parts.
	
	
	'''
	
	
	L	=	np.shape( Q )[0]
	
	h	=	int( np.sqrt(n) )
	D	=	np.zeros(  ( h, h )  )
	
	b	=	int( L/h )
	
	for i in range( h ):
		for j in range( h ):
			D[i, j]	=	np.sum( Q[ i*b : (i+1)*b, j*b : (j+1)*b ] )
	
	return D






def D2Q( D ):
	
	
	'''
	
	
	Parameters
	-----------
	D:		dictionary of dphs of all quadrants, resolved at single detector-modules.
	
	Returns
	-----------
	Q:		dph of the whole detector plane, resolved at single detector-modules.
	
	
	'''
	
	
	Q	=	np.zeros( (2*Q_n, 2*Q_n) )
	
	Q[   0:  Q_n,   0:  Q_n ]	=	D[0]
	Q[   0:  Q_n, Q_n:2*Q_n ]	=	D[1]
	Q[ Q_n:2*Q_n, Q_n:2*Q_n ]	=	D[2]
	Q[ Q_n:2*Q_n,   0:  Q_n ]	=	D[3]
	
	return Q







def Q2full( Q_DPHs ):
	
	
	'''
	
	
	Parameters
	-----------
	Q_DPHs:		A dictionary containing DPHs of each quadrant (2D arrays), resolved at single pixels.
	
	Returns
	-----------
	DPH_full:	A 2-D aray giving the full DPH including all the quadrants.
	
	
	'''
	
	
	DPH_full	=	np.zeros( (2*L_y, 2*L_y) ) 
	
	DPH_full[   0:	L_y,   0:  L_x ]	=	Q_DPHs[0]
	DPH_full[   0:	L_y, L_x:2*L_x ]	=	Q_DPHs[1]
	DPH_full[ L_y:2*L_y, L_x:2*L_x ]	=	Q_DPHs[2]
	DPH_full[ L_y:2*L_y,   0:  L_x ]	=	Q_DPHs[3]
	
	return DPH_full







def modid_pixid_to_detx_dety( quadid, detid, pixid ):
	
	
	'''
	
	
	Parameters
	-----------
	quadid.
	detid.
	pixid.
	
	Returns
	-----------
	detx.
	dety.
	
	
	'''
	
	
	detx	=	(detid % 4) * 16 + (pixid % 16)
	dety	=	(detid //4) * 16 + (pixid //16)
	if ( quadid == 0 or quadid == 3 ):
		dety	=	63 - dety
	else:
		detx	=	63 - detx
	
	return detx, dety







def renormalization( badpix_filename ):
	
	
	'''
	
	
	Parameters
	-----------
	badpix_filename:	Name of file whose renormalization parameters are to be calculated at the module level.
	
	Returns
	-----------
	normalization:		A dictionary, running over Quadrant indices, each object being a 2-D array.
						Each such array is a 2-D array with the Detector Module IDs of the CZTI plane carefully mapped into python indices.
						A DPH made from the observed data, needs to be divided by this DPH,	to account for the missing data from the flagged pixels.
	
	Comments
	-----------
	If this array is flattened, then the order of its running indices is to reversed for Quadrants 1 and 2.
	
	Assumptions
	-----------
	The flags for the CZT pixels are:
	Good-0; Spectroscopically bad-1
	Flickering-2; Noisy-3; Dead/Inactive-4
	
	
	'''
	
	
	normalization	=	{}
	
	hdu	=	fits.open( badpix_filename )
	for i in range( Q_n ):	
		data	=	hdu[i+1].data
		detID	=	data['DETID']
		flags	=	data['PIX_FLAG']
		
		norm	=	np.zeros( number_of_DMs_2D_per_Quadrant )
		for j in range( number_of_DMs_2D_per_Quadrant ):
			
			index_cut	=	np.where( detID == j )[0]
			detID_cut	=	detID[index_cut]
			flags_cut	=	flags[index_cut]
			active_num	=	len( np.where( flags_cut == 0 )[0] )
			norm[j]		=	active_num
		norm = norm / number_of_pixels_2D_per_Module
		
		if i == 1 or i == 2:	norm = norm[::-1]
		norm	=	norm.reshape( number_of_DMs_1D_per_Quadrant, number_of_DMs_1D_per_Quadrant )
		normalization[i]	=	norm
	hdu.close()
	
	
	return normalization






def reduced_chisquared( theoretical, observed, obs_err, constraints ):
	
	
	'''
	
	Parameters
	-----------
	theoretical:		Array of theoretical quantity.
	observed:			Array of observed quantity.
	obs_err:			Array of errors on observed quantity.
	constraints:		The number of constraints on the fitting.
	
	Returns
	-----------
	chisqrd:			The chisqrd.
	dof:				The number of degrees of freedom.
	reduced_chisquared:	The reduced chi-squared.
	
		
	'''
	
		
	chisqrd	=	np.sum(   ( (observed-theoretical)/obs_err )**2   )
	dof		=	len(observed)-constraints
	reduced_chisquared	=	chisqrd / dof
		
	return chisqrd, dof, reduced_chisquared








def create_lightcurves( evt_filename, t_start, t_stop, t_bin, t_offset, E_start, E_stop ):
	
	
	'''
	
	
	Parameters
	-----------
	evt_filename	:	The name of the event file from which the data is to be imported to make the light-curve.
	t_start			:	The start time for making the light-curve.
	t_stop			:	The stop time for making the light-curve.
	t_offset		:	The time from which the offset is to be calculated.
	E_start			:	Start of the energy range whose ligh-curve is required.
	E_stop			:	End of the energy range whose ligh-curve is required.
	
	Returns
	-----------
	Q_rebinnedtime	:		A dictionary of rebinned event-stamps for all quadrants.
	Q_countrates	:		A dictionary of binned count-rates, for all quadrants.
	
	
	'''
	
	
	hdu	=	fits.open( evt_filename )

	Q_time	=	{}	;	Q_energy	=	{}

	start_times	=	np.zeros( Q_n );	stop_times	=	np.zeros( Q_n )


	for i in range( Q_n ):
			
		Q_time[i]	=	hdu[i+1].data['TIME']
		Q_energy[i]	=	hdu[i+1].data['ENERGY']
		
		#	To apply the energy-cut.
		indX_Energy	=	np.where(  ( E_start <= Q_energy[i] ) & ( Q_energy[i] <= E_stop )  )[0]
		Q_time[i]	=	Q_time[i][indX_Energy]
		
		#	To apply the time-cut.
		index_cut	=	np.where(    ( t_start < Q_time[i] ) & ( Q_time[i] < t_stop )    )[0]
		Q_time[i]	=	Q_time[i][index_cut]
		
		Q_time[i]	=	Q_time[i] - t_offset
		
		start_times[i]	=	Q_time[i][0]
		stop_times[i]	=	Q_time[i][-1]
	
	hdu.close()
	
	min_start	=	np.min( start_times )	;	max_stop	=	np.max( stop_times )
	
	#	Artificially changing the start and stop times of registering events for different quadrants to minimum start time and maximum stop time.
	for i in range( Q_n ):
		Q_time[i]	=	redefine_boundaries( Q_time[i], min_start, max_stop )
	
	Q_rebinnedtime	=	{}	;	Q_countrates	=	{}
	
	for i in range( Q_n ):
		
		Q_rebinnedtime[i], temp	=	my_histogram( Q_time[i], t_bin )
		Q_countrates[i]	=	temp/t_bin
	
	return Q_rebinnedtime, Q_countrates










def create_lightcurves_with_livetime_correction( evt_filename, lvt_filename, t_start, t_stop, t_bin, t_offset, E_start, E_stop ):
	
	
	'''
	
	
	Parameters
	-----------
	evt_filename	:	The name of the event file from which the data is to be imported to make the light-curve.
	lvt_filename	:	The name of the livetime file from which the livetime information is to be extracted.
	t_start			:	The start time for making the light-curve.
	t_stop			:	The stop time for making the light-curve.
	t_offset		:	The time from which the offset is to be calculated.
	E_start			:	Start of the energy range whose ligh-curve is required.
	E_stop			:	End of the energy range whose ligh-curve is required.
	
	Returns
	-----------
	Q_time_tbin			:		A dictionary of rebinned event-stamps for all quadrants.
	Q_countrate_tbin	:		A dictionary of binned count-rates, for all quadrants.
	
	Comments
	-----------
	This function is to be used only if the livetime correction is applicable.
	
	
	'''
	
	
	hdu	=	fits.open( evt_filename )
	lvt	=	fits.open( lvt_filename )

	Q_time	=	{}	;	Q_energy	=	{}
	Q_lvt	=	{}	;	Q_exp	=	{}

	for i in range( Q_n ):
			
		Q_time[i]	=	hdu[i+1].data['TIME']
		Q_energy[i]	=	hdu[i+1].data['ENERGY']
		
		Q_lvt[i]	=	lvt[i+1].data['TIME']
		Q_exp[i]	=	lvt[i+1].data['FRACEXP']
		
		#	To apply the energy-cut.
		indX_Energy	=	np.where(  ( E_start <= Q_energy[i] ) & ( Q_energy[i] <= E_stop )  )[0]
		Q_time[i]	=	Q_time[i][indX_Energy]
		
		#	To apply the time-cut.
		index_cut	=	np.where(    ( t_start < Q_time[i] ) & ( Q_time[i] < t_stop )    )[0]
		Q_time[i]	=	Q_time[i][index_cut]
		
		# To choose only those livetime bins where data is available after applying energy and time cuts on the data.
		index_cut	=	np.where( (Q_lvt[i] > Q_time[i][0]) & (Q_lvt[i] < Q_time[i][-1]) )[0]
		Q_lvt[i]	=	Q_lvt[i][index_cut]
		Q_exp[i]	=	Q_exp[i][index_cut]
		
		#	To take the corresponding datasets.
		start_at	=	Q_lvt[i][0]		-	1/2.0
		stop_at		=	Q_lvt[i][-1]	+	1/2.0	
		Q_time[i], []	=	window( start_at, stop_at, Q_time[i], [] )
				
		#	To artificially set the first and last data points so that the binning is according to the livetime bins.
		Q_time[i][0]	=	start_at
		Q_time[i][-1]	=	stop_at
		
		#	To change the offset of both the time-axes.
		Q_time[i]	=	Q_time[i]	-	t_offset
		Q_lvt[i]	=	Q_lvt[i]	-	t_offset
		
	hdu.close()
	lvt.close()
	
	
	Q_time_1sec	=	{}	;	Q_countrate_1sec	=	{}
	
	#	Binning at 1 sec intervals, following the livetime time-stamps.
	for i in range( Q_n ):
		
		Q_time_1sec[i], temp	=	my_histogram( Q_time[i], 1 )
		Q_countrate_1sec[i]	=	temp/t_bin
		
		#	To put the countrates of the bins with livetime 0, to 0, by hand.
		mask	=	np.where( Q_exp[i] == 0 )[0]
		non_mask	=	np.where( Q_exp[i] != 0 )[0]
		Q_countrate_1sec[i][mask]	=	0
		
		#	To apply the livetime correction to the countrates corresponding to non-zero livetime.
		Q_countrate_1sec[i][non_mask] =	Q_countrate_1sec[i][non_mask] / Q_exp[i][non_mask]
	
	#	To add up the many 1 sec bins that make up the chosen t_bin.
	N	=	int(t_bin)
	
	Q_time_tbin	=	{}	;	Q_countrate_tbin	=	{}
	
	if N == 1:
		Q_time_tbin	=	Q_time_1sec
		Q_countrate_tbin	=	Q_countrate_1sec
		
	else:
		
		for i in range( Q_n ):
			
			rem	=	len(Q_time_1sec[i]) % N
			
			#	To remove the last bin with incomplete data for the required binning.
			if rem != 0:
				Q_time_tbin[i]			=	Q_time_1sec[i][:-rem]
				Q_countrate_tbin[i]	=	Q_countrate_1sec[i][:-rem]
								
			else:
				Q_time_tbin[i]			=	Q_time_1sec[i]
				Q_countrate_tbin[i]	=	Q_countrate_1sec[i]
							
			L	=	int( my_range(Q_time_tbin[i]) // N ) + 1
						
			Q_time_tbin[i]			=	Q_time_tbin[i].reshape( L, N )
			Q_countrate_tbin[i]	=	Q_countrate_tbin[i].reshape( L, N )
			
			Q_time_tbin[i]			=	Q_time_tbin[i].mean( axis = 1 )
			Q_countrate_tbin[i]	=	Q_countrate_tbin[i].sum( axis = 1 )
	
	#	To remove any data that has 0 countrate, i.e. due to livetime supression.
	for i in range( Q_n ):
		mask	=	np.where( Q_countrate_tbin[i]	== 0 )[0]
		Q_time_tbin[i]	=	np.delete( Q_time_tbin[i], mask )
		Q_countrate_tbin[i]	=	np.delete( Q_countrate_tbin[i], mask )
	
				
	return Q_time_tbin, Q_countrate_tbin










def detect_peaks( x, y, x_search, significance ):
	
	
	'''
	
	
	Parameters
	-----------
	x			:	x-axis data-points.
	y			:	Corresponding elements in the quantity in which peaks are to be searched for.
	x_search	:	The length of data in the x-axis over which the continuum is known to be stable, in same units as x.
	significance:	The significance level to be used for peak-detection.
	
	Returns
	-----------
	x_peaks:				The elements in "x" where significant peaks were detected in "y".
	y_peaks:				The corresponding elements in "y", i.e. the 'peaks.'
							For demonstration, plot "y" v/s "x" and "y_peaks" v/s "x_peaks" together.
	significance_of_peaks:	The deviation from the continuum of the detected peaks.		
	
	Assumptions
	-----------
	Lengths of "x" and "y" are equal.
	
	
	'''
	
	
	L		=	len(x)
	
	# To choose a region around each data point.
	x_bin	=	np.mean( x[1:] - x[:-1] )
	
	points	=	int( x_search / x_bin )
	
	median=[]
	for i in range(L):
		left	=	i - points
		right	=	i + points
		if left < 0:		left = 0
		if right> (L-1):	right= (L-1)
		
		selected_x, selected_y = window( x[left], x[right], x, y )
	
		# To calculate the median of the selected region: this gives the overall continuum for that point.		
		md = np.median(selected_y) 
		median.append(md)
	
	# To calculate the global noise, via the MAD (Median Absolute Deviation).
	cont_sub_y = y - median
	med	=	np.median(cont_sub_y)
	#~ std	=	np.median( np.abs(cont_sub_y - med) )
	#~ std	=	1.4826 * std
	std	=	np.std(cont_sub_y)
	
	# To select the regions with huge peaks in the data.
	r	=	np.where( cont_sub_y > ( significance*std + med ) )[0]
	x_around_peaks	=	x[r]
	y_around_peaks	=	y[r]
	significance_of_peaks	=	cont_sub_y[r]/std
	
	return x_around_peaks, y_around_peaks, significance_of_peaks










def GRB_band( E, alpha, beta, E_c, K ):
	
	
	'''
	
	
	Parameters
	-----------
	E:		Array containing values of Energy, in keV.
	alpha:	1st power-law index.
	beta:	2nd power-law index.
	E_c:	Characteristic energy, in keV.
	K:		Overall normalization.
	
	Returns
	----------
	spec:	Normalized band-spectrum over the full array.
	
	
	'''
	
	
	E_cutoff	=	(alpha - beta)*E_c
	E_lower 	=	E[ E <= E_cutoff ]
	spec_lower	=	K * ( (E_lower/100)**alpha ) * np.exp(- E_lower/E_c)
	E_upper		=	E[ E >  E_cutoff ]
	spec_upper	=	K * ( ((alpha-beta)*(E_c/100))**(alpha-beta) ) * ( (E_upper/100)**beta ) * np.exp( -(alpha-beta) )
	spec		=	np.append( spec_lower, spec_upper )	
	
	return spec






def GRB_powerlaw( E, Gamma, K ):
	
	
	'''
	
	
	Parameters
	-----------
	E:		Energy, in keV.
	Gamma:	Photon index.
	K:		overall normalization.
	
	Returns
	----------
	Normalized powerlaw-spectrum.
	
	
	'''
	
	
	spec	=	K * E**(-Gamma)
	
	
	return spec






def sort( primary, secondary ):
	
	
	'''
	
	
	Parameters
	-----------
	primary:	The array which is to be sorted.
	secondary:	Another array which is to be sorted according to the elements of 'primary'.
	
	Returns
	----------
	p:		Sorted primary.
	s:		Correspondingly sorted secondary.
	
	
	'''
	
	
	index	=	np.argsort( primary )
	p	=	primary[index]
	s	=	secondary[index]
	
	
	return p, s





def subtract_continuum( x, y, x_search ):
	
	
	'''
	
	
	Parameters
	-----------
	x			:	x-axis data-points.
	y			:	Corresponding elements in the quantity whose continuum is to be removed.
	x_search	:	The length of data in the x-axis over which the continuum is known to be stable, in same units as x.
	
	Returns
	-----------
	continuum:		The continuum.
	contsub_y:		The continuum-subtracted y-elements.
	
	Assumptions
	-----------
	Lengths of "x" and "y" are equal.
	
	
	'''
	
	
	L		=	len(x)
	
	# To choose a region around each data point.
	x_bin	=	np.mean( x[1:] - x[:-1] )
	
	points	=	int( x_search / x_bin )
	
	median	=	[]
	for i in range(L):
		left	=	i - points
		right	=	i + points
		if left < 0:		left = 0
		if right> (L-1):	right= (L-1)
		
		selected_x, selected_y = window( x[left], x[right], x, y )
	
		# To calculate the median of the selected region: this gives the overall continuum for that point.		
		md = np.median(selected_y) 
		median.append(md)
	median	=	np.array(median)
	
	continuum	=	median
	contsub_y	=	y - median
	
	return continuum, contsub_y





def fit_a_poisson( x, y ):
	
	
	'''
	
	
	Parameters
	-----------
	x:	The array containing the independent discrete variable.
	y:	The dependent variable which is to be attempted to fit to a Poisson distribution function.
	
	Returns
	-----------
	mean_best:	Best-fit mean.
	coeff_best:	Best-fit co-efficient.
	
	Comments
	-----------
	Assumes that the fit will not fail drastically!
	
	
	'''
	
	
	masks		=	np.where( y > 0 )[0]
	x_masked	=	x[masks]
	y_masked	=	y[masks]
	
	mean0		=	np.sum( y_masked * x_masked ) / np.sum(y_masked)
	coeff0		=	np.exp( -mean0 ) * np.sum(y_masked)
	
	popt, pcov	=	curve_fit( poisson, x_masked, y_masked, p0 = [mean0, coeff0] )
	mean_best	=	popt[0]
	coeff_best	=	popt[1]
	
	
	return mean_best, coeff_best





def fit_a_gaussian( x, y ):
	
	
	'''
	
	
	Parameters
	-----------
	x:	The array containing the independent discrete variable.
	y:	The dependent variable which is to be attempted to fit to a Poisson distribution function.
	
	Returns
	-----------
	mean_best:	Best-fit mean.
	sigma_best:	Best-fit sigma.
	coeff_best:	Best-fit co-efficient.
	
	Comments
	-----------
	Assumes that the fit will not fail drastically!
	
	
	'''
	
	
	masks		=	np.where( y > 0 )[0]
	x_masked	=	x[masks]
	y_masked	=	y[masks]
	
	coeff0		=	np.max(y_masked)
	mean0		=	x_masked[ np.where(y_masked==coeff0)[0][0] ]
	sigma0		=	np.sqrt(  np.sum( y_masked * (x_masked-mean0)**2 ) / np.sum(y_masked)  )
	
	popt, pcov	=	curve_fit( gaussian, x_masked, y_masked, p0 = [mean0, sigma0, coeff0] )
	mean_best	=	popt[0]
	sigma_best	=	np.abs( popt[1] )
	coeff_best	=	popt[2]
	
	
	return mean_best, sigma_best, coeff_best





def flag_pixels( dph, bad_x, bad_y ):
	
	
	'''
	
	
	Parameters
	-----------
	dph:		The 2_D array, representing the DPH that needs to be flagged.
	bad_x:		1-D array containing the detx of the bad-pixels.
	bad_y:		1-D array containing the dety of the corresponding pixels.
	
	Returns
	-----------
	flagged_dph:	The dph with the bad pixels flagged.
	
	
	'''
	
	
	for j in range( len(bad_x) ):
		
		x	=	bad_x[j]
		y	=	bad_y[j]
		
		flagged_dph	=	dph
		flagged_dph[ (L_y-1)-y , x ]	=	0
	
	
	return flagged_dph





def stats_over_array( array, N ):
	
	
	'''
	
	
	Parameters
	-----------
	array:	Array which is to be binned every "N" data points.
	N:		Number of points.
	
	Returns
	-----------
	array_sum:	Over the array binned every "N" data points, sum.
	array_mean:	Over the array binned every "N" data points, mean.
	array_std:	Over the array binned every "N" data points, standard deviation.
	array_var:	Over the array binned every "N" data points, varaiance.
	
	Comments
	-----------
	After rebinning, throws away data in the last bin, if it is incomplete.
	
	
	'''
	
	
	lx	=	len( array )
	rem	=	lx % N
	
	#	To remove the last bin with incomplete data for the required binning.
	if rem != 0:
		array	=	array[:-rem]
	
	L	=	int( lx // N )
	
	array_2D	=	array.reshape( L, N )
	array_sum	=	array_2D.sum( axis = 1 )
	array_mean	=	array_2D.mean( axis = 1 )
	array_std	=	array_2D.std( axis = 1 )
	array_var	=	array_2D.var( axis = 1 )
		
	return array_sum, array_mean, array_std, array_var




def create_full_DPH( detx, dety ):
	
	
	'''
	
	
	Parameters
	-----------
	detx:		Array of detx.
	dety:		Array of dety.
	
	Returns
	-----------
	image:		2-D array, containing the required python-indices mapped image.
	
	
	'''
	
	
	L_x	=	64	;	L_y	=	L_x	#	CZTI specifics.
	DPH	=	np.zeros( ( L_x, L_y ) )
	
	for k, x in enumerate( detx ):
		
		#	To account for the convention of python-indices and CZTI indices, x & y should be flipped.		
		DPH[ (L_y-1)-dety[k] , x ]	+=	1
	
	DPH	=	np.array( [DPH] )[0]
	
	
	return DPH




def veto_channel2energy( channel, i ):
	
	
	'''
	
	
	Parameters
	-----------
	channel:	1-D numpy array conaining a set of valuescontaining to PHA detected in the Veto detector.
	i:			Quadrant ID of the veto detector. This is required because the calibration parameters (gain and offset) are different for the different Qs.

	Returns
	-----------
	energy:		Corresponding energies.
	
	Comments
	-----------
	For using this on the CZT-tagged events, the channels are to be taken from the individual Quadrant files. To reduce the amount of data,
	the 8 bit veto channel numbers (0-255) have been contracted to 7 bits (0-127) in these files. Thus, one should do:
	channel	=	2 * channel
	before calling this function.
	
	
	'''
	
	
	if   i == 0:	energy = 5.591 * channel - 56.741
	elif i == 1:	energy = 5.594 * channel - 41.289
	elif i == 2:	energy = 5.943 * channel - 41.682
	elif i == 3:	energy = 5.222 * channel - 26.528
	
	
	return energy




def bin_secondary_along_primary_axis( x, y, x_start, x_stop, x_bin ):
	
	
	'''
	
	
	Parameters
	-----------
	x:			Data along primary axis, with respect to data along secondary axis is to be binned.
	y:			Data along secondary (to be binned).
	x_start:	Left  boundary of the binned primary.
	x_stop:		Right boundary of the binned primary.
	x_bin:		The bin along the primary.
	
	Returns
	-----------
	x_mid:		The middle bins of the rebinned primary axis.
	y_sum:		The secondary axis binned according to the bins along the primary axis.
	
	'''
	
	
	x_edges	=	np.arange( x_start, x_stop + x_bin, x_bin )
	
	y_sum	=	np.zeros( len(x_edges) - 1 )
	for j, l in enumerate( x_edges[:-1] ):
		
		r		=	x_edges[j+1] - 1e-3
		a, b	=	window( l, r, x, y )
		y_sum[j]=	np.sum(  b )
	
	x_mid	=	mid_array( x_edges )
	
	
	return x_mid, y_sum



def my_histogram_according_to_given_integers( array ):
	
	
	'''
	
	Parameters
	-----------
	array:		Array, which contains only integer values, which is to be histogram-ed according to these histograms.
	
	Returns
	-----------
	hist:		The number of elements along the unique elements of sorted "array".
	
	
	'''
	
	
	array	=	np.sort(array)
	hist	=	np.array( [] )
	for j, value in enumerate( np.unique(array) ):
		
		number	=	len( np.where( array == value )[0] )
		hist	=	np.append( hist, number )
	
	
	return hist
