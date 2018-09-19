"""

This python script takes as input L2 CZT data and corresponding L2 bunch data.
It operates four cleaning steps on the data:
1) bunchclean: which has been modified from before;
2) gross noisy pixel removal: same as before;
3) DPHclean: new step;
4) flickpixclean: similar to the 2nd step of pixclean, but with modifications.

It can be safely run on all data including GRBs, and the output is data cleaned upto the maximum level possible,
along with demonstrative lightcurves with livetime corrections first calculated from the GTIs and then updated after every cleaning step.


Requirements
--------------------------------
Python 2.7 or higher
numpy
astropy
matplotlib
"debduttaS_functions.py"		in the same directory as this script.
"detecting_clusters__ver2.py"	in the same directory as this script.


Instructions to use this script
--------------------------------
First save the path to the data files as a string in the variable "path_read", e.g.
" path_read = os.getcwd() + '/data/' "
goes to the "data" directory under the current directory.

Then set the names of the event file and the corresponding bunch file in the string variables "event_filename" and "bunch_filename", e.g.
event_filename	=	path_read + 'AS1G05_233T10_9000000570_04576cztM0_level2.fits'
bunch_filename	=	path_read + 'AS1G05_233T10_9000000570_04576cztM0_level2_bunch.fits'
which uses the directory structure as set before.

Then define the path where the plots will be saved, e.g.
" path_save		=	os.getcwd() + '/plots/final_pipeline/my_method/GRB160802A/' "

Define the variables
"t_start"	:	The time [UT, sec] you want to start the analysis.
"t_stop"	:	The time [UT, sec] you want to stop  the analysis.
"t_offset"	:	The offset [UT, sec] given to the plotted lightcurves.

You can also choose the values of other parameters as described in the document.

Finally, change directory to the one where the script is located, and run it:
" python final_pipeline--with_livetimes--ver3.py "


Comments about runtime
--------------------------------
The first step in the cleaning takes the maximum time because it runs a for-loop over all the bunches and cannot be vectorized.
It takes around 10 minutes per quadrant per orbit, but can vary depending on the version of the python modules installed in the user's machine,
and the level of optimization of these modules.



Author:			Debdutta Paul
Last updated:	06th November, 2016

"""


from __future__ import division
from matplotlib import cm
from astropy.io import fits, ascii
from astropy.table import Table
from astropy.stats import sigma_clip
import os
import numpy	as np
import numpy.ma as ma
import debduttaS_functions as mf
import detecting_clusters__ver2 as dc
import matplotlib.pyplot   as plt
plt.rc('axes', linewidth = 2)
plt.rc('font', family = 'serif', serif = 'cm10')
plt.rc('text', usetex = True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

P = np.pi # Dear old pi!
padding	 = 8 # The padding of the axes labels.
size_font = 18 # The fontsize in the images.





#######################################################################################################################################################
#	To define parameters for representation and cleaning procedures.
t_bin			=	1e0		#	in sec, the time-scale at which to finally bin the data for representation
t2				=	60*1e-6 #	in sec, to club consecutive bunches if the interval between them is less than this amount of time
t3				=	60*1e-6 #	in sec, to flag data post-bunch for this time-scale
gross_cutoff	=	5		#	"n" for n-sigma outlier flagging at the DPH level, i.e. gross noisy pixels
t_look			=	1e-1	#	in sec, the counts are binnned at this time-scale to search for clusters in the DPH
T_search		=	5e2		#	in sec, the de-trending time-scale
cutoff			=	2		#	"n" for n-sigma outlier search in the de-trended light-curve, whose DPHs are probed for structures
threshold		=	0.7		#	the most conservative case
allowable		=	3		#	if the quantity used for flagging is greater than this, the DPH is flagged to have a structure
ID				=	3		#	combo
flick_tbin		=	1e-1	#	in sec, time-scale for finding flickering pixels
flick_threshold	=	2		#	if counts from a pixel in "flick_tbin" is greater than this, that pixel is flagged for the entire time


#	To define parameters related to this source and file structures.

path_read			=	os.getcwd() + '/data/'
#~ badpix_filename		=	path_read + 'spect_bad_gain_flag.txt'




#~ bunch_filename	=	path_read + 'AS1G05_211T01_9000000464_03497cztM0_level2_bunch.fits'		#	GRB160521B
#~ event_filename	=	path_read + 'AS1G05_211T01_9000000464_03497cztM0_level2.fits'			#	GRB160521B
#~ path_save		=	os.getcwd() + '/plots/final_pipeline/my_method/GRB160521B/'
#~ GRB_trigger		=	201518037	#	UT, in sec:	GRB160521B
#~ T1	=	GRB_trigger - 0			#	UT, in sec:	GRB160521B
#~ T2	=	GRB_trigger + 9.984		#	UT, in sec:	GRB160521B
#~ 
#~ bunch_filename	=	path_read + 'AS1G05_045T01_9000000468_03558cztM0_level2_bunch.fits'		#	GRB160525C
#~ event_filename	=	path_read + 'AS1G05_045T01_9000000468_03558cztM0_level2.fits'			#	GRB160525C
#~ path_save		=	os.getcwd() + '/plots/final_pipeline/my_method/GRB160525C/'
#~ GRB_trigger		=	201869670	#	UT, in sec:	GRB160525C
#~ T1	=	GRB_trigger - 10	#	UT, in sec:	GRB160525C
#~ T2	=	GRB_trigger + 10	#	UT, in sec:	GRB160525C
#~ 
#~ bunch_filename	=	path_read + 'AS1G05_234T02_9000000474_03632cztM0_level2_bunch.fits'		#	GRB160530A
#~ event_filename	=	path_read + 'AS1G05_234T02_9000000474_03632cztM0_level2.fits'			#	GRB160530A
#~ path_save		=	os.getcwd() + '/plots/final_pipeline/my_method/GRB160530A/'
#~ GRB_trigger		=	202287826	#	UT, in sec:	GRB160530A
#~ T1	=	GRB_trigger - 0			#	UT, in sec:	GRB160530A
#~ T2	=	GRB_trigger + 38.912	#	UT, in sec:	GRB160530A
#~ 
#~ bunch_filename	=	path_read + 'AS1G05_247T01_9000000512_03983cztM0_level2_bunch.fits'		#	GRB160623A
#~ event_filename	=	path_read + 'AS1G05_247T01_9000000512_03983cztM0_level2.fits'			#	GRB160623A
#~ path_save		=	os.getcwd() + '/plots/final_pipeline/my_method/GRB160623A/'
#~ GRB_trigger		=	204353977	#	UT, in sec:	GRB160623A
#~ T1	=	GRB_trigger - 0			#	UT, in sec:	GRB160623A
#~ T2	=	GRB_trigger + 90.46		#	UT, in sec:	GRB160623A
#~ 
#~ bunch_filename	=	path_read + 'AS1G05_245T01_9000000532_04135cztM0_level2_bunch.fits'		#	GRB160703A
#~ event_filename	=	path_read + 'AS1G05_245T01_9000000532_04135cztM0_level2.fits'			#	GRB160703A
#~ path_save		=	os.getcwd() + '/plots/final_pipeline/my_method/GRB160703A/'
#~ GRB_trigger		=	205243805	#	UT, in sec:	GRB160703A
#~ T1	=	GRB_trigger - 26.43		#	UT, in sec:	GRB160703A
#~ T2	=	GRB_trigger + 52.57		#	UT, in sec:	GRB160703A
#~ 
#~ bunch_filename	=	path_read + 'AS1G05_230T02_9000000548_04391cztM0_level2_bunch.fits'		#	GRB160720A
#~ event_filename	=	path_read + 'AS1G05_230T02_9000000548_04391cztM0_level2.fits'			#	GRB160720A
#~ path_save		=	os.getcwd() + '/plots/final_pipeline/my_method/GRB160720A/'
#~ GRB_trigger		=	206735037	#	UT, in sec:	GRB160720A
#~ T1	=	GRB_trigger - 1.02		#	UT, in sec:	GRB160720A
#~ T2	=	GRB_trigger + 237		#	UT, in sec:	GRB160720A
#~ 
#~ bunch_filename	=	path_read + 'AS1G05_009T02_9000000618_04866cztM0_level2_bunch.fits'		#	GRB160821A
#~ event_filename	=	path_read + 'AS1G05_009T02_9000000618_04866cztM0_level2.fits'			#	GRB160821A
#~ path_save		=	os.getcwd() + '/plots/final_pipeline/my_method/GRB160821A/'
#~ GRB_trigger		=	209507670	#	UT, in sec:	GRB160821A
#~ T1	=	GRB_trigger - 4.1		#	UT, in sec:	GRB160821A
#~ T2	=	GRB_trigger + 208.328	#	UT, in sec:	GRB160821A
#~ 
#~ t_start	=	T1 - 200
#~ t_stop	=	T2 + 200
#~ t_offset=	GRB_trigger








event_filename	=	path_read + 'AS1G05_233T10_9000000570_04576cztM0_level2.fits'			#	GRB160802A
bunch_filename	=	path_read + 'AS1G05_233T10_9000000570_04576cztM0_level2_bunch.fits'		#	GRB160802A
path_save		=	os.getcwd() + '/plots/final_pipeline/my_method/GRB160802A/ver3/'
GRB_trigger		=	207814409	#	UT, in sec:	GRB160802A
T1	=	GRB_trigger - 0.3		#	UT, in sec:	GRB160802A
T2	=	GRB_trigger + 20.224	#	UT, in sec:	GRB160802A

t_start	=	T1 - 1e3			#	UT, in sec: the whole analysis starts from here
t_stop	=	T2 + 0.98*1e3		#	UT, in sec: the whole analysis stops here
#	t_start	=	T1 - 1e2		#	UT, in sec: the whole analysis starts from here
#	t_stop	=	T2 + 0.8e2		#	UT, in sec: the whole analysis stops here
t_offset=	GRB_trigger			#	UT, in sec: this is the offset of the lightcurves plotted


#######################################################################################################################################################





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


print '--------------------------------------------------------------------\n\n'
print 'Number of Quadrants:			' ,			Q_n
print 'Number of pixels 1D per Module:		' , number_of_pixels_1D_per_Module
print 'Number of pixels 2D per Module:		' , number_of_pixels_2D_per_Module
print 'Number of Modules 1D per Quadrant:	' , number_of_DMs_1D_per_Quadrant
print 'Number of Modules 2D per Quadrant:	' , number_of_DMs_2D_per_Quadrant
print 'Number of pixels 1D per Quadrant:	' , number_of_pixels_1D_per_Quadrant
print 'Number of pixels 2D per Quadrant:	' , number_of_pixels_2D_per_Quadrant
print 'Number of CZT pixels, 2D Total:		' , total_number_of_pixels
print '\n\n--------------------------------------------------------------------\n\n\n\n\n\n\n\n\n'


#######################################################################################################################################################







#######################################################################################################################################################

print '--------------------------------------------------------------------\n\n'
print 't_bin:			{:.1f} sec'.format(t_bin)
print 't2:			{:.1f} microsec'.format(t2*1e6)
print 't3:			{:.1f} microsec'.format(t3*1e6)
print 'gross_cutoff:		{:d}'.format(gross_cutoff)
print 't_look:			{:.1f} sec'.format(t_look)
print 'T_search:		{:.0f} sec'.format(T_search)
print 'cutoff:			', cutoff
print 'threshold:		', threshold
print 'allowable:		', allowable
print 'ID:			', ID
print 'flick_tbin:		{:.1f} sec'.format(flick_tbin)
print 'flick_threshold:	', flick_threshold
print '\n\n--------------------------------------------------------------------\n\n\n\n\n\n\n\n\n'


#######################################################################################################################################################






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


def get_bunchdata( bunch_starts, bunch_2nds, bunch_ends, bunch_lengths, indices ):
	
	bunch_starts_cut	=	bunch_starts[indices]
	bunch_2nds_cut		=	bunch_2nds[indices]
	bunch_ends_cut		=	bunch_ends[indices]
	bunch_lengths_cut	=	bunch_lengths[indices]
	
	return bunch_starts_cut, bunch_2nds_cut, bunch_ends_cut, bunch_lengths_cut


def get_data( time, detx, dety, energy, veto, indices ):
	
	time_cut	=	time[indices]
	detx_cut	=	detx[indices]
	dety_cut	=	dety[indices]
	energy_cut	=	energy[indices]
	veto_cut	=	veto[indices]
	
	return time_cut, detx_cut, dety_cut, energy_cut, veto_cut


def flagdata( time, detx, dety, energy, veto, indices_to_flag ):
	
	time	=	np.delete( time, indices_to_flag )
	detx	=	np.delete( detx, indices_to_flag )
	dety	=	np.delete( dety, indices_to_flag )
	energy	=	np.delete( energy, indices_to_flag )
	veto	=	np.delete( veto, indices_to_flag )
		
	return time, detx, dety, energy, veto


def grossnoisy( detx, dety, gross_cutoff ):

	full_DPH	=	mf.create_full_DPH( detx, dety )
	
	#	To check how many of the pixels are flagged by the onboard software.
	initially_flagged	=	len( np.where( full_DPH == 0 )[0] )
	print '\nAlready flagged pixels:			' ,  initially_flagged
	
	#	To mask the pixels with zero counts before flagging, for subsequent analysis.
	full_DPH	=	ma.masked_equal( full_DPH, 0 )
	
	#	To apply the iterative 5-sigma clipping based on the DPH.
	clipped_DPH	=	sigma_clip( data = full_DPH, sigma_lower = 100, sigma_upper = gross_cutoff )
	indices_flagged	=	np.where( ma.getmask( clipped_DPH ) == True )
	total_flagged	=	indices_flagged[0].shape[0]
	print 'Number of gross noisy pixels:		', total_flagged - initially_flagged
	print 'Total number of flagged pixels:		', total_flagged,'\n\n'
	detx_flag, dety_flag	=	reverse_mapping( indices_flagged[0], indices_flagged[1] )
	
	indices_to_flag	=	np.array( [] )
	for j, x in enumerate( detx_flag ):
		y	=	dety_flag[j]
		ind	=	np.where( (detx == x) & (dety==y) )[0]
		indices_to_flag	=	np.append( indices_to_flag, ind )
	
	#	To calculate overall livetime factor
	numerator	=	number_of_pixels_2D_per_Quadrant - total_flagged
	denominator	=	number_of_pixels_2D_per_Quadrant - initially_flagged
	overall_livetime_factor	=	numerator / denominator
	
	
	return indices_to_flag, overall_livetime_factor, total_flagged


def calculate_SNR( time, countrate, T1, T2 ):
	
	#	To calculate the time-cale used for binning the lightcurve.
	t_bin	=	np.diff(time)[0]
	
	#	To choose the data within the required intervals for background and GRB.
	GRB_start	=	T1 - 10
	GRB_stop	=	T2 + 10
	ind_BKG1	=	np.where( time < GRB_start )[0]
	ind_BKG2	=	np.where( GRB_stop <= time )[0]
	ind_GRB		=	np.where( ( GRB_start <= time ) & ( time <  GRB_stop ) )[0]
	
	time_BKG1	=	time[ind_BKG1]
	time_BKG2	=	time[ind_BKG2]
	time_GRB	=	time[ind_GRB ]
	counts_BKG1	=	countrate[ind_BKG1] * t_bin
	counts_BKG2	=	countrate[ind_BKG2] * t_bin
	counts_GRB	=	countrate[ind_GRB ] * t_bin
	
	time_BKG	=	np.append(  time_BKG1,   time_BKG2)
	counts_BKG	=	np.append(counts_BKG1, counts_BKG2)
	
	GRB_duration	=	GRB_stop - GRB_start
	BKG1_duration	=	GRB_start - time[0]
	BKG2_duration	=	time[-1] - GRB_stop
	
	L			=	len(counts_GRB)
	GRB_total	=	np.sum(counts_GRB)
	BKG_rate	=	( np.sum(counts_BKG1)/BKG1_duration + np.sum(counts_BKG2)/BKG2_duration ) / 2
	
	Signal		=	( GRB_total - BKG_rate*GRB_duration ) / L
	Noise		=	np.std( counts_BKG )
	Signal_error=	np.sqrt( GRB_total + BKG_rate*GRB_duration ) / L
	
	SNR			=	Signal / Noise
	SNR_error	=	Signal_error / Noise
	
	
	return round(SNR, 3), round(SNR_error, 3)


def get_single( time, detx, dety, energy, veto ):
	
	delta	=	np.diff( time )
	single_indices	=	np.where( delta > time_res )[0]
	
	time_single, detx_single, dety_single, energy_single, veto_single	=	get_data( time, detx, dety, energy, veto, single_indices )
	
	return time_single, detx_single, dety_single, energy_single, veto_single


def get_double( time, detx, dety, energy, veto ):
	
	delta	=	np.diff( time )
	double_indices		=	np.where( delta <= time_res )[0]	
	indices_multiple	=	np.where( np.diff(double_indices) < 2 )[0]
	indices_multiple	=	np.append( indices_multiple, indices_multiple+1 )
	
	#	print ( double_indices[indices_multiple+1] - double_indices[indices_multiple] == 1 ).all()
	#	print ( double_indices[indices_multiple+1] - double_indices[indices_multiple] == 1 ).size
	#	print np.where( double_indices[indices_multiple+2] - double_indices[indices_multiple+1] == 1 )[0].size
	
	double_indices	=	np.delete( double_indices, indices_multiple )
	double_indices	=	np.append( double_indices, double_indices+1 )
	double_indices	=	np.unique( double_indices )
	double_indices	=	np.sort( double_indices )
	time_double, detx_double, dety_double, energy_double, veto_double	=	get_data( time, detx, dety, energy, veto, double_indices )
	
	return time_double, detx_double, dety_double, energy_double, veto_double


def my_absdiff( n1, n2 ):
	
	difference	=	n1 - n2
	if n1 > n2: return	difference
	else:		return -difference


def get_Comptondouble( time, detx, dety, energy, veto ):
	
	delta			=	np.diff( time )
	double_indices		=	np.where( delta <= time_res )[0]
	indices_multiple	=	np.where( np.diff(double_indices) < 2 )[0]
	indices_multiple	=	np.append( indices_multiple, indices_multiple+1 )
	double_indices		=	np.delete( double_indices, indices_multiple )
	
	inds_to_delete	=	np.array([])
	for inds, l in enumerate(double_indices):
		diff_x	=	my_absdiff( detx[l+1], detx[l] )
		diff_y	=	my_absdiff( dety[l+1], dety[l] )
		if ( diff_x > 1 ) or ( diff_y > 1 ):
			inds_to_delete	=	np.append( inds_to_delete, inds )
		elif ( diff_x == 0 ) and ( diff_y == 0 ):
			inds_to_delete	=	np.append( inds_to_delete, inds )
	double_indices	=	np.delete( double_indices, inds_to_delete )
	
	double_indices	=	np.append( double_indices, double_indices+1 )
	double_indices	=	np.unique( double_indices )
	double_indices	=	np.sort( double_indices )
	time_double, detx_double, dety_double, energy_double, veto_double	=	get_data( time, detx, dety, energy, veto, double_indices )
	
	return time_double, detx_double, dety_double, energy_double, veto_double


def bin_data_into_energies( time, detx, dety, energy, veto, E_low, E_high ):
	
	index_cut	=	np.where( (E_low <= energy) & (energy <= E_high) )[0]
	time_cut, detx_cut, dety_cut, energy_cut, veto_cut	=	get_data( time, detx, dety, energy, veto, index_cut )
	
	return time_cut, detx_cut, dety_cut, energy_cut, veto_cut


def get_vetotaggedCZT_data( time, detx, dety, energy, veto ):
	
	index_get	=	np.where( veto != 0 )[0]
	time_cut, detx_cut, dety_cut, energy_cut, veto_cut	=	get_data( time, detx, dety, energy, veto, index_get )
	
	return time_cut, detx_cut, dety_cut, energy_cut, veto_cut


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
	For using on the tagged events, the PHA's are to be taken from the individual quadrant files. To reduce the amount of data,
	the 8 bit veto channel numbers (0-255) have been contracted to 7 bits (0-127) in the quadrant files. Thus, one should do:
	channel	=	2 * channel
	before transforming the channel to energy.
	
	
	'''
	
	
	if   i == 0:	energy = 5.591 * channel - 56.741
	elif i == 1:	energy = 5.594 * channel - 41.289
	elif i == 2:	energy = 5.943 * channel - 41.682
	elif i == 3:	energy = 5.222 * channel - 26.528
	
	
	return energy


#######################################################################################################################################################





#######################################################################################################################################################



#~ ##	Spectroscopically bad pixels data
#~ pixflags_table	=	np.loadtxt( badpix_filename )
#~ Q_id	=	pixflags_table[:, 0]
#~ mod_id	=	pixflags_table[:, 1]
#~ pix_id	=	pixflags_table[:, 2]
#~ flags	=	pixflags_table[:, 3]




##	Veto data
#	To extract the data.
dat		=	fits.open( event_filename )
Quad_id	=	dat[5].data['QuadID']
times	=	dat[5].data['Time']
counts	=	dat[5].data['VetoSpec']
dat.close()
#	To choose the good Veto channels only:
channel_lower	=	24	#	Strict Veto lower cutoff.
channel_upper	=	107	#	Saturation effect is known to dominate above this channel.
channels=	np.arange( channel_lower, channel_upper )
counts	=	counts[ :, channel_lower:channel_upper ]
counts	=	counts.transpose()

start_times	=	np.zeros(Q_n)
stop_times	=	np.zeros(Q_n)
for i in range( Q_n ):
	#	To convert from channel to energy information for this Q.
	energy	=	veto_channel2energy( channels, i )
	
	#	To extract the data for the specific Q.
	ind_Q	=	np.where( Quad_id == i )[0]
	t		=	times[ind_Q]
	c		=	counts[:, ind_Q]
	
	#	To select the data within the chosen time-window.
	ind_t	=	np.where( (t_start < t) & ( t < t_stop ) )[0]
	t		=	t[ind_t]
	c		=	c[:, ind_t]
	
	#	To create the lightcurve.
	energy_integrated_counts	=	np.sum(c, axis = 0)
	
	#~ plt.title( r'$ \rm{ t_{bin} } = $' + r'$ {0:.1f} \, $'.format(t_bin) + r'$ \rm{ s : \quad Q } $' + r'$ {:d} $'.format(i), fontsize = size_font )
	plt.xlabel( r'$ \rm{UT - t_{offset} \; [in \; sec]} $', fontsize = size_font )
	plt.ylabel( r'$ \rm{Counts / sec} $', fontsize = size_font )	
	plt.step( t-t_offset, energy_integrated_counts, color = 'k' )
	plt.savefig( path_save + 'Q{0:d}--veto.png'.format(i) )
	plt.clf()
	plt.close()
	
	start_times[i]	=	t.min()
	stop_times[i]	=	t.max()
t_start	=	round( start_times.max() - t_bin/2 )
t_stop	=	round(  stop_times.min() + t_bin/2 )
#~ print t_start, t_stop


## Livetime calculations at the start
expfracs_at_start	=	{}
livetime_mids	=	np.arange( t_start+t_bin/2, t_stop+t_bin/2, t_bin )
#	print livetime_mids.size
#	print livetime_mids[0], t_start, livetime_mids[-1], t_stop
#	print livetime_mids - t_start
expfracs_at_start	=	{}	
dat	=	fits.open( event_filename )
for i in range( Q_n ):
	#	print '##########################################################\n'
	#	print 'Quadrant {:d}...'.format(i), '\n\n'
	
	GTI_start	=	dat[i+9].data['START']
	GTI_stop	=	dat[i+9].data['STOP']
	
	#	print GTI_start
	#	print GTI_stop
	#	print ( GTI_start - GTI_start[0] >= 0 ).all()
	#	print ( GTI_stop - GTI_start > 0 ).all()
	#	print ( GTI_stop - GTI_start ), '\n\n'
	#	plt.hist( GTI_stop - GTI_start )
	#	plt.show()
	
	fracs	=	np.zeros( livetime_mids.size )
	for k, mids in enumerate( livetime_mids ):
		starts_at	=	mids - t_bin/2
		stops__at	=	mids + t_bin/2
		
		
		
		
		inds_fully_inside	=	np.where( (starts_at < GTI_start) & (GTI_start < stops__at) & (starts_at < GTI_stop) & (GTI_stop < stops__at) )[0]
		if inds_fully_inside.size > 0:
			#	print GTI_start[inds_fully_inside],  GTI_stop[inds_fully_inside]
			fracs[k]	+=	( GTI_stop[inds_fully_inside] - GTI_start[inds_fully_inside] ) / t_bin
			#	print k, inds_fully_inside, fracs[k]
		
		
		
		indices_GTIstarts_smaller	=	np.where( GTI_start < starts_at )[0]
		GTIstarts_smaller	=	GTI_start[indices_GTIstarts_smaller]
		GTIstops__smaller	=	GTI_stop[ indices_GTIstarts_smaller]
		
		indices_GTIstops__greater	=	np.where( stops__at < GTI_stop )[0]
		GTIstarts_greater	=	GTI_start[indices_GTIstops__greater]
		GTIstops__greater	=	GTI_stop[ indices_GTIstops__greater]
		
		#	print indices_GTIstarts_smaller
		#	print indices_GTIstops__greater
		
		if GTIstops__smaller[-1] > stops__at:
			fracs[k]	=	1
			#	print GTIstarts_smaller[-1] - starts_at
			#	print GTIstops__smaller[-1] - stops__at, '\n'
			#	print indices_GTIstarts_smaller[-1]-indices_GTIstops__greater[0]
			#	print k, 1
			
		elif ( GTIstops__smaller[-1] < starts_at ) and ( GTIstarts_greater[0] > stops__at ):
			fracs[k]	=	0
			#	print GTIstop__smaller[-1] - starts_at
			#	print GTIstarts_greater[0] - starts_at, '\n'
			#	print k, 0
		
		elif ( starts_at < GTIstops__smaller[-1] ) and ( GTIstops__smaller[-1] < stops__at ):
			fracs[k]	+=	( GTIstops__smaller[-1] - starts_at ) / t_bin
			#	print indices_GTIstarts_smaller[-1]
			#	print GTIstarts_smaller[-1]-starts_at, GTIstops__smaller[-1]-starts_at, GTIstarts_greater[0]-starts_at, GTIstops__greater[0]-starts_at
			if ( GTIstarts_greater[0]-starts_at < t_bin ):
				fracs[k]	=	fracs[k] + (stops__at - GTIstarts_greater[0])/t_bin
			#	print k, fracs[k]
			#	print '\n'
		
		elif ( starts_at < GTIstarts_greater[0]  ) and (  GTIstarts_greater[0] < stops__at ):
			fracs[k]	+=	( stops__at - GTIstarts_greater[0] ) / t_bin
			#	print GTIstarts_smaller[-1]-starts_at, GTIstops__smaller[-1]-starts_at, GTIstarts_greater[0]-starts_at, GTIstops__greater[0]-starts_at
			#	print k, fracs[k]
			#	print '\n'
		
		else:
			print 'SOMETHING IS WRONG!'
	
	#	print fracs
	#	print '\n\n'
	expfracs_at_start[i]	=	fracs
#	print expfracs_at_start[1].size
#	plt.plot( livetime_mids, expfracs_at_start[1], 'ro' )
#	plt.show()
#	print np.any(expfracs_at_start[1] == 0)
print '\n\n\n\n'


##	CZT data
hdu	=	fits.open( bunch_filename )
for i in range( Q_n ):
	#	To loop over the Quadrants.
	print '##########################################################\n'
	print 'Quadrant {:d}...'.format(i), '\n\n'
	
	############################## START ###############################
	#~ print '..........................................................\n'
	#~ print 'Very start...', '\n'
	
	#	To extract the data.
	time	=	dat[i+1].data['TIME']
	detx	=	dat[i+1].data['DETX']
	dety	=	dat[i+1].data['DETY']
	PHA		=	dat[i+1].data['PHA']
	veto	=	dat[i+1].data['veto']
	energy	=	PHA * (200.0/4096)*4
	
	#	print PHA
	#	print energy.min(), energy.max()
	
	#	To extract the bunch data.
	bunch_2nds		=	hdu[i+1].data['Time']
	bunch_starts	=	hdu[i+1].data['Time_dfs']
	bunch_ends		=	hdu[i+1].data['Time_dsl']
	bunch_lengths	=	hdu[i+1].data['NumEvent']
	bunch_starts	=	bunch_2nds - time_res * bunch_starts
	bunch_ends		=	bunch_2nds + time_res * bunch_ends
	bunch_lengths	=	bunch_lengths + 1
	
	#	To slice off data between t_start and t_stop.
	index_cut		=	np.where( (t_start < bunch_starts) & (bunch_starts < t_stop) )[0]
	bunch_starts, bunch_2nds, bunch_ends, bunch_lengths	=	get_bunchdata( bunch_starts, bunch_2nds, bunch_ends, bunch_lengths, index_cut )	
	index_cut		=	np.where( (t_start < time) & (time < t_stop) )[0]
	time, detx, dety, energy, veto	=	get_data( time, detx, dety, energy, veto, index_cut )
	
	############################## START ###############################
	
	
	
	########################### LIGHTCURVES ############################
	
	##	Bunches
	x, y	=	mf.my_histogram_according_to_given_boundaries( bunch_starts, t_bin, t_start, t_stop )
	y		=	y / t_bin
	
	#~ plt.title( r'$ \rm{ t_{bin} } = $' + r'$ \, {0:.1f} \, $'.format(t_bin) + r'$ \rm{ s : \quad Q } $' + r'$ {:d} $'.format(i), fontsize = size_font )
	plt.xlabel( r'$ \rm{UT - t_{offset} \; [in \; sec]} $', fontsize = size_font )
	plt.ylabel( r'$ \rm{Counts / sec} $', fontsize = size_font )
	plt.step( x-t_offset, y, color = 'k' )
	plt.savefig( path_save + 'Q{0:d}--bunches--{1:.1f}sec.png'.format(i, t_bin) )
	plt.clf()
	plt.close()
	
	
	
	#	All events
	x, y	=	mf.my_histogram_according_to_given_boundaries( time, t_bin, t_start, t_stop )
	y		=	y / t_bin
	inds_nonzero_lvts	=	np.where( expfracs_at_start[i] != 0 )[0]
	x	=	x[inds_nonzero_lvts]
	y	=	y[inds_nonzero_lvts] / expfracs_at_start[i][inds_nonzero_lvts]
	
	#~ plt.title( r'$ \rm{ t_{bin} } = $' + r'$ \, {0:.1f} \, $'.format(t_bin) + r'$ \rm{ s : \quad Q } $' + r'$ {:d} $'.format(i), fontsize = size_font )
	plt.xlabel( r'$ \rm{UT - t_{offset} \; [in \; sec]} $', fontsize = size_font )
	plt.ylabel( r'$ \rm{Counts / sec} $', fontsize = size_font )
	plt.step( x-t_offset, y, color = 'k' )
	plt.savefig( path_save + 'Q{0:d}--allevts_at_start--{1:.1f}sec.png'.format(i, t_bin) )
	plt.savefig( path_save + 'Q{0:d}--allevts_at_start--{1:.1f}sec.pdf'.format(i, t_bin) )
	plt.clf()
	plt.close()
	
	########################### LIGHTCURVES ############################
		
	
	#~ ################# SPECTROSCOPICALLY BAD PIXELS #####################
	#~ print '..........................................................\n'
	#~ print 'Spectroscopically bad pixels...', '\n'
	#~ 
	#~ Quad_arr	=	np.array([])
	#~ bad_detx	=	[]
	#~ bad_dety	=	[]
	#~ for k in range( number_of_pixels_2D_per_Quadrant ):
		#~ quadid	=	Q_id[	i*number_of_pixels_2D_per_Quadrant + k ]
		#~ detid	=	mod_id[ i*number_of_pixels_2D_per_Quadrant + k ]
		#~ pixid	=	pix_id[ i*number_of_pixels_2D_per_Quadrant + k ]
		#~ f		=	flags[	i*number_of_pixels_2D_per_Quadrant + k ]
		#~ x,	y	=	mf.modid_pixid_to_detx_dety( i, detid, pixid )
		#~ if f==1:
			#~ bad_detx.append(x)
			#~ bad_dety.append(y)
		#~ Quad_arr	=	np.append( Quad_arr, np.array([quadid]) )
	#~ indices_to_flag	=	np.array( [] )
	#~ for j, x in enumerate( bad_detx ):
		#~ y	=	bad_dety[j]
		#~ ind	=	np.where( (detx == x) & (dety==y) )[0]
		#~ indices_to_flag	=	np.append( indices_to_flag, ind )
	#~ time, detx, dety, energy, veto	=	flagdata( time, detx, dety, energy, veto, indices_to_flag )	
	#~ 
	#~ x, y	=	mf.my_histogram_according_to_given_boundaries( time, t_bin, t_start, t_stop )
	#~ 
	#~ SNR, SNR_error	=	calculate_SNR( x, y, T1, T2 )
	#~ print 'SNR:	{0:.3f}	error:	{1:.3f}'.format(SNR, SNR_error), '\n'
	#~ 
	#~ print '..........................................................\n'
	#~ ################# SPECTROSCOPICALLY BAD PIXELS #####################
	
	
	
	
	########################### BUNCHCLEAN #############################
	print '..........................................................\n'
	print 'New bunchclean...', '\n'
	
	L	=	len(bunch_starts)
	print '\nTotal number of bunches:	', L, '\n'
	
	#	To redefine bunches.
	time_interval_between_bunches	=	bunch_starts[1:] - bunch_ends[:-1]
	ind_bunch		=	np.where( ( time_interval_between_bunches < t2 ) )[0]
	bunch_starts	=	np.delete( bunch_starts, ind_bunch+1 )
	bunch_ends		=	np.delete( bunch_ends, ind_bunch )
	bunch_2nds		=	np.delete( bunch_2nds, ind_bunch+1 )
	bunch_lengths[ind_bunch]=	bunch_lengths[ind_bunch] + bunch_lengths[ind_bunch+1]
	bunch_lengths			=	np.delete( bunch_lengths, ind_bunch+1 )
	
	#	To update livetime after bunchclean
	bunch_diff	=	bunch_ends - bunch_starts
	
	expfracs_after_bunchclean	=	np.zeros( expfracs_at_start[i].size )
	for k, T in enumerate( livetime_mids ):
		expfrac_before	=	expfracs_at_start[i][k]
		
		inds	=	np.where( (T-t_bin/2 < bunch_starts) & (bunch_ends+t3 < T+t_bin/2) )[0]
		time_lost	=	np.sum( bunch_diff[inds] ) + t3*inds.size	#	for bunchclean and post bunchclean
		expfrac_updated	=	expfrac_before * (  1  -  (1/16)*(time_lost/t_bin)  )		#	Assuming that all data from exactly one module is lost
		#	print np.sum( bunch_diff[inds] ), inds.size*t3, time_lost
		#	print expfrac_before, expfrac_updated, '\n\n'
		expfracs_after_bunchclean[k]	=	expfrac_updated
	
	#~ #	To plot the livetime factors before and after bunchclean.
	#~ plt.plot( expfracs_at_start[i], 'bo', label = 'Start' )
	#~ plt.plot( expfracs_after_bunchclean, 'ro', label = 'Bunchclean' )
	#~ plt.legend( numpoints = 1, loc = 'best' )
	#~ plt.show()
	#~ print ( expfracs_after_bunchclean - expfracs_at_start[i] )
	
	
	#	To flag the data beween and post bunches.
	indices_to_flag	=	np.array([])
	for j, end in enumerate( bunch_ends ):
		#	To loop over the bunches.
		start	=	bunch_starts[j]
		
		inds	=	np.where( (start <= time) & (time <= end+t3) )[0]
		indices_to_flag	=	np.append( indices_to_flag, inds )
	time, detx, dety, energy, veto	=	flagdata( time, detx, dety, energy, veto, indices_to_flag )
	
	########################### BUNCHCLEAN #############################
	
	
	
	
	########################### LIGHTCURVES ############################
	
	##	Single events
	#~ time_single, detx_single, dety_single, energy_single, veto_single	=	get_single( time, detx, dety, energy, veto )
	#~ 
	#~ x, y	=	mf.my_histogram_according_to_given_boundaries( time_single, t_bin, t_start, t_stop )
	#~ y		=	y / t_bin
	#~ x	=	x[inds_nonzero_lvts]
	#~ y	=	y[inds_nonzero_lvts] / expfracs_after_bunchclean[inds_nonzero_lvts]
	#~ 
	#~ plt.title( r'$ \rm{ t_{bin} } = $' + r'$ \, {0:.1f} \, $'.format(t_bin) + r'$ \rm{ s : \quad Q } $' + r'$ {:d} $'.format(i), fontsize = size_font )
	#~ plt.xlabel( r'$ \rm{UT - t_{offset} \; [in \; sec]} $', fontsize = size_font )
	#~ plt.ylabel( r'$ \rm{Counts / sec} $', fontsize = size_font )
	#~ plt.step( x-t_offset, y, color = 'k' )
	#~ plt.savefig( path_save + 'Q{0:d}--sglevts--step1_after_bunchclean--{1:.1f}sec.png'.format(i, t_bin) )
	#~ plt.clf()
	#~ plt.close()
	#~ 
	#~ SNR, SNR_error	=	calculate_SNR( x, y, T1, T2 )
	#~ print 'SNR:	{0:.3f}	error:	{1:.3f}'.format(SNR, SNR_error), '\n'
	#~ 
	#~ 
	#~ 
	#~ ##	Double events
	#~ time_double, detx_double, dety_double, energy_double, veto_double	=	get_double( time, detx, dety, energy, veto )
	#~ 
	#~ x, y	=	mf.my_histogram_according_to_given_boundaries( time_double, t_bin, t_start, t_stop )
	#~ y		=	y / t_bin
	#~ x	=	x[inds_nonzero_lvts]
	#~ y	=	y[inds_nonzero_lvts] / expfracs_after_bunchclean[inds_nonzero_lvts]
	#~ 
	#~ plt.title( r'$ \rm{ t_{bin} } = $' + r'$ \, {0:.1f} \, $'.format(t_bin) + r'$ \rm{ s : \quad Q } $' + r'$ {:d} $'.format(i), fontsize = size_font )
	#~ plt.xlabel( r'$ \rm{UT - t_{offset} \; [in \; sec]} $', fontsize = size_font )
	#~ plt.ylabel( r'$ \rm{Counts / sec} $', fontsize = size_font )
	#~ plt.step( x-t_offset, y, color = 'k' )
	#~ plt.savefig( path_save + 'Q{0:d}--dblevts--step1_after_bunchclean--{1:.1f}sec.png'.format(i, t_bin) )
	#~ plt.clf()
	#~ plt.close()
	#~ 
	#~ SNR, SNR_error	=	calculate_SNR( x, y, T1, T2 )
	#~ print 'SNR:	{0:.3f}	error:	{1:.3f}'.format(SNR, SNR_error), '\n'
	
	print '..........................................................\n'
	########################### LIGHTCURVES ############################
	
	
	
	
	###################### GROSS NOISY PIXEL CLEAN #####################
	print '..........................................................\n'
	print 'Gross noisy pixels...', '\n'
	
	indices_to_flag, overall_livetime_factor, total_flagged_pixels	=	grossnoisy( detx, dety, gross_cutoff )
	time, detx, dety, energy, veto	=	flagdata( time, detx, dety, energy, veto, indices_to_flag )
	
	expfracs_after_grossnoisy	=	expfracs_after_bunchclean * overall_livetime_factor
	
	#~ #	To plot the livetime factors before and after grossnoisy.
	#~ plt.plot( expfracs_after_bunchclean, 'bo', label = 'Bunchclean' )
	#~ plt.plot( expfracs_after_grossnoisy, 'ro', label = 'Grossnoisy' )
	#~ plt.legend( numpoints = 1, loc = 'best' )
	#~ plt.show()
	#~ print ( expfracs_after_grossnoisy - expfracs_at_start[i] )
	
	###################### GROSS NOISY PIXEL CLEAN #####################
	
	
	
	
	########################### LIGHTCURVES ############################
	
	#~ ##	Single events
	time_single, detx_single, dety_single, energy_single, veto_single	=	get_single( time, detx, dety, energy, veto )
	
	x, y	=	mf.my_histogram_according_to_given_boundaries( time_single, t_bin, t_start, t_stop )
	y		=	y / t_bin
	
	#~ plt.title( r'$ \rm{ t_{bin} } = $' + r'$ \, {0:.1f} \, $'.format(t_bin) + r'$ \rm{ s : \quad Q } $' + r'$ {:d} $'.format(i), fontsize = size_font )
	#~ plt.xlabel( r'$ \rm{UT - t_{offset} \; [in \; sec]} $', fontsize = size_font )
	#~ plt.ylabel( r'$ \rm{Counts / sec} $', fontsize = size_font )
	#~ plt.step( x-t_offset, y, color = 'k', label = 'With' )
	#~ plt.legend()
	#~ plt.show()
	#~ plt.savefig( path_save + 'Q{0:d}--sglevts--step2_after_grossnoisy--{1:.1f}sec.png'.format(i, t_bin) )
	#~ plt.clf()
	#~ plt.close()
	
	SNR, SNR_error	=	calculate_SNR( x, y, T1, T2 )
	print 'SNR:	{0:.3f}	error:	{1:.3f}'.format(SNR, SNR_error), '\n'
	
	
	#~ ##	Double events
	#~ time_double, detx_double, dety_double, energy_double, veto_double	=	get_double( time, detx, dety, energy, veto )
	#~ 
	#~ x, y	=	mf.my_histogram_according_to_given_boundaries( time_double, t_bin, t_start, t_stop )
	#~ y		=	y / t_bin
	#~ x	=	x[inds_nonzero_lvts]
	#~ y	=	y[inds_nonzero_lvts] / expfracs_after_grossnoisy[inds_nonzero_lvts]
	#~ 
	#~ plt.title( r'$ \rm{ t_{bin} } = $' + r'$ \, {0:.1f} \, $'.format(t_bin) + r'$ \rm{ s : \quad Q } $' + r'$ {:d} $'.format(i), fontsize = size_font )
	#~ plt.xlabel( r'$ \rm{UT - t_{offset} \; [in \; sec]} $', fontsize = size_font )
	#~ plt.ylabel( r'$ \rm{Counts / sec} $', fontsize = size_font )
	#~ plt.step( x-t_offset, y, color = 'k' )
	#~ plt.savefig( path_save + 'Q{0:d}--dblevts--step2_after_grossnoisy--{1:.1f}sec.png'.format(i, t_bin) )
	#~ plt.clf()
	#~ plt.close()
	#~ 
	#~ SNR, SNR_error	=	calculate_SNR( x, y, T1, T2 )
	#~ print 'SNR:	{0:.3f}	error:	{1:.3f}'.format(SNR, SNR_error), '\n'
	
	print '..........................................................\n'
	########################### LIGHTCURVES ############################
	
	
	
	
	########################### DPHclean ###############################
	print '..........................................................\n'
	print 'DPHclean...', '\n'
	
	expfracs_after_DPHclean	=	expfracs_after_grossnoisy.copy()
	
	x, y	=	mf.my_histogram_according_to_given_boundaries( time, t_look, t_start, t_stop )
	y		=	y / t_look
	peaks_index, peaks_time, peaks_countrate, peaks_significance	=	mf.detect_peaks( x, y, T_search, cutoff )
	#~ plt.title( r'$ \rm{ t_{bin} } = $' + r'$ \, {0:.1f} \, $'.format(t_look) + r'$ \rm{ s : \quad Q } $' + r'$ {:d} $'.format(i), fontsize = size_font )
	plt.xlabel( r'$ \rm{UT - t_{offset} \; [in \; sec]} $', fontsize = size_font )
	plt.ylabel( r'$ \rm{Counts / sec} $', fontsize = size_font )
	plt.step( x-t_offset, y, color = 'k', lw = 0.5 )
	plt.plot( peaks_time-t_offset, peaks_countrate, 'ro' )
	#~ plt.savefig( path_save + 'Q{:d}--DPHclean_before--zoom.png'.format(i) )
	#~ plt.savefig( path_save + 'Q{:d}--DPHclean_before--zoom.pdf'.format(i) )
	plt.savefig( path_save + 'Q{:d}--DPHclean_before.png'.format(i) )
	plt.savefig( path_save + 'Q{:d}--DPHclean_before.pdf'.format(i) )
	plt.clf()
	plt.close()	
	
	
	time_binned, counts_binned, image_binned	=	mf.create_dph( time, detx, dety, t_look )
	print 'These many bins:	',	time_binned.size
	
	flag_indices	=	np.array( [] )
	times_to_flag	=	np.array( [] )
	#	To make DPH-investigation of all DPHs.
	for j, peaktime in enumerate( time_binned ):
		#	The j-th bin.
		#	To extract the required data around the peak.
		index		=	np.where(    ( (peaktime-t_look/2) < time ) & ( time < (peaktime+t_look/2) )    )[0]
		data_time	=	time[index]
		data_detx	=	detx[index]
		data_dety	=	dety[index]
		
		#	To first make a DPH for the chosen bin.
		DPH		=	mf.create_full_DPH( data_detx, data_dety )
		#	To decide whether the data in that region needs to be flagged.
		decision_flag, number_of_points, sum_of_measures, number_of_hot_pairs, cluster_image, bad_detx, bad_dety	=	dc.flag_the_dph( DPH, threshold, allowable, ID )
		
		#	Plotting specifications.
		def applyPlotStyle( mx ):
			bn	=	8
			ax.set_xticks( np.arange(0, L_x, bn) )
			ax.set_xticklabels( np.arange(0, L_x,  bn) )
			ax.set_yticks( np.arange(0, L_y, bn) )
			ax.set_yticklabels( np.arange(L_y, 0, -bn) )
			ax.grid( linewidth = 1.2 )
		
		if decision_flag	==	1:
			flag_indices	=	np.append( flag_indices, j )
			
			ind_lvt	=	mf.nearest( livetime_mids, peaktime )
			expfracs_after_DPHclean[ind_lvt]	=	expfracs_after_DPHclean[ind_lvt] * (  1  -  ( number_of_points/(number_of_pixels_2D_per_Quadrant - total_flagged_pixels) )*(t_look/t_bin)  )
			
			indices_to_flag	=	np.array( [] )
			for j, x in enumerate( bad_detx ):
				y	=	bad_dety[j]
				#	ind	=	np.where( (data_detx == x) & (data_dety==y) )[0]
				#	indices_to_flag	=	np.append( indices_to_flag, ind )
				
			
				#	To extract the image near the neighbourhood of the point, the neighbourhood is defined.
				m	=	(L_y - 1) - y
				n	=	x
				x_min	=	int(m - 1)
				x_max	=	int(m + 1)
				y_min	=	int(n - 1)
				y_max	=	int(n + 1)
				if x_min < 0:	x_min	=	0
				if y_min < 0:	y_min	=	0
				if x_max > L_x:	x_max	=	L_x
				if y_max > L_y:	y_max	=	L_y
				image_neighbourhood	=	DPH[ x_min : x_max+1, y_min : y_max+1 ]
				
				#	To check if there are any pixels in the neighbourhood registering non-zero events.
				sum_of_neighbourhood	=	np.sum( image_neighbourhood )
				if sum_of_neighbourhood > 2:
					ind	=	np.where( (data_detx == x) & (data_dety==y) )[0]
					indices_to_flag	=	np.append( indices_to_flag, ind )
			
			indices_to_flag	=	indices_to_flag.astype(int)
			data_time	=	data_time[indices_to_flag]
			data_detx	=	data_detx[indices_to_flag]
			data_dety	=	data_dety[indices_to_flag]
			
			times_to_flag	=	np.append( times_to_flag, data_time )
	
	print 'These many flagged:	',	flag_indices.size, '\n\n'
	
	#~ #	To plot the livetime factors before and after DPHclean.
	#~ plt.plot( expfracs_after_grossnoisy, 'bo', label = 'Grossnoisy' )
	#~ plt.plot( expfracs_after_DPHclean, 'ro', label = 'DPHclean' )
	#~ plt.legend( numpoints = 1, loc = 'best' )
	#~ plt.show()
	#~ print expfracs_after_DPHclean - expfracs_after_grossnoisy
	
	total_flag_indices	=	np.array( [] )
	for k, timestamp in enumerate(times_to_flag):
		ind = np.where( time == timestamp )[0]
		total_flag_indices = np.append( total_flag_indices, ind )
	total_flag_indices	=	np.unique( total_flag_indices )	
	
	time, detx, dety, energy, veto	=	flagdata( time, detx, dety, energy, veto, total_flag_indices )
	
	x, y	=	mf.my_histogram_according_to_given_boundaries( time, t_look, t_start, t_stop )
	y		=	y / t_look
	peaks_index, peaks_time, peaks_countrate, peaks_significance	=	mf.detect_peaks( x, y, T_search, cutoff )
	#~ plt.title( r'$ \rm{ t_{bin} } = $' + r'$ \, {0:.1f} \, $'.format(t_look) + r'$ \rm{ s : \quad Q } $' + r'$ {:d} $'.format(i), fontsize = size_font )
	plt.xlabel( r'$ \rm{UT - t_{offset} \; [in \; sec]} $', fontsize = size_font )
	plt.ylabel( r'$ \rm{Counts / sec} $', fontsize = size_font )
	plt.step( x-t_offset, y, color = 'k', lw = 0.5 )
	plt.plot( peaks_time-t_offset, peaks_countrate, 'ro' )
	#~ plt.savefig( path_save + 'Q{:d}--DPHclean_after--zoom.png'.format(i) )
	#~ plt.savefig( path_save + 'Q{:d}--DPHclean_after--zoom.pdf'.format(i) )
	plt.savefig( path_save + 'Q{:d}--DPHclean_after.png'.format(i) )
	plt.savefig( path_save + 'Q{:d}--DPHclean_after.pdf'.format(i) )
	plt.clf()
	plt.close()	
	
	
	########################### DPHclean ###############################
	
	
	
	########################### LIGHTCURVES ############################
	
	##	Single events
	time_single, detx_single, dety_single, energy_single, veto_single	=	get_single( time, detx, dety, energy, veto )
	
	x, y	=	mf.my_histogram_according_to_given_boundaries( time_single, t_bin, t_start, t_stop )
	y		=	y / t_bin
	x	=	x[inds_nonzero_lvts]
	y	=	y[inds_nonzero_lvts] / expfracs_after_DPHclean[inds_nonzero_lvts]
	
	#~ plt.title( r'$ \rm{ t_{bin} } = $' + r'$ \, {0:.1f} \, $'.format(t_bin) + r'$ \rm{ s : \quad Q } $' + r'$ {:d} $'.format(i), fontsize = size_font )
	plt.xlabel( r'$ \rm{UT - t_{offset} \; [in \; sec]} $', fontsize = size_font )
	plt.ylabel( r'$ \rm{Counts / sec} $', fontsize = size_font )
	plt.step( x-t_offset, y, color = 'k' )
	plt.savefig( path_save + 'Q{0:d}--sglevts--step3_after_DPHclean--{1:.1f}sec.png'.format(i, t_bin) )
	plt.clf()
	plt.close()
	
	cont, contsub	=	mf.subtract_continuum( x, y, T_search )
	contsub			=	contsub + cont.mean()
	
	plt.hist( contsub, bins = 100 )
	plt.savefig( path_save + 'Q{0:d}--allevts--step3_after_DPHclean--{1:.1f}sec_histogram.png'.format(i, t_bin) )
	plt.clf()
	plt.close()
	
	SNR, SNR_error	=	calculate_SNR( x, y, T1, T2 )
	print 'SNR:	{0:.3f}	error:	{1:.3f}'.format(SNR, SNR_error), '\n'
	
	
	
	##	Double events
	time_double, detx_double, dety_double, energy_double, veto_double	=	get_double( time, detx, dety, energy, veto )
	
	x, y	=	mf.my_histogram_according_to_given_boundaries( time_double, t_bin, t_start, t_stop )
	y		=	y / t_bin
	x	=	x[inds_nonzero_lvts]
	y	=	y[inds_nonzero_lvts] / expfracs_after_DPHclean[inds_nonzero_lvts]
	
	#~ plt.title( r'$ \rm{ t_{bin} } = $' + r'$ \, {0:.1f} \, $'.format(t_bin) + r'$ \rm{ s : \quad Q } $' + r'$ {:d} $'.format(i), fontsize = size_font )
	plt.xlabel( r'$ \rm{UT - t_{offset} \; [in \; sec]} $', fontsize = size_font )
	plt.ylabel( r'$ \rm{Counts / sec} $', fontsize = size_font )
	plt.step( x-t_offset, y, color = 'k' )
	plt.savefig( path_save + 'Q{0:d}--dblevts--step3_after_DPHclean--{1:.1f}sec.png'.format(i, t_bin) )
	plt.clf()
	plt.close()
	
	#~ SNR, SNR_error	=	calculate_SNR( x, y, T1, T2 )
	#~ print 'SNR:	{0:.3f}	error:	{1:.3f}'.format(SNR, SNR_error), '\n'
	
	print '..........................................................\n'
	########################### LIGHTCURVES ############################
	
	
	
	
	##################### FLICKERING PIXEL CLEAN #######################
	print '..........................................................\n'
	print 'Flickering pixels...', '\n'
	
	expfracs_after_flickpixclean	=	expfracs_after_DPHclean.copy()
	
	##	Pixel-wise lightcurve.
	sample = [ detx, dety, time ]
	h, edges	=	np.histogramdd(  sample, bins = ( np.arange( -0.5, L_x+0.5 ), np.arange ( -0.5, L_y+0.5 ), np.arange( time[0], time[-1], flick_tbin ) )  )
	h_all	=	h.flatten()
	
	bad_detx	=	np.array([])
	bad_dety	=	np.array([])	
	clipped_h	=	h > flick_threshold
	indices		=	np.where( clipped_h == True )
	bad_detx	=	indices[0]
	bad_dety	=	indices[1]
	bad_times	=	indices[2]
	mid_flick	=	mf.mid_array(edges[2])
	
	badx_flagged	=	np.array([])
	bady_flagged	=	np.array([])
	indices_to_flag	=	np.array([])
	for j, x in enumerate( bad_detx ):
		y	=	bad_dety[j]
		t	=	bad_times[j]
		t	=	mid_flick[t]
		
		#	##	To flag all the data from these pixels.
		#	ind	=	np.where( (detx == x) & (dety==y) )[0]
		#	indices_to_flag	=	np.append( indices_to_flag, ind )
		
		
		
		##	To flag only where this pixel was flickering.
		ind	=	np.where( (detx == x) & (dety == y) & ( t-flick_tbin/2 < time ) & ( time < t+flick_tbin/2 ) )[0]
		indices_bad	=	np.where( h[x,y] > flick_threshold )[0]
		
		if indices_bad.size == 1 and h[x,y][indices_bad] == flick_threshold+1:	
			pass
		else:
			indices_to_flag	=	np.append( indices_to_flag, ind )
			ind_lvt	=	mf.nearest( livetime_mids, t )
			expfracs_after_flickpixclean[ind_lvt]	=	expfracs_after_flickpixclean[ind_lvt] * (  1  -  (flick_tbin/t_bin) * (1/(number_of_pixels_2D_per_Quadrant - total_flagged_pixels))  )
			#	print ind_lvt
			
			#	To calculate the number of flagged pixels.
			if ( np.any(badx_flagged == x) == True ) and ( np.any(bady_flagged == y) == True ):
				pass
			else:
				badx_flagged	=	np.append( badx_flagged, x )
				bady_flagged	=	np.append( bady_flagged, y )
	
	flickering_instances		=	np.unique(indices_to_flag).size
	number_of_flickering_pixels	=	badx_flagged.size
	time, detx, dety, energy, veto	=	flagdata( time, detx, dety, energy, veto, indices_to_flag )
	
	
	##	Pixel-wise lightcurve.
	sample = [ detx, dety, time ]
	h, edges	=	np.histogramdd(  sample, bins = ( np.arange( -0.5, L_x+0.5 ), np.arange ( -0.5, L_y+0.5 ), np.arange( time[0], time[-1], flick_tbin ) )  )
	h_all	=	h.flatten()
	
	bad_detx	=	np.array([])
	bad_dety	=	np.array([])	
	clipped_h	=	h > flick_threshold
	indices		=	np.where( clipped_h == True )
	bad_detx	=	indices[0]
	bad_dety	=	indices[1]
	bad_times	=	indices[2]
	mid_flick	=	mf.mid_array(edges[2])
	
	badx_flagged	=	np.array([])
	bady_flagged	=	np.array([])
	indices_to_flag	=	np.array([])
	for j, x in enumerate( bad_detx ):
		y	=	bad_dety[j]
		if ( np.any(badx_flagged == x) == True ) and ( np.any(bady_flagged == y) == True ):
			pass
		else:
			t	=	bad_times[j]
			t	=	mid_flick[t]
			
			inds_all_in_that_bin	=	np.where(  ( t-flick_tbin/2 < time ) & ( time < t+flick_tbin/2 )  )[0]
			detx_in_that_bin	=	detx[inds_all_in_that_bin].astype(np.int16)
			dety_in_that_bin	=	dety[inds_all_in_that_bin].astype(np.int16)
			inds_nearby	=	np.where(   (np.absolute(detx_in_that_bin - x) == 1) & (np.absolute(dety_in_that_bin - y) == 1)   )[0]
			if inds_nearby.size > 0:
				nearby_detx	=	detx_in_that_bin[inds_nearby]
				nearby_dety	=	dety_in_that_bin[inds_nearby]
				nearby_unique_detx	=	np.array( [] )
				nearby_unique_dety	=	np.array( [] )
				for k, x_nearby in enumerate( nearby_detx ):
					y_nearby	=	nearby_dety[k]
					if ( np.any(nearby_unique_detx == x_nearby) == True ) and ( np.any(nearby_unique_dety == y_nearby) == True ):
						pass
					else:
						#	print x, y
						#	print x_nearby, nearby_dety[k]
						nearby_unique_detx	=	np.append( nearby_unique_detx, x_nearby )
						nearby_unique_dety	=	np.append( nearby_unique_dety, y_nearby )
						counts_in_this_nearby_pixel	=	np.where( (detx_in_that_bin[inds_nearby] == x_nearby) & (dety_in_that_bin[inds_nearby] == y_nearby) )[0].size
						
						#	print counts_in_this_nearby_pixel
						if counts_in_this_nearby_pixel > flick_threshold:
							#	print x, y, t
							#	print x_nearby, nearby_dety[k]
							#	print counts_in_this_nearby_pixel
							
							ind_1	=	np.where( (detx == x) & (dety == y) & ( t-flick_tbin/2 < time ) & ( time < t+flick_tbin/2 ) )[0]
							indices_to_flag	=	np.append( indices_to_flag, ind_1 )
							ind_2	=	np.where( (detx == x_nearby) & (dety == y_nearby) & ( t-flick_tbin/2 < time ) & ( time < t+flick_tbin/2 ) )[0]
							indices_to_flag	=	np.append( indices_to_flag, ind_2 )
							
							#	print ind_1.size, ind_2.size
							#	print '\n'
							
							if ( np.any(badx_flagged == x) == True ) and ( np.any(bady_flagged == y) == True ):
								pass
							else:
								
								ind_lvt	=	mf.nearest( livetime_mids, t )
								expfracs_after_flickpixclean[ind_lvt]	=	expfracs_after_flickpixclean[ind_lvt] * (  1  -  (flick_tbin/t_bin) * (2/(number_of_pixels_2D_per_Quadrant - total_flagged_pixels))  )
								#	print ind_lvt
								
								#	To calclulate the number of additonally flagged pixels.
								badx_flagged	=	np.append( badx_flagged, np.array( [x, x_nearby] ) )
								bady_flagged	=	np.append( bady_flagged, np.array( [y, y_nearby] ) )
	
	number_of_flickering_pixels	+=	badx_flagged.size
	flickering_instances		+=	np.unique(indices_to_flag).size
	print '\nNumber of flickering pixels:	', number_of_flickering_pixels
	print 'Number of flickering instances:	', flickering_instances, '\n\n'
	
	time, detx, dety, energy, veto	=	flagdata( time, detx, dety, energy, veto, indices_to_flag )
	
	#~ #	To plot the livetime factors before and after flickpixclean.
	#~ plt.plot( expfracs_after_DPHclean, 'bo', label = 'DPHclean' )
	#~ plt.plot( expfracs_after_flickpixclean, 'ro', label = 'flickpixclean' )
	#~ plt.legend( numpoints = 1, loc = 'best' )
	#~ plt.show()
	#~ print expfracs_after_flickpixclean - expfracs_after_DPHclean
		
	##################### FLICKERING PIXEL CLEAN #######################
	
	
	
	
	########################### LIGHTCURVES ############################
	
	## Veto-tagged CZT events
	time_vetotagged, detx_vetotagged, dety_vetotagged, energy_vetotagged, veto_vetotagged	=	get_vetotaggedCZT_data( time, detx, dety, energy, veto )
	
	x, y	=	mf.my_histogram_according_to_given_boundaries( time_vetotagged, t_bin, t_start, t_stop )
	y		=	y / t_bin
	x	=	x[inds_nonzero_lvts]
	y	=	y[inds_nonzero_lvts] / expfracs_after_flickpixclean[inds_nonzero_lvts]
	
	#	print x.size, livetime_mids.size, expfracs_after_flickpixclean.size
	
	#~ plt.title( r'$ \rm{ t_{bin} } = $' + r'$ \, {0:.1f} \, $'.format(t_bin) + r'$ \rm{ s : \quad Q } $' + r'$ {:d} $'.format(i), fontsize = size_font )
	plt.xlabel( r'$ \rm{UT - t_{offset} \; [in \; sec]} $', fontsize = size_font )
	plt.ylabel( r'$ \rm{Counts / sec} $', fontsize = size_font )
	plt.step( x-t_offset, y, color = 'k' )
	plt.savefig( path_save + 'Q{0:d}--vetotaggedCZT--{1:.1f}sec.png'.format(i, t_bin) )
	plt.clf()
	plt.close()
	
	
	
	##	Single events
	time_single, detx_single, dety_single, energy_single, veto_single	=	get_single( time, detx, dety, energy, veto )
	
	x, y	=	mf.my_histogram_according_to_given_boundaries( time_single, t_bin, t_start, t_stop )
	y		=	y / t_bin
	x	=	x[inds_nonzero_lvts]
	y	=	y[inds_nonzero_lvts] / expfracs_after_flickpixclean[inds_nonzero_lvts]
	#~ plt.title( r'$ \rm{ t_{bin} } = $' + r'$ \, {0:.1f} \, $'.format(t_bin) + r'$ \rm{ s : \quad Q } $' + r'$ {:d} $'.format(i), fontsize = size_font )
	plt.xlabel( r'$ \rm{UT - t_{offset} \; [in \; sec]} $', fontsize = size_font )
	plt.ylabel( r'$ \rm{Counts / sec} $', fontsize = size_font )	
	plt.step( x-t_offset, y, color = 'k' )
	plt.savefig( path_save + 'Q{0:d}--sglevts--cleaned_all--{1:.1f}sec.png'.format(i, t_bin) )
	plt.savefig( path_save + 'Q{0:d}--sglevts--cleaned_all--{1:.1f}sec.pdf'.format(i, t_bin) )
	plt.clf()
	plt.close()
	
	cont, contsub	=	mf.subtract_continuum( x, y, T_search )
	contsub			=	contsub + cont.mean()
	
	plt.hist( contsub, bins = 100 )
	plt.savefig( path_save + 'Q{0:d}--sglevts--cleaned_all--{1:.1f}sec_histogram.png'.format(i, t_bin) )
	plt.clf()
	plt.close()
	
	SNR, SNR_error	=	calculate_SNR( x, y, T1, T2 )
	print 'SNR:	{0:.3f}	error:	{1:.3f}'.format(SNR, SNR_error), '\n'
	
	
	#	Single events:	energy binned
	#	0 - 50 keV
	E_low	=	0	;	E_high	=	50
	time_here, detx_here, dety_here, energy_here, veto_here	=	bin_data_into_energies( time_single, detx_single, dety_single, energy_single, veto_single, E_low, E_high )
	x, y	=	mf.my_histogram_according_to_given_boundaries( time_here, t_bin, t_start, t_stop )
	y		=	y / t_bin
	x	=	x[inds_nonzero_lvts]
	y	=	y[inds_nonzero_lvts] / expfracs_after_flickpixclean[inds_nonzero_lvts]
	#~ plt.title( r'$ \rm{ t_{bin} } = $' + r'$ \, {0:.1f} \, $'.format(t_bin) + r'$ \rm{ s : \quad Q } $' + r'$ {:d} $'.format(i), fontsize = size_font )
	plt.xlabel( r'$ \rm{UT - t_{offset} \; [in \; sec]} $', fontsize = size_font )
	plt.ylabel( r'$ \rm{Counts / sec} $', fontsize = size_font )
	plt.step( x-t_offset, y, color = 'k' )
	plt.savefig( path_save + 'Q{0:d}--sglevts--cleaned_{1:d}to{2:d}_keV--{3:.1f}sec.png'.format(i, E_low, E_high, t_bin) )
	plt.clf()
	plt.close()
	#	51 - 100 keV
	E_low	=	51	;	E_high	=	100
	time_here, detx_here, dety_here, energy_here, veto_here	=	bin_data_into_energies( time_single, detx_single, dety_single, energy_single, veto_single, E_low, E_high )
	x, y	=	mf.my_histogram_according_to_given_boundaries( time_here, t_bin, t_start, t_stop )
	y		=	y / t_bin
	x	=	x[inds_nonzero_lvts]
	y	=	y[inds_nonzero_lvts] / expfracs_after_flickpixclean[inds_nonzero_lvts]
	#~ plt.title( r'$ \rm{ t_{bin} } = $' + r'$ \, {0:.1f} \, $'.format(t_bin) + r'$ \rm{ s : \quad Q } $' + r'$ {:d} $'.format(i), fontsize = size_font )
	plt.xlabel( r'$ \rm{UT - t_{offset} \; [in \; sec]} $', fontsize = size_font )
	plt.ylabel( r'$ \rm{Counts / sec} $', fontsize = size_font )
	plt.step( x-t_offset, y, color = 'k' )
	plt.savefig( path_save + 'Q{0:d}--sglevts--cleaned_{1:d}to{2:d}_keV--{3:.1f}sec.png'.format(i, E_low, E_high, t_bin) )
	plt.clf()
	plt.close()
	#	101 - 200 keV
	E_low	=	101	;	E_high	=	200
	time_here, detx_here, dety_here, energy_here, veto_here	=	bin_data_into_energies( time_single, detx_single, dety_single, energy_single, veto_single, E_low, E_high )
	x, y	=	mf.my_histogram_according_to_given_boundaries( time_here, t_bin, t_start, t_stop )
	y		=	y / t_bin
	x	=	x[inds_nonzero_lvts]
	y	=	y[inds_nonzero_lvts] / expfracs_after_flickpixclean[inds_nonzero_lvts]
	#~ plt.title( r'$ \rm{ t_{bin} } = $' + r'$ \, {0:.1f} \, $'.format(t_bin) + r'$ \rm{ s : \quad Q } $' + r'$ {:d} $'.format(i), fontsize = size_font )
	plt.xlabel( r'$ \rm{UT - t_{offset} \; [in \; sec]} $', fontsize = size_font )
	plt.ylabel( r'$ \rm{Counts / sec} $', fontsize = size_font )
	plt.step( x-t_offset, y, color = 'k' )
	plt.savefig( path_save + 'Q{0:d}--sglevts--cleaned_{1:d}to{2:d}_keV--{3:.1f}sec.png'.format(i, E_low, E_high, t_bin) )
	plt.clf()
	plt.close()
	
	
	
	##	Double events
	time_double, detx_double, dety_double, energy_double, veto_double	=	get_double( time, detx, dety, energy, veto )
	
	x, y	=	mf.my_histogram_according_to_given_boundaries( time_double, t_bin, t_start, t_stop )
	y		=	y / t_bin
	x	=	x[inds_nonzero_lvts]
	y	=	y[inds_nonzero_lvts] / expfracs_after_flickpixclean[inds_nonzero_lvts]
	#~ plt.title( r'$ \rm{ t_{bin} } = $' + r'$ \, {0:.1f} \, $'.format(t_bin) + r'$ \rm{ s : \quad Q } $' + r'$ {:d} $'.format(i), fontsize = size_font )
	plt.xlabel( r'$ \rm{UT - t_{offset} \; [in \; sec]} $', fontsize = size_font )
	plt.ylabel( r'$ \rm{Counts / sec} $', fontsize = size_font )	
	plt.step( x-t_offset, y, color = 'k' )
	plt.savefig( path_save + 'Q{0:d}--dblevts--cleaned_all--{1:.1f}sec.png'.format(i, t_bin) )
	plt.savefig( path_save + 'Q{0:d}--dblevts--cleaned_all--{1:.1f}sec.pdf'.format(i, t_bin) )
	plt.clf()
	plt.close()
	
	#~ SNR, SNR_error	=	calculate_SNR( x, y, T1, T2 )
	#~ print 'SNR:	{0:.3f}	error:	{1:.3f}'.format(SNR, SNR_error), '\n'
	
	
	
	##	Compton (rough) double events
	time_double, detx_double, dety_double, energy_double, veto_double	=	get_Comptondouble( time, detx, dety, energy, veto )
	
	x, y	=	mf.my_histogram_according_to_given_boundaries( time_double, t_bin, t_start, t_stop )
	y		=	y / t_bin
	x	=	x[inds_nonzero_lvts]
	y	=	y[inds_nonzero_lvts] / expfracs_after_flickpixclean[inds_nonzero_lvts]	
	#~ plt.title( r'$ \rm{ t_{bin} } = $' + r'$ \, {0:.1f} \, $'.format(t_bin) + r'$ \rm{ s : \quad Q } $' + r'$ {:d} $'.format(i), fontsize = size_font )
	plt.xlabel( r'$ \rm{UT - t_{offset} \; [in \; sec]} $', fontsize = size_font )
	plt.ylabel( r'$ \rm{Counts / sec} $', fontsize = size_font )	
	plt.step( x-t_offset, y, color = 'k' )
	plt.savefig( path_save + 'Q{0:d}--dblevts_roughCompton--cleaned_all--{1:.1f}sec.png'.format(i, t_bin) )
	plt.savefig( path_save + 'Q{0:d}--dblevts_roughCompton--cleaned_all--{1:.1f}sec.pdf'.format(i, t_bin) )
	plt.clf()
	plt.close()
	
	#~ SNR, SNR_error	=	calculate_SNR( x, y, T1, T2 )
	#~ print 'SNR:	{0:.3f}	error:	{1:.3f}'.format(SNR, SNR_error), '\n'
		
	print '..........................................................\n'
	########################### LIGHTCURVES ############################
	
	
	#	print 'Average time interval between cleaned events: ', round( np.diff(time).mean()*1e3, 3), 'millisecond.'
	
	
	print '\n##########################################################'
	print '\n\n\n\n'
dat.close()
hdu.close()

