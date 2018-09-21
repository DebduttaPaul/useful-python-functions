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
t3				=	50*1e-6 #	in sec, to flag data post-bunch for this time-scale
gross_cutoff	=	5		#	"n" for n-sigma outlier flagging at the DPH level, i.e. gross noisy pixels
t_look			=	1e-1	#	in sec, the counts are binnned at this time-scale to search for clusters in the DPH
T_search		=	5e2		#	in sec, the de-trending time-scale
cutoff			=	2		#	"n" for n-sigma outlier search in the de-trended light-curve, whose DPHs are probed for structures
threshold		=	0.7		#	the most conservative case
allowable		=	10		#	if the quantity used for flagging is greater than this, the DPH is flagged to have a structure
ID				=	3		#	combo
flick_tbin		=	1e-1	#	in sec, time-scale for finding flickering pixels
flick_threshold	=	6		#	if counts from a pixel in "flick_tbin" is greater than this, that pixel is flagged for the entire time


#	To define parameters related to this source and file structures.

path_read			=	os.getcwd() + '/data/'


bunch_filename	=	path_read + 'AS1G05_233T10_9000000570_04576cztM0_level2_bunch.fits'		#	GRB160802A
event_filename	=	path_read + 'AS1G05_233T10_9000000570_04576cztM0_level2.fits'			#	GRB160802A
path_save		=	os.getcwd() + '/plots/new/GRB160802A/bunchclean_new++/'
GRB_trigger		=	207814409	#	UT, in sec:	GRB160802A
T1	=	GRB_trigger - 0.3		#	UT, in sec:	GRB160802A
T2	=	GRB_trigger + 20.224	#	UT, in sec:	GRB160802A

#~ bunch_filename	=	path_read + 'AS1G05_009T02_9000000618_04866cztM0_level2_bunch.fits'		#	GRB160821A
#~ event_filename	=	path_read + 'AS1G05_009T02_9000000618_04866cztM0_level2.fits'			#	GRB160821A
#~ path_save		=	os.getcwd() + '/plots/new/GRB160821A/bunchclean_new++/'
#~ GRB_trigger		=	209507670	#	UT, in sec:	GRB160821A
#~ T1	=	GRB_trigger - 4.1		#	UT, in sec:	GRB160821A
#~ T2	=	GRB_trigger + 208.328	#	UT, in sec:	GRB160821A

t_start	=	T1 - 400
t_stop	=	T2 + 400
t_offset=	GRB_trigger

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


def flagdata( time, detx, dety, PHA, indices_to_flag ):
	
	time	=	np.delete( time, indices_to_flag )
	detx	=	np.delete( detx, indices_to_flag )
	dety	=	np.delete( dety, indices_to_flag )
	PHA		=	np.delete( PHA,  indices_to_flag )
		
	return time, detx, dety, PHA


#	Grossnoisy pixels flagging.
def grossnoisy( time, detx, dety, gross_cutoff ):

	full_DPH	=	mf.create_full_DPH( detx, dety )
	
	#	To check how many of the pixels are flagged by the onboard software.
	#~ initially_flagged	=	len( np.where( full_DPH == 0 )[0] )
	initially_masked	=	np.where( full_DPH == 0 )
	initially_flagged	=	np.shape(initially_masked)[1]
	initially_masked_x, initially_masked_y	=	reverse_mapping( initially_masked[0], initially_masked[1] )
	print '\nAlready flagged pixels:		' ,  initially_flagged
	
	#	To mask the pixels with zero counts before flagging, for subsequent analysis.
	full_DPH	=	ma.masked_equal( full_DPH, 0 )
	
	#	To apply the iterative 5-sigma clipping based on the DPH.
	clipped_DPH	=	sigma_clip( data = full_DPH, sigma_lower = 100, sigma_upper = gross_cutoff )
	indices_flagged	=	np.where( ma.getmask( clipped_DPH ) == True )
	total_flagged	=	indices_flagged[0].shape[0]
	print 'Number of flagged pixels:	', total_flagged - initially_flagged, '\n\n'
	detx_flag, dety_flag	=	reverse_mapping( indices_flagged[0], indices_flagged[1] )
	
	#	To make and plot lightcurves of the gross noisy pixels.
	sample = [ detx, dety, time ]
	h, edges	=	np.histogramdd(  sample, bins = ( np.arange( -0.5, L_x+0.5 ), np.arange ( -0.5, L_y+0.5 ), np.arange(time[0], time[-1], 1e0) )  )
	
	
	for k, x in enumerate( detx_flag ):
		y	=	dety_flag[k]
		index	=	np.where( (initially_masked_x == x) & (initially_masked_y == y) )
		test	=	np.shape(index)[1]
		if test == 0:
			plt.title( r'$ {0:d} : \; {1:d} , \; {2:d} $'.format(i, x, y), fontsize = size_font )
			plt.plot( mf.mid_array(edges[2])-GRB_trigger, h[x, y] )
			plt.show()
	
	
	indices_to_flag	=	np.array( [] )
	for j, x in enumerate( detx_flag ):
		y	=	dety_flag[j]
		ind	=	np.where( (detx == x) & (dety==y) )[0]
		indices_to_flag	=	np.append( indices_to_flag, ind )
	
	
	return indices_to_flag


#######################################################################################################################################################





#######################################################################################################################################################



dat	=	fits.open( event_filename )
hdu	=	fits.open( bunch_filename )
for i in range( Q_n ):
	#	To loop over the Quadrants.
	print '##########################################################\n'
	print 'Quadrant {:d}...'.format(i), '\n\n'
	
	############################## START ###############################
	
	#	To extract the data.
	time	=	dat[i+1].data['TIME']
	detx	=	dat[i+1].data['DETX']
	dety	=	dat[i+1].data['DETY']
	PHA		=	dat[i+1].data['PHA']
	
	#	To extract the bunch data.
	bunch_2nds		=	hdu[i+1].data['Time']
	bunch_starts	=	hdu[i+1].data['Time_dfs']
	bunch_ends		=	hdu[i+1].data['Time_dsl']
	bunch_lengths	=	hdu[i+1].data['NumEvent']
	bunch_starts	=	bunch_2nds - time_res * bunch_starts
	bunch_ends		=	bunch_2nds + time_res * bunch_ends
	
	#	To slice off data between t_start and t_stop.
	index_cut		=	np.where( (t_start < bunch_starts) & (bunch_starts < t_stop) )[0]
	bunch_starts	=	bunch_starts[index_cut]
	bunch_2nds		=	bunch_2nds[index_cut]
	bunch_ends		=	bunch_ends[index_cut]
	bunch_lengths	=	bunch_lengths[index_cut]
	index_cut		=	np.where( (t_start-1 < time) & (time < t_stop+1) )[0]
	time			=	time[index_cut]
	detx			=	detx[index_cut]
	dety			=	dety[index_cut]
	PHA				=	PHA[index_cut]
	
	############################## START ###############################
	
	
	
	
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
	
	#	To flag the data beween the bunches and post-bunch.
	indices_to_flag	=	np.array([])
	for j, end in enumerate( bunch_ends ):
		#	To loop over the bunches.
		start	=	bunch_starts[j]
		
		inds	=	np.where( (start <= time) & (time <= end+t3) )[0]
		indices_to_flag	=	np.append( indices_to_flag, inds )
	time, detx, dety, PHA	=	flagdata( time, detx, dety, PHA, indices_to_flag )
	
	print '..........................................................\n'
	########################### BUNCHCLEAN #############################
	
	
	
	
	###################### GROSS NOISY PIXEL CLEAN #####################
	print '..........................................................\n'
	print 'Gross noisy pixels...', '\n'
	
	indices_to_flag	=	grossnoisy( time, detx, dety, gross_cutoff )
	time, detx, dety, PHA	=	flagdata( time, detx, dety, PHA, indices_to_flag )
	
	print '..........................................................\n'
	###################### GROSS NOISY PIXEL CLEAN #####################
	
	
	print '\n##########################################################'
	print '\n\n\n\n'
dat.close()
hdu.close()

