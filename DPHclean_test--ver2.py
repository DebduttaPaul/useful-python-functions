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
allowable		=	3		#	if the quantity used for flagging is greater than this, the DPH is flagged to have a structure
ID				=	3		#	combo


#	To define parameters related to this source and file structures.

path_read			=	os.getcwd() + '/data/DPHclean/GRB160802A/2000_sec/'

path_save		=	os.getcwd() + '/plots/DPHclean/ver2/GRB160802A/'

GRB_trigger		=	207814409	#	UT, in sec:	GRB160802A

t_start	=	GRB_trigger - 1e3
t_stop	=	GRB_trigger + 1e3
t_offset=	GRB_trigger
#~ mf.mkdir_p( path_save + '/DPHs/not_flagged/' )
#~ mf.mkdir_p( path_save + '/DPHs/flagged/' )
#~ mf.mkdir_p( path_save + '/DPHs/flagged_merged/' )
#~ mf.mkdir_p( path_save + '/DDHs/merged/' )
#~ mf.mkdir_p( path_save + '/histograms/' )


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
print '\n\n--------------------------------------------------------------------\n\n\n\n\n\n\n\n\n'


#######################################################################################################################################################






#######################################################################################################################################################



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


def create_DDH( time, detx, dety ):
	
	
	index	=	np.argsort( time )
	time	=	time[index]
	detx	=	detx[index]
	dety	=	dety[index]
	diff	=	( time - time[0] ) * 1e3	#	delay is calculated in milliseconds.
	
	DDH	=	np.zeros( ( L_x, L_y ) )
	
	for k, x in enumerate( detx ):
		DDH[ (L_y-1)-dety[k] , x ]	= diff[k]
	
	
	return DDH


#######################################################################################################################################################





#######################################################################################################################################################

DPHstructures_energy = {}

for i in range( Q_n ):
	#	To loop over the Quadrants.
	print '##########################################################\n'
	print 'Quadrant {:d}...'.format(i), '\n\n'
	
	Q_data	=	ascii.read( path_read + 'data_before_DPHclean--Q{:d}.txt'.format(i), format = 'fixed_width' )
	time	=	Q_data['Time'].data
	detx	=	Q_data['detx'].data
	dety	=	Q_data['dety'].data
	energy	=	Q_data['Energy'].data
	veto	=	Q_data['Veto'].data
	
	########################### DPHclean ###############################
	print '..........................................................\n'
	print 'DPHclean...', '\n'
	
	x_t, y_c	=	mf.my_histogram_according_to_given_boundaries( time, t_look, t_start, t_stop )
	y_c			=	y_c / t_look
	
	#	To find the n-sigma peaks in the light-curve at defined time-bin.	
	peaks_index, peaks_time, peaks_countrate, peaks_significance	=	mf.detect_peaks( x_t, y_c, T_search, cutoff )
	how_many_outliers	=	peaks_time.size
	print '\nThese many outliers:	',	how_many_outliers
	
	total_flaginds	=	np.array( [] )
	flag_indices	=	np.array( [] )
	times_to_flag	=	np.array( [] )
	check_flagged	=	np.array( [] )
	check_unflagged	=	np.array( [] )
	#	To make dph-investigation of the peaks in the light-curve.
	for j, ind_peaks in enumerate( peaks_index ):
		#	The j-th peak found in the light-curve.
		peaktime	=	peaks_time[j]
		#	To extract the required data around the peak.
		index		=	np.where(    ( (peaktime-t_look/2) < time ) & ( time < (peaktime+t_look/2) )    )[0]
		data_time, data_detx, data_dety, data_energy, data_veto	=	get_data( time, detx, dety, energy, veto, index )
		
		#	To first make a DPH for the chosen bin.
		DPH		=	mf.create_full_DPH( data_detx, data_dety )
		#	To decide whether the data in that region needs to be flagged.
		decision_flag, number_of_points, sum_of_measures, number_of_hot_pairs, cluster_image, bad_detx, bad_dety	=	dc.flag_the_dph( DPH, threshold, allowable, ID )
		parameter_checked	=	sum_of_measures/number_of_points
		
		#	Plotting specifications.
		def applyPlotStyle( mx ):
			bn	=	8
			ax.set_xticks( np.arange(0, L_x, bn) )
			ax.set_xticklabels( np.arange(0, L_x,  bn) )
			ax.set_yticks( np.arange(0, L_y, bn) )
			ax.set_yticklabels( np.arange(L_y, 0, -bn) )
			ax.grid( linewidth = 1.2 )
		
		if decision_flag	==	1:
			#	print 'No. of timestamps, contributing:	', data_time.shape[0]
			check_flagged	=	np.append( check_flagged, parameter_checked )
			total_flaginds	=	np.append( total_flaginds, ind_peaks )
			flag_indices	=	np.append( flag_indices, j )
			
			indices_to_flag	=	np.array( [] )
			for k, x in enumerate( bad_detx ):
				y	=	bad_dety[k]
				
				#	To extract the image near the neighbourhood of the point, the neighbourhood is defined.
				m	=	(L_y - 1) - y
				n	=	x
				x_min	=	int( m - 1 )
				x_max	=	int( m + 1 )
				y_min	=	int( n - 1 )
				y_max	=	int( n + 1 )
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
			data_time, data_detx, data_dety, data_energy, data_veto	=	get_data( data_time, data_detx, data_dety, data_energy, data_veto, indices_to_flag )
			
			times_to_flag	=	np.append( times_to_flag, data_time )
			
			#~ #	To plot the DPH.
			#~ mx	=	int( DPH.max() )
			#~ fig, axes	=	plt.subplots( 1, 2 )
			#~ ax	=	axes[0]
			#~ applyPlotStyle( mx )
			#~ im	=	ax.imshow( DPH, clim = ( 0, mx ), cmap = cm.jet_r )
			#~ ax	=	axes[1]
			#~ applyPlotStyle( mx )
			#~ im = ax.imshow( mf.create_full_DPH(data_detx, data_dety), clim = ( 0, mx ), cmap = cm.jet_r )
			#~ plt.colorbar( im, ticks = range(0, mx+1), ax=axes.ravel().tolist(), orientation = 'horizontal' )
			#~ plt.savefig( path_save + 'DPHs/flagged/' + 'Q{0:d}--peak_{1:d}.png'.format(i, ind_peaks+1) )
			#~ plt.clf()
			#~ plt.close()
			#~ 
			#~ #	To make and plot the DDH.
			#~ DDH		=	create_DDH( data_time, data_detx, data_dety )
			#~ mx	=	int( DDH.max() )
			#~ fig, ax	=	plt.subplots( 1, 1 )
			#~ applyPlotStyle( mx )
			#~ im = ax.imshow( DDH, clim = ( 0, mx ), cmap = cm.jet_r )
			#~ plt.colorbar( im, ticks = range(0, mx+1, 10) )
			#~ plt.xlabel( r'$ \rm{ det_{x} } $', fontsize = size_font )
			#~ plt.ylabel( r'$ \rm{ det_{y} } $', fontsize = size_font )
			#~ plt.savefig( path_save + 'DDHs/' + 'Q{0:d}--peak_{1:d}.png'.format(i, ind_peaks+1) )
			#~ plt.clf()
			#~ plt.close()
			
		
		else:
			if np.isnan( parameter_checked ) == False:
				check_unflagged	=	np.append( check_unflagged, parameter_checked )
			
			#~ #	To plot the DPH.
			#~ mx	=	int( DPH.max() )
			#~ fig, axes	=	plt.subplots( 1, 2 )
			#~ ax	=	axes[0]
			#~ applyPlotStyle( mx )
			#~ im	=	ax.imshow( DPH, clim = ( 0, mx ), cmap = cm.jet_r )
			#~ ax	=	axes[1]
			#~ applyPlotStyle( mx )
			#~ im = ax.imshow( mf.create_full_DPH(bad_detx, bad_dety), clim = ( 0, mx ), cmap = cm.jet_r )
			#~ plt.colorbar( im, ticks = range(0, mx+1), ax=axes.ravel().tolist(), orientation = 'horizontal' )
			#~ plt.savefig( path_save + 'DPHs/not_flagged/' + 'Q{0:d}--peak_{1:d}.png'.format(i, ind_peaks+1) )
			#~ plt.clf()
			#~ plt.close()
	
	
	#	To make DPHs of data for which structures are seen over consecutive time-bins.
	total_flaginds	=	total_flaginds.astype(int)
	
	
	repeats	=	np.where( np.diff(total_flaginds) == 1 )[0]
	if repeats.size != 0:
		for l, r in enumerate(repeats[:-1]):
			
			
			if total_flaginds[r+2] - total_flaginds[r] == 2:
				indices_to_club	=	total_flaginds[ r : r+3 ]
				#	print 'indices to club:	', indices_to_club
				
				total_time	=	np.array( [] )
				total_detx	=	np.array( [] )
				total_dety	=	np.array( [] )
				for j in indices_to_club:
					#	The j-th bin in the light-curve.
					peaktime	=	x_t[j]
					#	To extract the required data around the peak.
					index		=	np.where(    ( (peaktime-t_look/2) < time ) & ( time < (peaktime+t_look/2) )    )[0]
					data_time, data_detx, data_dety, data_energy, data_veto	=	get_data( time, detx, dety, energy, veto, index )
					
					#	To first make a DPH for the chosen bin.
					DPH		=	mf.create_full_DPH( data_detx, data_dety )
					decision_flag, number_of_points, sum_of_measures, number_of_hot_pairs, cluster_image, bad_detx, bad_dety	=	dc.flag_the_dph( DPH, threshold, allowable, ID )
					
					indices_to_flag	=	np.array( [] )
					indices_to_flag	=	np.array( [] )
					for k, x in enumerate( bad_detx ):
						y	=	bad_dety[k]
				
						#	To extract the image near the neighbourhood of the point, the neighbourhood is defined.
						m	=	(L_y - 1) - y
						n	=	x
						x_min	=	int( m - 1 )
						x_max	=	int( m + 1 )
						y_min	=	int( n - 1 )
						y_max	=	int( n + 1 )
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
					data_time, data_detx, data_dety, data_energy, data_veto	=	get_data( data_time, data_detx, data_dety, data_energy, data_veto, indices_to_flag )
					total_time	=	np.append( total_time, data_time )
					total_detx	=	np.append( total_detx, data_detx )	;	total_detx	=	total_detx.astype(int)
					total_dety	=	np.append( total_dety, data_dety )	;	total_dety	=	total_dety.astype(int)
					
				#~ #	To make and plot the DPH.
				#~ DPH		=	mf.create_full_DPH( total_detx, total_dety )
				#~ mx	=	int( DPH.max() )
				#~ fig, ax	=	plt.subplots( 1, 1 )
				#~ applyPlotStyle( mx )
				#~ im	=	ax.imshow( DPH, clim = ( 0, mx ), cmap = cm.jet_r )
				#~ plt.colorbar( im, ticks = range(0, mx+1) )
				#~ plt.xlabel( r'$ \rm{ det_{x} } $', fontsize = size_font )
				#~ plt.ylabel( r'$ \rm{ det_{y} } $', fontsize = size_font )
				#~ plt.savefig( path_save + 'DPHs/flagged_merged/' + 'Q{0:d}--merged_{1:d}+{2:d}+{3:d}.png'.format( i, total_flaginds[r+1], total_flaginds[r+1]+1, total_flaginds[r+1]+2 ) )
				#~ plt.clf()
				#~ plt.close()
				#~ 
				#~ #	To make and plot the DDH.
				#~ DDH		=	create_DDH( total_time, total_detx, total_dety )
				#~ mx	=	int( DDH.max() )
				#~ fig, ax	=	plt.subplots( 1, 1 )
				#~ applyPlotStyle( mx )
				#~ im = ax.imshow( DDH, clim = ( 0, mx ), cmap = cm.jet_r )
				#~ plt.colorbar( im, ticks = range(0, mx+1, 10) )
				#~ plt.xlabel( r'$ \rm{ det_{x} } $', fontsize = size_font )
				#~ plt.ylabel( r'$ \rm{ det_{y} } $', fontsize = size_font )
				#~ plt.savefig( path_save + 'DDHs/merged/' + 'Q{0:d}--merged_{1:d}+{2:d}+{3:d}.png'.format( i, total_flaginds[r+1], total_flaginds[r+1]+1, total_flaginds[r+1]+2 ) )
				#~ plt.clf()
				#~ plt.close()
			
			
			else:
				indices_to_club	=	total_flaginds[ r : r+2 ]
				#	print 'indices to club:	', indices_to_club
				
				total_time	=	np.array( [] )
				total_detx	=	np.array( [] )
				total_dety	=	np.array( [] )
				for j in indices_to_club:
					#	The j-th bin in the light-curve.
					peaktime	=	x_t[j]
					#	To extract the required data around the peak.
					index		=	np.where(    ( (peaktime-t_look/2) < time ) & ( time < (peaktime+t_look/2) )    )[0]
					data_time, data_detx, data_dety, data_energy, data_veto	=	get_data( time, detx, dety, energy, veto, index )
					
					#	To first make a DPH for the chosen bin.
					DPH		=	mf.create_full_DPH( data_detx, data_dety )
					decision_flag, number_of_points, sum_of_measures, number_of_hot_pairs, cluster_image, bad_detx, bad_dety	=	dc.flag_the_dph( DPH, threshold, allowable, ID )
					
					indices_to_flag	=	np.array( [] )
					indices_to_flag	=	np.array( [] )
					for k, x in enumerate( bad_detx ):
						y	=	bad_dety[k]
				
						#	To extract the image near the neighbourhood of the point, the neighbourhood is defined.
						m	=	(L_y - 1) - y
						n	=	x
						x_min	=	int( m - 1 )
						x_max	=	int( m + 1 )
						y_min	=	int( n - 1 )
						y_max	=	int( n + 1 )
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
					data_time, data_detx, data_dety, data_energy, data_veto	=	get_data( data_time, data_detx, data_dety, data_energy, data_veto, indices_to_flag )
					total_time	=	np.append( total_time, data_time )
					total_detx	=	np.append( total_detx, data_detx )
					total_dety	=	np.append( total_dety, data_dety )
					
				#~ #	To make and plot the DPH.
				#~ DPH		=	mf.create_full_DPH( total_detx, total_dety )
				#~ mx	=	int( DPH.max() )
				#~ fig, ax	=	plt.subplots( 1, 1 )
				#~ applyPlotStyle( mx )
				#~ im	=	ax.imshow( DPH, clim = ( 0, mx ), cmap = cm.jet_r )
				#~ plt.colorbar( im, ticks = range(0, mx+1) )
				#~ plt.xlabel( r'$ \rm{ det_{x} } $', fontsize = size_font )
				#~ plt.ylabel( r'$ \rm{ det_{y} } $', fontsize = size_font )
				#~ plt.savefig( path_save + 'DPHs/flagged_merged/' + 'Q{0:d}--merged_{1:d}+{2:d}.png'.format( i, total_flaginds[r+1], total_flaginds[r+1]+1 ) )
				#~ plt.clf()
				#~ plt.close()
				#~ 
				#~ #	To make and plot the DDH.
				#~ DDH		=	create_DDH( total_time, total_detx, total_dety )
				#~ mx	=	int( DDH.max() )
				#~ fig, ax	=	plt.subplots( 1, 1 )
				#~ applyPlotStyle( mx )
				#~ im = ax.imshow( DDH, clim = ( 0, mx ), cmap = cm.jet_r )
				#~ plt.colorbar( im, ticks = range(0, mx+1, 10) )
				#~ plt.xlabel( r'$ \rm{ det_{x} } $', fontsize = size_font )
				#~ plt.ylabel( r'$ \rm{ det_{y} } $', fontsize = size_font )
				#~ plt.savefig( path_save + 'DDHs/merged/' + 'Q{0:d}--merged_{1:d}+{2:d}.png'.format( i, total_flaginds[r+1], total_flaginds[r+1]+1 ) )
				#~ plt.clf()
				#~ plt.close()
	
	
	print 'These many flagged:	',	flag_indices.size, '\n\n'
	
	
	#~ plt.hist( check_flagged, bins = 50, color = 'b' )
	#~ plt.savefig( path_save + 'histograms/Q{:d}--flagged.png'.format(i) )
	#~ plt.savefig( path_save + 'histograms/Q{:d}--flagged.pdf'.format(i) )
	#~ plt.clf()
	#~ plt.close()
	#~ 
	#~ plt.hist( check_unflagged, bins = 100, color = 'r' )
	#~ plt.savefig( path_save + 'histograms/Q{:d}--not_flagged.png'.format(i) )
	#~ plt.savefig( path_save + 'histograms/Q{:d}--not_flagged.pdf'.format(i) )
	#~ plt.clf()
	#~ plt.close()
	
	
	
	total_flag_indices	=	np.array( [] )
	for k, timestamp in enumerate(times_to_flag):
		ind = np.where( time == timestamp )[0]
		total_flag_indices = np.append( total_flag_indices, ind )
	total_flag_indices	=	np.unique( total_flag_indices )
	total_flag_indices	=	total_flag_indices.astype(int)
	
	Time, DetX, DetY, Energy, Veto	=	get_data( time, detx, dety, energy, veto, total_flag_indices )	# The data in the DPHstructures.	
	time, detx, dety, energy, veto	=	flagdata( time, detx, dety, energy, veto, total_flag_indices )	# Data cleaned of DPHstructures.
	
	DPHstructures_energy[i]	=	Energy
	
	print '..........................................................\n'
	########################### DPHclean ###############################	
	
	
	
	print '\n##########################################################'
	print '\n\n\n\n'

energy_bin = 10 ; energy_start = 0; energy_stop = 200	# in keV
total_DpHstructures_spectrum = np.zeros(  int( (energy_stop-energy_start)/energy_bin )  )
for i in range(Q_n):
	x, y = mf.my_histogram_according_to_given_boundaries( DPHstructures_energy[i], energy_bin, energy_start, energy_stop )
	total_DpHstructures_spectrum += y
	y = y / y.sum()	
	plt.plot( x, y, label = 'Q{0:d}'.format(i), lw = 2 )
plt.legend()
plt.xlabel( r'$ \rm{ Energy \; [keV] } $', fontsize = size_font )
plt.savefig( path_save + 'spectra/DPHstructures_spectrum--Qwise.png' )
plt.savefig( path_save + 'spectra/DPHstructures_spectrum--Qwise.pdf' )
plt.clf()
plt.close()

total_DpHstructures_spectrum = total_DpHstructures_spectrum / total_DpHstructures_spectrum.sum()
plt.plot( x, total_DpHstructures_spectrum, color = 'k', lw = 2 )
plt.xlabel( r'$ \rm{ Energy \; [keV] } $', fontsize = size_font )
plt.savefig( path_save + 'spectra/DPHstructures_spectrum--total.png' )
plt.savefig( path_save + 'spectra/DPHstructures_spectrum--total.pdf' )
plt.clf()
plt.close()


ascii.write( Table( [x, total_DpHstructures_spectrum], names = ['energy [keV]', 'normalized spectrum'] ), path_save + 'spectra/DPHstructures_spectrum--total.txt', format = 'fixed_width' )
