from __future__ import division
from astropy.io import fits, ascii
from astropy.table import Table
import os
import time		as 	t
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





#######################################################################################################################################################
#	To define parameters.

t_max			=	02*1e-3	#	in sec, the length of time for which co-added LCs are to be made and plotted
t2				=	60*1e-6 #	in sec, to club consecutive bunches if the interval between them is less than this amount of time
t3				=	60*1e-6 #	in sec, to flag data post-bunch for this time-scale
bunch_threshold	=	10		#	flag all data post bunch only if the bunch length is greater than this, otherwise flag data only from the affected modules

path_read			=	os.getcwd() + '/data/'


#~ bunch_filename		=	path_read + 'AS1G05_237T02_9000000422cztM0_level2_bunch.fits'	#	BKG
#~ event_filename		=	path_read + 'AS1G05_237T02_9000000422cztM0_level2.fits'			#	BKG
#~ #	path_save	=	os.getcwd() + '/plots/bunch_re-analysis/BKG/'
#~ path_save	=	os.getcwd() + '/plots/bunch_re-analysis/BKG--1st,_2nd_and_3rd_orbits/'
#~ t_start	=	198773802+1			#	UT, in sec:	1st orbit starts here
#~ #	t_stop	=	t_start + 4.1*1e3	#	covers first orbit
#~ #	t_stop	=	t_start + 1.1*1e4	#	covers first 2 orbits
#~ t_stop	=	t_start + 1.7*1e4	#	covers first 3 orbits
#~ t_offset=	t_start



bunch_filename	=	path_read + 'AS1G05_233T10_9000000570_04576cztM0_level2_bunch.fits'		#	GRB160802A
event_filename	=	path_read + 'AS1G05_233T10_9000000570_04576cztM0_level2.fits'			#	GRB160802A
path_save		=	os.getcwd() + '/plots/bunch_re-analysis/GRB160802A/'
GRB_trigger		=	207814409	#	UT, in sec:	GRB160802A
T1	=	GRB_trigger - 0.3		#	UT, in sec:	GRB160802A
T2	=	GRB_trigger + 20.224	#	UT, in sec:	GRB160802A
t_start	=	T1 - 3.2*1e3
t_stop	=	T2 + 1.2*1e3
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
print 't_max:			{:.1f} millisec'.format(t_max*1e3)
print 't2:			{:.1f} microsec'.format(t2*1e6)
print 't3:			{:.1f} microsec'.format(t3*1e6)
print 'bunch_threshold:	{:d}'.format(bunch_threshold)
print '\n\n--------------------------------------------------------------------\n\n\n\n\n\n\n\n\n'


#######################################################################################################################################################






#######################################################################################################################################################



def flagdata( time, detx, dety, DetID, indices_to_flag ):
	
	time	=	np.delete( time, indices_to_flag )
	detx	=	np.delete( detx, indices_to_flag )
	dety	=	np.delete( dety, indices_to_flag )
	DetID	=	np.delete( DetID,indices_to_flag )
		
	return time, detx, dety, DetID



#######################################################################################################################################################






#######################################################################################################################################################


dat	=	fits.open( event_filename )
hdu	=	fits.open( bunch_filename )
#~ for i in range( Q_n ):
for i in [1]:
	#	To loop over the Quadrants.
	print '##########################################################\n'
	print 'Quadrant {:d}...'.format(i), '\n\n'
	
	############################## START ###############################
	
	#	To extract the data.
	time	=	dat[i+1].data['TIME']
	detx	=	dat[i+1].data['DETX']
	dety	=	dat[i+1].data['DETY']
	DetID	=	dat[i+1].data['DetID']
	
	#	To extract the bunch data.	
	bunch_2nds		=	hdu[i+1].data['Time']
	bunch_starts	=	hdu[i+1].data['Time_dfs']
	bunch_ends		=	hdu[i+1].data['Time_dsl']
	bunch_lengths	=	hdu[i+1].data['NumEvent']
	DetId1			=	hdu[i+1].data['DetId1']
	DetId2			=	hdu[i+1].data['DetId2']
	DetId3			=	hdu[i+1].data['DetId3']
	DetId4			=	hdu[i+1].data['DetId4']
	bunch_starts	=	bunch_2nds - time_res * bunch_starts
	bunch_ends		=	bunch_2nds + time_res * bunch_ends
	
	#	To slice off data between t_start and t_stop.
	index_cut		=	np.where( (t_start < bunch_starts) & (bunch_starts < t_stop) )[0]
	bunch_starts	=	bunch_starts[index_cut]
	bunch_2nds		=	bunch_2nds[index_cut]
	bunch_ends		=	bunch_ends[index_cut]
	bunch_lengths	=	bunch_lengths[index_cut]
	DetId1			=	DetId1[index_cut]
	DetId2			=	DetId2[index_cut]
	DetId3			=	DetId3[index_cut]
	DetId4			=	DetId4[index_cut]
	index_cut		=	np.where( (t_start-1 < time) & (time < t_stop+1) )[0]
	time			=	time[index_cut]
	detx			=	detx[index_cut]
	dety			=	dety[index_cut]
	DetID			=	DetID[index_cut]
	
	############################## START ###############################
	
	
	
	
	####################### CO-ADDED LIGHTCURVES #######################
	lc	=	np.zeros( int(t_max // time_res) )
	lt	=	np.zeros( int(t_max // time_res) )
	
	start_time	=	t.time()
	for j, end in enumerate( bunch_ends[:-1] ):
		#	To loop over the bunches.
		start	=	bunch_starts[j+1]
		if ( start - end > 0 ):
			times, []	=	mf.window( end, start, time, [] )
			if times.size > 0:
				#	To make the light-curve from the data within the bunches.
				x, y	=	mf.my_histogram_according_to_given_boundaries( times-end, time_res, 0, t_max )
				lc	=	lc + y
				
				ind	=	np.where( y > 0 )[0]
				z	=	np.zeros( int(t_max // time_res) )
				z[ind]	=	1
				lt		=	lt + z
	print '\n{:.3f} mins elapsed to make co-added LCs from raw data.'.format( (t.time() - start_time) / 60 )
	
	non_zero	=	np.where( lc != 0 )[0]
	plt.xlabel( r'$ \rm{ \tau \; [20 \, \mu s] } $', fontsize = size_font )
	plt.ylabel( r'$ \rm{ Coadded \; Data } $', fontsize = size_font )
	plt.xticks( range(10, 100, 10) )
	plt.plot( x[non_zero]/time_res, lc[non_zero]/lt[non_zero], color = 'k' )
	plt.savefig( path_save + 'Q{:d}--LC--raw_data.png'.format(i) )
	plt.savefig( path_save + 'Q{:d}--LC--raw_data.pdf'.format(i) )
	plt.clf()
	plt.close()
	
	ascii.write( Table( [x[non_zero]/time_res, lc[non_zero], lt[non_zero]], names = ['time [in 20us]', 'co-added data', 'exposure'] ), path_save + 'co-added_data--before_bunchclean.txt', format = 'fixed_width', overwrite = True )
	
	####################### CO-ADDED LIGHTCURVES #######################
	
	
	
	
	######################## BUNCH REDEFINITION ########################
	
	
	#	Histogram of bunches.
	time_interval_between_bunches	=	bunch_starts[1:] - bunch_ends[:-1]
	ind_smalls	=	np.where( time_interval_between_bunches <= t_max )[0]
	timeint_btwn_bunches	=	time_interval_between_bunches[ind_smalls] / time_res
	#~ hist_x, hist_y	=	mf.my_histogram_according_to_given_boundaries( timeint_btwn_bunches, bin_size, bin_start, bin_stop )
	
	#~ plt.title( r'$ \rm{ Histogram : \quad Q } $' + r'$ {:d} $'.format(i), fontsize = size_font )
	plt.xlabel( r'$ \rm{ \Delta T \; [20 \, \mu s] } $', fontsize = size_font )
	plt.xticks( range(10, 100, 10) )
	plt.hist( timeint_btwn_bunches, bins = int(2e2), color = 'k' )
	plt.savefig( path_save  + 'Q{:d}--Interval_between_Bunches_Histogram--raw_data.png'.format(i) )
	plt.savefig( path_save  + 'Q{:d}--Interval_between_Bunches_Histogram--raw_data.pdf'.format(i) )
	plt.clf()
	plt.close()
	
	##	To redefine bunches.
	time_interval_between_bunches	=	bunch_starts[1:] - bunch_ends[:-1]
	ind_bunch		=	np.where( ( time_interval_between_bunches < t2 ) )[0]
	bunch_starts	=	np.delete( bunch_starts, ind_bunch+1 )
	bunch_ends		=	np.delete( bunch_ends, ind_bunch )
	bunch_2nds		=	np.delete( bunch_2nds, ind_bunch+1 )
	DetId1			=	np.delete( DetId1, ind_bunch+1 )
	DetId2			=	np.delete( DetId2, ind_bunch+1 )
	DetId3			=	np.delete( DetId3, ind_bunch+1 )
	DetId4			=	np.delete( DetId4, ind_bunch+1 )
	bunch_lengths[ind_bunch]=	bunch_lengths[ind_bunch] + bunch_lengths[ind_bunch+1]
	bunch_lengths			=	np.delete( bunch_lengths, ind_bunch+1 )
	
	#	Histogram of bunches.
	time_interval_between_bunches	=	bunch_starts[1:] - bunch_ends[:-1]
	ind_smalls	=	np.where( time_interval_between_bunches <= t_max )[0]
	timeint_btwn_bunches	=	time_interval_between_bunches[ind_smalls] / time_res
	#~ hist_x, hist_y	=	mf.my_histogram_according_to_given_boundaries( timeint_btwn_bunches, bin_size, bin_start, bin_stop )
	
	#~ plt.title( r'$ \rm{ Histogram : \quad Q } $' + r'$ {:d} $'.format(i), fontsize = size_font )
	plt.xlabel( r'$ \rm{ \Delta T \; [20 \, \mu s] } $', fontsize = size_font )
	plt.xticks( range(10, 100, 10) )
	plt.hist( timeint_btwn_bunches, bins = int(2e2), color = 'k' )
	plt.savefig( path_save  + 'Q{:d}--Interval_between_Bunches_Histogram--after_bunch_redefinition.png'.format(i) )
	plt.savefig( path_save  + 'Q{:d}--Interval_between_Bunches_Histogram--after_bunch_redefinition.pdf'.format(i) )
	plt.clf()
	plt.close()
	
	######################## BUNCH REDEFINITION ########################
	
	
	
	
	
	
	
	
	
	########################### BUNCHCLEAN #############################
	
	#	#	To plot the light-curve of the number of bunches.
	#	plt.hist( bunch_starts - t_offset, bins = 5e2 )
	#	plt.hist( time - t_offset, bins = 5e2 )
	#	plt.show()
	
	L	=	len(bunch_starts)
	print 'Total number of bunches:	', L
	
	##	New bunchclean.
	start_time	=	t.time()
	indices_to_flag	=	np.array([])
	#	To flag the data beween the bunches and post-bunch.
	for j, end in enumerate( bunch_ends ):
		#	To loop over the bunches.
		start	=	bunch_starts[j]
		
		inds	=	np.where( (start <= time) & (time <= end+t3) )[0]
		indices_to_flag	=	np.append( indices_to_flag, inds )
	time, detx, dety, DetID	=	flagdata( time, detx, dety, DetID, indices_to_flag )
	print '\n{:.3f} mins elapsed to flag the data between and post bunches.'.format( (t.time() - start_time) / 60 )	
	
	
	
	
	
	#~ ##New bunchclean, modified.
	#~ start_time	=	t.time()
	#~ #	To flag the data beween the bunches.
	#~ indices_to_flag	=	np.array([])
	#~ for j, end in enumerate( bunch_ends ):
		#~ #	To loop over the bunches.
		#~ start	=	bunch_starts[j]
		#~ 
		#~ inds	=	np.where( (start <= time) & (time <= end) )[0]
		#~ indices_to_flag	=	np.append( indices_to_flag, inds )
	#~ print '\n{:.3f} mins elapsed to flag the data between the bunches.'.format( (t.time() - start_time) / 60 )
	#~ time, detx, dety, DetID	=	flagdata( time, detx, dety, DetID, indices_to_flag )
	#~ 
	#~ #	To flag the data post bunches.
	#~ indices_to_flag	=	np.array([])
	#~ start_time	=	t.time()
	#~ #	To flag the data post each bunch for t3.
	#~ for j, end in enumerate( bunch_ends ):
		#~ #	To loop over the bunches.
		#~ length	=	bunch_lengths[j]
		#~ 
		#~ if length > bunch_threshold:
			#~ index	=	np.where( (end <= time) & (time <= end+t3) )[0]
			#~ if index.size != 0:
				#~ inds	=	index
		#~ 
		#~ else:
			#~ index	=	np.where( (end <= time) & (time <= end+t3) )[0]
			#~ 
			#~ if index.size != 0:				
				#~ Times	=	time[index]
				#~ ModIDs	=	DetID[index]
				#~ 
				#~ detIDs	=	np.unique( np.array([ DetId1[j], DetId2[j], DetId3[j], DetId4[j] ]) )
				#~ 
				#~ #	print '\n'
				#~ #	print ModIDs
				#~ #	print detIDs
				#~ #	print '\n'
				#~ 
				#~ indices	=	np.array([])
				#~ for detID in detIDs:
					#~ #	To loop over all the Detector Modules which have been affected.
					#~ ind	=	np.where( ModIDs == detID )[0]
					#~ if ind.size != 0:
						#~ indices	=	np.append( indices, ind )
				#~ if indices.size != 0:
					#~ indices		=	indices.astype(int)
					#~ times_to_flag	=	Times[indices]
					#~ ModIDs_to_flag	=	ModIDs[indices]
					#~ 
					#~ mask	=	np.in1d( time, times_to_flag )
					#~ inds	=	np.where( mask == True )
					#~ 
					#~ 
		#~ indices_to_flag	=	np.append( indices_to_flag, inds )
	#~ 
	#~ time, detx, dety, DetID	=	flagdata( time, detx, dety, DetID, indices_to_flag )
	#~ print '{:.3f} mins elapsed to flag the data post bunches.'.format( (t.time() - start_time) / 60 )
	
	
	########################### BUNCHCLEAN #############################
	
	
	
	####################### CO-ADDED LIGHTCURVES #######################
	lc	=	np.zeros( int(t_max // time_res) )
	lt	=	np.zeros( int(t_max // time_res) )
	
	start_time	=	t.time()
	for j, end in enumerate( bunch_ends[:-1] ):
		#	To loop over the bunches.
		start	=	bunch_starts[j+1]
		if ( start - end > 0 ):
			times, []	=	mf.window( end, start, time, [] )
			if len(times) > 0:
				#	To make the light-curve from the data within the bunches.
				x, y	=	mf.my_histogram_according_to_given_boundaries( times-end, time_res, 0, t_max )
				lc	=	lc + y
				
				ind	=	np.where( y > 0 )[0]
				z	=	np.zeros( int(t_max // time_res) )
				z[ind]	=	1
				lt		=	lt + z
	print '\n{:.3f} mins elapsed to make co-added LC after new bunchclean.'.format( (t.time() - start_time) / 60 )
	#	print '\n{:.3f} mins elapsed to make co-added LC after new, modified, bunchclean.'.format( (t.time() - start_time) / 60 )
	
	non_zero	=	np.where( lc != 0 )[0]
	plt.xlabel( r'$ \rm{ \tau \; [20 \, \mu s] } $', fontsize = size_font )	
	plt.ylabel( r'$ \rm{ Coadded \; Data } $', fontsize = size_font )
	plt.xticks( range(10, 100, 10) )
	plt.plot( x[non_zero]/time_res, lc[non_zero]/lt[non_zero], color = 'k' )
	plt.savefig( path_save + 'Q{:d}--LC--after_new_bunchclean.png'.format(i) )
	plt.savefig( path_save + 'Q{:d}--LC--after_new_bunchclean.pdf'.format(i) )
	#	plt.savefig( path_save + 'Q{:d}--LC--after_new_bunchclean_modified.png'.format(i) )
	plt.clf()
	plt.close()
		
	ascii.write( Table( [x[non_zero]/time_res, lc[non_zero], lt[non_zero]], names = ['time [in 20us]', 'co-added data', 'exposure'] ), path_save + 'co-added_data--after_bunchclean.txt', format = 'fixed_width', overwrite = True )
	
	####################### CO-ADDED LIGHTCURVES #######################
	
	
	print '\n##########################################################'
	print '\n\n\n\n'
dat.close()
hdu.close()
