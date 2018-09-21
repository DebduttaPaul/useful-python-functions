"""

Author:			Debdutta Paul
Last updated:	02nd September, 2016

"""


from __future__ import division
import os, warnings
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import debduttaS_functions as mf
from astropy.io import fits, ascii
from astropy.table import Table
from astropy.stats import sigma_clip
from matplotlib import cm
from scipy import interpolate
from scipy.optimize import curve_fit

plt.rc('axes', linewidth=2)
plt.rc('font', family='serif', serif='cm10')
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

#	Ignoring silly warnings.
warnings.filterwarnings("ignore")

P = np.pi # Dear old pi!
padding	 = 8 # The padding of the axes labels.
size_font = 18 # The fontsize in the images.


################################################################################################################################################################################
#	General note: to test substitute code for cztpixclean.
################################################################################################################################################################################








################################################################################################################################################################################
#	To set the various CZTI specifics.

time_res = 20 * 1e-6 # Temporal resolution of CZTI, in second.
L_x = 64; L_y = L_x # Dimensions of the CZTI quadrants in x & y directions.
Q_n	=	4	# Number of quadrants.
number_of_pixels_per_module		=	256
number_of_modules_per_quadrant	=	4**2
number_of_pixels_per_quadrant	=	number_of_pixels_per_module * number_of_modules_per_quadrant
total_number_of_pixels			=	number_of_pixels_per_quadrant * Q_n

#~ print 'Number of quadrants:		' , Q_n
#~ print 'Number of pixels per module:	' , number_of_pixels_per_module
#~ print 'Number of modules per quadrant:	' , number_of_modules_per_quadrant
#~ print 'Number of pixels per quadrant:	' , number_of_pixels_per_quadrant
#~ print 'Total number of pixels:		' , total_number_of_pixels, '\n\n\n\n\n\n'


################################################################################################################################################################################






################################################################################################################################################################################
#	To define functions for latter use.

def create_full_dph( detx, dety, data ):
	
	
	'''
	
	
	Parameters
	-----------
	detx:		Array of detx.
	dety:		Array of dety.
	data:		Array of corresponding data.
	
	Returns
	-----------
	image:		2-D array, containing the required python-indices mapped image.
	
	
	'''
	
	
	L_x	=	64	;	L_y	=	L_x	#	CZTI specifics.
	image	=	np.zeros( ( L_x, L_y ) )
	
	for k, x in enumerate( detx ):
		
		#	To account for the convention of python-indices and CZTI indices, x & y should be flipped.		
		image[ (L_y-1)-dety[k] , x ]	=	data[k]
	
	image	=	np.array( [image] )[0]
	
	
	return image


################################################################################################################################################################################





################################################################################################################################################################################
#	For manual inputs of user-defined parameters.

evt_filename	=	'AS1G05_237T02_9000000422cztM0_level2_bc.evt'		#	the post bunch-clean file, background March.
lvt_filename	=	'AS1G05_237T02_9000000422cztM0_level2_livetime.evt' #	the corresponding livetime file.

caldb_filename	=	'AS1czteff_area20160401v03.fits'					#	the file from where the effective area of CZTI pixels are extracted.

t_bin	=	1e2	;	dec	=	1	#	the counts are binnned at this time-scale

t_offset=	198773802.0			#	in UT sec, 1st orbit starts here
t_start	=	t_offset + 31700	#	in UT sec, relatively clean part of 6th orbit only
t_stop	=	t_offset + 35000	#	in UT sec, relatively clean part of 6th orbit only

#~ t_offset=	198773802.0			#	in UT sec, 1st orbit starts here.
#~ t_start	=	198773802.0     	#	in UT sec, all orbits.
#~ t_stop	=	198900315.828   	#	in UT sec, all orbits.

E_low	=	0		#	in keV.
E_high	=	300		#	in keV, all energies considered.

path_save	=	os.getcwd() + '/plots/effective_area/'


################################################################################################################################################################################





################################################################################################################################################################################




#	Plotting specifications.
def applyPlotStyle( mx ):
	bn	=	8
	ax.set_xticks( np.arange(0, L_x, bn) )
	ax.set_xticklabels( np.arange(0, L_x,  bn) )
	ax.set_yticks( np.arange(0, L_y, bn) )
	ax.set_yticklabels( np.arange(L_y, 0, -bn) )
	ax.grid( linewidth = 1.2 )

effarea_maps	=	{}
hdu		=	fits.open( caldb_filename )

for i in range( Q_n ):
	#	To loop over the quadrants.
	detx	=	hdu[i+1].data['DETX']
	dety	=	hdu[i+1].data['DETY']
	area	=	hdu[i+1].data['AREA']
	
	print area.shape
	
	areaeff	=	np.sum( area, axis = 1 )
	areaeff	=	areaeff / areaeff.mean()
	
	effarea_map		=	create_full_dph( detx, dety, areaeff )
	effarea_maps[i]	=	effarea_map
	
	#~ #	To plot the maps.
	#~ mx	=	int( effarea_map.max() )
	#~ fig, ax	=	plt.subplots()
	#~ applyPlotStyle( mx )
	#~ im = ax.imshow( effarea_map, clim = ( 0, mx ), cmap = cm.jet_r )
	#~ plt.colorbar( im )
	#~ plt.title( r'$ \rm{Q} $' + r'$ \rm{:d} $'.format(i), fontsize = size_font )
	#~ plt.savefig( path_save + 'effective_area_map--Q{:d}'.format(i) )
	#~ plt.clf()
	#~ plt.close()


hdu.close()






hdu	=	fits.open( evt_filename )
Q_time	=	{}
Q_detx	=	{}
Q_dety	=	{}
Q_energy	=	{}

for i in range( Q_n ):
	#	To loop over the quadrants.
	print '**********************************************'
	print '\nQuadrant {:d}...\n'.format(i)
	
	
	##	To extract data and apply cuts.
	
	#	To extract all relevant data.
	Q_time[i]	=	hdu[i+1].data['TIME']
	Q_detx[i]	=	hdu[i+1].data['DETX']
	Q_dety[i]	=	hdu[i+1].data['DETY']
	Q_energy[i]	=	hdu[i+1].data['ENERGY']
	
	#	To apply the energy-cut.
	indX_Energy	=	np.where(  ( E_low <= Q_energy[i] ) & ( Q_energy[i] <= E_high )  )[0]
	Q_time[i]	=	Q_time[i][indX_Energy]
	Q_detx[i]	=	Q_detx[i][indX_Energy]
	Q_dety[i]	=	Q_dety[i][indX_Energy]
	Q_energy[i]	=	Q_energy[i][indX_Energy]
	
	#	To apply the time-cut.
	index_cut	=	np.where(    ( t_start < Q_time[i] ) & ( Q_time[i] < t_stop )    )[0]
	Q_time[i]	=	Q_time[i][index_cut]
	Q_detx[i]	=	Q_detx[i][index_cut]
	Q_dety[i]	=	Q_dety[i][index_cut]
	Q_energy[i]	=	Q_energy[i][index_cut]
	Q_time[i]	=	Q_time[i] - t_offset
	
	
	
	
	#	To create the DPHs.
	times, counts, DPHs	=	mf.create_dph( Q_time[i], Q_detx[i], Q_dety[i], t_bin )
	
	#	To plot the light-curve.
	#	plt.step( times, counts )
	#	plt.show()
	
	#	To create the full DPH without light-curve variation taken into account.
	full_DPH	=	np.sum( DPHs, axis = 0 )
	
	#	To create the full DPI, defined as the DPH scaled by the total counts.
	counts	=	counts / counts.mean()
	bins	=	DPHs.shape[0]
	full_DPI=	np.zeros( (L_x, L_y) )
	for j in range( bins ):
		full_DPI += DPHs[j] / counts[j]
	
	#~ #	To correct the DPI for the effective area (which also takes into account the QEs) of the pixels.
	#~ full_DPI	=	full_DPI / effarea_maps[i]
	#	Since doing this here is over-correcting the banana pixels and resulting in many more pixels getting flagged, we are not going to do this at this step.
	
	#	To check how many of the pixels are flagged by the onboard software.
	initially_flagged	=	len( np.where( full_DPI == 0 )[0] )
	print 'Already flagged pixels:		' ,  initially_flagged, '\n'
	
	#	To mask the pixels with zero counts before flagging, for subsequent analysis.
	full_DPI	=	ma.masked_equal( full_DPI, 0 )
	
	#	To check the statistics of the data before further cleaning.
	min		=	full_DPI.min()
	mean	=	full_DPI.mean()
	sigma	=	full_DPI.std()
	print 'Before:		min {0:.1f},	mean {1:.1f},	sigma {2:.1f}'.format(min, mean, sigma)
	
	#	To plot the variation with the pixels.
	#	plt.plot( full_DPI.flatten() )
	#	plt.show()
	
	#	To apply the iterative 5-sigma clipping based on the DPI.
	clipped_full_DPI	=	sigma_clip( data = full_DPI, sigma_lower = 100, sigma_upper = 5 )
	
	#	To extract statistics of the flagged data.
	min		=	clipped_full_DPI.min()
	mean	=	clipped_full_DPI.mean()
	sigma	=	clipped_full_DPI.std()
	print 'After :		min {0:.1f},	mean {1:.1f},	sigma {2:.1f}'.format(min, mean, sigma)
	
	#	To plot the resultant DPI against the pixels.
	#	plt.plot( clipped_full_DPI.flatten() )
	#	plt.show()
	
	#	To check the statistics of the thus-cleaned data.
	print '\n', 'Max allowed: {0:.1f},	Max data: {1:.1f}'.format( (mean + 5*sigma), clipped_full_DPI.flatten().max() ), '\n'
	indices_flagged	=	np.where( ma.getmask( clipped_full_DPI ) == True )
	total_flagged	=	indices_flagged[0].shape[0]
	print 'Number of flagged pixels:	', total_flagged - initially_flagged
	
	#~ #	To plot histogram of the counts, before and after cleaning.
	#~ h_bin	=	30
	#~ x1, y1	=	mf.my_histogram( full_DPI.flatten(), h_bin )
	#~ x2, y2	=	mf.my_histogram( clipped_full_DPI.flatten(), h_bin )
	#~ plt.xlim( 0, x2.max() )
	#~ plt.title( r'$ \rm{Q} $' + r'$ \rm{:d} $'.format(i), fontsize = size_font )
	#~ plt.xlabel( r'$ \rm{ Scaled \; counts } $', fontsize = size_font )
	#~ plt.ylabel( r'$ \rm{ Frequency } $', fontsize = size_font )
	#~ plt.plot( x1, y1, 'r-', lw = 2, label =  'Input' )
	#~ plt.plot( x2, y2, 'ko', lw = 2, label = 'Output' )
	#~ plt.legend( numpoints = 1 )
	#~ plt.show()
	
	#~ #	To plot the DPIs before and after cleaning.
	#~ mx	=	int( clipped_full_DPI.max() )
	#~ fig, axes	=	plt.subplots( 1, 2 )
	#~ ax	=	axes[0]
	#~ applyPlotStyle( mx )
	#~ im = ax.imshow( full_DPI, clim = ( 0, mx ), cmap = cm.jet_r )
	#~ ax	=	axes[1]
	#~ applyPlotStyle( mx )
	#~ im = ax.imshow( clipped_full_DPI, clim = ( 0, mx ), cmap = cm.jet_r )
	#~ plt.colorbar( im, ax = axes.ravel().tolist(), orientation = 'horizontal' )
	#~ plt.suptitle( r'$ \rm{ Left, Before; \; Right, After: \quad Q } $' + r'$ {0:d} $'.format(i), fontsize = size_font-2 )
	#~ plt.show()
	
	#	To now correct the DPI for the effective area (which also takes into account the QEs) of the pixels.
	clipped_full_DPI	=	clipped_full_DPI / effarea_maps[i]
	
	#	To plot histogram of the counts.
	h_bin	=	30
	x, y	=	mf.my_histogram( clipped_full_DPI.flatten(), h_bin )
	plt.xlim( 0, x.max() )
	plt.title( r'$ \rm{Q} $' + r'$ \rm{:d} $'.format(i), fontsize = size_font )
	plt.xlabel( r'$ \rm{ Scaled \; counts } $', fontsize = size_font )
	plt.ylabel( r'$ \rm{ Frequency } $', fontsize = size_font )
	plt.plot( x, y, 'r-', lw = 2 )
	plt.legend( numpoints = 1 )
	plt.show()
	
	#	To plot the DPI.
	mx	=	int( clipped_full_DPI.max() )
	fig, ax	=	plt.subplots()
	applyPlotStyle( mx )
	im = ax.imshow( clipped_full_DPI, clim = ( 0, mx ), cmap = cm.jet_r )
	plt.colorbar( im )
	plt.title( r'$ \rm{Q} $' + r'$ \rm{:d} $'.format(i), fontsize = size_font )
	plt.show()
	
	
	
		
	print '\n\n'
	
hdu.close()
