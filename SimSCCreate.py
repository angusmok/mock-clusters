# !/usr/bin/env python

#------------------------------------------------------------------------------
###
#------------------------------------------------------------------------------
###
#------------------------------------------------------------------------------

###
# Declares Libraries
###

import string
import math
import sys
import os
###
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate

#------------------------------------------------------------------------------
###
#------------------------------------------------------------------------------
###
#------------------------------------------------------------------------------

###
# Set Parameters
###

# Set general matplotlib properties
A = plt.rcParams.keys()
plt.rc('font', family = 'serif', size = 26)
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['lines.markersize'] = 8
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.labelsize'] = 28
plt.rcParams['ytick.labelsize'] = 28
plt.rcParams['xtick.major.size'] = 20
plt.rcParams['xtick.minor.size'] = 12
plt.rcParams['ytick.major.size'] = 20
plt.rcParams['ytick.minor.size'] = 12
plt.rcParams['xtick.major.width'] = 4
plt.rcParams['xtick.minor.width'] = 2
plt.rcParams['ytick.major.width'] = 4
plt.rcParams['ytick.minor.width'] = 2
plt.rcParams['legend.numpoints'] = 1

# Other General Parameters
np.random.seed(145)
Ntotal = 1000000
maglimit = -9

# Age Distribution Parameters
age_distribution = [1E6, 1E10]
age_distribution_numbin = 100
age_distribution_gauss = np.power(10, 8.699)
age_width_gauss = np.power(10, 8.0)
age_base_infant = 5.0 # 10.0 = 90% (equivalent to continuous_logspace), 5.0 = 80%, 1.0 = 10%

# Mass Distribution Parameters
mass_distribution = [1E3, 1E8]
mass_distribution_numbin = 100
mass_distribution_trun7 = 1E7
mass_distribution_trun6 = 1E6
mass_distribution_trun5 = 1E5

# Destruction Parameters
dest_rate = -6. # value in log solar masses per year  (-1 kills most, -6 typical, -10 almost nothing))
dest_t4 = 7.9E9 # avg of M51 and M33; M51 = 7.85 (7.1E7), M33 = 8.8 (6.3E8), MW = 8.75 (5.6E8), SMC = 9.9 (7.9E9)
dest_gamma = 0.61 

# Plotting Parameters
mass_bins_log = np.power(10, np.linspace(2, 9, num = 29))
mass_range_log = [1E2, 1E9]
mass_bins_lin = np.linspace(1E2, 1E9, num = 29)
mass_range_lin = [1E2, 1E9]
###
age_bins_log = np.power(10, np.linspace(5, 11, num = 25))
age_range_log = [1E5, 1E11]
age_bins_lin = np.linspace(1E5, 1E10, num = 21)
age_range_lin = [1E5, 1E10]
###
colour_range = [-2, 10]
output_contlin = 0
output_contlog = 1
output_gauss = 0
output_infant = 0

# Generate Derived Paramters
random_age = np.random.uniform(size = Ntotal)
random_mass = np.random.uniform(size = Ntotal)
random_num = range(0, len(random_age))
###
age_distribution_start = np.nanmin(age_distribution)
age_distribution_end = np.nanmax(age_distribution)
age_distribution_array = np.power(10, np.linspace(np.log10(age_distribution_start), np.log10(age_distribution_end), num = age_distribution_numbin))
age_alpha = -np.log10(age_base_infant)
###
mass_distribution_start = np.nanmin(mass_distribution)
mass_distribution_end = np.nanmax(mass_distribution)
mass_distribution_array = np.power(10, np.linspace(np.log10(mass_distribution_start), np.log10(mass_distribution_end), num = mass_distribution_numbin))

# Other tasks:
# Duplicates terminal to log file
import datetime
now = datetime.datetime.now()
class Logger(object):
	def __init__(self):
		self.terminal = sys.stdout
		self.log = open('./Logs/SimSCCreate_' + str(now)[:10] + '.log', 'w')

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message) 

	def flush(self):
		pass    

sys.stdout = Logger()

# Create directories, if not present
if not os.path.exists('./Logs/'):
	os.makedirs('./Logs')
if not os.path.exists('./SimSCOutput/'):
	os.makedirs('./SimSCOutput/')
if not os.path.exists('./SimSCFigures/'):
	os.makedirs('./SimSCFigures')

#------------------------------------------------------------------------------
###
#------------------------------------------------------------------------------
###
#------------------------------------------------------------------------------

###
# Define Functions
###

# Func: Write simulated data to file
def outputclusterinformation(filename, array1, array2):

	###
	#
	#
	#
	###

	f1 = open('./SimSCOutput/' + filename + '.txt', 'w')
	if len(array1) != len(array2):
		print('Error in input arrays for ' + filename)
	else:
		for i in range(0, len(array1)):
			if array1[i] > 0 and array2[i] > 0:
				f1.write('{:.6e}; {:.6e}\n'.format(array1[i], array2[i]))
	f1.close()

	return 0

# Func: Sample from distributions and generate mock catalogues
def samplefromdistribution(array1, value1, normfun):

	###
	#
	#
	#
	###

	cumulative = []
	for i in range(0, len(array1)):
		cumulative.append(integrate.quad(normfun, value1, array1[i])[0])
	cumulative_interp = interpolate.interp1d(cumulative, array1)
	arrayout = cumulative_interp(random_mass)

	# Add in Gaussian noise (15%)
	array_noise1 = np.random.normal(loc = 0, scale = arrayout * 0.15, size = len(arrayout))
	arrayout_mnoise1 = arrayout + array_noise1

	# Add in Gaussian noise (50%)
	array_noise2 = np.random.normal(loc = 0, scale = arrayout * 0.50, size = len(arrayout))
	arrayout_mnoise2 = arrayout + array_noise2

	return arrayout, arrayout_mnoise1, arrayout_mnoise2, cumulative

# Func: Apply various destruction models
def massdestruction(massarray, dest_rate, dest_gamma, array1, array2, array3, array4):

	###
	#
	#
	#
	###

	mass_array1_massinddest = massarray - (np.power(10, dest_rate) * array1)
	mass_array2_massinddest = massarray - (np.power(10, dest_rate) * array2)
	mass_array3_massinddest = massarray - (np.power(10, dest_rate) * array3)
	mass_array4_massinddest = massarray - (np.power(10, dest_rate) * array4)
	
	lamer_time = dest_t4 * np.power((massarray / 1E4), dest_gamma)
	mass_array1_lamerdest = np.where(array1 < lamer_time, massarray, -1.)
	mass_array2_lamerdest = np.where(array2 < lamer_time, massarray, -1.)
	mass_array3_lamerdest = np.where(array3 < lamer_time, massarray, -1.)
	mass_array4_lamerdest = np.where(array4 < lamer_time, massarray, -1.)

	return mass_array1_massinddest, mass_array2_massinddest, mass_array3_massinddest, mass_array4_massinddest, mass_array1_lamerdest, mass_array2_lamerdest, mass_array3_lamerdest, mass_array4_lamerdest

# Func: Apply mass correction to V-mag arrays
def vmagmasscorr(vmagarray, array1, array2, array3, array4, array5, array6, array7, array8, array9, array10, array11, array12, array13, array14, array15, array16, array17, array18, array19, array20):

	###
	#
	#
	#
	###

	array1_mcorr = vmagarray - (2.5 * np.log10(np.where(array1 > 1, array1, 1)))
	array2_mcorr = vmagarray - (2.5 * np.log10(np.where(array2 > 1, array2, 1)))
	array3_mcorr = vmagarray - (2.5 * np.log10(np.where(array3 > 1, array3, 1)))
	array4_mcorr = vmagarray - (2.5 * np.log10(np.where(array4 > 1, array4, 1)))
	array5_mcorr = vmagarray - (2.5 * np.log10(np.where(array5 > 1, array5, 1)))
	array6_mcorr = vmagarray - (2.5 * np.log10(np.where(array6 > 1, array6, 1)))
	array7_mcorr = vmagarray - (2.5 * np.log10(np.where(array7 > 1, array7, 1)))
	array8_mcorr = vmagarray - (2.5 * np.log10(np.where(array8 > 1, array8, 1)))
	array9_mcorr = vmagarray - (2.5 * np.log10(np.where(array9 > 1, array9, 1)))
	array10_mcorr = vmagarray - (2.5 * np.log10(np.where(array10 > 1, array10, 1)))
	array11_mcorr = vmagarray - (2.5 * np.log10(np.where(array11 > 1, array11, 1)))
	array12_mcorr = vmagarray - (2.5 * np.log10(np.where(array12 > 1, array12, 1)))
	array13_mcorr = vmagarray - (2.5 * np.log10(np.where(array13 > 1, array13, 1)))
	array14_mcorr = vmagarray - (2.5 * np.log10(np.where(array14 > 1, array14, 1)))
	array15_mcorr = vmagarray - (2.5 * np.log10(np.where(array15 > 1, array15, 1)))
	array16_mcorr = vmagarray - (2.5 * np.log10(np.where(array16 > 1, array16, 1)))
	array17_mcorr = vmagarray - (2.5 * np.log10(np.where(array17 > 1, array17, 1)))
	array18_mcorr = vmagarray - (2.5 * np.log10(np.where(array18 > 1, array18, 1)))
	array19_mcorr = vmagarray - (2.5 * np.log10(np.where(array19 > 1, array19, 1)))
	array20_mcorr = vmagarray - (2.5 * np.log10(np.where(array20 > 1, array20, 1)))

	return array1_mcorr, array2_mcorr, array3_mcorr, array4_mcorr, array5_mcorr, array6_mcorr, array7_mcorr, array8_mcorr, array9_mcorr, array10_mcorr, array11_mcorr, array12_mcorr, array13_mcorr, array14_mcorr, array15_mcorr, array16_mcorr, array17_mcorr, array18_mcorr, array19_mcorr, array20_mcorr

#------------------------------------------------------------------------------
###
#------------------------------------------------------------------------------
###
#------------------------------------------------------------------------------

###
# Main Code
###

###
# Step 1 - Create Age Distribution
###

print("Step 1 - Create Age Distribution")
print("A - Continous Formation (log), B - Continuous Formation (linear), C - Gaussian Burst, D - Infant Mortality")
###
age_contlog = np.power(10, (np.random.uniform(low = np.log10(age_distribution_start), high = np.log10(age_distribution_end), size = Ntotal)))
age_contlin = np.random.uniform(low = age_distribution_start, high = age_distribution_end, size = Ntotal)
age_gauss = np.random.normal(loc = age_distribution_gauss, scale = age_width_gauss, size = Ntotal)

# Create infant mortality age distribution
infantmortality = lambda x: np.power(x, - (age_alpha + 1))
infantmortality_norm = integrate.quad(infantmortality, age_distribution_start, age_distribution_end)
infantmortality_normfun = lambda x: np.power(x, - (age_alpha + 1)) / infantmortality_norm[0]
print ('Infant Mortality Function Normalization: {:.2f}'.format(infantmortality_norm[0]))
age_infant, age_infant_mnoise1, age_infant_mnoise2, age_infant_cumulative = samplefromdistribution(age_distribution_array, age_distribution_start, infantmortality_normfun)

###
# Step 2 - Create Mass Distribution
###

print("Step 2 - Create Mass Distribution")
print("A - Power Law, B - Schechter Function, C - Truncated Power Law")
###

# Create Normalized Probability Functions
powerlaw = lambda x: np.power(x, -2)
powerlaw_norm = integrate.quad(powerlaw, mass_distribution_start, mass_distribution_end)
powerlaw_normfun = lambda x: np.power(x, -2) / powerlaw_norm[0]
print ('Power Law Mass Function Normalization: {:.2e}'.format(powerlaw_norm[0]))
powerlaw_mass, powerlaw_mass_mnoise1, powerlaw_mass_mnoise2, powerlaw_cumulative = samplefromdistribution(mass_distribution_array, mass_distribution_start, powerlaw_normfun)
###
schechter5 = lambda x: np.power(x / mass_distribution_trun5, -2) * np.exp(-(x / mass_distribution_trun5))
schechter5_norm = integrate.quad(schechter5, mass_distribution_start, mass_distribution_end)
schechter5_normfun = lambda x: (np.power(x / mass_distribution_trun5, -2) * np.exp(-(x / mass_distribution_trun5))) / schechter5_norm[0]
print ('Schechter (1E5) Mass Function Normalization: {:.2e}'.format(schechter5_norm[0]))
schechter5_mass, schechter5_mass_mnoise1, schechter5_mass_mnoise2, schechter5_cumulative = samplefromdistribution(mass_distribution_array, mass_distribution_start, schechter5_normfun)
###
schechter6 = lambda x: np.power(x / mass_distribution_trun6, -2) * np.exp(-(x / mass_distribution_trun6))
schechter6_norm = integrate.quad(schechter6, mass_distribution_start, mass_distribution_end)
schechter6_normfun = lambda x: (np.power(x / mass_distribution_trun6, -2) * np.exp(-(x / mass_distribution_trun6))) / schechter6_norm[0]
print ('Schechter (1E6) Mass Function Normalization: {:.2e}'.format(schechter6_norm[0]))
schechter6_mass, schechter6_mass_mnoise1, schechter6_mass_mnoise2, schechter6_cumulative = samplefromdistribution(mass_distribution_array, mass_distribution_start, schechter6_normfun)
###
schechter7 = lambda x: np.power(x / mass_distribution_trun7, -2) * np.exp(-(x / mass_distribution_trun7))
schechter7_norm = integrate.quad(schechter7, mass_distribution_start, mass_distribution_end)
schechter7_normfun = lambda x: (np.power(x / mass_distribution_trun7, -2) * np.exp(-(x / mass_distribution_trun7))) / schechter7_norm[0]
print ('Schechter (1E7) Mass Function Normalization: {:.2e}'.format(schechter7_norm[0]))
schechter7_mass, schechter7_mass_mnoise1, schechter7_mass_mnoise2, schechter7_cumulative = samplefromdistribution(mass_distribution_array, mass_distribution_start, schechter7_normfun)
###
truncated5 = lambda x: np.power(x / mass_distribution_trun5, -2) - 1
truncated5_norm = integrate.quad(truncated5, mass_distribution_start, mass_distribution_trun5)
truncated5_normfun = lambda x: (np.power(x / mass_distribution_trun5, -2) - 1) / truncated5_norm[0]
print ('Truncated (1E5) Mass Function Normalization: {:.2e}'.format(truncated5_norm[0]))
mass_distribution_array5 = np.power(10, np.linspace(np.log10(mass_distribution_start), 5, num = mass_distribution_numbin))
truncated5_mass, truncated5_mass_mnoise1, truncated5_mass_mnoise2, truncated5_cumulative = samplefromdistribution(mass_distribution_array5, mass_distribution_start, truncated5_normfun)
###
truncated6 = lambda x: np.power(x / mass_distribution_trun6, -2) - 1
truncated6_norm = integrate.quad(truncated6, mass_distribution_start, mass_distribution_trun6)
truncated6_normfun = lambda x: (np.power(x / mass_distribution_trun6, -2) - 1) / truncated6_norm[0]
print ('Truncated (1E6) Mass Function Normalization: {:.2e}'.format(truncated6_norm[0]))
mass_distribution_array6 = np.power(10, np.linspace(np.log10(mass_distribution_start), 6, num = mass_distribution_numbin))
truncated6_mass, truncated6_mass_mnoise1, truncated6_mass_mnoise2, truncated6_cumulative = samplefromdistribution(mass_distribution_array6, mass_distribution_start, truncated6_normfun)
###
truncated7 = lambda x: np.power(x / mass_distribution_trun7, -2) - 1
truncated7_norm = integrate.quad(truncated7, mass_distribution_start, mass_distribution_trun7)
truncated7_normfun = lambda x: (np.power(x / mass_distribution_trun7, -2) - 1) / truncated7_norm[0]
print ('Truncated (1E7) Mass Function Normalization: {:.2e}'.format(truncated7_norm[0]))
mass_distribution_array7 = np.power(10, np.linspace(np.log10(mass_distribution_start), 7, num = mass_distribution_numbin))
truncated7_mass, truncated7_mass_mnoise1, truncated7_mass_mnoise2, truncated7_cumulative = samplefromdistribution(mass_distribution_array7, mass_distribution_start, truncated7_normfun)

###
# Step 3 - Read in Model Data
###

print("Step 3 - Read in Model Data")
###

# M62 = solar metallicity
# log-age-yr[0], Mbol[1], Umag[2], Bmag[3], Vmag[4], Kmag[5], 14-V[6], 17-V[7], 22-V[8], 27-V[9], U-J[10], J-F[11], F-N[12], U-B[13], B-V[14]
cb07_basel_m62_chap = np.genfromtxt('./cb2007_lr_BaSeL_m62_chab_ssp.1color', skip_header = 30)
print(cb07_basel_m62_chap)
vmag_interp = interpolate.interp1d(np.power(10, cb07_basel_m62_chap[:,0]), cb07_basel_m62_chap[:,4])

###
# Step 4 - Implement Destruction Law
###

print("Step 4 - Implement Destruction Law")
print("A - Mass Dependent, B - Mass Independent")

# Apply cluster mass destruction laws
powerlaw_mass_contlog_massinddest, powerlaw_mass_contlin_massinddest, powerlaw_mass_gauss_massinddest, powerlaw_mass_infant_massinddest, powerlaw_mass_contlog_lamerdest, powerlaw_mass_contlin_lamerdest, powerlaw_mass_gauss_lamerdest, powerlaw_mass_infant_lamerdest = massdestruction(powerlaw_mass, dest_rate, dest_gamma, age_contlog, age_contlin, age_gauss, age_infant)
schechter5_mass_contlog_massinddest, schechter5_mass_contlin_massinddest, schechter5_mass_gauss_massinddest, schechter5_mass_infant_massinddest, schechter5_mass_contlog_lamerdest, schechter5_mass_contlin_lamerdest, schechter5_mass_gauss_lamerdest, schechter5_mass_infant_lamerdest = massdestruction(schechter5_mass, dest_rate, dest_gamma, age_contlog, age_contlin, age_gauss, age_infant)
schechter6_mass_contlog_massinddest, schechter6_mass_contlin_massinddest, schechter6_mass_gauss_massinddest, schechter6_mass_infant_massinddest, schechter6_mass_contlog_lamerdest, schechter6_mass_contlin_lamerdest, schechter6_mass_gauss_lamerdest, schechter6_mass_infant_lamerdest = massdestruction(schechter6_mass, dest_rate, dest_gamma, age_contlog, age_contlin, age_gauss, age_infant)
schechter7_mass_contlog_massinddest, schechter7_mass_contlin_massinddest, schechter7_mass_gauss_massinddest, schechter7_mass_infant_massinddest, schechter7_mass_contlog_lamerdest, schechter7_mass_contlin_lamerdest, schechter7_mass_gauss_lamerdest, schechter7_mass_infant_lamerdest = massdestruction(schechter7_mass, dest_rate, dest_gamma, age_contlog, age_contlin, age_gauss, age_infant)

# Apply mass correction to V-mag
vmag_contlog = vmag_interp(age_contlog)
vmag_contlin = vmag_interp(age_contlin)
vmag_gauss = vmag_interp(age_gauss)
vmag_infant = vmag_interp(age_infant)

# Apply vmag correction
vmag_contlog_powerlaw_mcorr, vmag_contlog_powerlaw_mnoise1_mcorr, vmag_contlog_powerlaw_mnoise2_mcorr, vmag_contlog_powerlaw_massinddest_mcorr, vmag_contlog_powerlaw_lamerdest_mcorr, vmag_contlog_schechter5_mcorr, vmag_contlog_schechter5_mnoise1_mcorr, vmag_contlog_schechter5_mnoise2_mcorr, vmag_contlog_schechter5_massinddest_mcorr, vmag_contlog_schechter5_lamerdest_mcorr, vmag_contlog_schechter6_mcorr, vmag_contlog_schechter6_mnoise1_mcorr, vmag_contlog_schechter6_mnoise2_mcorr, vmag_contlog_schechter6_massinddest_mcorr, vmag_contlog_schechter6_lamerdest_mcorr, vmag_contlog_schechter7_mcorr, vmag_contlog_schechter7_mnoise1_mcorr, vmag_contlog_schechter7_mnoise2_mcorr, vmag_contlog_schechter7_massinddest_mcorr, vmag_contlog_schechter7_lamerdest_mcorr = vmagmasscorr(vmag_contlog, powerlaw_mass, powerlaw_mass_mnoise1, powerlaw_mass_mnoise2, powerlaw_mass_contlog_massinddest, powerlaw_mass_contlog_lamerdest, schechter5_mass, schechter5_mass_mnoise1, schechter5_mass_mnoise2, schechter5_mass_contlog_massinddest, schechter5_mass_contlog_lamerdest, schechter6_mass, schechter6_mass_mnoise1, schechter6_mass_mnoise2, schechter6_mass_contlog_massinddest, schechter6_mass_contlog_lamerdest, schechter7_mass, schechter7_mass_mnoise1, schechter7_mass_mnoise2, schechter7_mass_contlog_massinddest, schechter7_mass_contlog_lamerdest)
vmag_contlin_powerlaw_mcorr, vmag_contlin_powerlaw_mnoise1_mcorr, vmag_contlin_powerlaw_mnoise2_mcorr, vmag_contlin_powerlaw_massinddest_mcorr, vmag_contlin_powerlaw_lamerdest_mcorr, vmag_contlin_schechter5_mcorr, vmag_contlin_schechter5_mnoise1_mcorr, vmag_contlin_schechter5_mnoise2_mcorr, vmag_contlin_schechter5_massinddest_mcorr, vmag_contlin_schechter5_lamerdest_mcorr, vmag_contlin_schechter6_mcorr, vmag_contlin_schechter6_mnoise1_mcorr, vmag_contlin_schechter6_mnoise2_mcorr, vmag_contlin_schechter6_massinddest_mcorr, vmag_contlin_schechter6_lamerdest_mcorr, vmag_contlin_schechter7_mcorr, vmag_contlin_schechter7_mnoise1_mcorr, vmag_contlin_schechter7_mnoise2_mcorr, vmag_contlin_schechter7_massinddest_mcorr, vmag_contlin_schechter7_lamerdest_mcorr = vmagmasscorr(vmag_contlin, powerlaw_mass, powerlaw_mass_mnoise1, powerlaw_mass_mnoise2, powerlaw_mass_contlin_massinddest, powerlaw_mass_contlin_lamerdest, schechter5_mass, schechter5_mass_mnoise1, schechter5_mass_mnoise2, schechter5_mass_contlin_massinddest, schechter5_mass_contlin_lamerdest, schechter6_mass, schechter6_mass_mnoise1, schechter6_mass_mnoise2, schechter6_mass_contlin_massinddest, schechter6_mass_contlin_lamerdest, schechter7_mass, schechter7_mass_mnoise1, schechter7_mass_mnoise2, schechter7_mass_contlin_massinddest, schechter7_mass_contlin_lamerdest)
vmag_gauss_powerlaw_mcorr, vmag_gauss_powerlaw_mnoise1_mcorr, vmag_gauss_powerlaw_mnoise2_mcorr, vmag_gauss_powerlaw_massinddest_mcorr, vmag_gauss_powerlaw_lamerdest_mcorr, vmag_gauss_schechter5_mcorr, vmag_gauss_schechter5_mnoise1_mcorr, vmag_gauss_schechter5_mnoise2_mcorr, vmag_gauss_schechter5_massinddest_mcorr, vmag_gauss_schechter5_lamerdest_mcorr, vmag_gauss_schechter6_mcorr, vmag_gauss_schechter6_mnoise1_mcorr, vmag_gauss_schechter6_mnoise2_mcorr, vmag_gauss_schechter6_massinddest_mcorr, vmag_gauss_schechter6_lamerdest_mcorr, vmag_gauss_schechter7_mcorr, vmag_gauss_schechter7_mnoise1_mcorr, vmag_gauss_schechter7_mnoise2_mcorr, vmag_gauss_schechter7_massinddest_mcorr, vmag_gauss_schechter7_lamerdest_mcorr = vmagmasscorr(vmag_gauss, powerlaw_mass, powerlaw_mass_mnoise1, powerlaw_mass_mnoise2, powerlaw_mass_gauss_massinddest, powerlaw_mass_gauss_lamerdest, schechter5_mass, schechter5_mass_mnoise1, schechter5_mass_mnoise2, schechter5_mass_gauss_massinddest, schechter5_mass_gauss_lamerdest, schechter6_mass, schechter6_mass_mnoise1, schechter6_mass_mnoise2, schechter6_mass_gauss_massinddest, schechter6_mass_gauss_lamerdest, schechter7_mass, schechter7_mass_mnoise1, schechter7_mass_mnoise2, schechter7_mass_gauss_massinddest, schechter7_mass_gauss_lamerdest)
vmag_infant_powerlaw_mcorr, vmag_infant_powerlaw_mnoise1_mcorr, vmag_infant_powerlaw_mnoise2_mcorr, vmag_infant_powerlaw_massinddest_mcorr, vmag_infant_powerlaw_lamerdest_mcorr, vmag_infant_schechter5_mcorr, vmag_infant_schechter5_mnoise1_mcorr, vmag_infant_schechter5_mnoise2_mcorr, vmag_infant_schechter5_massinddest_mcorr, vmag_infant_schechter5_lamerdest_mcorr, vmag_infant_schechter6_mcorr, vmag_infant_schechter6_mnoise1_mcorr, vmag_infant_schechter6_mnoise2_mcorr, vmag_infant_schechter6_massinddest_mcorr, vmag_infant_schechter6_lamerdest_mcorr, vmag_infant_schechter7_mcorr, vmag_infant_schechter7_mnoise1_mcorr, vmag_infant_schechter7_mnoise2_mcorr, vmag_infant_schechter7_massinddest_mcorr, vmag_infant_schechter7_lamerdest_mcorr = vmagmasscorr(vmag_infant, powerlaw_mass, powerlaw_mass_mnoise1, powerlaw_mass_mnoise2, powerlaw_mass_infant_massinddest, powerlaw_mass_infant_lamerdest, schechter5_mass, schechter5_mass_mnoise1, schechter5_mass_mnoise2, schechter5_mass_infant_massinddest, schechter5_mass_infant_lamerdest, schechter6_mass, schechter6_mass_mnoise1, schechter6_mass_mnoise2, schechter6_mass_infant_massinddest, schechter6_mass_infant_lamerdest, schechter7_mass, schechter7_mass_mnoise1, schechter7_mass_mnoise2, schechter7_mass_infant_massinddest, schechter7_mass_infant_lamerdest)

# Output cluster data information
if output_contlog > 0:

	outputclusterinformation('powerlaw_contlog_nodest', powerlaw_mass, age_contlog)
	# outputclusterinformation('powerlaw_contlog_nodest_mnoise1', powerlaw_mass_mnoise1, age_contlog)
	# outputclusterinformation('powerlaw_contlog_nodest_mnoise2', powerlaw_mass_mnoise2, age_contlog)
	outputclusterinformation('schechter5_contlog_nodest', schechter5_mass, age_contlog)
	# outputclusterinformation('schechter5_contlog_nodest_mnoise1', schechter5_mass_mnoise1, age_contlog)
	# outputclusterinformation('schechter5_contlog_nodest_mnoise2', schechter5_mass_mnoise2, age_contlog)
	outputclusterinformation('schechter6_contlog_nodest', schechter6_mass, age_contlog)
	# outputclusterinformation('schechter6_contlog_nodest_mnoise1', schechter6_mass_mnoise1, age_contlog)
	# outputclusterinformation('schechter6_contlog_nodest_mnoise2', schechter6_mass_mnoise2, age_contlog)
	outputclusterinformation('schechter7_contlog_nodest', schechter7_mass, age_contlog)
	# outputclusterinformation('schechter7_contlog_nodest_mnoise1', schechter7_mass_mnoise1, age_contlog)
	# outputclusterinformation('schechter7_contlog_nodest_mnoise2', schechter7_mass_mnoise2, age_contlog)
	###
	outputclusterinformation('truncated5_contlog_nodest', truncated5_mass, age_contlog)
	outputclusterinformation('truncated6_contlog_nodest', truncated6_mass, age_contlog)
	outputclusterinformation('truncated7_contlog_nodest', truncated7_mass, age_contlog)
	###
	outputclusterinformation('powerlaw_contlin_nodest', powerlaw_mass, age_contlin)

# If flag > 2, then also output mass destruction catalogues
if output_contlog > 2:

	outputclusterinformation('powerlaw_contlog_massinddest', powerlaw_mass_contlog_massinddest, age_contlog)
	outputclusterinformation('powerlaw_contlog_lamerdest', powerlaw_mass_contlog_lamerdest, age_contlog)
	outputclusterinformation('schechter5_contlog_massinddest', schechter5_mass_contlog_massinddest, age_contlog)
	outputclusterinformation('schechter5_contlog_lamerdest', schechter5_mass_contlog_lamerdest, age_contlog)
	outputclusterinformation('schechter6_contlog_massinddest', schechter6_mass_contlog_massinddest, age_contlog)
	outputclusterinformation('schechter6_contlog_lamerdest', schechter6_mass_contlog_lamerdest, age_contlog)
	outputclusterinformation('schechter7_contlog_massinddest', schechter7_mass_contlog_massinddest, age_contlog)
	outputclusterinformation('schechter7_contlog_lamerdest', schechter7_mass_contlog_lamerdest, age_contlog)

if output_contlin > 0:

	outputclusterinformation('powerlaw_contlin_nodest', powerlaw_mass, age_contlin)
	outputclusterinformation('powerlaw_contlin_nodest_mnoise1', powerlaw_mass_mnoise1, age_contlin)
	outputclusterinformation('powerlaw_contlin_nodest_mnoise2', powerlaw_mass_mnoise2, age_contlin)
	outputclusterinformation('schechter5_contlin_nodest', schechter5_mass, age_contlin)
	outputclusterinformation('schechter5_contlin_nodest_mnoise1', schechter5_mass_mnoise1, age_contlin)
	outputclusterinformation('schechter5_contlin_nodest_mnoise2', schechter5_mass_mnoise2, age_contlin)
	outputclusterinformation('schechter6_contlin_nodest', schechter6_mass, age_contlin)
	outputclusterinformation('schechter6_contlin_nodest_mnoise1', schechter6_mass_mnoise1, age_contlin)
	outputclusterinformation('schechter6_contlin_nodest_mnoise2', schechter6_mass_mnoise2, age_contlin)
	outputclusterinformation('schechter7_contlin_nodest', schechter7_mass, age_contlin)
	outputclusterinformation('schechter7_contlin_nodest_mnoise1', schechter7_mass_mnoise1, age_contlin)
	outputclusterinformation('schechter7_contlin_nodest_mnoise2', schechter7_mass_mnoise2, age_contlin)

if output_contlin > 2:

	outputclusterinformation('powerlaw_contlin_massinddest', powerlaw_mass_contlin_massinddest, age_contlin)
	outputclusterinformation('powerlaw_contlin_lamerdest', powerlaw_mass_contlin_lamerdest, age_contlin)
	outputclusterinformation('schechter5_contlin_massinddest', schechter5_mass_contlin_massinddest, age_contlin)
	outputclusterinformation('schechter5_contlin_lamerdest', schechter5_mass_contlin_lamerdest, age_contlin)
	outputclusterinformation('schechter6_contlin_massinddest', schechter6_mass_contlin_massinddest, age_contlin)
	outputclusterinformation('schechter6_contlin_lamerdest', schechter6_mass_contlin_lamerdest, age_contlin)
	outputclusterinformation('schechter7_contlin_massinddest', schechter7_mass_contlin_massinddest, age_contlin)
	outputclusterinformation('schechter7_contlin_lamerdest', schechter7_mass_contlin_lamerdest, age_contlin)

if output_gauss > 0:

	outputclusterinformation('powerlaw_gauss_nodest', powerlaw_mass, age_gauss)
	outputclusterinformation('powerlaw_gauss_nodest_mnoise1', powerlaw_mass_mnoise1, age_gauss)
	outputclusterinformation('powerlaw_gauss_nodest_mnoise2', powerlaw_mass_mnoise2, age_gauss)
	outputclusterinformation('schechter5_gauss_nodest', schechter5_mass, age_gauss)
	outputclusterinformation('schechter5_gauss_nodest_mnoise1', schechter5_mass_mnoise1, age_gauss)
	outputclusterinformation('schechter5_gauss_nodest_mnoise2', schechter5_mass_mnoise2, age_gauss)
	outputclusterinformation('schechter6_gauss_nodest', schechter6_mass, age_gauss)
	outputclusterinformation('schechter6_gauss_nodest_mnoise1', schechter6_mass_mnoise1, age_gauss)
	outputclusterinformation('schechter6_gauss_nodest_mnoise2', schechter6_mass_mnoise2, age_gauss)
	outputclusterinformation('schechter7_gauss_nodest', schechter7_mass, age_gauss)
	outputclusterinformation('schechter7_gauss_nodest_mnoise1', schechter7_mass_mnoise1, age_gauss)
	outputclusterinformation('schechter7_gauss_nodest_mnoise2', schechter7_mass_mnoise2, age_gauss)

if output_gauss > 2:

	outputclusterinformation('powerlaw_gauss_massinddest', powerlaw_mass_gauss_massinddest, age_gauss)
	outputclusterinformation('powerlaw_gauss_lamerdest', powerlaw_mass_gauss_lamerdest, age_gauss)
	outputclusterinformation('schechter5_gauss_massinddest', schechter5_mass_gauss_massinddest, age_gauss)
	outputclusterinformation('schechter5_gauss_lamerdest', schechter5_mass_gauss_lamerdest, age_gauss)
	outputclusterinformation('schechter6_gauss_massinddest', schechter6_mass_gauss_massinddest, age_gauss)
	outputclusterinformation('schechter6_gauss_lamerdest', schechter6_mass_gauss_lamerdest, age_gauss)
	outputclusterinformation('schechter7_gauss_massinddest', schechter7_mass_gauss_massinddest, age_gauss)
	outputclusterinformation('schechter7_gauss_lamerdest', schechter7_mass_gauss_lamerdest, age_gauss)

if output_infant > 0:

	outputclusterinformation('powerlaw_infant_nodest', powerlaw_mass, age_infant)
	outputclusterinformation('powerlaw_infant_nodest_mnoise1', powerlaw_mass_mnoise1, age_infant)
	outputclusterinformation('powerlaw_infant_nodest_mnoise2', powerlaw_mass_mnoise2, age_infant)
	outputclusterinformation('schechter5_infant_nodest', schechter5_mass, age_infant)
	outputclusterinformation('schechter5_infant_nodest_mnoise1', schechter5_mass_mnoise1, age_infant)
	outputclusterinformation('schechter5_infant_nodest_mnoise2', schechter5_mass_mnoise2, age_infant)
	outputclusterinformation('schechter6_infant_nodest', schechter6_mass, age_infant)
	outputclusterinformation('schechter6_infant_nodest_mnoise1', schechter6_mass_mnoise1, age_infant)
	outputclusterinformation('schechter6_infant_nodest_mnoise2', schechter6_mass_mnoise2, age_infant)
	outputclusterinformation('schechter7_infant_nodest', schechter7_mass, age_infant)
	outputclusterinformation('schechter7_infant_nodest_mnoise1', schechter7_mass_mnoise1, age_infant)
	outputclusterinformation('schechter7_infant_nodest_mnoise2', schechter7_mass_mnoise2, age_infant)

if output_infant > 2:

	outputclusterinformation('powerlaw_infant_massinddest', powerlaw_mass_infant_massinddest, age_infant)
	outputclusterinformation('powerlaw_infant_lamerdest', powerlaw_mass_infant_lamerdest, age_infant)
	outputclusterinformation('schechter5_infant_massinddest', schechter5_mass_infant_massinddest, age_infant)
	outputclusterinformation('schechter5_infant_lamerdest', schechter5_mass_infant_lamerdest, age_infant)
	outputclusterinformation('schechter6_infant_massinddest', schechter6_mass_infant_massinddest, age_infant)
	outputclusterinformation('schechter6_infant_lamerdest', schechter6_mass_infant_lamerdest, age_infant)
	outputclusterinformation('schechter7_infant_massinddest', schechter7_mass_infant_massinddest, age_infant)
	outputclusterinformation('schechter7_infant_lamerdest', schechter7_mass_infant_lamerdest, age_infant)

###
# Step 5 - Make Diagnostic Plots
###

print("Step 5 - Make Diagnostic Plots")
###

print('>>> 00A_Age_Distributions')
fig = plt.figure(figsize = (12, 12))
ax1 = fig.add_subplot(111)
###
plt.hist([age_contlog], bins = age_bins_log, color = ['k'], histtype = 'step', stacked = True, label = 'Continuous (log)')
plt.hist([age_contlin], bins = age_bins_log, color = ['b'], histtype = 'step', stacked = True, label = 'Continuous (linear)')
plt.hist([age_gauss], bins = age_bins_log, color = ['r'], histtype = 'step', stacked = True, label = 'Gaussian Burst')
plt.hist([age_infant], bins = age_bins_log, color = ['g'], histtype = 'step', stacked = True, label = 'Infant Mortality')
###
plt.legend(loc = 'upper right')
plt.xlabel(r'Age (yr)')
plt.ylabel(r'N')
plt.xscale('log', nonposx = 'clip')
plt.yscale('log', nonposy = 'clip')
plt.axis(age_range_log + [1, Ntotal * 2.5])
plt.savefig('./SimSCFigures/00A_Age_Distributions.png')
plt.close()

### --<>--<>--<>-- ###

print('>>> 00B_Age_Distributions_Scatter')
fig = plt.figure(figsize = (12, 12))
###
ax1 = fig.add_subplot(221)
plt.plot(random_num, age_contlog, 'ko', markersize = 2, label = 'Continuous (log)')
plt.title('Continuous (log)')
plt.yscale('log', nonposy = 'clip')
ax1 = fig.add_subplot(222)
plt.plot(random_num, age_contlin, 'bo', markersize = 2, label = 'Continuous (linear)')
plt.title('Continuous (linear)')
plt.yscale('log', nonposy = 'clip')
ax1 = fig.add_subplot(223)
plt.plot(random_num, age_gauss, 'ro', markersize = 2, label = 'Gaussian Burst')
plt.title('Gaussian Burst')
plt.yscale('log', nonposy = 'clip')
ax1 = fig.add_subplot(224)
plt.plot(random_num, age_infant, 'go', markersize = 2, label = 'Infant Mortality')
plt.title('Infant Mortality')
plt.yscale('log', nonposy = 'clip')
###
plt.xlabel(r'N')
plt.ylabel(r'Age (yr)')
plt.savefig('./SimSCFigures/00B_Age_Distributions_Scatter.png')
plt.close()

### --<>--<>--<>-- ###

print('>>> 01_PDF')
fig = plt.figure(figsize = (12, 12))
ax1 = fig.add_subplot(111)
###
plt.plot(mass_distribution_array, powerlaw(mass_distribution_array) / powerlaw_norm[0], 'k-', label = 'Power Law')
plt.plot(mass_distribution_array, schechter5(mass_distribution_array) / schechter5_norm[0], 'b-', label = 'Schechter Function (1E5)')
plt.plot(mass_distribution_array, schechter6(mass_distribution_array) / schechter6_norm[0], 'g-', label = 'Schechter Function (1E6)')
plt.plot(mass_distribution_array, schechter7(mass_distribution_array) / schechter7_norm[0], 'r-', label = 'Schechter Function (1E7)')
plt.plot(mass_distribution_array, truncated6(mass_distribution_array) / truncated6_norm[0], 'g:', label = 'Truncated Power Law (1E6)')
###
plt.legend(loc = 'upper right')
plt.xlabel(r'M [M$_\odot$]')
plt.ylabel(r'Probability')
plt.xscale('log', nonposx = 'clip')
plt.yscale('log', nonposy = 'clip')
plt.axis(mass_distribution + [1E-15, 1E-3])
plt.savefig('./SimSCFigures/01_PDF.png')
plt.close()

### --<>--<>--<>-- ###

print('>>> 02_CDF')
fig = plt.figure(figsize = (12, 12))
ax1 = fig.add_subplot(111)
###
plt.plot(mass_distribution_array, powerlaw_cumulative, 'k-', label = 'Power Law')
plt.plot(mass_distribution_array, schechter5_cumulative, 'b-', label = 'Schechter Function (1E5)')
plt.plot(mass_distribution_array, schechter6_cumulative, 'g-', label = 'Schechter Function (1E6)')
plt.plot(mass_distribution_array, schechter7_cumulative, 'r-', label = 'Schechter Function (1E7)')
###
plt.legend(loc = 'upper right')
plt.xlabel(r'M [M$_\odot$]')
plt.ylabel(r'Probability')
plt.xscale('log', nonposx = 'clip')
plt.yscale('log', nonposy = 'clip')
plt.axis(mass_distribution + [1E-4, 5])
plt.savefig('./SimSCFigures/02_CDF.png')
plt.close()

### --<>--<>--<>-- ###

print('>>> 03A_Initial_Mass')
fig = plt.figure(figsize = (12, 12))
ax1 = fig.add_subplot(111)
###
plt.plot(random_num, powerlaw_mass, 'ko', label = 'Power Law', markersize = 1.5, alpha = 0.4)
plt.plot(random_num, schechter5_mass, 'bo', label = 'Schechter Function (1E5)', markersize = 1.5, alpha = 0.4)
plt.plot(random_num, schechter6_mass, 'go', label = 'Schechter Function (1E6)', markersize = 1.5, alpha = 0.4)
plt.plot(random_num, schechter7_mass, 'ro', label = 'Schechter Function (1E7)', markersize = 1.5, alpha = 0.4)
###
plt.legend(loc = 'upper right')
plt.xlabel(r'N')
plt.ylabel(r'M [M$_\odot$]')
# plt.xscale('log', nonposx = 'clip')
plt.yscale('log', nonposy = 'clip')
plt.axis([0, Ntotal] + mass_distribution)
plt.savefig('./SimSCFigures/03A_Initial_Mass.png')
plt.close()

### --<>--<>--<>-- ###

print('>>> 03B_Initial_Mass_Hist')
fig = plt.figure(figsize = (12, 12))
ax1 = fig.add_subplot(111)
###
plt.hist([powerlaw_mass], bins = mass_bins_log, color = ['k'], histtype = 'step', stacked = True, label = 'Power Law')
plt.hist([powerlaw_mass_mnoise1], bins = mass_bins_log, color = ['k'], histtype = 'step', stacked = True, label = 'Power Law + Error', alpha = 0.5)
plt.hist([schechter5_mass], bins = mass_bins_log, color = ['b'], histtype = 'step', stacked = True, label = 'Schechter Function (1E5)')
plt.hist([schechter5_mass_mnoise1], bins = mass_bins_log, color = ['b'], histtype = 'step', stacked = True, label = 'Schechter Function (1E5) + Error', alpha = 0.5)
plt.hist([schechter6_mass], bins = mass_bins_log, color = ['g'], histtype = 'step', stacked = True, label = 'Schechter Function (1E6)')
plt.hist([schechter6_mass_mnoise1], bins = mass_bins_log, color = ['g'], histtype = 'step', stacked = True, label = 'Schechter Function (1E6) + Error', alpha = 0.5)
plt.hist([schechter7_mass], bins = mass_bins_log, color = ['r'], histtype = 'step', stacked = True, label = 'Schechter Function (1E7)')
plt.hist([schechter7_mass_mnoise1], bins = mass_bins_log, color = ['r'], histtype = 'step', stacked = True, label = 'Schechter Function (1E7) + Error', alpha = 0.5)
###
plt.legend(loc = 'upper right')
plt.xlabel(r'M [M$_\odot$]')
plt.ylabel(r'N')
plt.xscale('log', nonposx = 'clip')
plt.yscale('log', nonposy = 'clip')
plt.axis(mass_distribution + [1, 1E6])
plt.savefig('./SimSCFigures/03B_Initial_Mass_Hist.png')
plt.close()

### --<>--<>--<>-- ###
print('>>> 03C_Initial_Mass_ContLin')
fig = plt.figure(figsize = (12, 12))
ax1 = fig.add_subplot(111)
###
plt.plot(age_contlin, powerlaw_mass, 'ko', label = 'Power Law', markersize = 1.5, alpha = 0.4)
# plt.plot(age_contlin, schechter5_mass, 'bo', label = 'Schechter Function (1E5)', markersize = 1.5, alpha = 0.4)
# plt.plot(age_contlin, schechter6_mass, 'go', label = 'Schechter Function (1E6)', markersize = 1.5, alpha = 0.4)
# plt.plot(age_contlin, schechter7_mass, 'ro', label = 'Schechter Function (1E7)', markersize = 1.5, alpha = 0.4)
###
plt.legend(loc = 'upper right')
plt.xlabel(r'Age (yr)')
plt.ylabel(r'M [M$_\odot$]')
plt.xscale('log', nonposx = 'clip')
plt.yscale('log', nonposy = 'clip')
plt.axis(age_distribution + mass_distribution)
plt.savefig('./SimSCFigures/03C_Initial_Mass_ContLin.png')
plt.close()

### --<>--<>--<>-- ###
print('>>> 03D_Initial_Mass_ContLog')
fig = plt.figure(figsize = (12, 12))
ax1 = fig.add_subplot(111)
###
plt.plot(age_contlog, powerlaw_mass, 'ko', label = 'Power Law', markersize = 1.5, alpha = 0.4)
# plt.plot(age_contlog, schechter5_mass, 'bo', label = 'Schechter Function (1E5)', markersize = 1.5, alpha = 0.4)
# plt.plot(age_contlog, schechter6_mass, 'go', label = 'Schechter Function (1E6)', markersize = 1.5, alpha = 0.4)
# plt.plot(age_contlog, schechter7_mass, 'ro', label = 'Schechter Function (1E7)', markersize = 1.5, alpha = 0.4)
###
plt.legend(loc = 'upper right')
plt.xlabel(r'Age (yr)')
plt.ylabel(r'M [M$_\odot$]')
plt.xscale('log', nonposx = 'clip')
plt.yscale('log', nonposy = 'clip')
plt.axis(age_distribution + mass_distribution)
plt.savefig('./SimSCFigures/03C_Initial_Mass_ContLog.png')
plt.close()

### --<>--<>--<>-- ###

print('>>> 04A_BC03')
fig = plt.figure(figsize = (12, 12))
ax1 = fig.add_subplot(111)
###
plt.plot(np.power(10, cb07_basel_m62_chap[:,0]), cb07_basel_m62_chap[:,3], 'b-', label = 'Bmag')
plt.plot(np.power(10, cb07_basel_m62_chap[:,0]), cb07_basel_m62_chap[:,4], 'g-', label = 'Vmag')
plt.plot(np.power(10, cb07_basel_m62_chap[:,0]), cb07_basel_m62_chap[:,5], 'r-', label = 'Kmag')
###
plt.legend(loc = 'upper right')
plt.xlabel(r'Age (yr)')
plt.ylabel(r'Magnitude')
plt.xscale('log', nonposx = 'clip')
# plt.yscale('log', nonposy = 'clip')
plt.axis(age_distribution + colour_range)
plt.savefig('./SimSCFigures/04A_BC03.png')
plt.close()

### --<>--<>--<>-- ###

print('>>> 05A11_MassAge_PowerLaw_ContLin')
fig = plt.figure(figsize = (12, 12))
ax1 = fig.add_subplot(111)
###
plt.scatter(age_contlin, powerlaw_mass, c = vmag_contlin_powerlaw_mcorr)
plt.colorbar()
###
plt.xlabel(r'Age (yr)')
plt.ylabel(r'M [M$_\odot$]')
plt.xscale('log', nonposx = 'clip')
plt.yscale('log', nonposy = 'clip')
plt.axis(age_distribution + [1E1, 1E8])
plt.savefig('./SimSCFigures/05A11_MassAge_PowerLaw_ContLin.png')
plt.close()

### --<>--<>--<>-- ###

print('>>> 05A12_MassAge_PowerLaw_ContLin_MagCut')
fig = plt.figure(figsize = (12, 12))
ax1 = fig.add_subplot(111)
###
plt.scatter(age_contlin[vmag_contlin_powerlaw_mcorr < maglimit], powerlaw_mass[vmag_contlin_powerlaw_mcorr < maglimit], c = vmag_contlin_powerlaw_mcorr[vmag_contlin_powerlaw_mcorr < maglimit])
plt.colorbar()
###
plt.xlabel(r'Age (yr)')
plt.ylabel(r'M [M$_\odot$]')
plt.xscale('log', nonposx = 'clip')
plt.yscale('log', nonposy = 'clip')
plt.axis(age_distribution + [1E1, 1E8])
plt.savefig('./SimSCFigures/05A12_MassAge_PowerLaw_ContLin_MagCut.png')
plt.close()

### --<>--<>--<>-- ###

print('>>> 05B11_MassAge_PowerLaw_ContLog')
fig = plt.figure(figsize = (12, 12))
ax1 = fig.add_subplot(111)
###
plt.scatter(age_contlog, powerlaw_mass, c = vmag_contlog_powerlaw_mcorr)
plt.colorbar()
###
plt.xlabel(r'Age (yr)')
plt.ylabel(r'M [M$_\odot$]')
plt.xscale('log', nonposx = 'clip')
plt.yscale('log', nonposy = 'clip')
plt.axis(age_distribution + [1E1, 1E8])
plt.savefig('./SimSCFigures/05B11_MassAge_PowerLaw_ContLog.png')
plt.close()

### --<>--<>--<>-- ###

print('>>> 05B12_MassAge_PowerLaw_ContLog_MagCut')
fig = plt.figure(figsize = (12, 12))
ax1 = fig.add_subplot(111)
###
plt.scatter(age_contlog[vmag_contlog_powerlaw_mcorr < maglimit], powerlaw_mass[vmag_contlog_powerlaw_mcorr < maglimit], c = vmag_contlog_powerlaw_mcorr[vmag_contlog_powerlaw_mcorr < maglimit])
plt.colorbar()
###
plt.xlabel(r'Age (yr)')
plt.ylabel(r'M [M$_\odot$]')
plt.xscale('log', nonposx = 'clip')
plt.yscale('log', nonposy = 'clip')
plt.axis(age_distribution + [1E1, 1E8])
plt.savefig('./SimSCFigures/05B12_MassAge_PowerLaw_ContLog_MagCut.png')
plt.close()

### --<>--<>--<>-- ###

print('>>> 05B21_MassAge_PowerLaw_ContLog_MassIndep')
fig = plt.figure(figsize = (12, 12))
ax1 = fig.add_subplot(111)
###
plt.scatter(age_contlog[powerlaw_mass_contlog_massinddest > 0], powerlaw_mass_contlog_massinddest[powerlaw_mass_contlog_massinddest > 0], c = vmag_contlog_powerlaw_lamerdest_mcorr[powerlaw_mass_contlog_massinddest > 0])
plt.colorbar()
###
plt.xlabel(r'Age (yr)')
plt.ylabel(r'M [M$_\odot$]')
plt.xscale('log', nonposx = 'clip')
plt.yscale('log', nonposy = 'clip')
plt.axis(age_distribution + [1E1, 1E8])
plt.savefig('./SimSCFigures/05B21_MassAge_PowerLaw_ContLog_MassIndep.png')
plt.close()

### --<>--<>--<>-- ###

print('>>> 05B22_MassAge_PowerLaw_ContLog_MassIndep_MagCut')
fig = plt.figure(figsize = (12, 12))
ax1 = fig.add_subplot(111)
###
plt.scatter(age_contlog[vmag_contlog_powerlaw_mcorr < maglimit], powerlaw_mass_contlog_massinddest[vmag_contlog_powerlaw_mcorr < maglimit], c = vmag_contlog_powerlaw_lamerdest_mcorr[vmag_contlog_powerlaw_mcorr < maglimit])
plt.colorbar()
###
plt.xlabel(r'Age (yr)')
plt.ylabel(r'M [M$_\odot$]')
plt.xscale('log', nonposx = 'clip')
plt.yscale('log', nonposy = 'clip')
plt.axis(age_distribution + [1E1, 1E8])
plt.savefig('./SimSCFigures/05B22_MassAge_PowerLaw_ContLog_MassIndep_MagCut.png')
plt.close()

### --<>--<>--<>-- ###

print('>>> 05B31_MassAge_PowerLaw_ContLog_Lamers')
fig = plt.figure(figsize = (12, 12))
ax1 = fig.add_subplot(111)
###
plt.scatter(age_contlog[powerlaw_mass_contlog_lamerdest > 0], powerlaw_mass_contlog_lamerdest[powerlaw_mass_contlog_lamerdest > 0], c = vmag_contlog_powerlaw_lamerdest_mcorr[powerlaw_mass_contlog_lamerdest > 0])
plt.colorbar()
###
plt.xlabel(r'Age (yr)')
plt.ylabel(r'M [M$_\odot$]')
plt.xscale('log', nonposx = 'clip')
plt.yscale('log', nonposy = 'clip')
plt.axis(age_distribution + [1E1, 1E8])
plt.savefig('./SimSCFigures/05B31_MassAge_PowerLaw_ContLog_Lamers.png')
plt.close()

### --<>--<>--<>-- ###

print('>>> 05B32_MassAge_PowerLaw_ContLog_Lamers_MagCut')
fig = plt.figure(figsize = (12, 12))
ax1 = fig.add_subplot(111)
###
plt.scatter(age_contlog[vmag_contlog_powerlaw_mcorr < maglimit], powerlaw_mass_contlog_lamerdest[vmag_contlog_powerlaw_mcorr < maglimit], c = vmag_contlog_powerlaw_lamerdest_mcorr[vmag_contlog_powerlaw_mcorr < maglimit])
plt.colorbar()
###
plt.xlabel(r'Age (yr)')
plt.ylabel(r'M [M$_\odot$]')
plt.xscale('log', nonposx = 'clip')
plt.yscale('log', nonposy = 'clip')
plt.axis(age_distribution + [1E1, 1E8])
plt.savefig('./SimSCFigures/05B32_MassAge_PowerLaw_ContLog_Lamers_MagCut.png')
plt.close()

### --<>--<>--<>-- ###

# Test the number of simulated galaxies required (above a certain mass limit)
age_contlog_age1 = age_contlog[age_contlog < 1E7]
age_contlog_age2 = age_contlog[np.where((age_contlog > 1E7) & (age_contlog < 1E8))]
age_contlog_age3 = age_contlog[np.where((age_contlog > 1E8) & (age_contlog < 4 * 1E8))]
###
powerlaw_mass_age1 = powerlaw_mass[age_contlog < 1E7]
powerlaw_mass_age2 = powerlaw_mass[np.where((age_contlog > 1E7) & (age_contlog < 1E8))]
powerlaw_mass_age3 = powerlaw_mass[np.where((age_contlog > 1E8) & (age_contlog < 4 * 1E8))]

### --<>--<>--<>-- ###

print('>>> 06_Simulate_PowerLaw_N30')
fig = plt.figure(figsize = (12, 12))
ax1 = fig.add_subplot(111)
###
plt.plot(random_num, np.cumsum(np.where((age_contlog < 1E7) & (powerlaw_mass > 1E3), 1, 0)), 'k--', label = r'$\tau <$ 10 Myr')
plt.plot(random_num, np.cumsum(np.where((age_contlog > 1E7) & (age_contlog < 1E8) & (powerlaw_mass > 1E3), 1, 0)), 'r--', label = r'10 Myr $< \tau <$ 100 Myr')
plt.plot(random_num, np.cumsum(np.where((age_contlog > 1E8) & (age_contlog < 4 * 1E8) & (powerlaw_mass > 1E3), 1, 0)), 'g--', label = r'100 Myr $< \tau <$ 400 Myr')
###
plt.xlabel(r'N')
plt.legend(loc = 'upper right')
plt.ylabel(r'N (> 10$^3$ M$_\odot$)')
plt.xscale('log', nonposx = 'clip')
plt.yscale('log', nonposy = 'clip')
plt.axis([np.power(10, -0.25), np.power(10, 5), np.power(10, -0.25), np.power(10, 5)])
plt.savefig('./SimSCFigures/06_Simulate_PowerLaw_N30.png')
plt.close()

### --<>--<>--<>-- ###

print('>>> 06_Simulate_PowerLaw_N40')
fig = plt.figure(figsize = (12, 12))
ax1 = fig.add_subplot(111)
###
plt.plot(random_num, np.cumsum(np.where((age_contlog < 1E7) & (powerlaw_mass > 1E4), 1, 0)), 'k--', label = r'$\tau <$ 10 Myr')
plt.plot(random_num, np.cumsum(np.where((age_contlog > 1E7) & (age_contlog < 1E8) & (powerlaw_mass > 1E4), 1, 0)), 'r--', label = r'10 Myr $< \tau <$ 100 Myr')
plt.plot(random_num, np.cumsum(np.where((age_contlog > 1E8) & (age_contlog < 4 * 1E8) & (powerlaw_mass > 1E4), 1, 0)), 'g--', label = r'100 Myr $< \tau <$ 400 Myr')
###
plt.xlabel(r'N')
plt.legend(loc = 'upper right')
plt.ylabel(r'N (> 10$^4$ M$_\odot$)')
plt.xscale('log', nonposx = 'clip')
plt.yscale('log', nonposy = 'clip')
plt.axis([np.power(10, -0.25), np.power(10, 5), np.power(10, -0.25), np.power(10, 5)])
plt.savefig('./SimSCFigures/06_Simulate_PowerLaw_N40.png')
plt.close()

### --<>--<>--<>-- ###

print('>>> 06_Simulate_PowerLaw_N45')
fig = plt.figure(figsize = (12, 12))
ax1 = fig.add_subplot(111)
###
plt.plot(random_num, np.cumsum(np.where((age_contlog < 1E7) & (powerlaw_mass > np.power(10, 4.5)), 1, 0)), 'k--', label = r'$\tau <$ 10 Myr')
plt.plot(random_num, np.cumsum(np.where((age_contlog > 1E7) & (age_contlog < 1E8) & (powerlaw_mass > np.power(10, 4.5)), 1, 0)), 'r--', label = r'10 Myr $< \tau <$ 100 Myr')
plt.plot(random_num, np.cumsum(np.where((age_contlog > 1E8) & (age_contlog < 4 * 1E8) & (powerlaw_mass > np.power(10, 4.5)), 1, 0)), 'g--', label = r'100 Myr $< \tau <$ 400 Myr')
###
plt.xlabel(r'N')
plt.legend(loc = 'upper right')
plt.ylabel(r'N (> 10$^{4.5}$ M$_\odot$)')
plt.xscale('log', nonposx = 'clip')
plt.yscale('log', nonposy = 'clip')
plt.axis([np.power(10, -0.25), np.power(10, 5), np.power(10, -0.25), np.power(10, 5)])
plt.savefig('./SimSCFigures/06_Simulate_PowerLaw_N45.png')
plt.close()

### --<>--<>--<>-- ###

print('>>> 06_Simulate_PowerLaw_N50')
fig = plt.figure(figsize = (12, 12))
ax1 = fig.add_subplot(111)
###
plt.plot(random_num, np.cumsum(np.where((age_contlog < 1E7) & (powerlaw_mass > 1E5), 1, 0)), 'k--', label = r'$\tau <$ 10 Myr')
plt.plot(random_num, np.cumsum(np.where((age_contlog > 1E7) & (age_contlog < 1E8) & (powerlaw_mass > 1E5), 1, 0)), 'r--', label = r'10 Myr $< \tau <$ 100 Myr')
plt.plot(random_num, np.cumsum(np.where((age_contlog > 1E8) & (age_contlog < 4 * 1E8) & (powerlaw_mass > 1E5), 1, 0)), 'g--', label = r'100 Myr $< \tau <$ 400 Myr')
###
plt.legend(loc = 'upper right')
plt.xlabel(r'N')
plt.ylabel(r'N (> 10$^5$ M$_\odot$)')
plt.xscale('log', nonposx = 'clip')
plt.yscale('log', nonposy = 'clip')
plt.axis([np.power(10, -0.25), np.power(10, 5), np.power(10, -0.25), np.power(10, 5)])
plt.savefig('./SimSCFigures/06_Simulate_PowerLaw_N50.png')
plt.close()

### --<>--<>--<>-- ###

print('>>> 07B11_MassAge_Schechter5_ContLog')
fig = plt.figure(figsize = (12, 12))
ax1 = fig.add_subplot(111)
###
plt.scatter(age_contlog, schechter5_mass, c = vmag_contlog_schechter5_mcorr)
plt.colorbar()
###
plt.xlabel(r'Age (yr)')
plt.ylabel(r'M [M$_\odot$]')
plt.xscale('log', nonposx = 'clip')
plt.yscale('log', nonposy = 'clip')
plt.axis(age_distribution + [1E1, 1E8])
plt.savefig('./SimSCFigures/07B11_MassAge_Schechter5_ContLog.png')
plt.close()

### --<>--<>--<>-- ###

print('>>> 07B12_MassAge_Schechter5_ContLog_MagCut')
fig = plt.figure(figsize = (12, 12))
ax1 = fig.add_subplot(111)
###
plt.scatter(age_contlog[vmag_contlog_schechter5_mcorr < maglimit], schechter5_mass[vmag_contlog_schechter5_mcorr < maglimit], c = vmag_contlog_schechter5_mcorr[vmag_contlog_schechter5_mcorr < maglimit])
plt.colorbar()
###
plt.xlabel(r'Age (yr)')
plt.ylabel(r'M [M$_\odot$]')
plt.xscale('log', nonposx = 'clip')
plt.yscale('log', nonposy = 'clip')
plt.axis(age_distribution + [1E1, 1E8])
plt.savefig('./SimSCFigures/07B12_MassAge_Schechter5_ContLog_MagCut.png')
plt.close()

### --<>--<>--<>-- ###

print('>>> 07B21_MassAge_Schechter5_ContLog_MassIndep')
fig = plt.figure(figsize = (12, 12))
ax1 = fig.add_subplot(111)
###
plt.scatter(age_contlog[schechter5_mass_contlog_massinddest > 0], schechter5_mass_contlog_massinddest[schechter5_mass_contlog_massinddest > 0], c = vmag_contlog_schechter5_lamerdest_mcorr[schechter5_mass_contlog_massinddest > 0])
plt.colorbar()
###
plt.xlabel(r'Age (yr)')
plt.ylabel(r'M [M$_\odot$]')
plt.xscale('log', nonposx = 'clip')
plt.yscale('log', nonposy = 'clip')
plt.axis(age_distribution + [1E1, 1E8])
plt.savefig('./SimSCFigures/07B21_MassAge_Schechter5_ContLog_MassIndep.png')
plt.close()

### --<>--<>--<>-- ###

print('>>> 07B22_MassAge_Schechter5_ContLog_MassIndep_MagCut')
fig = plt.figure(figsize = (12, 12))
ax1 = fig.add_subplot(111)
###
plt.scatter(age_contlog[vmag_contlog_schechter5_mcorr < maglimit], schechter5_mass_contlog_massinddest[vmag_contlog_schechter5_mcorr < maglimit], c = vmag_contlog_schechter5_lamerdest_mcorr[vmag_contlog_schechter5_mcorr < maglimit])
plt.colorbar()
###
plt.xlabel(r'Age (yr)')
plt.ylabel(r'M [M$_\odot$]')
plt.xscale('log', nonposx = 'clip')
plt.yscale('log', nonposy = 'clip')
plt.axis(age_distribution + [1E1, 1E8])
plt.savefig('./SimSCFigures/07B22_MassAge_Schechter5_ContLog_MassIndep_MagCut.png')
plt.close()

### --<>--<>--<>-- ###

print('>>> 07B31_MassAge_Schechter5_ContLog_Lamers')
fig = plt.figure(figsize = (12, 12))
ax1 = fig.add_subplot(111)
###
plt.scatter(age_contlog[schechter5_mass_contlog_lamerdest > 0], schechter5_mass_contlog_lamerdest[schechter5_mass_contlog_lamerdest > 0], c = vmag_contlog_schechter5_lamerdest_mcorr[schechter5_mass_contlog_lamerdest > 0])
plt.colorbar()
###
plt.xlabel(r'Age (yr)')
plt.ylabel(r'M [M$_\odot$]')
plt.xscale('log', nonposx = 'clip')
plt.yscale('log', nonposy = 'clip')
plt.axis(age_distribution + [1E1, 1E8])
plt.savefig('./SimSCFigures/07B31_MassAge_Schechter5_ContLog_Lamers.png')
plt.close()

### --<>--<>--<>-- ###

print('>>> 07B32_MassAge_Schechter5_ContLog_Lamers_MagCut')
fig = plt.figure(figsize = (12, 12))
ax1 = fig.add_subplot(111)
###
plt.scatter(age_contlog[vmag_contlog_schechter5_mcorr < maglimit], schechter5_mass_contlog_lamerdest[vmag_contlog_schechter5_mcorr < maglimit], c = vmag_contlog_schechter5_lamerdest_mcorr[vmag_contlog_schechter5_mcorr < maglimit])
plt.colorbar()
###
plt.xlabel(r'Age (yr)')
plt.ylabel(r'M [M$_\odot$]')
plt.xscale('log', nonposx = 'clip')
plt.yscale('log', nonposy = 'clip')
plt.axis(age_distribution + [1E1, 1E8])
plt.savefig('./SimSCFigures/07B32_MassAge_Schechter5_ContLog_Lamers_MagCut.png')
plt.close()
