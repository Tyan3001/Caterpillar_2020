import numpy as np
import hdf_data_functions as hdf

files = hdf.get_filenames('competitionfiles')


channel = 'ch_74'
cor = hdf.get_channel_correlations(channel, files[4], .8)

max_cor = ['0', '0', 0]

print(cor)

file = 'COOLCAT_20110614_060338_45_20110614_060338_450.hdf'

#for cvls in cor:
#    if cvls[1] != channel:
#        if cvls[2] > max_cor[2]:
#            max_cor = cvls



print(max_cor)


#beanz = hdf.predict_channel_file(file, max_cor[1], channel)
#print (beanz)


hdf.plot_day_regression(file, 'ch_101')


#predict channel file


