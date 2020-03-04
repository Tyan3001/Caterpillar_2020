import hdf_data_functions as hdf 
import numpy as np
import h5py
import z_test as z
import matplotlib.pyplot as plt
#files = hdf.get_filenames('competitionfiles')
#for f in files:
#    print(f)

#nf = hdf.filter_new_file(str('competitionfiles/' + files[0]))



#hdf.write_entire_files_to_csv(files)

#dictionary = hdf.read_csv()
#print(dictionary)
#print(numpy_array)

#csvdict = hdf.read_csv()

#for d in csvdict:
#    for key in d.keys():
#        if key == 'ch_1':
#            print(d[key][0]) 

means = hdf.get_all_means()

clean_means = means
print(clean_means)

for pair in clean_means:
    interval = z.construct_95conf_interval(pair[1])
    pair[1] = z.filter_dset(pair[1], interval)
x = [i for i in range(1, len(clean_means[0][1]) + 1)]
print("\n\n\n")
print(clean_means[0][1])

plt.plot(x, clean_means[0][1])
#plt.plot(x, means[0][1])


plt.xlabel(clean_means[0][0])
plt.ylabel('means')
plt.title('Bruh u gae')
#plt.show()
           
hf = h5py.File('new_mean_data.h5', 'w')
for pair in clean_means:
    y = [i for i in range(1, len(pair[1]) + 1)]
    hf.create_dataset(pair[0], pair[1])
hf.close()
#print(means)
