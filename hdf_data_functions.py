import h5py
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing 
import csv
from itertools import chain
from scipy.stats import spearmanr
import random
import sys

cwd = os.getcwd()





def plot_day_regression(filename, channel):
    filepath = str('competitionfiles/' + filename)
    f = h5py.File(filepath, 'r')
    chanIDs = f['DYNAMIC DATA']
    dset = chanIDs[channel]['MEASURED']
    arr = [i for i in range(len(dset))]
    plt.plot(arr, dset[()])
    plt.show
















def reject_outliers(data, m = 2):
    return data[abs(data - np.mean(data)) <= m * np.std(data)]


def find_expected(file_name, ch2):
    files = get_filenames()
    loc = 0
    i = 0
    for x in files:
        if file_name == x:
            loc = i
        i+=1
    
    correlations = get_channel_correlations(ch2, file_name[loc - 1], .8)
    
    #for y in correlations:
       
    return 



def sigmoi(x, deriv = False):
    if(deriv == True):
        return x * (1-x)
    return 1 / (1 + np.exp(-x))



def predict_channel_file(f, ch1, ch2):
    
    the_set = get_set( f, ch1, ch2)
     
    in_set = np.array(the_set[0])
    out_set = np.array(the_set[1])
    fin_test = np.array(the_set[2])
    expected = nnpredict2(in_set, out_set, fin_test)
    return expected


def nnpredict(input_sets, output_set, final_test):
    input_sets = input_sets

    weights = 2*np.random.random((100,1)) - 1

    for x in range(0, 5):
        inset = input_sets
        sigmoid_test = sigmoi(np.dot(inset, weights))
        print(sigmoid_test)
        sig_test_err = output_set - sigmoid_test

        weight_adjust = sig_test_err * sigmoi(sigmoid_test, True)

        weights = weights + np.dot(inset.T, weight_adjust)
    print(" Neural Nut test done")
    print(weights)
    expected_out = sigmoi(np.dot(final_test, weights))
    print(expected_out)
    return expected_out
    



def nnpredict2(input_sets, output_set, final_test):
    
    weights = 2*np.random.random((100,1)) - 1
    print("\n\n\nOUTPUT:")
    print(output_set)
    for x in range(5):

        inset = input_sets

        sigmoid_test = np.dot(inset, weights)
        print(sigmoid_test)
        sig_test_err = output_set - sigmoid_test

        weight_adjust = sig_test_err

        weights = weights + np.dot(inset.T, weight_adjust)

    print(" Neural Nut test done")
   # print(weights)
   
    expected_out = sigmoi(np.dot(final_test, weights))
    print(expected_out)
    return expected_out
 




def get_set(file_name, ch1, ch2):
    #ch1 is correlational channel, ch2 is wanted output
    files = get_filenames('competitionfiles/')
    location = 0
    i = 0
    inputs_other = []
    means_other = []
    #inputs_this = []


    for f in files:
        if file_name == f:
            location = 1
        i += 1
    

    filepath = str('competitionfiles/' + file_name)
    f = h5py.File(filepath, 'r')
    chanIDs = f['DYNAMIC DATA']
    dset = chanIDs[ch1]['MEASURED']
    inputs_this = np.array(pick_100_points(dset))
    f.close

    for x in range(-5, 5):
        i = location + x * 5
        if (i != location):
            filepath = str('competitionfiles/' + files[i])
            print(i)
            f = h5py.File(filepath, 'r')
            chanIDs = f['DYNAMIC DATA']
            print("open file")
            dset1 = chanIDs[ch1]['MEASURED']
            print("dset gotten")
            mi = (np.min(dset1[()]))
            ma = (np.max(dset1[()]))
            if (mi != ma):
                oset = np.array(pick_100_points(dset1))
                print("pickpoints")
                inputs_other.append(oset)
                dset2 = chanIDs[ch2]['MEASURED']
                #if mean == 0 and len == 0:
                
                mean = sum(dset2[()]) / len(dset2[()])
                means_other.append(mean)
                print("done")
            else:
                i -= 1
                
            f.close()
    
    out = [inputs_other, means_other, inputs_this] * 10000
    return out


def pick_100_points(data):
    x = random.sample(range(int(np.min(data[()])), int(np.max(data[()]))), 100)
    print (x)
    return x









def get_channel_correlations(channel, filename, thresh):
    significant = []   
    filepath = str('competitionfiles/' + filename)
    f = h5py.File(filepath, 'r')
    ch1 = channel
    ch2 = 'ch_10'
    chanIDs = f['DYNAMIC DATA']
    dset1 = chanIDs[ch1]['MEASURED']
    checked = []

    chanIDs2 = np.array(chanIDs)
    for ids in chanIDs2:
        if ids not in checked:
            dset2 = chanIDs[ids]['MEASURED']
            np.seterr(divide = 'ignore', invalid = 'ignore')
            corr, _ = spearmanr(dset1[()], dset2[()])
            if abs(corr) > thresh:
                c = [channel, ids, corr]
                significant.append(c)
    
    #print(banana)




    #print(dset.shape)
    #all_dat2d.append(dset)
    #print(all_dat2d)
    #all_dat = all_dat2d.flatten()
    #flatten_list = list(chain.from_iterable(all_dat2d))
    #print (flatten_list)
    #plt.plot(flatten_list[()])
    #plt.title("Value of " + ChannelName)
    #plt.xlabel("Datapoint #")
    #plt.ylabel("Value")
    #plt.show()

    #Close the file
    f.close() 
    return significant






#returns an array of the data in the specified hdf file
def get_data_array(filename):
    f = h5py.File(filename, 'r')
    chanIDs = f['DYNAMIC DATA']
    Mean = 0
    output = {}
    for id in chanIDs.keys():
        dset = np.array(chanIDs[id]['MEASURED'])
        if  np.min(dset) != 0 and np.max(dset) != 0:
            new_dset = reject_outliers(dset)
            if len(new_dset) == 0:
                print(dset)
                print(len(dset))
            else:
                new_dset = (new_dset - new_dset.mean())/new_dset.std()
                if str(np.min(new_dset)) != "nan" and str(np.max(new_dset)) != "nan" and str(np.mean(new_dset)) != "nan":
                    output.update({str(id): (np.mean(new_dset), np.min(new_dset), np.max(new_dset), len(new_dset))})

    f.close()
    return output

#returns an array of all the filenames in the specified directory
def get_filenames(directory):
    filename_array = []
    for filename in os.listdir(directory):
        if filename.endswith(".hdf"):
            filename_array.append(filename)
            #print(filename)
    #print('done with get_filenames')
    return filename_array;
def write_entire_files_to_csv(filename_array):
    with open('full_dat.csv', 'a') as out:
        fw = csv.writer(out)
        for filename in filename_array:
            f = h5py.File(str('competitionfiles/' + filename), 'r')
            chanIDs = f['DYNAMIC DATA']
            Mean = 0
            #output = {}
            fw.writerow([filename])
            for id in chanIDs.keys():
                dset = np.array(chanIDs[id]['MEASURED'])
                if  np.min(dset) != 0 and np.max(dset) != 0:
                    new_dset = reject_outliers(dset)
                    if len(new_dset) == 0:
                        print(dset)
                        print(len(dset))
                    else:
                        new_dset = (new_dset - new_dset.mean())/new_dset.std()
                        if str(np.min(new_dset)) != "nan" and str(np.max(new_dset)) != "nan" and str(np.mean(new_dset)) != "nan":
                        #output.update({str(id): (np.mean(new_dset), np.min(new_dset), np.max(new_dset), len(new_dset))})
                            row = (str(id) + "," + str(np.mean(new_dset)) + "," + str(np.min(new_dset)) + "," + 
                                    str(np.max(new_dset)) + "," + str(len(new_dset)))
                            fw.writerow([row])
                        print(id)
    return 0

def Get_Dictionary():
    i = 1
    filename_array = get_filenames('competitionfiles')
    List_Of_Dictionaries = list()
    for filename in filename_array:
        print(filename)
        List_Of_Dictionaries.append(get_data_array(str("competitionfiles/" + filename)))
        write_to_csv(get_data_array(str("competitionfiles/" + filename)), filename)
        print(i)
        i += 1
    return List_Of_Dictionaries
    #make a list of means , find the mean of the means(sample mean), std_err = list.std()/sqrt(len(list)), z-score = dp - sample mean / std_err, z.cdf(-abs(z-score))
    #95%

def write_to_csv(data_array, filename):
    with open('full_file_data_array.csv', 'a') as out:
        write = csv.writer(out)
        write.writerow((filename, data_array))
    return

def filter_new_file(filename):
    f = h5py.File(filename, 'r')
    chanIDs = f['DYNAMIC DATA']
    filt_data = []
    for id in chanIDs.keys():
        dset = np.array(chanIDs[id]['MEASURED'])
        if  np.min(dset) != 0 and np.max(dset) != 0:
            new_dset = reject_outliers(dset)
            new_dat = np.array([id, new_dset])
            print(new_dat)
            filt_data.append(new_dat)
    with open('newdat.csv', 'a') as out:
        write = csv.writer(out)
        write.writerow(filt_data)
    return filt_data

def read_csv():
    Dictionary_List = [{} for sub in range(885)]
    iterator = 0
    with open('full_dat.csv', 'r') as FILE:
        for line in FILE:
            if line.find("COOLCAT") == -1:
                line = re.sub(r'^"|"$', '', line)
                Line_List = line.split(",")
                Dictionary_List[iterator].update({Line_List[0]: (float(Line_List[1]), float(Line_List[2]), float(Line_List[3]), float(Line_List[4]))})
            else:
                iterator = iterator + 1
    return Dictionary_List

def get_means(channel_key):
    data = read_csv()
    i = 1
    for d in data:
        for key in d.keys():
            if key == channel_key:
                i += 1
    means = np.zeros(i)
    j = 0
    for d in data:
        for key in d.keys():
            if key == channel_key:
                means[j] = d[key][0]
                j += 1
    return means 

def get_all_means():
    data = read_csv()
    
    #get all the data channel keys that exist
    all_keys = []
    for d in data:
        for k in d.keys():
            if k not in all_keys:
                all_keys.append(k)

    length = len(data)
    array = []
    i = 0
    for channel_key in all_keys:
        for d in data:
            i = 0
            for key in d.keys():
                if key == channel_key:
                    i += 1
    z = 0
    print(i)
    #means = np.zeros(i)
    for channel_key in all_keys:
        m = [channel_key, get_means(channel_key)]
        array.append(m)
        z += 1
    return array 





