#!/usr/bin/python
#

import numpy
import os
import sys
from scipy import stats

aValues = [10, 20, 30]
etaValues = [20, 41, 82]
cflValues = [60, 70, 80, 90, 99]
pValues = [1, 2]
errorTypes = ["L1", "L2", "Li"]

resultsPath = str(sys.argv[1])
deployPath = str(sys.argv[2])

p_e_path = deployPath + '/table_error_4'
if not os.path.isdir(p_e_path) :

    os.mkdir(p_e_path)
    os.chdir(p_e_path)

    data_path = deployPath + '/dr_error_collect'

    error = numpy.fromfile(data_path + "/error.npy")

    # This is dangerous, a change in the saved shape must be
    # reproduced here.
    error = error.reshape((len(etaValues),len(cflValues),len(pValues),len(aValues),len(errorTypes)))
    
    type2 = numpy.dtype([
        ('error_type', str, 10),
        ('a', str, 10),
        ('rate_reg_best', numpy.float64, 1),
        ('rate_reg_worst', numpy.float64, 1),
        ('rate_reg_mean', numpy.float64, 1),
        ('R2_reg_worst', numpy.float64, 1),
        ('finest_best', numpy.float64, 1),
        ('finest_worst', numpy.float64, 1)])

    for pIndex, pV in enumerate(pValues) :
        
        proc_data_2 = numpy.empty([0,1],dtype=type2)

        for aIndex, aV in enumerate(aValues) :            
            for etIndex, error_type in enumerate(errorTypes) :

                rate_reg_sum = 0
                rate_reg_best = 0
                rate_reg_worst = "inf"
                R2_worst = "inf"
                finest_worst = 0
                finest_best = "inf"
                counter = 0

                for cflIndex, cflV in enumerate(cflValues) :

                    counter += 1
 
                    rate_reg, intercept, R, p_value, std_err \
                        = stats.linregress(-numpy.log(etaValues), numpy.log(error[:,cflIndex,pIndex,aIndex,etIndex]))
                    R2 = R**2
                    rate_reg_sum += rate_reg
                    rate_reg_worst = min(rate_reg_worst, rate_reg) 
                    rate_reg_best = max(rate_reg_best, rate_reg) 
                    R2_worst = min(R2_worst, R2)

                    finest_worst = max(error[-1,cflIndex,pIndex,aIndex,etIndex],finest_worst)
                    finest_best = min(error[-1,cflIndex,pIndex,aIndex,etIndex],finest_best)
                    
                rate_reg_mean = rate_reg_sum / counter

                proc_data_2 = numpy.row_stack((proc_data_2,
                            numpy.array(
                                [(error_type,
                                  aV,
                                  rate_reg_best,
                                  rate_reg_worst,
                                  rate_reg_mean,
                                  R2_worst,
                                  finest_best,
                                  finest_worst)],
                                dtype=type2)))

        header = ' '.join(proc_data_2.dtype.names)
        outfile_name = "conv_rates_stats_as-p" + str(pV) + ".dat"
        with open(outfile_name,'w') as outfile :
            outfile.write("% " + 'rate_reg_best' + " & " +\
                          'rate_reg_worst' + " & " +\
                          'rate_reg_mean' + " & " +\
                          'R2_reg_worst' + " & " +\
                          'finest_best' + " & " +\
                          'finest_worst' + " \\\\\n")
            prev_a = ""
            for row in proc_data_2 :
                if row['a'] != prev_a :
                    outfile.write('\\midrule\n')
                    outfile.write('\multirow{2}{*}{$a=' + row['a'][0] + '$} &\n')
                prev_a = row['a']
                if row['error_type'] == "L1" :
                    outfile.write('$E_1$ &\n')                  
                elif row['error_type'] == "Li" :
                    outfile.write('& $E_\infty$ &\n')
                elif row['error_type'] == "L2" :
                    continue
                outfile.write("{:.4f}".format(row['rate_reg_best'][0]) + " & " +\
                              "{:.4f}".format(row['rate_reg_worst'][0]) + " & " +\
                              "{:.4f}".format(row['rate_reg_mean'][0]) + " & " +\
                              "{:.4f}".format(row['R2_reg_worst'][0]) + " & " +\
                              "{:.4f}".format(row['finest_best'][0]) + " & " +\
                              "{:.4f}".format(row['finest_worst'][0]) + " \\\\\n")
        outfile.close()
