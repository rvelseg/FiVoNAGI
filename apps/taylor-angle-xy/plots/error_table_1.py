#!/usr/bin/python
#

import numpy
import os
import sys
from scipy import stats

angleValues = ["00000", "PIo32", "PIo16", "PIo08", "PIo04"]
etaValues = [5, 10, 20, 41, 82]
cflValues = [60, 70, 80, 90, 99]
pValues = [1, 2]
errorTypes = ["L1", "L2", "Li"]

resultsPath = str(sys.argv[1])
deployPath = str(sys.argv[2])

p_e_path = deployPath + '/table_error_1'
if not os.path.isdir(p_e_path) :

    os.mkdir(p_e_path)
    os.chdir(p_e_path)

    data_path = deployPath + '/dr_error_etas_CPU'

    error = numpy.fromfile(data_path + "/error.npy")

    # This is dangerous, a change in the saved shape must be
    # reproduced here.
    error = error.reshape((len(etaValues),len(cflValues),len(pValues),len(angleValues),len(errorTypes)))
    
    type2 = numpy.dtype([
        ('error_type', str, 10),
        ('rate_last_best', numpy.float64, 1),
        ('rate_last_worst', numpy.float64, 1),
        ('rate_last_mean', numpy.float64, 1),
        ('rate_reg_best', numpy.float64, 1),
        ('rate_reg_worst', numpy.float64, 1),
        ('rate_reg_mean', numpy.float64, 1),
        ('R2_reg_worst', numpy.float64, 1)])

    for pIndex, pV in enumerate(pValues) :

        proc_data_2 = numpy.empty([0,1],dtype=type2)
        
        for etIndex, error_type in enumerate(errorTypes) :

            rate_last_sum = 0
            rate_last_best = 0
            rate_last_worst = "inf"
            rate_reg_sum = 0
            rate_reg_best = 0
            rate_reg_worst = "inf"
            R2_worst = "inf"
            counter = 0

            for angleIndex, angleV in enumerate(angleValues) :
                for cflIndex, cflV in enumerate(cflValues) :

                    counter += 1
 
                    rate_last = (1/numpy.log(2)) * numpy.log( error[-2,cflIndex,pIndex,angleIndex,etIndex] \
                              / error[-1,cflIndex,pIndex,angleIndex,etIndex] )
                    rate_last_sum += rate_last
                    rate_last_worst = min(rate_last_worst, rate_last) 
                    rate_last_best = max(rate_last_best, rate_last) 

                    rate_reg, intercept, R, p_value, std_err \
                        = stats.linregress(-numpy.log(etaValues), numpy.log(error[:,cflIndex,pIndex,angleIndex,etIndex]))
                    R2 = R**2
                    rate_reg_sum += rate_reg
                    rate_reg_worst = min(rate_reg_worst, rate_reg) 
                    rate_reg_best = max(rate_reg_best, rate_reg) 
                    R2_worst = min(R2_worst, R2)
                    
            rate_last_mean = rate_last_sum / counter
            rate_reg_mean = rate_reg_sum / counter

            proc_data_2 = numpy.row_stack((proc_data_2,
                        numpy.array(
                            [(error_type,
                              rate_last_best,
                              rate_last_worst,
                              rate_last_mean,
                              rate_reg_best,
                              rate_reg_worst,
                              rate_reg_mean,
                              R2_worst)],
                            dtype=type2)))

        header = ' '.join(proc_data_2.dtype.names)
        outfile_name = "conv_rates_stats" + str(pV) + ".dat"
        with open(outfile_name,'w') as outfile :
            outfile.write("# " + header + '\n')
            for row in proc_data_2 : 
                numpy.savetxt(outfile,
                              row,
                              fmt="%s %f %f %f %f %f %f %f")
        outfile.close()

    
