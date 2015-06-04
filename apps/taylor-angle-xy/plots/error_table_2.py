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

p_e_path = deployPath + '/table_error_2'
if not os.path.isdir(p_e_path) :

    os.mkdir(p_e_path)
    os.chdir(p_e_path)

    data_path = deployPath + '/dr_error_etas_CPU'

    error = numpy.fromfile(data_path + "/error.npy")

    # This is dangerous, a change in the saved shape must be
    # reproduced here.
    error = error.reshape((len(etaValues),len(cflValues),len(pValues),len(angleValues),len(errorTypes)))
    
    type3 = numpy.dtype([
        ('error_type', str, 10),
        ('exec_name', str, 30),
        ('rate_last', numpy.float64, 1),
        ('rate_reg', numpy.float64, 1),
        ('intercept_reg', numpy.float64, 1),
        ('R2_reg', numpy.float64, 1)])

    proc_data_3 = numpy.empty([0,1],dtype=type3)

    for pIndex, pV in enumerate(pValues) :
        for etIndex, error_type in enumerate(errorTypes) :
            for angleIndex, angleV in enumerate(angleValues) :
                for cflIndex, cflV in enumerate(cflValues) :

                    rate_last = (1/numpy.log(2)) * numpy.log( error[-2,cflIndex,pIndex,angleIndex,etIndex] \
                              / error[-1,cflIndex,pIndex,angleIndex,etIndex] )

                    rate_reg, intercept, R, p_value, std_err \
                        = stats.linregress(-numpy.log(etaValues), numpy.log(error[:,cflIndex,pIndex,angleIndex,etIndex]))
                    R2 = R**2

                    proc_data_3 = numpy.row_stack((proc_data_3,
                                numpy.array(
                                    [(error_type,
                                      'cfl{0}-p{1}-angle{2}'.format(cflV,pV,angleV),
                                      rate_last,
                                      rate_reg,
                                      intercept,
                                      R2)],
                                    dtype=type3)))

    header = ' '.join(proc_data_3.dtype.names)
    outfile_name = "conv_rates_all.dat"
    with open(outfile_name,'w') as outfile :
       outfile.write("# " + header + '\n')
       for row in proc_data_3 : 
           numpy.savetxt(outfile,
                         row,
                         fmt="%s %s %f %f %f %f")
    outfile.close()
