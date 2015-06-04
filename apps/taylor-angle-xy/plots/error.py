#!/usr/bin/python
#

import numpy 
from scipy import stats
import os
import sys
import glob
import re

angleValues = ["00000", "PIo32", "PIo16", "PIo08", "PIo04"]
etaValues = [5, 10, 20, 41, 82]
cflValues = [60, 70, 80, 90, 99]
pValues = [1, 2]
errorTypes = ["L1", "L2", "Li"]
error = numpy.ndarray((len(etaValues),len(cflValues),len(pValues),len(angleValues),len(errorTypes)))

resultsPath = str(sys.argv[1])
deployPath = str(sys.argv[2])

p_e_path = deployPath + '/dr_error_etas_CPU'
if not os.path.isdir(p_e_path) :

    os.mkdir(p_e_path)
    os.chdir(p_e_path)

    type1 = numpy.dtype([
        ('frame', int, 1),
        ('T', numpy.float64, 1),
        ('error_L1', numpy.float64, 1),
        ('error_L2', numpy.float64, 1),
        ('error_Li', numpy.float64, 1)])

    for angleIndex, angleV in enumerate(angleValues) :
        for cflIndex, cflV in enumerate(cflValues) :
            for pIndex, pV in enumerate(pValues) :
                for etaIndex, etaV in enumerate(etaValues) :
                    
                    proc_data_1 = numpy.empty([0,1],dtype=type1)

                    execName = 'eta{0}-cfl{1}-p{2}-angle{3}'.format(etaV,cflV,pV,angleV)
                    refPath = resultsPath + "/d-" + execName

                    ls_u1_cut = glob.glob(refPath + "/u1-cut-*.dat")
                    ls_u1_cut.sort()
                    fm = open(ls_u1_cut[-1],'r')
                    line = 1
                    while line :
                        line = fm.readline()
                        scan = re.search("^#\s*T\s*=\s*([0-9\.\-\+eE]+).*",line)
                        if scan is not None :
                            T = float(scan.group(1))
                            T_int = int(T+0.5)
                            break
                    fm.close()

                    if (T_int != 100) :
                        sys.stderr.write("ERROR: wrong time T = " + str(T) + '\n')
                        sys.exit(1)
                        
                    scan = re.search("u1-cut-([0-9]+)\.dat",ls_u1_cut[-1])
                    if scan is not None :
                        frame = int(scan.group(1))
                    else :
                        sys.stderr.write('ERROR: frame number not found\n')
                        sys.exit(1)                        
                        
                    load = numpy.loadtxt(ls_u1_cut[-1])

                    xi = load[:,1]
                    numeric = load[:,2]
                    analytic = load[:,3]

                    for etIndex, error_type in enumerate(errorTypes) :
                        tmp1 = 0
                        tmp2 = 0
                        for i in range(xi.size) :
                            if error_type == "L1" :
                                tmp1 += abs(analytic[i] - numeric[i])
                                tmp2 += abs(analytic[i])
                            elif error_type == "L2" :
                                tmp1 += (analytic[i] - numeric[i])**2
                                tmp2 += (analytic[i])**2
                            elif error_type == "Li" :
                                tmp1 = max(tmp1, abs(analytic[i] - numeric[i]))
                                tmp2 = max(tmp2, abs(analytic[i]))
                            else :
                                sys.stderr.write("ERROR: wrong error type " + error_type + '\n')
                                sys.exit(1)

                        if error_type == "L1" :
                            error[etaIndex,cflIndex,pIndex,angleIndex,etIndex] = tmp1/tmp2
                        elif error_type == "L2" :
                            error[etaIndex,cflIndex,pIndex,angleIndex,etIndex] = (tmp1/tmp2)**(0.5)
                        elif error_type == "Li" :
                            error[etaIndex,cflIndex,pIndex,angleIndex,etIndex] = tmp1/tmp2
                        else :
                            sys.stderr.write("ERROR: wrong error type " + error_type + '\n')
                            sys.exit(1)
                    
                    proc_data_1 = numpy.row_stack((proc_data_1,
                                numpy.array(
                                    [(frame,
                                      T,
                                      error[etaIndex,cflIndex,pIndex,angleIndex,0],
                                      error[etaIndex,cflIndex,pIndex,angleIndex,1],
                                      error[etaIndex,cflIndex,pIndex,angleIndex,2])],
                                    dtype=type1)))

                    execDir = "d-" + execName
                    os.mkdir(execDir)
                    
                    header = ' '.join(proc_data_1.dtype.names)
                    outfile_name = execDir + \
                                  "/u1-error.dat"
                    with open(outfile_name,'w') as outfile :
                       outfile.write("# " + header + '\n')
                       for row in proc_data_1 : 
                           numpy.savetxt(outfile,
                                         row,
                                         fmt="%d %f %f %f %f")
                    outfile.close()

                    error.tofile("error.npy")
