#!/usr/bin/python
#

import numpy
from scipy import interpolate, stats
import os
import sys

etaValues = [20, 41, 82]
cflValues = [60, 70, 80, 90, 99]
cflTex= ["0.60", "0.80", "0.99"]
pValues = [1, 2]
aValues = [10, 20, 30]
AMPL = 0.0004217
errorTypes = ["L1", "L2", "Li"]
error = numpy.ndarray((len(etaValues),len(cflValues),len(pValues),len(aValues),len(errorTypes)))

# the minimum and maximun values of x for which errors are measured
xMinErr = [0.1, 0.1, 0.1]
xMaxErr = [1.8,  1.8,  1.8]

ref_path = os.getcwd() + '/reference_data/extracted'
# ref_path = os.getcwd() + '/reference_data/provided'
resultsPath = str(sys.argv[1])
deployPath = str(sys.argv[2])

os.chdir(deployPath)

p_e_path = './dr_error_collect'
if not os.path.isdir(p_e_path) :

    os.mkdir(p_e_path)
    os.chdir(p_e_path)

    type1 = numpy.dtype([
        ('error_L1', numpy.float64, 1),
        ('error_L2', numpy.float64, 1),
        ('error_Li', numpy.float64, 1)])

    for aIndex, aV in enumerate(aValues) :

        # refFile = ref_path + '/ppp_{0}.dat'.format(aV)
        refFile = ref_path + '/data-2-a{0}-d.dat'.format(aV)

        load = numpy.loadtxt(refFile)

        ref_x1 = load[:,0]
        ref_u1 = load[:,1]

        f = interpolate.interp1d(ref_x1,ref_u1, kind='slinear')

        for etaIndex, etaV in enumerate(etaValues) :
            for cflIndex, cflV in enumerate(cflValues) :
                for pIndex, pV in enumerate(pValues) :

                    proc_data_1 = numpy.empty([0,1],dtype=type1)

                    execName = 'eta{0}-cfl{1}-p{2}-a{3}'.format(etaV,cflV,pV,aV)
                    numFile = resultsPath + '/' \
                      + 'd-' + execName + '/' \
                      + execName + '.dat'

                    load = numpy.loadtxt(numFile)

                    num_x1 = load[:,0]
                    num_u1 = load[:,1] / AMPL

                    i=0
                    for x in num_x1 :
                        if x <= f.x[-1] :
                            i=i+1
                        else:
                            i=i-1
                            break

                    int_x1 = num_x1[0:i]   
                    int_u1 = f(int_x1)            # interpolated reference

                    for etIndex, error_type in enumerate(errorTypes) :
                        tmp1 = 0
                        tmp2 = 0
                        for i in range(int_x1.size) :
                            if ( int_x1[i] >= xMinErr[aIndex] and \
                             int_x1[i] <= xMaxErr[aIndex] ) :
                                if error_type == "L1" :
                                    tmp1 += abs(int_u1[i] - num_u1[i])
                                    tmp2 += abs(int_u1[i])
                                elif error_type == "L2" :
                                    tmp1 += (int_u1[i] - num_u1[i])**2
                                    tmp2 += (int_u1[i])**2
                                elif error_type == "Li" :
                                    tmp1 = max(tmp1, abs(int_u1[i] - num_u1[i]))
                                    tmp2 = max(tmp2, abs(int_u1[i]))
                                else :
                                    sys.stderr.write("ERROR: wrong error type " + error_type + '\n')
                                    sys.exit(1)
                        
                        if error_type == "L1" :
                            error[etaIndex,cflIndex,pIndex,aIndex,etIndex] = tmp1/tmp2
                        elif error_type == "L2" :
                            error[etaIndex,cflIndex,pIndex,aIndex,etIndex] = (tmp1/tmp2)**(0.5)
                        elif error_type == "Li" :
                            error[etaIndex,cflIndex,pIndex,aIndex,etIndex] = tmp1/tmp2
                        else :
                            sys.stderr.write("ERROR: wrong error type " + error_type + '\n')
                            sys.exit(1)

                    proc_data_1 = numpy.row_stack((proc_data_1,
                                numpy.array(
                                    [(error[etaIndex,cflIndex,pIndex,aIndex,0],
                                      error[etaIndex,cflIndex,pIndex,aIndex,1],
                                      error[etaIndex,cflIndex,pIndex,aIndex,2])],
                                    dtype=type1)))
                    
                    header = ' '.join(proc_data_1.dtype.names)
                    outfile_name = "u1-error-" + execName + ".dat"
                    with open(outfile_name,'w') as outfile :
                       outfile.write("# " + header + '\n')
                       for row in proc_data_1 : 
                           numpy.savetxt(outfile,
                                         row,
                                         fmt="%f")
                    outfile.close()

                    error.tofile("error.npy")

