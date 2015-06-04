#!/usr/bin/python
#

import numpy 
from scipy import interpolate, stats
import os
import sys
import glob
import re

etaValues = [20, 41, 82]
cflValues = [60, 70, 80, 90, 99]
pValues = [1, 2]
aValues = [10, 20, 30]
errorTypes = ["L1", "L2", "Li"]

resultsPath = str(sys.argv[1])
deployPath = str(sys.argv[2])

p_e_path = deployPath + '/dr_error_etas'
if not os.path.isdir(p_e_path) :

    os.mkdir(p_e_path)
    os.chdir(p_e_path)

    for aIndex, aV in enumerate(aValues) :
        for cflIndex, cflV in enumerate(cflValues) :
            for pIndex, pV in enumerate(pValues) :
                
                outfile_name = 'error-cfl{0}-p{1}-a{2}.dat'.format(cflV,pV,aV)
                outfile = open(outfile_name,'w')
                outfile.write("# eta error_L1 error_L2 error_Li\n")

                for etaIndex, etaV in enumerate(etaValues) :

                    execName = 'eta{0}-cfl{1}-p{2}-a{3}'.format(etaV,cflV,pV,aV)
                    refPath = deployPath + "/dr_error_collect" + \
                              "/u1-error-" + execName + ".dat"

                    fm = open(refPath,'r')
                    line = 1
                    while line :
                        last_line = line                        
                        line = fm.readline()
                    fm.close()

                    outfile.write(str(etaV) + " " + last_line)
                outfile.close()
