#!/usr/bin/python
#

import numpy 
from scipy import interpolate, stats
import os
import sys
import glob
import re

angleValues = ["00000", "PIo32", "PIo16", "PIo08", "PIo04"]
etaValues = [5, 10, 20, 41, 82]
cflValues = [60, 70, 80, 90, 99]
pValues = [1, 2]

resultsPath = str(sys.argv[1])
deployPath = str(sys.argv[2])

p_e_path = deployPath + '/dr_error_cfls'
if not os.path.isdir(p_e_path) :

    os.mkdir(p_e_path)
    os.chdir(p_e_path)

    for angleIndex, angleV in enumerate(angleValues) :
        for etaIndex, etaV in enumerate(etaValues) :
            for pIndex, pV in enumerate(pValues) :
                
                outfile_name = 'error-eta{0}-p{1}-angle{2}.dat'.format(etaV,pV,angleV)
                outfile = open(outfile_name,'w')
                outfile.write("# cfl frame T error_L1 error_L2 error_Li\n")

                for cflIndex, cflV in enumerate(cflValues) :

                    execName = 'eta{0}-cfl{1}-p{2}-angle{3}'.format(etaV,cflV,pV,angleV)
                    refPath = deployPath + "/dr_error_etas_CPU/d-" + \
                              execName + "/u1-error.dat"

                    fm = open(refPath,'r')
                    line = 1
                    while line :
                        last_line = line
                        line = fm.readline()
                    fm.close()

                    outfile.write(str(cflV) + " " + last_line)
                outfile.close()
