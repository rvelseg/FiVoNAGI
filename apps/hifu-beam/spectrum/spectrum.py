#!/usr/bin/python
#

from scipy import interpolate
import numpy
import pylab
import os
import sys

etaValues = [20, 41, 82]
cflValues = [60, 70, 80, 90, 99]
pValues = [1, 2]
aValues = [10, 20, 30]
micValues = [2, 3, 4] # mic 1 were omitted to widen the time series

resultsPath = "../../../results/hifu-beam/data/"
# resultsPath = str(sys.argv[1])

if not os.path.isdir(resultsPath) :
    print "results path not found: " + resultsPath
    sys.exit(1)
print "resultsPath found: " + resultsPath

for aIndex, aV in enumerate(aValues) :
    for etaIndex, etaV in enumerate(etaValues) :
        for cflIndex, cflV in enumerate(cflValues) :
            for pIndex, pV in enumerate(pValues) :
                for micIndex, micV in enumerate(micValues) :
                    
                    execName = 'eta{0}-cfl{1}-p{2}-a{3}'.format(etaV,cflV,pV,aV)
                    # print execName + " : mic " + str(micV)

                    micFile = resultsPath + '/' \
                      + 'd-' + execName + '/' \
                      + 'microphone_' + str(micV) + '.dat'

                    load = numpy.loadtxt(micFile)

                    t = load[:,1]
                    u1 = load[:,2]  # use `/AMPL` to normalize amplitude to one

                    f = interpolate.interp1d(t,u1, kind='slinear')

                    dtmin = max(t)
                    for i in range(0,t.size) :
                        if t[i] == 0 or t[i-1] == 0 : continue
                        if dtmin > t[i] - t[i-1] : dtmin = t[i] - t[i-1]

                    t_int = numpy.linspace(0,max(t),int(max(t)/dtmin))
                    u1_int = f(t_int)

                    trigger = max(u1)*0.8
                    imin = 0
                    imax = t_int.size - 1
                    for i in range(0,t_int.size) :
                        if ( imin == 0
                             and u1_int[i] > trigger ) :
                            imin = i
                        if ( i >= 1
                             and u1_int[i] > trigger
                             and u1_int[i-1] < trigger ) :
                            imax = i

                    widener = 30
                    t_int = t_int[imin-etaV*widener:imax+etaV*widener]
                    u1_int = u1_int[imin-etaV*widener:imax+etaV*widener]

                    freq_axis = numpy.linspace(0,(t_int.size/2)*(1/(t_int[-1]-t_int[0])),t_int.size/2)
                    norm = 2.0/t_int.size

                    spec = norm * numpy.fft.fft(u1_int)

                    # if execName == "eta82-cfl99-p2-a30" : 
                    #     fig, ax = pylab.subplots(2)
                    #     ax[0].plot(t,u1,'y-',t_int,u1_int,'b-')
                    #     ax[0].set_xlim([t_int[0]-4,t_int[-1]+4])
                    #     ax[0].set_title(execName + ', microphone' + str(micV))
                    #     ax[1].plot(freq_axis,abs(spec[:(t_int.size/2)]),'-')
                    #     ax[1].set_xlim(0,20)
                    #     ax[1].set_yscale('log')
                    #     ax[1].set_title('spectrum')
                    #     pylab.show()
                    
                    type1 = numpy.dtype([('freq', numpy.float64, 1), ('amplitude', numpy.float64, 1)])
                    proc_data = numpy.empty([0,1],dtype=type1)
                    for i in range(0,freq_axis.size) :
                        proc_data_row = numpy.array([(freq_axis[i], abs(spec[i]))], dtype=type1)
                        proc_data = numpy.row_stack((proc_data, proc_data_row))
                    
                    outfile = file(resultsPath + '/' + 'd-' + execName + '/'
                                   + 'microphone_' + str(micV) + '_spec.dat', 'w')
                    outfile.write('# freq amplitude\n')
                    for i in range(0,proc_data.size) : 
                        outfile.write('{0} {1}\n'.format(float(proc_data[i]['freq']), 
                                                         float(proc_data[i]['amplitude'])))
                    outfile.close()
              
