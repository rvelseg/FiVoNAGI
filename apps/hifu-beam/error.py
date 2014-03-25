import numpy as np
from scipy import signal, interpolate
import matplotlib.pyplot as plt
from scipy.signal import kaiserord, lfilter, firwin, freqz
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show
import os
import sys

etaValues = [20, 41, 82]
cflValues = [60, 70, 80, 90, 99]
cflTex=["0.60", "0.70", "0.80", "0.90", "0.99"]
pValues = [1, 2]
aValues = [10, 20, 30]
AMPL = 0.0004217
error = np.ndarray((3,5,2,3)) # array for saving measured errors

# the minimum and maximun values of x for which error is measured
xMinErr = [0.1, 0.1, 0.1]
xMaxErr = [1.8,  1.8,  1.8]

resultsPath = str(sys.argv[1])
if not os.path.isdir(resultsPath) :
    print "results path not found: " + resultsPath
    sys.exit(1)
print "resultsPath found: " + resultsPath

deployPath = str(sys.argv[2])
if not os.path.isdir(deployPath) :
    print "deploy path not found: " + deployPath
    sys.exit(1)
print "deployPath found: " + deployPath

referencePath = str(sys.argv[3]) + "/extracted"
if not os.path.isdir(referencePath) :
    print "reference path not found: " + referencePath
    sys.exit(1)
print "referencePath found: " + referencePath

plots_export_path = deployPath + '/plots_export'
if not os.path.isdir(plots_export_path) :
    os.mkdir(plots_export_path)

plots_error_path = deployPath + '/plots_error'
if not os.path.isdir(plots_error_path) :
    os.mkdir(plots_error_path)

os.chdir(plots_error_path)

for aIndex, aV in enumerate(aValues) :

    refFile = referencePath + '/data-2-a{0}.csv'.format(aV)

    load = np.loadtxt(refFile)

    ref_x1 = load[:,0]
    ref_u1 = load[:,1]

    f = interpolate.interp1d(ref_x1,ref_u1, kind='slinear')

    for etaIndex, etaV in enumerate(etaValues) :
        for cflIndex, cflV in enumerate(cflValues) :
            for pIndex, pV in enumerate(pValues) :
            
                execName = 'eta{0}-cfl{1}-p{2}-a{3}'.format(etaV,cflV,pV,aV)
                numFile = resultsPath + '/' \
                  + 'd-' + execName + '/' \
                  + execName + '.dat'

                load = np.loadtxt(numFile)

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
                int_u1 = f(int_x1)

                # plt.plot(ref_x1, ref_u1, 'bo')
                # plt.plot(int_x1, int_u1,'ro')
                # plt.show()

                # from http://wiki.scipy.org/Cookbook/FIRFilter

                sample_rate = 1/(int_x1[1]-int_x1[0])
                t = int_x1
                x = int_u1
                # print "sample_rate = {0}".format(sample_rate)

                nyq_rate = sample_rate / 2.0
                width = 5.0/nyq_rate
                ripple_db = 30.0
                N, beta = kaiserord(ripple_db, width)
                cutoff_hz = 60.0
                taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))
                filtered_x = lfilter(taps, 1.0, x)

                # The phase delay of the filtered signal.
                delay = 0.5 * (N-1) / sample_rate

                # # Plot the raw data
                # plot(ref_x1, ref_u1, 'bo')
                # # Plot the original signal (interpolated).
                # plot(t, x, 'ro')
                # # Plot the filtered signal, shifted to compensate for the phase delay.
                # plot(t-delay, filtered_x, 'r-')
                # # Plot just the "good" part of the filtered signal.  The first N-1
                # # samples are "corrupted" by the initial conditions.
                # plot(t[N-1:]-delay, filtered_x[N-1:], 'go', linewidth=4)
                # plot(num_x1,num_u1,'ko')
                # grid(True)
                # show()

                fil_x1 = t[0.5 * (N-1):]-delay
                fil_u1 = filtered_x[0.5 * (N-1):]

                tmp1 = 0
                tmp2 = 0
                for i in range(fil_x1.size) :
                    if ( fil_x1[i] >= xMinErr[aIndex] and \
                         fil_x1[i] <= xMaxErr[aIndex] ) :
                        tmp1 = tmp1 + (fil_u1[i] - num_u1[i])**2
                        tmp2 = tmp2 + (fil_u1[i])**2
                tmp1 = tmp1**(0.5)
                tmp2 = tmp2**(0.5)

                if tmp2 != 0 :
                    error[etaIndex,cflIndex,pIndex,aIndex] = tmp1/tmp2
                else :
                    error[etaIndex,cflIndex,pIndex,aIndex] = 0

for pIndex, pV in enumerate(pValues) :
    gpstring = " "
    gpps = " "
    gpps2 = " "
    for etaIndex, etaV in enumerate(etaValues) :
        for cflIndex, cflV in enumerate(cflValues) :
            fileName = 'error-eta{0}-cfl{1}-p{2}.dat'.format(etaV,cflV,pV)
            fileW = open(fileName, 'w')
            line = '# a error \n'
            fileW.write(line)
            lcn = '$\\eta={0}$'.format(etaV) + \
              ', $\\nu_W={0}$'.format(cflTex[cflIndex])

            for aIndex, aV in enumerate(aValues) :
                line = str(aV) + ' ' + \
                  str(error[etaIndex,cflIndex,pIndex,aIndex]) + \
                  '\n'
                fileW.write(line)
            fileW.close()
            gpps = gpps + " \"{0}\" using 1:2 title \"{0}\" ,".format(fileName)
            gpps2 = gpps2 + " \"{0}\" using 1:2 title '{1}' ,".format(fileName, lcn)
    gpps = gpps[:-1]
    gpps2 = gpps2[:-1]
    gpstring = """

set term png size 800,600
set output "error-{0}.png"
set title "error-{0}"

set ylabel 'E'
set xrange [5:35]
set logscale y
set key outside right
set xlabel "a"
plot {1}


set terminal postscript eps size 3.5,2.62 enhanced color \
    font 'Helvetica,10' linewidth 1
set output "error-{0}.eps"

set ylabel 'E'
set xlabel 'a'
set key outside right
set xrange [5:35]
set logscale y
plot {2}

set terminal epslatex size 4.5in,3.3in standalone color colortext
set output "error-{0}-sa.tex"

unset title
set ylabel '$E$'
set xlabel '$a$'
set key outside right
set xrange [5:35]
set logscale y
plot {2}


""".format(pV,gpps,gpps2)

    print gpstring
    
    fileW = open('tmp.gp', 'w')
    fileW.write(gpstring)
    fileW.close()
    
    os.system('gnuplot tmp.gp')
    os.remove('tmp.gp')

    os.system('pdflatex error-{0}-sa.tex'.format(pV))

src = os.path.join(resultsPath, plots_error_path, 'error-2-sa.pdf')
dst = os.path.join(resultsPath, plots_export_path, 'error-2-sa.pdf')
os.rename(src, dst)



    

    
