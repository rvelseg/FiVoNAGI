import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

aValues = [10, 20, 30]

for aIndex, aV in enumerate(aValues) :

    refFile = './data-2-a{0}.csv'.format(aV)

    load = np.loadtxt(refFile)

    ref_x1 = load[:,0]
    ref_u1 = load[:,1]

    f = interpolate.interp1d(ref_x1,ref_u1, kind='slinear')

    int_x1 = np.linspace(0,max(ref_x1),1000)
    int_u1 = f(int_x1)

    plt.plot(ref_x1, ref_u1, 'b-', label="direct")
    plt.plot(int_x1, int_u1, 'r-', label="interpolated")
    plt.legend(loc='upper right')
    plt.show()

    outfilename = './data-2-a{0}-d.dat'.format(aV)
    print outfilename
    outfile = open(outfilename, 'w')
    for i,x in enumerate(int_x1) :
        outfile.write(str(x) + " " + str(int_u1[i]) + '\n')
    outfile.close()
