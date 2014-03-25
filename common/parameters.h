#ifndef INC_PARAMETERS_H
#define INC_PARAMETERS_H

// simulation parameters
static const DATATYPEV AMPL=0.1;
static const DATATYPEV FREC=512.0;
// the plot range will be [-PLOTSCALE*AMPL, PLOTSCALE*AMPL] 
static const DATATYPEV PLOTSCALE=1.0;
static const DATATYPEV CFLMAX=0.5;
static const DATATYPEV CFLWHISH=0.45;
static const DATATYPEV DTINI=0.01;
// It's important NX and NY to be powers of 2 for reduction to
// work properly.
static const int NX=1024;
static const int NY=256;
static const DATATYPEV XMAX = 2.0*10.0/32.0;
static const DATATYPEV YMAX = 0.5*10.0/32.0;

static const DATATYPEV BETA=5.0;
// for air DIFFDELTA 0.0000001
static const DATATYPEV DIFFDELTA=0.0000001;

// stop and ask for a key to continue on every step af the simulation
static const int stepStop = 1;

// pause wait pauseDuration and continue on every step of the
// simulation
static const int stepPause = 0;

// in miliseconds
static const int pauseDuration = 1000;

static const DATATYPEV DTPRINT =  0.000625;


#endif  /* INC_PARAMETERS_H */
