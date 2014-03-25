#ifndef INC_PARAMETERS_H
#define INC_PARAMETERS_H

// hifu-beam

// For comparison with CLAWPACK see: clawpack-4.6.1/apps/acoustics/2d/example1/setrun.py

#include ROOT_PATH(/common/numbers.h)
#include <string>
using namespace std;

// simulation parameters

// this should be a power of two, ZOOM=1 is no zoom;
static const int ZOOM = 2*ETAC;

static const DATATYPEV AMPL=0.0004217;
static const DATATYPEV FREC=1.0;
// the plot range (blue lines) will be [-PLOTSCALE*AMPL, PLOTSCALE*AMPL] 
static const DATATYPEV PLOTSCALE=1.0;
static const DATATYPEV CFLMAX=1.0;
static const DATATYPEV CFLWHISH=CFLW/100.0;
static const DATATYPEV DTINI=0.005;
// It's important NX and NY to be powers of 2 for reduction to
// work properly.
static const int NX=512*ETAC;
static const int NY=1024*ETAC;
static const DATATYPEV XMAX = 12.5*2.0; // in lambda units
static const DATATYPEV YMAX = 12.5*4.0; // in lambda units
static const DATATYPEV ETA = NX/XMAX;

static const DATATYPEV BETA=4.8;
// for air DIFFDELTA 0.0000001
static const DATATYPEV DIFFDELTA=0.0002974;

// stop and ask for a key to continue before the simulation
static const int initStop = 0;

// do a pause of stepPause miliseconds on every step
// of the simulation, use zero for no pause. 
static const int stepPause = 0;

// time between frames
static const DATATYPEV DTPRINT = 1.0;

// stop and ask for a key to continue on every frame 
static const int frameStop = 0;

// Export data to files on every displayed frame
#ifdef EXPORT // path where the files will be saved
static const int frameExport = 1;
#else
static const int frameExport = 0;
#endif /* EXPORT */

// Path where the files will be saved
#ifdef DEPLOY 
static const string deployPath = QUOTE(DEPLOY);
#else
static const string deployPath = ".";
#endif /* DEPLOY */

// if you set this parameter to zero the simulation won't stop by
// itself
static const float finalTime = 100;

// dimensions of the diplayed window
static const int NXW=NX;
static const int NYW=NY;

// left-botom position of the displayed window
static const int NXWLB=NX/2-NXW/2;
static const int NYWLB=NY/2-NYW/2; 
// jcut
static const int JCUT=2;


// moveDomain parameters

static const DATATYPEV MDVX = 1.0; // domain movement x velocity
static const DATATYPEV MDVY = 0.0; // domain movement y velocity
static const DATATYPEV MDT = XMAX*0.5; // domain movement starting time
// TIDE is below

// Implementation Specific Parameters
#ifndef ISPAM
#define ISPAM 30
#endif  /* ISPAM */

static const DATATYPEV ISPW = Num2*PI*FREC; // = 2 pi
static const DATATYPEV ISPlambda = Num2*PI/ISPW; // = 1

static const DATATYPEV ISPa = ISPAM*ISPlambda; // = ISPAM // in lambda units
static const DATATYPEV ISPF = 50.0*ISPlambda/(XMAX/(NX-1)); // in number of points // focal distance

///// dependent on ISP

static const DATATYPEV TIDE =  floor( 2 + 4/FREC + (ISPa*ISPa*(XMAX/(NX-1))) / (2.0*ISPF*(YMAX/(NY-1))*(YMAX/(NY-1))) );

#endif  /* INC_PARAMETERS_H */
