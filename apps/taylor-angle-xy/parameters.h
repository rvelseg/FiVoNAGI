#ifndef INC_PARAMETERS_H
#define INC_PARAMETERS_H

// taylor-angle-xy

// For comparison with CLAWPACK see: clawpack-4.6.1/apps/acoustics/2d/example1/setrun.py

#include ROOT_PATH(/common/numbers.h)
#include <string>
using namespace std;

// simulation parameters

// this should be a power of two, ZOOM=1 is no zoom;
static const int ZOOM = 1*ETAC;

// static const DATATYPEV AMPL=0.0; // see below
static const DATATYPEV FREC=1.0; // not used in this implementation
// the plot range  [-PLOTSCALE*AMPL, PLOTSCALE*AMPL] 
static const DATATYPEV PLOTSCALE=3.0;
static const DATATYPEV CFLMAX=1.0;
static const DATATYPEV CFLWHISH=CFLW/100.0;
static const DATATYPEV DTINI=0.005;
// It's important NX and NY to be powers of 2 for reduction to
// work properly.
static const int NX=256*ETAC;
static const int NY=256*ETAC;
static const DATATYPEV XMAX = 12.5; 
static const DATATYPEV YMAX = 12.5; 
static const DATATYPEV ETA = NX/XMAX;

static const DATATYPEV BETA=1; 
// for air DIFFDELTA 0.0000001
static const DATATYPEV DIFFDELTA=0.0001; 

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
static const int JCUT=NY/2; // not used in this implementation

// moveDomain parameters
#ifndef COS
#define COS 1
#endif  /* COS */

#ifndef SIN
#define SIN 0
#endif  /* COS */

static const DATATYPEV MDVX = COS; // domain movement x velocity
static const DATATYPEV MDVY = SIN; // domain movement y velocity
static const DATATYPEV MDT = XMAX*0.5/COS; // domain movement starting time 
static const DATATYPEV TIDE = 4 + (YMAX*0.5-MDT*SIN)*SIN; // time delay: time variable will be initialized as -TIDE


// Implementation Specific Parameters (ISP)
#ifndef ANGLE
#define ANGLE 0
#endif  /* ANGLE */

static const DATATYPEV ISPRHOA = - DIFFDELTA / BETA;
static const DATATYPEV ISPTHETA = ANGLE; // angle (should be between 0 and PI/4) 
static const DATATYPEV ISPC = 2.1972245773362196; // this determines the L value

///// dependent on ISP

static const DATATYPEV AMPL = - ISPRHOA; // AMPL should be positive


#endif  /* INC_PARAMETERS_H */
