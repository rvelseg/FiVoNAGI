#ifndef INC_INIT_H
#define INC_INIT_H

// taylor-angle-xy

// For comparison with CLAWPACK see: clawpack-4.6.1/apps/acoustics/2d/example1/qinit.f

#include ROOT_PATH(/common/debug_tools.h)

template <typename TTT>
void init
(TTT *u1, TTT *u2, TTT *u3,
 TTT *dx, TTT *dy, TTT *dt, 
 GPUGD_VARSFD) {
  TTT rhop;
  for (int i=0; i < NX; i++) { 
    for (int j=0; j < NY; j++) {

      rhop = ISPRHOA
	* tanh(ISPC*(static_cast<TTT>(i)*(*dx)*cos(ISPTHETA)
		     + (static_cast<TTT>(j-NY/2)*(*dy)+MDT*sin(ISPTHETA))
		   *sin(ISPTHETA)
		     + TIDE)); 
    
      u1[i+j*NX] = rhop; //CB:rho1
      u2[i+j*NX] = rhop*cos(ISPTHETA); //CB:rho1
      u3[i+j*NX] = rhop*sin(ISPTHETA); //CB:rho1
#if DEBUG >= 1
      debug2[i+j*NX] = Num0;
#endif /* DEBUG */
    }
  }
}

#endif /* INC_INIT_H */
