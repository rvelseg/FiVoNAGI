#ifndef INC_INIT_H
#define INC_INIT_H

// common

#include "../common/debug_tools.h"

template <typename TTT>
void init
(TTT *u1, TTT *u2, TTT *u3,
 TTT *dx, TTT *dy, TTT *dt, 
 GPUGD_VARSFD) {
 
  for (int i=0; i < NX; i++) { 
    for (int j=0; j < NY; j++) {
      u1[i+j*NX] = Num0; //CB:rho1
      u2[i+j*NX] = Num0;
      u3[i+j*NX] = Num0;
#if DEBUG >= 1
      debug2[i+j*NX] = Num0;
#endif /* DEBUG */
    }
  }
}

#endif /* INC_INIT_H */
