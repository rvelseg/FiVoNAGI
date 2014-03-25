#ifndef INC_NUMBERS_H
#define INC_NUMBERS_H

#ifdef DATATYPEV

static const DATATYPEV Num0 = 0;
static const DATATYPEV Num0p5 = 0.5;
static const DATATYPEV Num1 = 1;
static const DATATYPEV Num2 = 2;
static const DATATYPEV Num4 = 4;

#ifdef PRECISION
#if PRECISION == 1
static const DATATYPEV PI = 3.1415926;
#elif PRECISION == 2
static const DATATYPEV PI = 3.141592653589793;
#else /* PRECISION value */
# error unresolved PRECISION value
#endif /* PRECISION value */
#else /* PRECISION */
# error unresolved PRECISION definition
#endif /* PRECISION */

#else
# error unresolved DATATYPEV for number definitions

#endif /* ifdef DATATYPEV */

#endif /* INC_NUMBERS_H */
