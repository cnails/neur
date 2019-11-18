#include <iostream>
#include <cmath>
/* bydlo language */
#define sprosi std::cin
#define skazhi std::cout
#define op << std::endl
#define v >>
#define na <<
#define iz <<
#define i &&
#define ili ||
#define poka while(
#define ne !
#define dumai {}
#define verni return
#define uhodi return
#define ravno ==
#define ne_ravno !=
#define esli if(
#define to )
#define inache else
#define glavnoe main()
#define nachni {
#define konchi }
#define chetno % 2 == 0
#define nechetno %2 != 0
#define bolshe >
#define menshe <
#define delai )
#define dlya for (
#define nakin ++
#define otozhmi --
#define cho_tam_v *
#define dalshe continue
#define zabei break
/* types */
#define fakt bool
#define chislo int
#define bez_znaka unsigned
#define veshch double
#define stroka char*
#define fraza const char*
#define nichego void
#define v_fakt (bool)
#define v_chislo (int)
#define v_bez_znaka (unsigned)
#define v_veshch (double)
#define da true
#define net false
chislo glavnoe
nachni
   veshch a, b, c, s;
   sprosi v a v b v c;
   veshch p = (a + b + c) / 2.0;
   s = sqrt(p * (p - a) * (p - b) * (p - c));
   skazhi iz s;
konchi;
