# Risk23 AKA Grande minus chart

- risk_learn_v2_numpy_clean = Tale mislim da je zadnja stable. izracun in iteracije
- risk-plot = plotter. Notri vpises spemenljivke ki si jih dobil iz risk-learn in ti izrise graf v stilu Cowencka.
- return_on_straight_DCA_ris_learn.py = pac samo DCA kupocvanje in nic prodajnja...vsak x cas da notri y kolicino denarja. To je za preverjanje ucinkovitosti modela.
- backtest = mislim, da sem s tem potrjeval delovanje risk-learna (nisem siguren)
- ostale stvari so samo poskusni filei
 
Risk-learn sem poskusal pretvorit v cython zaradi optimizacije, ker pandas je drek za mnozicne iteracije (millions upon billions). Delno sem ga uspel pretvorit v numpy  but to no actual performance increase (verjetno nisem nic spremenil kar ima dejansiki efekt...lack of knowledge)
Se en TODO je bil da poskusim z izracuni na vec jedrih procesorja...dropped because fuck you that's why
