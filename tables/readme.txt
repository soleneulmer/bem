J/A+A/vol/page Beyond the exoplanet mass-radius relation  (Ulmer-Moll, 20XX)
================================================================================
Beyond the exoplanet mass-radius relation
S. Ulmer-Moll, N. C. Santos, P. Figueira, J. Brinchmann, and J. P. Faria
<Astronomy and Astrophysics, ..., ..., (20XX)>
=20XXA&A...xxx..xxxX
================================================================================
ADC_Keywords: Planets
Keywords: Planetary systems –
          Planets and satellites: fundamental parameters –
	  methods: data analysis

Abstract:
Context. The mass and radius are two fundamental properties to characterize
exoplanets but only for a relatively small fraction of exoplanets
are they both available. The mass is often derived from radial velocity
measurements while the radius is almost always measured with the transit method.
For a large number of exoplanets, either the radius or the mass is unknown,
while the host star has been characterized.
Several mass-radius relations dependent on the planet’s type have been
published which often allow to predict the radius,
as well as a bayesian code which forecasts the radius of an exoplanet
given the mass or vice versa.
Aims. Our goal is to derive the radius of exoplanets using only observables
extracted from spectra used primarily to determine radial velocities
and spectral parameters. Our objective is to obtain a mass-radius relation
that is independent of the planet’s type.
Methods. We work with a database of confirmed exoplanets with known radii
and masses as well as the planets from our Solar System.
Using random forests, a machine learning algorithm, we compute the radius of
exoplanets and compare the results to the published radii.
Our code, BEM, is available online. On top of this, we also explore
how the radius estimates compare to previously published mass-radius relations.
Results. The estimated radii reproduces the spread in radius found
for high mass planets better than previous mass-radius relations.
The average error on the radius is 1.8 R⊕ across the whole range of radii
from 1 to 22 R⊕ . We found that a random forest algorithm is able to derive
reliable radii especially for planets between 4 and 20 R⊕,
for which the error is smaller than 25%. The algorithm has a low bias
but still a high variance, which could be reduced by limiting
the growth of the forest or adding more data.
Conclusions. The random forest algorithm is a promising method to derive
exoplanet properties. We show that the exoplanet’s mass and equilibrium
temperature are the relevant properties which constrain the radius,
and do it with higher accuracy than the previous methods.

Description:
Parameters of planets used for the training set 

File Summary:
--------------------------------------------------------------------------------
 FileName      Lrecl    Records   Explanations
--------------------------------------------------------------------------------
ReadMe            ..          .   This file
training.dat     147        380   Parameters of planets in training set
testing.dat      146        128   Parameters of planets in testing set
--------------------------------------------------------------------------------


Byte-by-byte Description of file: training.dat
--------------------------------------------------------------------------------
Bytes   Format Units             Label            Explanations
--------------------------------------------------------------------------------
1-20     A20   ---		 name		  Name of the planet
21-34    F8.5  Earth mass        mass             Planetary mass
35-48    F8.5  Earth radius	 radius		  Planetary radius
49-64    F8.5  AU                semi_major_axis  Semi major axis
65-82    F8.5  Kelvin       	 temp_eq          Planet equilibrium temperature
83-96    F8.5  Solar luminosity  star_luminosity  Stellar luminosity
97-112   F8.5  Solar mass	 star_mass	  Stellar mass
113-128  F8.5  Solar radius  	 star_radius	  Stellar radius
129-146  F8.5  Kelvin     	 star_teff	  Stellar effective temperature
--------------------------------------------------------------------------------
Note (1): The following parameters were taken from the exoplanet.eu database
          on April 15, 2019: planetary mass, planetary radius,
	  semi major axis, stellar mass, stellar radius,
	  and stellar effective temperature
--------------------------------------------------------------------------------

Byte-by-byte Description of file: testing.dat
--------------------------------------------------------------------------------
Bytes   Format Units             Label            Explanations
--------------------------------------------------------------------------------
1-20     A20   ---		 name		  Name of the planet
21-34    F8.5  Earth mass        mass             Planetary mass
35-48    F8.5  Earth radius	 radius		  Planetary radius
49-64    F8.5  AU                semi_major_axis  Semi major axis
65-82    F8.5  Kelvin       	 temp_eq          Planet equilibrium temperature
83-96    F8.5  Solar luminosity  star_luminosity  Stellar luminosity
97-112   F8.5  Solar mass	 star_mass	  Stellar mass
113-128  F8.5  Solar radius  	 star_radius	  Stellar radius
129-146  F8.5  Kelvin     	 star_teff	  Stellar effective temperature
--------------------------------------------------------------------------------
Note (1): The following parameters were taken from the exoplanet.eu database
          on April 15, 2019: planetary mass, planetary radius,
	  semi major axis, stellar mass, stellar radius,
	  and stellar effective temperature
--------------------------------------------------------------------------------

References:
  The Extrasolar Planets Encyclopaedia, www.exoplanet.eu
================================================================================
