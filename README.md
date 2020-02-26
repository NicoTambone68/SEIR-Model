# SEIR-Model
A SEIR Compartmental model in Python

Compartmental modeling allows for numerical reasoning about epidemics. It was proposed in 1927 by Kermack and McKendrick for mathematical modelling of infectious diseases. The SEIR model divides the population into the following compartments:

* Susceptibles  S : individuals who are susceptible to the disease and can be infected
* Exposed  E : individuals who have been exposed to the virus
* Infected  I : individuals who carry an infection
* Recovered  R : individuals who have been infected and have recovered and possibly have acquired immunity as well.

Individuals move from one compartment to the other. From the stage of recovered they either have acquired immunity or become susceptible again, depending on the disease.

![SEIR Compartments](https://institutefordiseasemodeling.github.io/Documentation/general/_images/SEIR-SEIRS.png)
(Image courtesy of the Institute for Disease Modeling)


The module allows for simulations based on different parameters set by the user. The model implements the second order Runge-Kutta algorithm for the numerical solution of differential equations. Running a simulation will result in a chart describing the curves of any of the S-E-I-R compartments, as in the following picture. A chart describing the effective reproductive number will be shown as well. Please read the included Jupyter Notebook for a more comprehensive description of the model. 

![Model Output](https://github.com/NicoTambone68/SEIR-Model/blob/master/SEIR-demo.png)

## Requirements

* Python 3.7 with Numpy and Matplotlib are required to use the module SEIR.py
* Jupyter Notebook is required to read the presentation SEIR.jpynb

It is suggested to install [Anaconda (Python 3.7 version)](https://www.anaconda.com/distribution/) to get a fully functional environment.

