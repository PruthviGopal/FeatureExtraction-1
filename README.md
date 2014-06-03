FeatureExtraction
=================
This module was created in terms of my Bachelor Thesis, where to goal was to obtaind a fast working implementation of Accelerated Kernel Feature Analysis, both in Python and in C.
Furthermore, with Supervised Learning, I will try to detect malicious applications by training Support Vector Machines on sample data sets.

Couple of notes:
1) I will not upload the data sets to github. The methods can be applied to any other application as well.
2) The proposed implementation used the following toolkits, that are not typically installed:
	- Numpy (Python)
	- Scipy (Python)
	- Scikit-learn (Python)
	- Gnu Scientific Libraries (C)
   Please refer to the website of those toolkits for further information.
3) The C-Code can be compiled using gcc -o akfa akfa.c -lgsl -lgslcblas -lm -pg -lrt



