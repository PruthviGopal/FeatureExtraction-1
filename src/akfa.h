/* ------------- Imports -------------*/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>

/* ------------- Return Values -------------*/

#define akfa_success 1
#define akfa_allocation_error -1
#define akfa_computation_error -2
#define akfa_matrix_computation_success 2

/* ------------- Prototypes -------------*/

/*The given data set
 * This method is used to compute the Kernel Matrix for a given data set.
 * A Gaussian Kernel is used with this.
 *
 * Parameters:
 * 	dataset:		gsl_matrix* (n_samples, n_dimensions),
 * 					The given data set
 * 	K:				gsl_matrix* (n_samples, n_samples),
 * 					Allocated space to hold the Kernel Matrix
 * 	n_samples:		int,
 * 					Number of Samples in the data set
 * 	n_dimensions:	int,
 * 					Number of Dimensions each point has in the data set
 * 	sigma:			double,
 * 					Value needed for the computation of the Gaussian Kernel
 *
 *	Returns:
 *
 *	akfa_matrix_computation_success, if sucessfull
 *
 */
int computeKernelMatrix(gsl_matrix* dataset, gsl_matrix* K, int n_samples, int n_dimensions, double sigma);

/*
 * This method is used to iniate AKFA without a previosly given Kernel Matrix.
 * A Kernel Matrix is computed and the akfa is started.
 *
 * Parameters:
 *  components:		gsl_matrix* (n_features, n_samples),
 *  				Allocated space to hold the computed features
 * 	dataset:		gsl_matrix* (n_samples, n_dimensions),
 * 					The given data set
 * 	n_dimensions:	int,
 * 					Number of Dimensions each point has in the data set
 * 	n_samples:		int,
 * 					Number of Samples in the data set
 * 	delta:			double,
 * 					Cut-off factor for AKFA
 * 	sigma:			double,
 * 					Value needed for the computation of the Gaussian Kernel
 *
 *	Returns:
 *
 *	akfa_success, if sucessfull
 *
 */
int akfa_no_K(gsl_matrix* components, gsl_matrix* dataset, int n_dimensions, int n_samples, int n_features, double delta, double sigma);

/*
 * This method is performs AKFA to extract a number of features from a given data set.
 *
 * Parameters:
 *  K:				gsl_matrix* (n_samples, n_samples),
 * 					Allocated space that holds the Kernel Matrix
 *  components:		gsl_matrix* (n_features, n_samples),
 *  				Allocated space to hold the computed features
 * 	dataset:		gsl_matrix* (n_samples, n_dimensions),
 * 					The given data set
 * 	n_dimensions:	int,
 * 					Number of Dimensions each point has in the data set
 * 	n_samples:		int,
 * 					Number of Samples in the data set
 * 	delta:			double,
 * 					Cut-off factor for AKFA
 * 	sigma:			double,
 * 					Value needed for the computation of the Gaussian Kernel
 *
 *	Returns:
 *
 *	akfa_success, if sucessfull
 *
 */
int akfa(gsl_matrix* K, gsl_matrix* components, gsl_matrix* dataset, int n_dimensions, int n_samples, int n_features, double delta, double sigma);

/*
 * This function is used to write a matrix into a file
 *  Parameters:
 *  stream:	FILE*,
 * 			The opened file in which the matrix is to be written
 *  m:		gsl_matrix* (n_features, n_samples),
 *  		Allocated space to hold the computed features
 * 	fmt:	char*,
 * 			A list of delimiters
 */
int my_gsl_matrix_fprintf(FILE *stream,gsl_matrix *m,char *fmt);
