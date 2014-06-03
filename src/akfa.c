#include "akfa.h"

/*
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
int computeKernelMatrix(gsl_matrix* dataset, gsl_matrix* K, int n_samples, int n_dimensions, double sigma)
{
	/*
	 * Allocate two vectors to hold the two rows, that are needed for the computation of one point
	 */
	gsl_vector* xI = gsl_vector_calloc(n_dimensions);
	gsl_vector* xJ = gsl_vector_calloc(n_dimensions);
	int i,j;
	double kVal;
	for(i = 0; i < n_samples; i++) {
		for( j = 0; j < n_samples; j++) {
			/* Extract two rows from matrix*/
			gsl_matrix_get_row(xI,dataset,i);
			gsl_matrix_get_row(xJ,dataset,j);
			gsl_vector_sub(xI,xJ);
			/* Compute Euclidian Norm */
			kVal = gsl_blas_dnrm2(xI);
			gsl_matrix_set(K,i,j,exp( -((kVal*kVal)/(2*sigma*sigma))));
		}
	}
	/* Two Vectors no longer needed. Free */
	gsl_vector_free(xI);
	gsl_vector_free(xJ);
	return akfa_matrix_computation_success;
}
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
int akfa_no_K(gsl_matrix* components, gsl_matrix* dataset, int n_dimensions, int n_samples, int n_features, double delta, double sigma) {
	gsl_matrix* K = gsl_matrix_calloc (n_samples, n_samples);
	computeKernelMatrix(dataset,K, n_samples,n_dimensions, sigma);
	return akfa(K,components,dataset,n_dimensions,n_samples,n_features,delta,sigma);
}

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
int akfa(gsl_matrix* K, gsl_matrix* components, gsl_matrix* dataset, int n_dimensions, int n_samples, int n_features, double delta, double sigma) {
	gsl_vector* feature = gsl_vector_calloc(n_samples);
	/* This variable will be used to count the number of features that have been extracted*/
	int featureCount = 0;
	/* This vector will hold the indices of the vector, that has been chosen as a new feature*/
	int chosenVectorIndex;
	int j,i,k, idx;
	double maxValue, sum,fact, q;
	/* An additional vector for extracting the vector corresponding to the j-th chosen feature */
	gsl_vector* vec  = gsl_vector_alloc (n_samples);
	gsl_vector* tmp  = gsl_vector_alloc (n_samples);
	/*Features are extracting using AKFA*/
	for(featureCount = 0; featureCount < n_features; featureCount++) {
		idx = 0;
		/* Find the direction of highest variance */
		maxValue = 0.0;
		for(i = 0; i < n_samples; i++) {
			sum = 0.0;
			/* Check for cut-off condition*/
			if ( gsl_matrix_get(K,i,i) <= delta) {
				continue;
			}
			gsl_matrix_get_row(vec,K,i);
			gsl_vector_mul(vec,vec);
			for( j = 0; j < n_samples; j++) {
				sum = sum + gsl_vector_get(vec,j);
			}
			fact = 1/(n_samples*gsl_matrix_get(K,i,i));
			sum = sum * fact;
			/* Check if new highest variance has been found */
			if ( sum > maxValue) {
				maxValue = sum;
				idx = i;
			}
		}
		/* Store chosen feature */
		gsl_matrix_get_row(feature,K,idx);
		gsl_vector_scale(feature,gsl_blas_dnrm2(feature));
		gsl_matrix_set_row (components, featureCount, feature);
		/* Update the Kernel Matrix */
		if ( featureCount == n_features-1) {
			continue;
		}
		for( i = 0; i < n_samples; i++) {
			/* Update the Kernel Matrix vector-wise, not element-wise*/
			gsl_vector_memcpy(vec, feature);
			fact = gsl_vector_get(feature,i)/gsl_vector_get(feature,idx);
			gsl_vector_scale (vec, fact);
			gsl_matrix_get_row(tmp, K,i);
			gsl_vector_sub(tmp,vec);
			gsl_matrix_set_row(K,i,tmp);
		}

	}
	/* Free stuff after work*/
	gsl_matrix_free(K);
	gsl_vector_free(feature);
	gsl_vector_free(vec);
	gsl_vector_free(tmp);
	/* return successfully*/
	return akfa_success;
}

int my_gsl_matrix_fprintf(FILE *stream,gsl_matrix *m,char *fmt)
{
        size_t rows=m->size1;
        size_t cols=m->size2;
        size_t row,col,ml;
        int fill;
        char buf[100];
        gsl_vector *maxlen;

        maxlen=gsl_vector_alloc(cols);
        for (col=0;col<cols;++col) {
                ml=0;
                for (row=0;row<rows;++row) {
                        sprintf(buf,fmt,gsl_matrix_get(m,row,col));
                        if (strlen(buf)>ml)
                                ml=strlen(buf);
                }
                gsl_vector_set(maxlen,col,ml);
        }

        for (row=0;row<rows;++row) {
                for (col=0;col<cols;++col) {
                        sprintf(buf,fmt,gsl_matrix_get(m,row,col));
                        fprintf(stream,"%s",buf);
                        fill=gsl_vector_get(maxlen,col)+1-strlen(buf);
                        while (--fill>=0)
                                fprintf(stream," ");
                }
                fprintf(stream,"\n");
        }
        gsl_vector_free(maxlen);
        return 0;
}



/*
 *
 */
int main(int argc, char *argv[]) {
		int i = 0, j;
		double diff;
		int err;
		clock_t start, end;
		int numberOfSamples = strtol(argv[1], NULL, 10);
		int numberOfDimensions = strtol(argv[2], NULL, 10);
		int numberOfFeatures = strtol(argv[3], NULL, 10);
		FILE* datei;
		FILE* output;
		FILE* kernel;
		double elapsed;
		double delta = atof(argv[7]);
		datei = fopen(argv[4] ,"r");
		output = fopen(argv[5], "w");
		kernel = fopen(argv[6], "r");
		gsl_matrix* chosenFeatures = gsl_matrix_alloc (numberOfFeatures, numberOfSamples);
		gsl_matrix* dataset = gsl_matrix_alloc (numberOfSamples, numberOfDimensions);
		gsl_vector* point = gsl_vector_alloc(numberOfDimensions);
		gsl_matrix* K = gsl_matrix_calloc (numberOfSamples, numberOfSamples);
		/* Read the matrix from .txt file specified in argv[4]*/
		gsl_matrix_fscanf(datei, dataset);
		gsl_matrix_fscanf(kernel, K);
		/* Gives my the i-th data point:  gsl_matrix_get_row(point,dataset,0);*/
		/* Set value i,j: gsl_matrix_set(dataset,0,0,a);*/
		//gsl_matrix_set(dataset,0,0,a);

		for( i = 0; i < 400; i ++ ) {
			printf("%f\n",gsl_matrix_get(K,i,i));
			printf("%f\n",gsl_matrix_get(dataset,i,i));
		}




		err = akfa(K,chosenFeatures, dataset, numberOfDimensions, numberOfSamples, numberOfFeatures, 0.0, 4);


		my_gsl_matrix_fprintf(output, chosenFeatures, "%f");

		/*printf("%d\n", errno);*/
		return EXIT_SUCCESS;

		/* gcc -o test test.c -lgsl -lgslcblas -lm -pg -lrt
		 *
		 */


}

