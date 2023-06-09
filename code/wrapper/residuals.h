#define M_PI 

struct mystruct_t {

	// 1D I/O array
	double *X;
	// 2D I/O array
	double **Z;
};

void fastResiduals(struct mystruct_t *data, int ROWS, int ZCOLS, double bandwidth, double *residuals);