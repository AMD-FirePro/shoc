//#define CHECK_RESULTS

#include <math.h>
#include <stdlib.h>
#ifndef _WIN32
#include <sys/time.h>
#endif
#include <cassert>
#include <iostream>
#include <sstream>

#include "OpenCLDeviceInfo.h"
#include "Event.h"
#include "OptionParser.h"
#include "support.h"
#include "ResultDatabase.h"

#ifdef _WIN32
#define srand48(A) srand( (unsigned int)(A) )
#define drand48() ((double)rand()/((double)RAND_MAX+1.0))
#endif
using namespace std;

#define CL_BAIL_ON_ERROR(err) \
{                             \
    CL_CHECK_ERROR(err);      \
    if (err != CL_SUCCESS)    \
        return;               \
}

// Forward declaration
template <class T> inline std::string toString (const T& t){
    std::stringstream ss;
    ss << t;
    return ss.str();
}

template <class T>
void runTest(const string& testName, cl_device_id dev, cl_context ctx,
        cl_command_queue queue, ResultDatabase& resultDB, OptionParser& op,
        const string& compileFlags);

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing
//
// Arguments:
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//
// ****************************************************************************
void
addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("KiB", OPT_INT, "0", "data size (in Kibibytes)");
}

#ifdef CHECK_RESULTS
template <class T>
int DGEMM(char *transa, char *transb, int *m, int *n, int *k, T *alpha, T *a, int *lda, T *b, int *ldb, T *beta, T *c, int *ldc)
{
    /* System generated locals */
    int i1, i2, i3;

    /* Local variables */
    int info;
    bool nota, notb;
    T temp;
    int i, j, l, ncola;
    int nrowa, nrowb;


/*  Purpose
    =======
    DGEMM  performs one of the matrix-matrix operations
       C := alpha*op( A )*op( B ) + beta*C,
    where  op( X ) is one of
       op( X ) = X   or   op( X ) = X',
    alpha and beta are scalars, and A, B and C are matrices, with op( A )
    an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.

    Parameters
    ==========
    TRANSA - CHARACTER*1.
             On entry, TRANSA specifies the form of op( A ) to be used in
             the matrix multiplication as follows:
                TRANSA = 'N' or 'n',  op( A ) = A.
                TRANSA = 'T' or 't',  op( A ) = A'.
                TRANSA = 'C' or 'c',  op( A ) = A'.
             Unchanged on exit.

    TRANSB - CHARACTER*1.
             On entry, TRANSB specifies the form of op( B ) to be used in
             the matrix multiplication as follows:
                TRANSB = 'N' or 'n',  op( B ) = B.
                TRANSB = 'T' or 't',  op( B ) = B'.
                TRANSB = 'C' or 'c',  op( B ) = B'.
             Unchanged on exit.

    M      - INTEGER.
             On entry,  M  specifies  the number  of rows  of the  matrix
             op( A )  and of the  matrix  C.  M  must  be at least  zero.
             Unchanged on exit.

    N      - INTEGER.
             On entry,  N  specifies the number  of columns of the matrix
             op( B ) and the number of columns of the matrix C. N must be
             at least zero.
             Unchanged on exit.

    K      - INTEGER.
             On entry,  K  specifies  the number of columns of the matrix
             op( A ) and the number of rows of the matrix op( B ). K must
             be at least  zero.
             Unchanged on exit.

    ALPHA  - DOUBLE PRECISION.
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    A      - DOUBLE PRECISION array of DIMENSION ( LDA, ka ), where ka is
             k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
             Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
             part of the array  A  must contain the matrix  A,  otherwise
             the leading  k by m  part of the array  A  must contain  the
             matrix A.
             Unchanged on exit.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. When  TRANSA = 'N' or 'n' then
             LDA must be at least  max( 1, m ), otherwise  LDA must be at
             least  max( 1, k ).
             Unchanged on exit.

    B      - DOUBLE PRECISION array of DIMENSION ( LDB, kb ), where kb is
             n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
             Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
             part of the array  B  must contain the matrix  B,  otherwise
             the leading  n by k  part of the array  B  must contain  the
             matrix B.
             Unchanged on exit.

    LDB    - INTEGER.
             On entry, LDB specifies the first dimension of B as declared
             in the calling (sub) program. When  TRANSB = 'N' or 'n' then
             LDB must be at least  max( 1, k ), otherwise  LDB must be at
             least  max( 1, n ).
             Unchanged on exit.

    BETA   - DOUBLE PRECISION.
             On entry,  BETA  specifies the scalar  beta.  When  BETA  is
             supplied as zero then C need not be set on input.
             Unchanged on exit.

    C      - DOUBLE PRECISION array of DIMENSION ( LDC, n ).
             Before entry, the leading  m by n  part of the array  C must
             contain the matrix  C,  except when  beta  is zero, in which
             case C need not be set on entry.
             On exit, the array  C  is overwritten by the  m by n  matrix
             ( alpha*op( A )*op( B ) + beta*C ).

    LDC    - INTEGER.
             On entry, LDC specifies the first dimension of C as declared
             in  the  calling  (sub)  program.   LDC  must  be  at  least
             max( 1, m ).
             Unchanged on exit.


    Level 3 Blas routine.

    -- Written on 8-February-1989.
       Jack Dongarra, Argonne National Laboratory.
       Iain Duff, AERE Harwell.
       Jeremy Du Croz, Numerical Algorithms Group Ltd.
       Sven Hammarling, Numerical Algorithms Group Ltd.

       Set  NOTA  and  NOTB  as  true if  A  and  B  respectively are not
       transposed and set  NROWA, NCOLA and  NROWB  as the number of rows
       and  columns of  A  and the  number of  rows  of  B  respectively.

   Parameter adjustments
       Function Body */

#define A(I,J) a[(I)-1 + ((J)-1)* ( *lda)]
#define B(I,J) b[(I)-1 + ((J)-1)* ( *ldb)]
#define C(I,J) c[(I)-1 + ((J)-1)* ( *ldc)]
//#define MAX(a,b) { __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a > _b ? _a : _b; }

    nota = (*transa == 'N') || (*transa == 'n');
    notb = (*transb == 'N') || (*transb == 'n');
        if (nota) {
        nrowa = *m;
        ncola = *k;
    } else {
        nrowa = *k;
        ncola = *m;
    }
    if (notb) {
        nrowb = *k;
    } else {
        nrowb = *n;
    }

/*     Test the input parameters. */

    info = 0;
    if (! nota && ! ((*transa == 'C') || (*transa == 'c')) && ! ((*transa == 'T') || (*transa == 't'))) {
        info = 1;
    } else if (! notb && ! ((*transb == 'C') || (*transb == 'c')) && ! ((*transb == 'T') || (*transb == 't'))) {
        info = 2;
    } else if (*m < 0) {
        info = 3;
    } else if (*n < 0) {
        info = 4;
    } else if (*k < 0) {
        info = 5;
    } else if (*lda < nrowa) { // MAX(1,nrowa)) {
        info = 8;
    } else if (*ldb < nrowb) { // MAX(1,nrowb)) {
        info = 10;
    } else if (*ldc < *m) { // max(1,*m)) {
        info = 13;
    }
    if (info != 0) {
        printf("Error in DGEMM with parameter Nr: %d\n", info);
   //     return 0;
    }

/*     Quick return if possible. */

    if (*m == 0 || *n == 0 || (*alpha == 0. || *k == 0) && *beta == 1.) {
        return 0;
    }

/*     And if  alpha.eq.zero. */

    if (*alpha == 0.) {
        if (*beta == 0.) {
            i1 = *n;
            for (j = 1; j <= *n; ++j) {
                i2 = *m;
                for (i = 1; i <= *m; ++i) {
                    C(i,j) = 0.;
                }
            }
        } else {
            i1 = *n;
            for (j = 1; j <= *n; ++j) {
                i2 = *m;
                for (i = 1; i <= *m; ++i) {
                    C(i,j) = *beta * C(i,j);
                }
            }
        }
        return 0;
    }

/*     Start the operations. */

    if (notb) {
        if (nota) {

/*           Form  C := alpha*A*B + beta*C. */

            i1 = *n;
            for (j = 1; j <= *n; ++j) {
                if (*beta == 0.) {
                    i2 = *m;
                    for (i = 1; i <= *m; ++i) {
                        C(i,j) = 0.;
                    }
                } else if (*beta != 1.) {
                    i2 = *m;
                    for (i = 1; i <= *m; ++i) {
                        C(i,j) = *beta * C(i,j);
                    }
                }
                i2 = *k;
                for (l = 1; l <= *k; ++l) {
                    if (B(l,j) != 0.) {
                        temp = *alpha * B(l,j);
                        i3 = *m;
                        for (i = 1; i <= *m; ++i) {
                            C(i,j) += temp * A(i,l);
                        }
                    }
                }
            }
        } else {

/*           Form  C := alpha*A'*B + beta*C */

            i1 = *n;
            for (j = 1; j <= *n; ++j) {
                i2 = *m;
                for (i = 1; i <= *m; ++i) {
                    temp = 0.;
                    i3 = *k;
                    for (l = 1; l <= *k; ++l) {
                        temp += A(l,i) * B(l,j);
                    }
                    if (*beta == 0.) {
                        C(i,j) = *alpha * temp;
                    } else {
                        C(i,j) = *alpha * temp + *beta * C(i,j);
                    }
                }
            }
        }
    } else {
        if (nota) {

/*           Form  C := alpha*A*B' + beta*C */

            i1 = *n;
            for (j = 1; j <= *n; ++j) {
                if (*beta == 0.) {
                    i2 = *m;
                    for (i = 1; i <= *m; ++i) {
                        C(i,j) = 0.;
                    }
                } else if (*beta != 1.) {
                    i2 = *m;
                    for (i = 1; i <= *m; ++i) {
                        C(i,j) = *beta * C(i,j);
                    }
                }
                i2 = *k;
                for (l = 1; l <= *k; ++l) {
                    if (B(j,l) != 0.) {
                        temp = *alpha * B(j,l);
                        i3 = *m;
                        for (i = 1; i <= *m; ++i) {
                            C(i,j) += temp * A(i,l);
                        }
                    }
                }
            }
        } else {

/*           Form  C := alpha*A'*B' + beta*C */

            i1 = *n;
            for (j = 1; j <= *n; ++j) {
                i2 = *m;
                for (i = 1; i <= *m; ++i) {
                    temp = 0.;
                    i3 = *k;
                    for (l = 1; l <= *k; ++l) {
                        temp += A(l,i) * B(j,l);
                    }
                    if (*beta == 0.) {
                        C(i,j) = *alpha * temp;
                    } else {
                        C(i,j) = *alpha * temp + *beta * C(i,j);
                    }
                }
            }
        }
    }

    return 0;

/*     End of DGEMM . */

}

template <class T>
bool comparesquare(T *refData, T *data, const int height, const int width, const int ld, T *epsilon, T *error)
{
    T my_error = -1;
    int i, j;
    bool is_correct = true;
    for(j = 0; j < height; ++j)
    {
        for(i = 0; i < width; ++i)
        {
            double rel_diff = fabs((refData[j*ld+i] - data[j*ld+i])/refData[j*ld+i]);
            my_error = (rel_diff > my_error) ? rel_diff : my_error;
            is_correct = (rel_diff <= *epsilon);
            if( ! is_correct )
            {
//              printf("ERROR! ref[%d][%d]=%f data[%d][%d]=%f\n",j,i,refData[j*ld+i],j,i,data[j*ld+i]);
//              printf("ERROR! ref[%d][%d]=%f data[%d][%d]=%f\n",j,i+1,refData[j*ld+i+1],j,i+1,data[j*ld+i+1]);
                break;
            }
        }
        if( ! is_correct )
            break;
    }
    *error = my_error;

    if( is_correct )
        is_correct = (my_error < *epsilon);

    return is_correct;
}

template <class T>
bool comparelinear(T *refData, T *data, const int length, T *epsilon, T *error)
{
    T my_error = -1;
    int i;
    bool is_correct = true;
    for(i = 0; i < length; ++i)
    {
        double rel_diff = fabs((refData[i] - data[i])/refData[i]);
        my_error = (rel_diff > my_error) ? rel_diff : my_error;
        is_correct = (rel_diff <= *epsilon);
        if( ! is_correct )
        {
//          printf("ERROR! ref[%d]=%f data[%d]=%f\n",i,refData[i],i,data[i]);
//          printf("ERROR! ref[%d]=%f data[%d]=%f\n",i+1,refData[i+1],i+1,data[i+1]);
            break;
        }
    }
    *error = my_error;

    if( is_correct )
        is_correct = (my_error < *epsilon);

    return is_correct;
}
#endif

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Benchmarks the GEMM codes
//
// Arguments:
//   dev: the opencl device id to use for the benchmark
//   ctx: the opencl context to use for the benchmark
//   queue: the opencl command queue to issue commands to
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
//
// Returns:  nothing
//
// Programmer: Anthony Danalis
// Creation: August 26, 2009
//
// Modifications:
//   Jeremy Meredith, Thu Sep 24 17:30:18 EDT 2009
//   Use implicit include of source file instead of runtime loading.
//
//   Jeremy Meredith, Thu Aug 19 13:59:09 EDT 2010
//   Added transfer vs computation equivalence calculation.
//
//   Jeremy Meredith, Thu Aug 19 14:16:49 EDT 2010
//   Use pinned memory for better PCIe speeds.
//
// ****************************************************************************
extern const char *cl_source_gemmN;

void
RunBenchmark(cl_device_id dev,
                  cl_context ctx,
                  cl_command_queue queue,
                  ResultDatabase &resultDB,
                  OptionParser &op)
{
    // Always run single precision test
    // OpenCL doesn't support templated kernels, so we have to use macros
    runTest<float>("SGEMM", dev, ctx, queue, resultDB, op,
            "-DSINGLE_PRECISION");

    // If double precision is supported, run the DP test
    if (checkExtension(dev, "cl_khr_fp64"))
    {
        cout << "DP Supported\n";
        runTest<double>("DGEMM", dev, ctx, queue, resultDB, op,
                "-DK_DOUBLE_PRECISION ");
    }
    else if (checkExtension(dev, "cl_amd_fp64"))
    {
        cout << "DP Supported\n";
        runTest<double>("DGEMM", dev, ctx, queue, resultDB, op,
                "-DAMD_DOUBLE_PRECISION ");
    }
    else
    {
        cout << "DP Not Supported\n";
        char atts[1024] = "DP_Not_Supported";
        // resultDB requires neg entry for every possible result
        int passes = op.getOptionInt("passes");
        for (; passes > 0; --passes) {
            for (int i = 0; i < 2; i++) {
                const char transb = i ? 'T' : 'N';
                string testName="DGEMM";
                resultDB.AddResult(testName+"-"+transb, atts, "GFlops", FLT_MAX);
                resultDB.AddResult(testName+"-"+transb+"_PCIe", atts, "GFlops", FLT_MAX);
                resultDB.AddResult(testName+"-"+transb+"_Parity", atts, "N", FLT_MAX);
            }
        }
    }
}

template <class T>
void runTest(const string& testName, cl_device_id dev, cl_context ctx,
        cl_command_queue queue, ResultDatabase& resultDB, OptionParser& op,
        const string& compileFlags)
{

    int N;
    if (op.getOptionInt("KiB") == 0)
    {
        int probSizes[4] = { 1, 4, 8, 16 };
        N = probSizes[op.getOptionInt("size")-1] * 1024 / sizeof(T);
    } else {
        N = op.getOptionInt("KiB") * 1024 / sizeof(T);
    }

    cl_int err;
    int waitForEvents = 1;
    size_t m = N, n = N, k = N;
    size_t lda, ldb, ldc;
    const T alpha = 1;
    const T beta = -1;
    int i, j;

    lda = ldb = ldc = N;

    cl_uint numDimensions = 0;
    clGetDeviceInfo (dev, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                             sizeof(cl_uint), &numDimensions, NULL);
    size_t *maxWorkSizes = new size_t[numDimensions];
    clGetDeviceInfo (dev, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                       sizeof(size_t)*numDimensions, maxWorkSizes, NULL);

    if (numDimensions<2 || maxWorkSizes[0]<16 || maxWorkSizes[1] < 4)
    {
        cout << "SGEMM needs a 2-dimensional work group size of at least {16,4}." << endl;
        int passes = op.getOptionInt("passes");
        char atts[1024] = "GSize_Not_Supported";
        for (; passes > 0; --passes) {
            for (i = 0; i < 2; i++) {
                const char transb = i ? 'T' : 'N';
                resultDB.AddResult(testName+"-"+transb, atts, "GFlops", FLT_MAX);
                resultDB.AddResult(testName+"-"+transb+"_PCIe", atts, "GFlops", FLT_MAX);
                resultDB.AddResult(testName+"-"+transb+"_Parity", atts, "N", FLT_MAX);
            }
        }
        return;
    }

    size_t localWorkSize[2] = {16,4};


    // Create program object
    cl_program prog = clCreateProgramWithSource(ctx, 1,
                                 &cl_source_gemmN, NULL, &err);
    CL_CHECK_ERROR(err);

    string flags = compileFlags + " -cl-mad-enable";
    err = clBuildProgram(prog, 0, NULL, flags.c_str(), NULL,
            NULL);
    CL_CHECK_ERROR(err);

    // If compilation fails, print error messages and return
    if (err != CL_SUCCESS) {
        char log[5000];
        size_t retsize = 0;
        err =  clGetProgramBuildInfo (prog, dev, CL_PROGRAM_BUILD_LOG,
                5000*sizeof(char),  log, &retsize);

        CL_CHECK_ERROR(err);
        cout << "Build error." << endl;
        cout << "Retsize: " << retsize << endl;
        cout << "Log: " << log << endl;
        exit(-1);
    }

    // Generate the kernel objects
    cl_kernel sgemmNN = clCreateKernel(prog, "sgemmNN", &err);
    CL_CHECK_ERROR(err);

    cl_kernel sgemmNT = clCreateKernel(prog, "sgemmNT", &err);
    CL_CHECK_ERROR(err);

    // Allocate memory for the matrices
    T *A, *B, *C;
#ifdef CHECK_RESULTS
    T *C_gold;
#endif
    cl_mem Aobj, Bobj, Cobj;
    if (true) // pinned
    {
        Aobj = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
			       sizeof(T)*N*N, NULL, &err);
        CL_CHECK_ERROR(err);
        A =(T*)clEnqueueMapBuffer(queue,Aobj,true,CL_MAP_READ|CL_MAP_WRITE,
				       0,sizeof(T)*N*N,0, NULL,NULL,&err);
        CL_CHECK_ERROR(err);

        Bobj = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
			       sizeof(T)*N*N, NULL, &err);
        CL_CHECK_ERROR(err);
        B =(T*)clEnqueueMapBuffer(queue,Bobj,true,CL_MAP_READ|CL_MAP_WRITE,
				       0,sizeof(T)*N*N,0, NULL,NULL,&err);
        CL_CHECK_ERROR(err);

        Cobj = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
			       sizeof(T)*N*N, NULL, &err);
        CL_CHECK_ERROR(err);
        C =(T*)clEnqueueMapBuffer(queue,Cobj,true,CL_MAP_READ|CL_MAP_WRITE,
				       0,sizeof(T)*N*N,0, NULL,NULL,&err);
        CL_CHECK_ERROR(err);
    }
    else
    {
	A = (T*)malloc( N*N*sizeof( T ) );
	B = (T*)malloc( N*N*sizeof( T ) );
	C = (T*)malloc( N*N*sizeof( T ) );
    }
#ifdef CHECK_RESULTS
    C_gold = (T*)malloc( N*N*sizeof( T ) );
#endif

    // Initialize inputs
    srand48(13579862);
    for(i=0; i<m; ++i){
        for(j=0; j<k; ++j){
            A[i*k+j] = (T)(0.5 + drand48()*1.5);
        }
    }

    for(i=0; i<k; ++i){
        for(j=0; j<n; ++j){
            B[i*n+j] = (T)(0.5 + drand48()*1.5);
        }
    }

    for(i=0; i<m; ++i){
        for(j=0; j<n; ++j){
            C[i*n+j] = (T)(0.5 + drand48()*1.5);
#ifdef CHECK_RESULTS
            C_gold[i*n+j] = C[i*n+j];
#endif
        }
    }

    // Pass A and B to the GPU and create a GPU buffer for C
    cl_mem Agpu = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                 m*k * sizeof(T), NULL, &err);
    CL_BAIL_ON_ERROR(err);
    cl_mem Bgpu = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                 k*n * sizeof(T), NULL, &err);
    CL_BAIL_ON_ERROR(err);
    cl_mem Cgpu = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                 m*n * sizeof(T), NULL, &err);
    CL_BAIL_ON_ERROR(err);


    // Set arguments to the sgemmNN kernel
    err = clSetKernelArg(sgemmNN, 0, sizeof(cl_mem), (void*)&Agpu);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNN, 1, sizeof(int), (void*)&lda);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNN, 2, sizeof(cl_mem), (void*)&Bgpu);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNN, 3, sizeof(int), (void*)&ldb);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNN, 4, sizeof(cl_mem), (void*)&Cgpu);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNN, 5, sizeof(int), (void*)&ldc);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNN, 6, sizeof(int), (void*)&k);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNN, 7, sizeof(T), (void*)&alpha);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNN, 8, sizeof(T), (void*)&beta);
    CL_BAIL_ON_ERROR(err);

    // Pass arguments to the sgemmNT kernel
    err = clSetKernelArg(sgemmNT, 0, sizeof(cl_mem), (void*)&Agpu);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNT, 1, sizeof(int), (void*)&lda);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNT, 2, sizeof(cl_mem), (void*)&Bgpu);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNT, 3, sizeof(int), (void*)&ldb);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNT, 4, sizeof(cl_mem), (void*)&Cgpu);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNT, 5, sizeof(int), (void*)&ldc);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNT, 6, sizeof(int), (void*)&k);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNT, 7, sizeof(T), (void*)&alpha);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNT, 8, sizeof(T), (void*)&beta);
    CL_BAIL_ON_ERROR(err);

    const size_t globalWorkSize[2] = {m/4,n/4};

    int passes = op.getOptionInt("passes");

    // Run NN
    for (int i = 0; i < passes; i++) {
        Event evDownload1("Download A");
        Event evUpload("Upload");
        Event evNN("sgemmNN");

        err = clEnqueueWriteBuffer(queue, Agpu, CL_TRUE, 0, m*n*sizeof(T),
                A, 0, NULL, &evDownload1.CLEvent());
        err = clEnqueueWriteBuffer(queue, Bgpu, CL_TRUE, 0, m*n*sizeof(T),
                B, 0, NULL, NULL);
        err = clEnqueueWriteBuffer(queue, Cgpu, CL_TRUE, 0, m*n*sizeof(T),
                C, 0, NULL, NULL);

        // Wait until data transfers finish
        clFinish(queue);
        CL_BAIL_ON_ERROR(err);

        //Launch Kernels
        err = clEnqueueNDRangeKernel(queue, sgemmNN, 2, NULL, globalWorkSize,
                                     localWorkSize, 0, NULL, &evNN.CLEvent());
        clFinish(queue);
        CL_BAIL_ON_ERROR(err);

        err = clEnqueueReadBuffer(queue, Cgpu, CL_TRUE, 0, m*n*sizeof(T),
                C, 0, NULL, &evUpload.CLEvent());
        clFinish(queue);
        CL_BAIL_ON_ERROR(err);

        evNN.FillTimingInfo();
        evDownload1.FillTimingInfo();
        evUpload.FillTimingInfo();

        double user_wait_time = 0.0;
        double gemm_pure_time = 0.0;

        gemm_pure_time = evNN.SubmitEndRuntime();
        user_wait_time = evUpload.EndTime() - evDownload1.QueuedTime();
        double transfer_time = user_wait_time - gemm_pure_time;
        double flops = 2.0*(double)N*N*N;
        resultDB.AddResult(testName+"-N", toString(N), "GFLOPS",
                flops / gemm_pure_time);
        resultDB.AddResult(testName+"-N_PCIe", toString(N), "GFLOPS",
                flops / user_wait_time);
        resultDB.AddResult(testName+"-N_Parity", toString(N), "N",
                transfer_time / gemm_pure_time);
    }
#ifdef CHECK_RESULTS
    T error;
    T epsilon;
    if (sizeof(T) == 4) {
       epsilon = 1e-3;
    } else {
       epsilon = 1e-6;
    }
    // Check result with computation made on the CPU
    T al = 0, be = 1;
    for (int i = 0; i < passes; i++) {
        al = alpha + beta * al;
        be = beta * be;
    }
    int LDA=lda, LDB=ldb, LDC=ldc, KK=k;
    std::cout << "INFO: Case NN m=" << m << " n=" << n << " k=" << k << " alpha=" << alpha << " beta=" << beta << " ";
    DGEMM<T>((char *)"N", (char *)"N", &LDA, &LDB, &KK, &al, A, &LDA, B, &LDB, &be, C_gold, &LDC);
    if(comparesquare<T>(C_gold, C, ldc, ldc, ldc, &epsilon, &error)) {
//  if(comparelinear<T>(C_gold, C, ldc*ldc, &epsilon, &error)) {
         std::cout << "INFO: Passed! (Error=" << error << ") < (Epsilon=" << epsilon << ")\n";
    } else {
         std::cout << "INFO: Failed! (Error=" << error << ") > (Epsilon=" << epsilon << ")\n";
    }
#endif
    for(i=0; i<m; ++i){
        for(j=0; j<n; ++j){
            C[i*n+j] = (T)(0.5 + drand48()*1.5);
#ifdef CHECK_RESULTS
            C_gold[i*n+j] = C[i*n+j];
#endif
        }
    }

    // Run NT
    for (int i = 0; i < passes; i++) {
        Event evDownload1("Download A");
        Event evUpload("Upload");
        Event evNT("sgemmNT");

        err = clEnqueueWriteBuffer(queue, Agpu, CL_TRUE, 0, m*n*sizeof(T),
                A, 0, NULL, &evDownload1.CLEvent());
        err = clEnqueueWriteBuffer(queue, Bgpu, CL_TRUE, 0, m*n*sizeof(T),
                B, 0, NULL, NULL);
        err = clEnqueueWriteBuffer(queue, Cgpu, CL_TRUE, 0, m*n*sizeof(T),
                C, 0, NULL, NULL);
        clFinish(queue);
        CL_BAIL_ON_ERROR(err);

        //Launch Kernels
        err = clEnqueueNDRangeKernel(queue, sgemmNT, 2, NULL, globalWorkSize,
                                     localWorkSize, 0, NULL, &evNT.CLEvent());
        clFinish(queue);
        CL_BAIL_ON_ERROR(err);

        err = clEnqueueReadBuffer(queue, Cgpu, CL_TRUE, 0, m*n*sizeof(T),
                C, 0, NULL, &evUpload.CLEvent());
        clFinish(queue);
        CL_BAIL_ON_ERROR(err);

        evNT.FillTimingInfo();
        evDownload1.FillTimingInfo();
        evUpload.FillTimingInfo();

        double user_wait_time = 0.0;
        double gemm_pure_time = 0.0;

        gemm_pure_time = evNT.SubmitEndRuntime();
        user_wait_time = evUpload.EndTime() - evDownload1.QueuedTime();
        double transfer_time = user_wait_time - gemm_pure_time;
        double flops = 2.0*(double)N*N*N;
        resultDB.AddResult(testName+"-T", toString(N), "GFLOPS",
                flops / gemm_pure_time);
        resultDB.AddResult(testName+"-T_PCIe", toString(N), "GFLOPS",
                flops / user_wait_time);
        resultDB.AddResult(testName+"-T_Parity", toString(N), "N",
                transfer_time / gemm_pure_time);
    }
#ifdef CHECK_RESULTS
    // Check result with computation made on the CPU
    std::cout << "INFO: Case NT m=" << m << " n=" << n << " k=" << k << " alpha=" << alpha << " beta=" << beta << " ";
    DGEMM<T>((char *)"N", (char *)"T", &LDA, &LDB, &KK, &al, A, &LDA, B, &LDB, &be, C_gold, &LDC);
    if(comparesquare<T>(C_gold, C, ldc, ldc, ldc, &epsilon, &error)) {
//  if(comparelinear<T>(C_gold, C, ldc*ldc, &epsilon, &error)) {
         std::cout << "INFO: Passed! (Error=" << error << ") < (Epsilon=" << epsilon << ")\n";
    } else {
         std::cout << "INFO: Failed! (Error=" << error << ") > (Epsilon=" << epsilon << ")\n";
    }
#endif
    for(i=0; i<m; ++i){
        for(j=0; j<n; ++j){
            C[i*n+j] = (T)(0.5 + drand48()*1.5);
#ifdef CHECK_RESULTS
            C_gold[i*n+j] = C[i*n+j];
#endif
        }
    }

    if (true) // pinned
    {
        err = clReleaseMemObject(Aobj);
        CL_CHECK_ERROR(err);
        err = clReleaseMemObject(Bobj);
        CL_CHECK_ERROR(err);
        err = clReleaseMemObject(Cobj);
        CL_CHECK_ERROR(err);
    }
    else
    {
	free(A);
	free(B);
	free(C);
#ifdef CHECK_RESULTS
	free(C_gold);
#endif
    }

    err = clReleaseProgram(prog);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(sgemmNN);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(sgemmNT);
    CL_CHECK_ERROR(err);

    err = clReleaseMemObject(Agpu);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(Bgpu);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(Cgpu);
    CL_CHECK_ERROR(err);

}
