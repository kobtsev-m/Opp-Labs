#include <iostream>
#include <cmath>
#include <omp.h>

#define EPSILON (10e-4)
#define INF (10e6)


double randDouble(double max) {
    return static_cast<double>(rand()) / static_cast<double>(RAND_MAX / max);
}

void fillDataTest(double *A, double *x, double *b, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[j + i*n] = i == j ? 2.0 : 1.0;
        }
    }
    for (int i = 0; i < n; ++i) {
        x[i] = 0.0;
        b[i] = (double)(n+1);
    }
}

void fillData(double *A, double *x, double *b, int n) {
    double u[n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[j + i*n] = (double)(i == j ? i*i : i+j);
        }
    }
    for (int i = 0; i < n; ++i) {
        x[i] = randDouble(1000.0);
        u[i] = randDouble(1000.0);
    }
    for (int i = 0; i < n; ++i) {
        b[i] = 0.0;
        for (int j = 0; j < n; ++j) {
            b[i] += A[j + i*n] * u[j];
        }
    }
}

void ompV1(double *A, double *x, double *b, int n) {

    double yn[n], lenYn, lenB, tn1, tn2;

    for (int k = 0; k < INF; ++k) {

        lenYn = 0.0;
        lenB = 0.0;
        tn1 = 0.0;
        tn2 = 0.0;

        // Calculating Yn
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            yn[i] = -b[i];
            for (int j = 0; j < n; ++j) {
                yn[i] += A[j + i * n] * x[j];
            }
        }

        // Checking if solution if found
        #pragma omp parallel for schedule(static) reduction(+: lenYn, lenB)
        for (int i = 0; i < n; ++i) {
            lenYn += yn[i] * yn[i];
            lenB += b[i] * b[i];
        }
        if (sqrt(lenYn / lenB) < EPSILON) {
            break;
        }

        // Calculating Tn
        #pragma omp parallel for schedule(static) reduction(+: tn1, tn2)]
        for (int i = 0; i < n; ++i) {
            double AynTmp = 0.0;
            for (int j = 0; j < n; ++j) {
                AynTmp += A[j + i * n] * yn[j];
            }
            tn1 += yn[i] * AynTmp;
            tn2 += AynTmp * AynTmp;
        }
        double tn = tn1 / tn2;

        // Calculating next x
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            x[i] -= yn[i] * tn;
        }
    }

    delete[] yn;
}

void ompV2(double *A, double *x, double *b, int n) {

    double yn[n], lenYn, lenB, tn1, tn2;
    bool running = true;

    #pragma omp parallel
    {
        for (int k = 0; k < INF && running; ++k) {

            #pragma omp single
            {
                lenYn = 0.0;
                lenB = 0.0;
                tn1 = 0.0;
                tn2 = 0.0;
            }

            // Calculating Yn
            #pragma omp for
            for (int i = 0; i < n; ++i) {
                yn[i] = -b[i];
                for (int j = 0; j < n; ++j) {
                    yn[i] += A[j + i*n] * x[j];
                }
            }

            // Check if solution is found
            #pragma omp for reduction(+: lenYn, lenB)
            for (int i = 0; i < n; ++i) {
                lenYn += yn[i] * yn[i];
                lenB += b[i] * b[i];
            }
            #pragma omp single
            {
                if (sqrt(lenYn / lenB) < EPSILON) {
                    running = false;
                }
            }

            if (running) {
                // Calculating Tn
                #pragma omp for reduction(+: tn1, tn2)
                for (int i = 0; i < n; ++i) {
                    double AynTmp = 0.0;
                    for (int j = 0; j < n; ++j) {
                        AynTmp += A[j + i * n] * yn[j];
                    }
                    tn1 += yn[i] * AynTmp;
                    tn2 += AynTmp * AynTmp;
                }
                double tn = tn1 / tn2;

                // Calculating next x
                #pragma omp for
                for (int i = 0; i < n; ++i) {
                    x[i] -= yn[i] * tn;
                }
            }
        }
    }
}

void printAnswer(double* x, int n) {
    for (int i = 0; i < n; ++i) {
        printf("x%d: %.1f\n", i, x[i]);
    }
}

void printWorkTime(double startTime, double endTime) {
    printf("Working time: %.2f seconds\n", endTime - startTime);
}

int main(int argc, char *argv[]) {

    if (argc != 3) {
        printf("Wrong arguments number\n");
        return 0;
    }

    int variant = atoi(argv[1]);
    int n = atoi(argv[2]);

    auto *A = new double[n * n];
    auto *x = new double[n];
    auto *b = new double[n];

    fillData(A, x, b, n);

    double startTime = omp_get_wtime();
    variant == 1 ? ompV1(A, x, b, n) : ompV2(A, x, b, n);
    double endTime = omp_get_wtime();

    printAnswer(x, 10);
    printWorkTime(startTime, endTime);

    delete[] A;
    delete[] x;
    delete[] b;

    return 0;
}
