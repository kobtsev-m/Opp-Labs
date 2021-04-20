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
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[j + i*n] = (double)(i == j ? i*i : i+j);
        }
    }
    auto *u = new double[n];
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
    delete[] u;
}

void omp1(double *A, double *x, double *b, int n) {

    auto *yn = new double[n];
    double lenYn = 0.0, lenB = 0.0, tn1 = 0.0, tn2 = 0.0;

    for (int k = 0; k < INF; ++k) {

        // Calculating Yn
        #pragma omp parallel for
        // #pragma omp parallel for schedule(static, 1000)
        // #pragma omp parallel for schedule(static, 3000)
        // #pragma omp parallel for schedule(dynamic, 1000)
        // #pragma omp parallel for schedule(dynamic, 3000)
        // #pragma omp parallel for schedule(guided, 1000)
        // #pragma omp parallel for schedule(guided, 3000)
        for (int i = 0; i < n; ++i) {
            yn[i] = -b[i];
            for (int j = 0; j < n; ++j) {
                yn[i] += A[j + i * n] * x[j];
            }
        }

        // Checking if solution if found
        #pragma omp parallel for reduction(+: lenYn) reduction(+: lenB)
        // #pragma omp parallel for schedule(static, 1000) reduction(+: tn1) reduction(+: tn2)
        // #pragma omp parallel for schedule(static, 3000) reduction(+: tn1) reduction(+: tn2)
        // #pragma omp parallel for schedule(dynamic, 1000) reduction(+: tn1) reduction(+: tn2)
        // #pragma omp parallel for schedule(dynamic, 3000) reduction(+: tn1) reduction(+: tn2)
        // #pragma omp parallel for schedule(guided, 1000) reduction(+: tn1) reduction(+: tn2)
        // #pragma omp parallel for schedule(guided, 3000) reduction(+: tn1) reduction(+: tn2)
        for (int i = 0; i < n; ++i) {
            lenYn += yn[i] * yn[i];
            lenB += b[i] * b[i];
        }
        if (sqrt(lenYn / lenB) < EPSILON) {
            break;
        }

        // Calculating Tn
        #pragma omp parallel for reduction(+: tn1) reduction(+: tn2)
        // #pragma omp parallel for schedule(static, 1000) reduction(+: tn1) reduction(+: tn2)
        // #pragma omp parallel for schedule(static, 3000) reduction(+: tn1) reduction(+: tn2)
        // #pragma omp parallel for schedule(dynamic, 1000) reduction(+: tn1) reduction(+: tn2)
        // #pragma omp parallel for schedule(dynamic, 3000) reduction(+: tn1) reduction(+: tn2)
        // #pragma omp parallel for schedule(guided, 1000) reduction(+: tn1) reduction(+: tn2)
        // #pragma omp parallel for schedule(guided, 3000) reduction(+: tn1) reduction(+: tn2)
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
        #pragma omp parallel for
        // #pragma omp parallel for schedule(static, 1000)
        // #pragma omp parallel for schedule(static, 3000)
        // #pragma omp parallel for schedule(dynamic, 1000)
        // #pragma omp parallel for schedule(dynamic, 3000)
        // #pragma omp parallel for schedule(guided, 1000)
        // #pragma omp parallel for schedule(guided, 3000)
        for (int i = 0; i < n; ++i) {
            x[i] -= yn[i] * tn;
        }

        lenYn = 0.0;
        lenB = 0.0;
        tn1 = 0.0;
        tn2 = 0.0;
    }

    delete[] yn;
}

void omp2(double *A, double *x, double *b, int n) {

    auto *yn = new double[n];
    double lenYn = 0.0, lenB = 0.0, tn1 = 0.0, tn2 = 0.0;

    bool running = true;

    #pragma omp parallel
    {
        for (int k = 0; k < INF && running; ++k) {

            // Calculating Yn
            #pragma omp for
            for (int i = 0; i < n; ++i) {
                yn[i] = -b[i];
                for (int j = 0; j < n; ++j) {
                    yn[i] += A[j + i*n] * x[j];
                }
            }

            // Check if solution is found
            #pragma omp for reduction(+: lenYn) reduction(+: lenB)
            for (int i = 0; i < n; ++i) {
                lenYn += yn[i] * yn[i];
                lenB += b[i] * b[i];
            }
            #pragma omp barrier
            {
                if (sqrt(lenYn / lenB) < EPSILON) {
                    running = false;
                }
            }

            if (running) {
                // Calculating Tn
                #pragma omp for reduction(+: tn1) reduction(+: tn2)
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

            #pragma omp barrier
            {
                lenYn = 0.0;
                lenB = 0.0;
                tn1 = 0.0;
                tn2 = 0.0;
            }
        }
    }

    delete[] yn;
}

void printAnswer(double* x) {
    for (int i = 0; i < 10; ++i) {
        printf("x%d: %.1f\n", i, x[i]);
    }
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
    variant == 1 ? omp1(A, x, b, n) : omp2(A, x, b, n);
    double endTime = omp_get_wtime();

    delete[] A;
    delete[] x;
    delete[] b;

    printf("Working time: %.2f seconds\n", endTime - startTime);

    return 0;
}
