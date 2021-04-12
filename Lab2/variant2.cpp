#include <sys/time.h>
#include <iostream>
#include <cmath>

#define EPSILON (10e-4)
#define INF (10e6)

double randDouble(double max) {
    return static_cast<double>(rand()) / static_cast<double>(RAND_MAX / max);
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

bool isSolutionFound(double *yn, double *b, int n) {
    auto *length = new double[2]();
    #pragma omp for
    for (int i = 0; i < n; ++i) {
        length[0] += yn[i] * yn[i];
        length[1] += b[i] * b[i];
    }
    bool isFound = sqrt(length[0] / length[1]) < EPSILON;
    delete[] length;
    return isFound;
}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        printf("Wrong arguments number\n");
        return 0;
    }

    int n = atoi(argv[1]);

    auto *A = new double[n * n];
    auto *x = new double[n];
    auto *b = new double[n];

    fillData(A, x, b, n);

    struct timeval startTime, endTime;
    gettimeofday(&startTime, nullptr);

    volatile bool running = true;

    #pragma omp parallel shared(running)
    {
        for (int k = 0; k < INF && running; ++k) {
            // Calculating Yn
            auto *yn = new double[n];
            #pragma omp for
            for (int i = 0; i < n; ++i) {
                yn[i] = -b[i];
                for (int j = 0; j < n; ++j) {
                    yn[i] += A[j + i*n] * x[j];
                }
            }

            // Check if solution is found on all threads
            #pragma omp barrier
            {
                if (isSolutionFound(yn, b, n)) {
                    running = false;
                }
            }

            if (running) {
                // Calculating Tn
                auto *tn = new double[2]();
                double AynTmp;
                #pragma omp for
                for (int i = 0; i < n; ++i) {
                    AynTmp = 0.0;
                    for (int j = 0; j < n; ++j) {
                        AynTmp += A[j + i * n] * yn[j];
                    }
                    tn[0] += yn[i] * AynTmp;
                    tn[1] += AynTmp * AynTmp;
                }
                double tnResult = tn[0] / tn[1];
                delete[] tn;

                // Calculating next x
                #pragma omp for
                for (int i = 0; i < n; ++i) {
                    x[i] -= yn[i] * tnResult;
                }
            }
            // Deallocating memory
            delete[] yn;
        }
    }

    gettimeofday(&endTime, nullptr);

    delete[] A;
    delete[] x;
    delete[] b;

    double deltaSec = (endTime.tv_sec - startTime.tv_sec);
    double deltaUSec = (endTime.tv_usec - startTime.tv_usec);
    double delta = deltaSec + deltaUSec / 10e6;

    printf("Working time: %.2f seconds\n", delta);

    return 0;
}
