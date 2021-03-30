#include <iostream>
#include <cmath>
#include <ctime>

#define EPSILON (10e-5)
#define INF (10e6)

void fillData(double *A, double *x, double *b, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[j + i*n] = (i == j) ? 2.0 : 1.0;
        }
        x[i] = 0;
        b[i] = n + 1;
    }
}

double* calculateYn(double *A, double *x, double *b, int n) {
    auto *yn = new double[n];
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        yn[i] = -b[i];
        for (int j = 0; j < n; ++j) {
            yn[i] += A[j + i*n] * x[i];
        }
    }
    return yn;
}

bool isSolutionFound(double *yn, double *b, int n) {
    auto *length = new double[2]();
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        length[0] += yn[i];
        length[1] += b[i];
    }
    bool isFound = sqrt(std::abs(length[0] / length[1])) < EPSILON;
    delete[] length;
    return isFound;
}

double calculateTn(double *A, double *yn, int n) {
    auto *tn = new double[2]();
    double AynTmp;
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        AynTmp = 0.0;
        for (int j = 0; j < n; ++j) {
            AynTmp += A[j + i*n] * yn[i];
        }
        tn[0] += yn[i] * AynTmp;
        tn[1] += AynTmp * AynTmp;
    }
    double tnResult = tn[0] / tn[1];
    delete[] tn;
    return tnResult;
}

void calculateNextX(double *x, double *yn, double tn, int n) {
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        x[i] -= yn[i] * tn;
    }
}

void iter(int n) {

    auto *A = new double[n * n];
    auto *x = new double[n];
    auto *b = new double[n];

    fillData(A, x, b, n);

    for (int k = 0; k < INF; ++k) {
        double *yn = calculateYn(A, x, b, n);

        if (isSolutionFound(yn, b, n)) {
            delete[] yn;
            return;
        }

        double tn = calculateTn(A, yn, n);

        calculateNextX(x, yn, tn, n);
        delete[] yn;
    }

    delete[] A;
    delete[] x;
    delete[] b;
}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        printf("Wrong arguments number\n");
        return 0;
    }

    int n = atoi(argv[1]);

    clock_t startTime = clock();
    iter(n);
    clock_t endTime = clock();

    double elapsedTime = (endTime - startTime) / CLOCKS_PER_SEC;
    printf("Working time: %.2f seconds\n", elapsedTime);

    return 0;
}
