#include <sys/time.h>
#include <iostream>
#include <cmath>

#define EPSILON (10e-3)
#define INF (10e6)

double randDouble(double max) {
    return static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / max));
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

double* calculateYn(double *A, double *x, double *b, int n) {
    auto *yn = new double[n];
    for (int i = 0; i < n; ++i) {
        yn[i] = -b[i];
        for (int j = 0; j < n; ++j) {
            yn[i] += A[j + i*n] * x[j];
        }
    }
    return yn;
}

bool isSolutionFound(double *yn, double *b, int n) {
    auto *length = new double[2]();
    for (int i = 0; i < n; ++i) {
        length[0] += yn[i] * yn[i];
        length[1] += b[i] * b[i];
    }
    bool isFound = sqrt(length[0] / length[1]) < EPSILON;
    delete[] length;
    return isFound;
}

double calculateTn(double *A, double *yn, int n) {
    auto *tn = new double[2]();
    double AynTmp;
    for (int i = 0; i < n; ++i) {
        AynTmp = 0.0;
        for (int j = 0; j < n; ++j) {
            AynTmp += A[j + i*n] * yn[j];
        }
        tn[0] += yn[i] * AynTmp;
        tn[1] += AynTmp * AynTmp;
    }
    double tnResult = tn[0] / tn[1];
    delete[] tn;
    return tnResult;
}

void calculateNextX(double *x, double *yn, double tn, int n) {
    for (int i = 0; i < n; ++i) {
        x[i] -= yn[i] * tn;
    }
}

double calculateElapsedTime(struct timeval startTime, struct timeval endTime) {
    double deltaSec = (endTime.tv_sec - startTime.tv_sec);
    double deltaUSec = (endTime.tv_usec - startTime.tv_usec);
    return deltaSec + deltaUSec/10e6;
}

double iter(int n) {

    auto *A = new double[n * n];
    auto *x = new double[n];
    auto *b = new double[n];
    struct timeval startTime, endTime;

    fillData(A, x, b, n);

    gettimeofday(&startTime, nullptr);

    for (int k = 0; k < INF; ++k) {
        double *yn = calculateYn(A, x, b, n);

        if (isSolutionFound(yn, b, n)) {
            delete[] yn;
            break;
        }

        double tn = calculateTn(A, yn, n);

        calculateNextX(x, yn, tn, n);
        delete[] yn;
    }

    gettimeofday(&endTime, nullptr);

    delete[] A;
    delete[] x;
    delete[] b;

    return calculateElapsedTime(startTime, endTime);
}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        printf("Wrong arguments number\n");
        return 0;
    }

    int n = atoi(argv[1]);
    printf("Working time: %.2f seconds\n", iter(n));

    return 0;
}
