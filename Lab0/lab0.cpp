#include <sys/time.h>
#include <iostream>
#include <cmath>

#define EPSILON (10e-3)
#define INF (10e6)


double randDouble(double max) {
    return static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / max));
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
    double lengthYn = 0.0, lengthB = 0.0;
    for (int i = 0; i < n; ++i) {
        lengthYn += yn[i] * yn[i];
        lengthB += b[i] * b[i];
    }
    return sqrt(lengthYn / lengthB) < EPSILON;
}

double calculateTn(double *A, double *yn, int n) {
    double tn1 = 0.0, tn2 = 0.0;
    double AynTmp;
    for (int i = 0; i < n; ++i) {
        AynTmp = 0.0;
        for (int j = 0; j < n; ++j) {
            AynTmp += A[j + i*n] * yn[j];
        }
        tn1 += yn[i] * AynTmp;
        tn2 += AynTmp * AynTmp;
    }
    return tn1 / tn2;
}

void calculateNextX(double *x, double *yn, double tn, int n) {
    for (int i = 0; i < n; ++i) {
        x[i] -= yn[i] * tn;
    }
}

void printAnswer(double *x, int n) {
    for (int i = 0; i < n; ++i) {
        printf("x%d: %.1f", i, x[i]);
    }
}

void calculateWorkTime(struct timeval startTime, struct timeval endTime) {
    double deltaSec = (endTime.tv_sec - startTime.tv_sec);
    double deltaUSec = (endTime.tv_usec - startTime.tv_usec);
    printf("Working time: %.2f seconds\n", deltaSec + deltaUSec / 1e6);
}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        printf("Wrong arguments number\n");
        return 0;
    }

    int n = atoi(argv[1]);

    struct timeval startTime, endTime;
    auto *A = new double[n * n];
    auto *x = new double[n];
    auto *b = new double[n];

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

    printAnswer(x, 10);
    printWorkTime(startTime, endTime);

    delete[] A;
    delete[] x;
    delete[] b;

    return 0;
}
