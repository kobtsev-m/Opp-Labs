#include <iostream>

#define EPSILON (1e-5)

void initMatrix(double **matrix, int n) {
    for (int i = 0; i < n; ++i) {
        matrix[i] = new double [n];
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = i == j ? 2.0 : 1.0;
        }
    }
}

void clearMatrix(double **matrix, int n) {
    for (int i = 0; i < n; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

void initVectors(double *x, double *b, int n) {
    for (int i = 0; i < n; ++i) {
        x[i] = 0;
        b[i] = n+1;
    }
}

double* calculateYn(double **a, const double *xn, const double *b, int n) {
    auto *yn = new double [n];
    for (int i = 0; i < n; ++i) {
        yn[i] = -b[i];
        for (int j = 0; j < n; ++j) {
            yn[i] += a[i][j] * xn[i];
        }
    }
    return yn;
}

double calculateTn(double **a, const double *yn, int n) {
    double tnNumerator = 0.0;
    double tnDenominator = 0.0;
    auto *ayn = new double [n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            ayn[i] += a[i][j] * yn[i];
        }
        tnNumerator += yn[i] * ayn[i];
        tnDenominator += ayn[i] * ayn[i];
    }
    delete[] ayn;
    return tnNumerator / tnDenominator;
}

void calculateNextX(double *xn, const double *yn, double tn, int n) {
    for (int i = 0; i < n; ++i) {
        xn[i] -= tn * yn[i];
    }
}

bool isSolutionFound(const double *yn, const double *b, int n) {
    double lengthYn = 0.0;
    double lengthB = 0.0;
    for (int i = 0; i < n; ++i) {
        lengthYn += yn[i] * yn[i];
        lengthB += b[i] * b[i];
    }
    return (lengthYn / lengthB) < EPSILON;
}

void iterate(double **matrix_a, double *xn, double *b, int n) {
    while (true) {
        auto *yn = calculateYn(matrix_a, xn, b, n);
        if (isSolutionFound(yn, b, n)) {
            break;
        }
        double tn = calculateTn(matrix_a, yn, n);
        calculateNextX(xn, yn, tn, n);
        delete[] yn;
    }
}

void printAnswer(const double *x, int n) {
    for (int i = 0; i < n; ++i) {
        printf("x%d: %.2f\n", i, x[i]);
    }
}

int main(int argc, char* argv[]) {

    int n = argc > 1 ? strtol(argv[1], nullptr, 9) : 3;

    auto **matrix_a = new double *[n];
    auto *x = new double [n];
    auto *b = new double [n];

    initMatrix(matrix_a, n);
    initVectors(x, b, n);

    iterate(matrix_a, x, b, n);
    printAnswer(x, n);

    clearMatrix(matrix_a, n);
    delete[] x;
    delete[] b;

    return 0;
}
