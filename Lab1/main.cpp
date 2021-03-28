#include <iostream>
#include <cmath>
#include <mpi.h>

#define EPSILON (1e-5)

void copyMem(double *ptrTo, double *ptrFrom, int k) {
    for (int i = 0; i < k; ++i) {
        ptrTo[i] = ptrFrom[i];
    }
}

void MPI1_fillData(double *A, double *x, double *b, int n, int m, int idx) {
    for (int i = idx; i < idx + m; ++i) {
        for (int j = 0; j < n; ++j) {
            A[j + i*n] = (i == j) ? 2.0 : 1.0;
        }
    }
    for (int i = 0; i < n; ++i) {
        x[i] = 0;
        b[i] = n+1;
    }
}

double* MPI1_calculateYn(double *A, double *xn, double *b, int n, int m, int idx) {
    auto *yn = new double[n];
    for (int i = idx; i < idx + m; ++i) {
        yn[i] = -b[i];
        for (int j = 0; j < n; ++j) {
            yn[i] += A[j + i*n] * xn[i];
        }
    }
    return yn;
}

bool MPI1_isSolutionFound(double *yn, double *b, int n) {
    double *length = new double[2] {0.0, 0.0};
    for (int i = 0; i < n; ++i) {
        length[0] += yn[i] * yn[i];
        length[1] += b[i] * b[i];
    }
    return sqrt(length[0]) / sqrt(length[1]) < EPSILON;
}

double* MPI1_calculateTn(double *A, double *yn, int n, int m, int idx) {
    auto *tn = new double[2] {0.0, 0.0};
    auto *Ayn = new double[n];
    for (int i = idx; i < idx + m; ++i) {
        for (int j = 0; j < n; ++j) {
            Ayn[i] += A[j + i*n] * yn[i];
        }
        tn[0] += yn[i] * Ayn[i];
        tn[1] += Ayn[i] * Ayn[i];
    }
    delete[] Ayn;
    return tn;
}

void MPI1_calculateNextX(double *x, double *yn, double tn, int n) {
    for (int i = 0; i < n; ++i) {
        x[i] -= yn[i] * tn;
    }
}

double* mpi1(int n, int m, int idx) {

    auto *A = new double[n * m];
    auto *x = new double[n];
    auto *b = new double[n];

    MPI1_fillData(A, x, b, n, m, idx);

    int k = 0;
    while (k++ < 10) {
        // y(n) = Ax(n) - b
        auto *yn = MPI1_calculateYn(A, x, b, n, m, idx);
        auto *ynFinal = new double[n];
        MPI_Allreduce(yn, ynFinal, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        delete[] yn;

        // Check |y(n)| / |b| < Epsilon
        if (MPI1_isSolutionFound(ynFinal, b, n)) {
            break;
        }

        // t(n) = (y(n), Ay(n)) / (Ay(n), Ay(n))
        double *tn = MPI1_calculateTn(A, ynFinal, n, m, idx);
        auto *tnFinal = new double[2];
        MPI_Allreduce(tn, tnFinal, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double tnResult = tnFinal[0] / tnFinal[1];
        delete[] tn;
        delete[] tnFinal;

        // x(n+1) = x(n) - t(n)*y(n)
        MPI1_calculateNextX(x, ynFinal, tnResult, n);
        delete[] ynFinal;
    }

    delete[] A;
    delete[] b;

    return x;
}

void MPI2_fillData(double *A, double *x, double *b, int n, int m, int idx) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            A[j + i*n] = (i == idx + j) ? 2.0 : 1.0;
        }
    }
    for (int i = 0; i < m; ++i) {
        x[i] = 0;
        b[i] = n+1;
    }
}

double* MPI2_calculateYn(double *A, double *xn, double *b, int n, int m, int idx) {
    auto *yn = new double[n];
    for (int i = 0; i < n; ++i) {
        if (idx <= i  && i < idx + m) {
            yn[i] = -b[i - idx];
        }
        for (int j = 0; j < m; ++j) {
            yn[i] += A[j + i*n] * xn[j];
        }
    }
    return yn;
}

double* MPI2_getVectorsLength(double *yn, double *b, int m) {
    double *length = new double[2] {0.0, 0.0};
    for (int i = 0; i < m; ++i) {
        length[0] += yn[i] * yn[i];
        length[1] += b[i] * b[i];
    }
    return length;
}

double* MPI2_calculateAyn(double *A, double *yn, int n, int m) {
    auto *Ayn = new double[n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            Ayn[i] += A[j + i*n] * yn[j];
        }
    }
    return Ayn;
}

double* MPI2_calculateTn(double *yn, double *Ayn, int m) {
    double *tn = new double[2] {0.0, 0.0};
    for (int i = 0; i < m; ++i) {
        tn[0] += yn[i] * Ayn[i];
        tn[1] += Ayn[i] * Ayn[i];
    }
    return tn;
}

double* MPI2_calculateNextX(double *xn, double *yn, double tn, int n, int m, int idx) {
    auto *nextX = new double[n];
    for (int i = 0; i < m; ++i) {
        nextX[idx + i] = xn[i] - tn * yn[i];
    }
    return nextX;
}

double* mpi2(int n, int m, int idx) {

    auto *A = new double[n * m];
    auto *x = new double[m];
    auto *b = new double[m];

    MPI2_fillData(A, x, b, n, m, idx);

    while (true) {
        // y(n) = Ax(n) - b
        auto *yn = MPI2_calculateYn(A, x, b, n, m, idx);
        auto *ynFinal = new double[n];
        MPI_Allreduce(yn, ynFinal, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        delete[] yn;

        // y(n) cut
        auto *ynCut = new double [m];
        copyMem(ynCut, &ynFinal[idx], m);
        delete[] ynFinal;

        // Check |y(n)| / |b| < Epsilon
        auto *length = MPI2_getVectorsLength(ynCut, b, m);
        auto *lengthFinal = new double[2];
        MPI_Allreduce(length, lengthFinal, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (sqrt(lengthFinal[0]) / sqrt(lengthFinal[1]) < EPSILON) {
            break;
        }
        delete[] length;
        delete[] lengthFinal;

        // Ay(n) calculation
        double *Ayn = MPI2_calculateAyn(A, ynCut, n, m);
        auto *AynFinal = new double [n];
        MPI_Allreduce(Ayn, AynFinal, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        delete[] Ayn;

        // Ay(n) cut
        auto *AynCut = new double [m];
        copyMem(AynCut, &AynFinal[idx], m);
        delete[] AynFinal;

        // t(n) = (y(n), Ay(n)) / (Ay(n), Ay(n))
        double *tn = MPI2_calculateTn(ynCut, AynCut, m);
        auto *tnFinal = new double[2];
        MPI_Allreduce(tn, tnFinal, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double tnResult = tnFinal[0] / tnFinal[1];
        delete[] AynCut;
        delete[] tn;
        delete[] tnFinal;

        // x(n+1) = x(n) - t(n)*y(n)
        auto *xNext = MPI2_calculateNextX(x, ynCut, tnResult, n, m, idx);
        auto *xNextFinal = new double [n];
        MPI_Allreduce(xNext, xNextFinal, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        delete[] ynCut;
        delete[] xNext;

        // x(n) cut
        copyMem(x, &xNextFinal[idx], m);
        delete[] xNextFinal;
    }

    // Final x calculation
    auto *result = new double [n];
    copyMem(result, &x[idx], m);
    auto *resultFinal = new double [n];
    MPI_Allreduce(result, resultFinal, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    delete[] result;

    delete[] A;
    delete[] x;
    delete[] b;

    return resultFinal;
}

void printAnswer(double *x, int n) {
    for (int i = 0; i < n; ++i) {
        printf("x%d: %.2f\n", i, x[i]);
    }
}

int calculateIdx(int n, int procTotal, int procRank) {
    int idx = 0;
    for (int i = 0; i < procRank; ++i) {
        idx += (n - idx) / (procTotal - i);
    }
    return idx;
}

int main(int argc, char* argv[]) {

    if (argc != 2) {
        std::cout << "Please, enter mpi variant: 1, 2" << std::endl;
        return 0;
    }

    int procTotal, procRank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &procTotal);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

    int n = 10;
    int idx = calculateIdx(n, procTotal, procRank);
    int m = (n - idx) / (procTotal - procRank);


    bool isFirstVariant = atoi(argv[1]) == 1;
    double *result = isFirstVariant ? mpi1(n, m, idx) : mpi2(n, m, idx);

    MPI_Finalize();

    printAnswer(result, n);
    delete[] result;

    return 0;
}
