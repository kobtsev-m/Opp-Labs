#include <iostream>
#include <cmath>
#include <mpi.h>

#define EPSILON (10e-3)
#define INF (10e6)


double randDouble(double max) {
    return static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / max));
}

void copyMem(double *ptrTo, double *ptrFrom, int k) {
    for (int i = 0; i < k; ++i) {
        ptrTo[i] = ptrFrom[i];
    }
}

void mpi1_fillData(double *A, double *x, double *b, int n, int m, int idx) {
    double u[n], tmpB[n];
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            A[j + i*n] = (double)(idx + i == j ? j*j : idx+i+j);
        }
    }
    for (int i = 0; i < n; ++i) {
        x[i] = randDouble(1000.0);
        u[i] = randDouble(1000.0);
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            tmpB[idx + i] += A[j + i*n] * u[j];
        }
    }
    MPI_Allreduce(tmpB, b, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

double* mpi1_calculateYn(double *A, double *xn, double *b, int n, int m, int idx) {
    auto *yn = new double[n]();
    for (int i = 0; i < m; ++i) {
        yn[idx + i] -= b[idx + i];
        for (int j = 0; j < n; ++j) {
            yn[idx + i] += A[j + i*n] * xn[j];
        }
    }
    return yn;
}

bool mpi1_isSolutionFound(double *yn, double *b, int n, int idx) {
    double lengthYn = 0.0, lengthB = 0.0;
    for (int i = 0; i < n; ++i) {
        lengthYn += yn[i] * yn[i];
        lengthB += b[i] * b[i];
    }
    return sqrt(lengthYn / lengthB) < EPSILON;
}

double* mpi1_calculateTn(double *A, double *yn, int n, int m, int idx) {
    auto *tn = new double[2]();
    double AynTmp;
    for (int i = 0; i < m; ++i) {
        AynTmp = 0.0;
        for (int j = 0; j < n; ++j) {
            AynTmp += A[j + i*n] * yn[j];
        }
        tn[0] += yn[idx + i] * AynTmp;
        tn[1] += AynTmp * AynTmp;
    }
    return tn;
}

void mpi1_calculateNextX(double *x, double *yn, double tn, int n) {
    for (int i = 0; i < n; ++i) {
        x[i] -= yn[i] * tn;
    }
}

double mpi1(int n, int m, int idx) {

    auto *A = new double[n * m];
    auto *x = new double[n];
    auto *b = new double[n];

    mpi1_fillData(A, x, b, n, m, idx);

    double startTime = MPI_Wtime();

    for (int k = 0; k < INF; ++k) {
        // y(n) = Ax(n) - b
        auto *yn = mpi1_calculateYn(A, x, b, n, m, idx);
        double ynFinal[n];
        MPI_Allreduce(yn, ynFinal, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        delete[] yn;

        // Check |y(n)| / |b| < Epsilon
        if (mpi1_isSolutionFound(ynFinal, b, n, idx)) {
            break;
        }

        // t(n) = (y(n), Ay(n)) / (Ay(n), Ay(n))
        auto *tn = mpi1_calculateTn(A, ynFinal, n, m, idx);
        double tnFinal[2];
        MPI_Allreduce(tn, tnFinal, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        delete[] tn;

        // x(n+1) = x(n) - t(n)*y(n)
        mpi1_calculateNextX(x, ynFinal, tnFinal[0] / tnFinal[1], n);
    }

    double endTime = MPI_Wtime();

    delete[] A;
    delete[] b;
    delete[] x;

    return endTime - startTime;
}

void mpi2_fillData(double *A, double *x, double *b, int n, int m, int idx) {
    double u[m], tmpB[n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            A[j + i*m] = (double)(i == idx + j ? i*i : idx+i+j);
        }
    }
    for (int i = 0; i < m; ++i) {
        x[i] = randDouble(1000.0);
        u[i] = randDouble(1000.0);
    }
    for (int i = 0; i < n; ++i) {
        tmpB[i] = 0.0;
        for (int j = 0; j < m; ++j) {
            tmpB[i] += A[j + i*m] * u[j];
        }
    }
    double finalB[n];
    MPI_Allreduce(tmpB, finalB, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    copyMem(b, &finalB[idx], m);
}

double* mpi2_calculateYn(double *A, double *xn, double *b, int n, int m, int idx) {
    auto *yn = new double[n]();
    for (int i = 0; i < n; ++i) {
        if (idx <= i  && i < idx + m) {
            yn[i] = -b[i - idx];
        }
        for (int j = 0; j < m; ++j) {
            yn[i] += A[j + i*m] * xn[j];
        }
    }
    return yn;
}

double* mpi2_getVectorsLength(double *yn, double *b, int m) {
    double *length = new double[2]();
    for (int i = 0; i < m; ++i) {
        length[0] += yn[i] * yn[i];
        length[1] += b[i] * b[i];
    }
    return length;
}

double* mpi2_calculateAyn(double *A, double *yn, int n, int m) {
    auto *Ayn = new double[n]();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            Ayn[i] += A[j + i*m] * yn[j];
        }
    }
    return Ayn;
}

double* mpi2_calculateTn(double *yn, double *Ayn, int m) {
    double *tn = new double[2]();
    for (int i = 0; i < m; ++i) {
        tn[0] += yn[i] * Ayn[i];
        tn[1] += Ayn[i] * Ayn[i];
    }
    return tn;
}

double* mpi2_calculateNextX(double *xn, double *yn, double tn, int n, int m, int idx) {
    auto *xNext = new double[n]();
    for (int i = 0; i < m; ++i) {
        xNext[idx + i] = xn[i] - tn * yn[i];
    }
    return xNext;
}

double mpi2(int n, int m, int idx) {

    auto *A = new double[m * n];
    auto *x = new double[m];
    auto *b = new double[m];
    auto *xFinal = new double[n];

    mpi2_fillData(A, x, b, n, m, idx);

    double startTime = MPI_Wtime();

    for (int k = 0; k < INF; ++k) {
        // y(n) = Ax(n) - b
        auto *yn = mpi2_calculateYn(A, x, b, n, m, idx);
        double ynFinal[n];
        MPI_Allreduce(yn, ynFinal, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        delete[] yn;

        // y(n) cut
        double ynCut[m];
        copyMem(ynCut, &ynFinal[idx], m);

        // Calculating |y(n)| / |b| < Epsilon
        auto *length = mpi2_getVectorsLength(ynCut, b, m);
        double lengthFinal[2];
        MPI_Allreduce(length, lengthFinal, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        delete[] length;

        // Check if solution is found
        if (sqrt(lengthFinal[0] / lengthFinal[1]) < EPSILON) {
            break;
        }

        // Ay(n) calculation
        auto *Ayn = mpi2_calculateAyn(A, ynCut, n, m);
        double AynFinal[n];
        MPI_Allreduce(Ayn, AynFinal, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        delete[] Ayn;

        // Ay(n) cut
        double AynCut[m];
        copyMem(AynCut, &AynFinal[idx], m);

        // t(n) = (y(n), Ay(n)) / (Ay(n), Ay(n))
        auto *tn = mpi2_calculateTn(ynCut, AynCut, m);
        double tnFinal[2];
        MPI_Allreduce(tn, tnFinal, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        delete[] tn;

        // x(n+1) = x(n) - t(n)*y(n)
        auto *nextX = mpi2_calculateNextX(x, ynCut, tnFinal[0] / tnFinal[1], n, m, idx);
        MPI_Allreduce(nextX, xFinal, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        delete[] nextX;

        // x(n) cut
        copyMem(x, &xFinal[idx], m);
    }

    auto endTime = MPI_Wtime();

    delete[] A;
    delete[] x;
    delete[] b;
    delete[] xFinal;

    return endTime - startTime;
}

int calculateChunkDisplacement(int n, int procTotal, int procRank) {
    int idx = 0;
    for (int i = 0; i < procRank; ++i) {
        idx += (n - idx) / (procTotal - i);
    }
    return idx;
}

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    int procTotal, procRank;
    MPI_Comm_size(MPI_COMM_WORLD, &procTotal);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

    if (argc != 3) {
        if (procRank == 0) {
            printf("Wrong arguments number\n");
        }
        MPI_Finalize();
        return 0;
    }

    int variant = atoi(argv[1]);
    int n = atoi(argv[2]);

    int idx = calculateChunkDisplacement(n, procTotal, procRank);
    int m = (n - idx) / (procTotal - procRank);

    double elapsedTime = variant == 1 ? mpi1(n, m, idx) : mpi2(n, m, idx);

    if (procRank == 0) {
        printf("Work time: %.2f seconds\n", elapsedTime);
    }

    MPI_Finalize();
    return 0;
}
