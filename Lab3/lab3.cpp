#include <iostream>
#include <mpi.h>


void fillData(double *A, double *B, int n1, int n2, int n3) {
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n2; ++j) {
            A[i*n2 + j] = (i % 2) ? 1.0 : 2.0;
        }
    }
    for (int i = 0; i < n2; ++i) {
        for (int j = 0; j < n3; ++j) {
            B[i*n3 + j] = 2.0;
        }
    }
}

int calculateM(int n, int procTotal, int procRank) {
    int m, idx = 0;
    for (int i = 0; i < procRank + 1; ++i) {
        m = (n - idx) / (procTotal - i);
        idx += m;
    }
    return m;
}

void mulMatrix(double *A, double *B, double *C, int n1, int n2, int n3) {
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n3; ++j) {
            for (int k = 0; k < n2; ++k) {
                C[i*n3 + j] += A[i*n2 + k] * B[k*n3 + j];
            }
        }
    }
}

void printAnswer(double *C, int n1, int n3) {
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n3; ++j) {
            printf("%.1f ", C[i*n3 + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    MPI_Comm matrixComm, rowsComm, colsComm;
    int procRows = 4;
    int procCols = 4;

    int matrixDims[2] = {procRows, procCols};
    int matrixPeriods[2] = {false, false};
    int matrixReorder = true;
    int rowsDims[2] = {procRows, 0};
    int colsDims[2] = {0, procCols};
    int coords[2];

    int n1 = 24;
    int n2 = 16;
    int n3 = 16;

    double *A = nullptr, *B = nullptr, *tmpC = nullptr, *C;
    double *blockA, *blockB, *blockC;

    // Создание декартовой топологии размером procRows x procCols
    MPI_Cart_create(
        MPI_COMM_WORLD,
        2,
        matrixDims,
        matrixPeriods,
        matrixReorder,
        &matrixComm
    );

    // Выделение коммутаторов под строки и столбцы
    MPI_Cart_sub(matrixComm, rowsDims, &rowsComm);
    MPI_Cart_sub(matrixComm, colsDims, &colsComm);

    // Получение координат текцщего процесса
    MPI_Cart_get(matrixComm, 2, matrixDims, matrixPeriods, coords);

    int xRank = coords[0];
    int yRank = coords[1];

    // Подсчёт количества строк и столбцов, выделяемых на процесс
    int rowsM = calculateM(n1, procRows, xRank);
    int colsM = calculateM(n2, procCols, yRank);
    int blockSize = rowsM * colsM;

    if (!xRank && !yRank) {
        A = new double[n1 * n2];
        B = new double[n2 * n3];
        tmpC = new double[n1 * n3];
        C = new double[n1 * n3];
        fillData(A, B, n1, n2, n3);
    }

    blockA = new double[rowsM * n2];
    blockB = new double[colsM * n3];
    blockC = new double[blockSize]();

    // Раздача строк матриц A и B по первому столдцу и первой строке процессов соответсенно
    if (!yRank) {
        MPI_Scatter(A, rowsM * n2, MPI_DOUBLE, blockA, rowsM * n2, MPI_DOUBLE, 0, rowsComm);
    }
    if (!xRank) {
        MPI_Scatter(B, colsM * n3, MPI_DOUBLE, blockB, colsM * n3, MPI_DOUBLE, 0, colsComm);
    }

    // Ожидание завершения и раздача матриц останым процессам в декартовой системе
    MPI_Barrier(matrixComm);
    MPI_Bcast(blockA, rowsM * n2, MPI_DOUBLE, 0, colsComm);
    MPI_Bcast(blockB, colsM * n3, MPI_DOUBLE, 0, rowsComm);

    // Ожидание завершения и подсчётов блока на каждом процессе
    MPI_Barrier(matrixComm);
    mulMatrix(blockA, blockB, blockC, rowsM, n2, colsM);

    // Сборка матрицы C со всех процессов
    MPI_Gather(blockC, blockSize, MPI_DOUBLE, tmpC, blockSize, MPI_DOUBLE, 0, matrixComm);

    delete[] blockA;
    delete[] blockB;
    delete[] blockC;

    // Корректировка результатов после сбора матрицы MPI_Gather
    if (!xRank && !yRank) {
        int k = 0;
        for (int x = 0; x < procRows; ++x) {
            for (int y = 0; y < procCols; ++y) {
                for (int i = x * rowsM; i < (x + 1) * rowsM; ++i) {
                    for (int j = y * colsM; j < (y + 1) * colsM; ++j) {
                        C[i * n3 + j] = tmpC[k++];
                    }
                }
            }
        }
    }

    // Вывод результата и очистка памяти
    if (!xRank && !yRank) {
        printAnswer(C, n1, n3);
        delete[] A;
        delete[] B;
        delete[] C;
    }

    MPI_Finalize();

    return 0;
}