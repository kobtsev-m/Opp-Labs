#include <iostream>
#include <mpi.h>
#include <cmath>

const double EPSILON = 10e-8;
const double A_CONST = 10e5;

const int NX = 128;
const int NY = 8;
const int NZ = 8;

const double X0 = -1.0;
const double Y0 = -1.0;
const double Z0 = -1.0;
const double PHI0 = 0.0;

const double HX = 2.0 / (double)(NX - 1);
const double HY = 2.0 / (double)(NY - 1);
const double HZ = 2.0 / (double)(NZ - 1);


int calculateChunkSize(int n, int pSize, int pRank) {
    int m = 0, idx = 0;
    for (int i = 0; i < pRank + 1; ++i) {
        idx += m;
        m = (n - idx) / (pSize - i);
    }
    return m + 2;
}

int calculateChunkDiplacement(int n, int pSize, int pRank) {
    int m = 0, idx = 0;
    for (int i = 0; i < pRank + 1; ++i) {
        idx += m;
        m = (n - idx) / (pSize - i);
    }
    return idx + 2*pRank;
}

double calculatePhi(int i, int j, int k) {
    double x = X0 + (double)i*HX;
    double y = Y0 + (double)j*HY;
    double z = Z0 + (double)k*HZ;
    return x*x + y*y + z*z;
}

double calculateRo(double **prevPhi, int i, int j, int k) {
    return 6 - A_CONST*prevPhi[i][j*NZ + k];
}

void calculateNextPhi(double **phi, double **prevPhi, int i, int j, int k) {
    double xPart = (prevPhi[i + 1][j*NZ + k] + prevPhi[i - 1][j*NZ + k]) / (HX*HX);
    double yPart = (prevPhi[i][(j+1)*NZ + k] + prevPhi[i][(j - 1)*NZ + k]) / (HY*HY);
    double zPart = (prevPhi[i][j*NZ + (k+1)] + prevPhi[i][j*NZ + (k - 1)]) / (HZ*HZ);
    phi[i][j*NZ + k] = xPart + yPart + zPart - calculateRo(prevPhi, i, j, k);
    phi[i][j*NZ + k] /= (2/(HX*HX) + 2/(HY*HY) + 2/(HZ*HZ) + A_CONST);
}

bool checkForInnerBorder(int pSize, int pRank, int chunkSize, int i, int j, int k) {
    return (pRank == 0 && i == 1) ||
           (pRank == pSize - 1 && i == chunkSize - 2) ||
           (j == 0 || k == 0) ||
           (j == NY - 1 || k == NZ - 1);
}

bool checkForOuterBorder(int pSize, int pRank, int chunkSize, int i) {
    return (pRank == 0 && i == 0) ||
           (pRank == pSize - 1 && i == chunkSize - 1);
}

void fillInitialData(double **phi, int pSize, int pRank, int chunkSize) {
    for (int i = 1; i < chunkSize - 2; ++i) {
        for (int j = 0; j < NY; ++j) {
            for (int k = 0; k < NZ; ++k) {
                if (checkForInnerBorder(pSize, pRank, chunkSize, i, j, k)) {
                    int idx = calculateChunkDiplacement(NX, pSize, pRank);
                    phi[i][j*NZ + k] = calculatePhi(idx + i, j, k);
                } else{
                    phi[i][j*NZ + k] = PHI0;
                }
            }
        }
    }
}

double calculateEpsilon(double** fi, double** prevPhi, int chunkSize) {
    double maxDiff = 0, tmpDiff;
    for (int i = 1; i < chunkSize - 1; ++i) {
        for (int j = 1; j < NY - 1; ++j) {
            for (int k = 1; k < NZ - 1; ++k) {
                tmpDiff = std::abs(fi[i][j*NZ + k] - prevPhi[i][j*NZ + k]);
                maxDiff = tmpDiff > maxDiff ? tmpDiff : maxDiff;
            }
        }
    }
    return maxDiff;
}

double calculateAccuracy(double** phi,int pRank, int pSize, int chunkSize) {
    double maxDiff = 0, tmpDiff;
    for (int i = 0; i < chunkSize; ++i) {
        for (int j = 0; j < NY; ++j) {
            for (int k = 0; k < NZ; ++k) {
                if (checkForOuterBorder(pSize, pRank, chunkSize, i)) {
                    continue;
                }
                int idx = calculateChunkDiplacement(NX, pSize, pRank);
                tmpDiff = std::abs(phi[i][j*NZ+k] - calculatePhi(idx + i, j, k));
                maxDiff = tmpDiff > maxDiff ? tmpDiff : maxDiff;
            }
        }
    }
    return maxDiff;
}

int main(int argc, char* argv[]) {

    MPI_Init(&argc,&argv);

    int pSize, pRank;
    MPI_Comm_size(MPI_COMM_WORLD, &pSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &pRank);

    int sendRecvTag = 0;
    MPI_Request sendRecvRequest[4];
    MPI_Status sendRecvStatus[4];

    int chunkSize = calculateChunkSize(NX, pSize, pRank);
    int iter = 0;

    double **phi = new double* [chunkSize];
    double **prevPhi = new double* [chunkSize];
    for (int i = 0; i < chunkSize; ++i) {
        phi[i] = new double [NY * NZ];
        prevPhi[i] = new double [NY * NZ];
    }

    double maxDiff = EPSILON, localDiff;
    double maxAccuracy, localAccuracy;
    double startTime, endTime;

    fillInitialData(phi, pSize, pRank, chunkSize);

    startTime = MPI_Wtime();

    for (; maxDiff >= EPSILON; ++iter) {

        // Копирование значений в prevPhi
        for (int i = 0; i < chunkSize; ++i) {
            for (int j = 0; j < NY; ++j) {
                for (int k = 0; k < NZ; ++k) {
                    prevPhi[i][j*NZ + k] = phi[i][j*NZ + k];
                }
            }
        }

        // Подсчёт значений на границах
        for (int j = 0; j < NY; ++j) {
            for (int k = 0; k < NZ; ++k) {
                if (pRank != 0) {
                    calculateNextPhi(phi, prevPhi, 1, j, k);
                }
                if (pRank != pSize - 1) {
                    calculateNextPhi(phi, prevPhi, chunkSize - 2, j, k);
                }
            }
        }

        // Рассылка границ соседним процессам
        if (pRank != 0) {
            MPI_Isend(
                phi[1], NY * NZ, MPI_DOUBLE, pRank - 1, sendRecvTag,
                MPI_COMM_WORLD, &sendRecvRequest[0]
            );
        }
        if (pRank != pSize - 1) {
            MPI_Isend(
                phi[chunkSize - 2], NY * NZ, MPI_DOUBLE, pRank + 1, sendRecvTag,
                MPI_COMM_WORLD, &sendRecvRequest[1]
            );
        }

        // Подсчёт остальных значений на текущей итерации
        for (int i = 2; i < chunkSize - 2; ++i) {
            for (int j = 0; j < NY; ++j) {
                for (int k = 0; k < NZ; ++k) {
                    calculateNextPhi(phi, prevPhi, i, j, k);
                }
            }
        }

        // Получение границ с соседних процессов
        if (pRank != 0) {
            MPI_Irecv(
                phi[chunkSize - 1], NY * NZ, MPI_DOUBLE, pRank - 1, sendRecvTag,
                MPI_COMM_WORLD, &sendRecvRequest[2]
            );
        }
        if (pRank != pSize - 1) {
            MPI_Irecv(
                phi[0], NY * NZ, MPI_DOUBLE, pRank + 1, sendRecvTag,
                MPI_COMM_WORLD, &sendRecvRequest[3]
            );
        }

        // Ожидание завершения неблокирущих операций
        if (pRank != 0) {
            MPI_Wait(&sendRecvRequest[0], &sendRecvStatus[0]);
            MPI_Wait(&sendRecvRequest[2], &sendRecvStatus[2]);
        }
        if (pRank != pSize - 1) {
            MPI_Wait(&sendRecvRequest[1], &sendRecvStatus[1]);
            MPI_Wait(&sendRecvRequest[3], &sendRecvStatus[3]);
        }

        // Вычисление максимального модуля разности для проверки условия сходимсоти
        localDiff = calculateEpsilon(phi, prevPhi, chunkSize);
        MPI_Allreduce(&localDiff, &maxDiff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }

    endTime = MPI_Wtime();

    // Подсчёт погрешности вычисленной функции phi
    localAccuracy = calculateAccuracy(phi, pRank, pSize, chunkSize);
    MPI_Allreduce(&localAccuracy, &maxAccuracy, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // Вывод результата
    if (pRank == 0) {
        printf("Iterations: %d\n", iter);
        printf("Accuracy: %.2f\n", maxAccuracy);
        printf("Work time: %.2f\n", endTime - startTime);
    }

    // Очистка памяти
    for (int i = 0; i < chunkSize; ++i) {
        delete[] prevPhi[i];
        delete[] phi[i];
    }
    delete[] prevPhi;
    delete[] phi;

    MPI_Finalize();
    return 0;
}
