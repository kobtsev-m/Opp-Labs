#include <iostream>
#include <cmath>
#include <cstring>
#include <mpi.h>
#include <pthread.h>

#define SUCCESS_STATUS 0
#define ERROR_STATUS 1

#define REQUEST_TAG 0
#define ANSWER_TAG 1

#define TASK_SENT 0
#define ASK_FOR_TASK 1
#define NO_TASKS 2
#define WORK_DONE 3

const int MAIN_PROC = 0;
const int ITERS_TOTAL = 5;
const int TASKS_ON_PROC = 250;
const int L_CONST = 1000;
const int MUTEX_TOTAL = 3;

pthread_mutex_t taskListMutex;
pthread_mutex_t ownIdxMutex;
pthread_mutex_t givenTasksMutex;

pthread_mutex_t mutexList[] = {
    taskListMutex,
    ownIdxMutex,
    givenTasksMutex
};

int pRank, pSize;
int* taskList;
int givenTasks, ownTaskIdx;

int getTasksToGive() {
    return pSize / (pRank + 1);
}

void calculate(double *localRes, int taskIdx) {
    for (int i = 0; i < taskList[taskIdx]; ++i) {
        *localRes += exp(sin(i));
    }
}

void refreshTaskList(int iter) {
    pthread_mutex_lock(&taskListMutex);
    for (int i = pRank * TASKS_ON_PROC; i < (pRank + 1) * TASKS_ON_PROC; ++i) {
        taskList[i] = std::abs(50 - i % TASKS_ON_PROC) * std::abs(pRank - (iter % pSize)) * L_CONST;
    }
    pthread_mutex_unlock(&taskListMutex);
}

bool getNewTasks(int sponsor, int *receivedTasks, int *otherTaskIdx) {
    int sendMsg = ASK_FOR_TASK, recvMsg;
    MPI_Send(&sendMsg, 1, MPI_INT, sponsor, REQUEST_TAG, MPI_COMM_WORLD);
    MPI_Recv(&recvMsg, 1, MPI_INT, sponsor, ANSWER_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (recvMsg == TASK_SENT) {
        MPI_Recv(
            receivedTasks, 1, MPI_INT,
            sponsor, ANSWER_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );
        pthread_mutex_lock(&taskListMutex);
        MPI_Recv(
            &taskList[sponsor * TASKS_ON_PROC], *receivedTasks, MPI_INT,
            sponsor, ANSWER_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );
        pthread_mutex_unlock(&taskListMutex);
        *otherTaskIdx = sponsor * TASKS_ON_PROC;
        return true;
    }
    return false;
}

void* workingTask(void*) {
    int localTasksCounter, tasksCounter;
    double startTime, localTime, maxTime, minTime;
    double avaregeImbalance = 0.0;
    double localRes = 0.0, globalRes;

    for (int iter = 0; iter < ITERS_TOTAL; ++iter) {
        if (pRank == MAIN_PROC) {
            printf("-----------ITER %d-----------\n", iter);
        }

        refreshTaskList(iter);
        localTasksCounter = 0;

        pthread_mutex_lock(&ownIdxMutex);
        ownTaskIdx = TASKS_ON_PROC * pRank;
        pthread_mutex_unlock(&ownIdxMutex);
        pthread_mutex_lock(&givenTasksMutex);
        givenTasks = 0;
        pthread_mutex_unlock(&givenTasksMutex);

        startTime = MPI_Wtime();

        while (ownTaskIdx < TASKS_ON_PROC * (pRank + 1) - givenTasks) {
            calculate(&localRes, ownTaskIdx);
            pthread_mutex_lock(&ownIdxMutex);
            ownTaskIdx++;
            pthread_mutex_unlock(&ownIdxMutex);
            localTasksCounter++;
        }

        bool areNewTasks;
        do {
            areNewTasks = false;
            for (int currRank = 0; currRank < pSize; ++currRank) {
                int receivedTasks, otherTaskIdx;
                if (currRank != pRank && getNewTasks(currRank, &receivedTasks, &otherTaskIdx)) {
                    for (int i = 0; i < receivedTasks; ++i) {
                        calculate(&localRes, otherTaskIdx);
                        otherTaskIdx++;
                        localTasksCounter++;
                    }
                    areNewTasks = true;
                    break;
                }
            }
        } while (areNewTasks);

        localTime = MPI_Wtime() - startTime;

        printf("Elapsed time on proc %d: %.2f\n", pRank, localTime);
        MPI_Barrier(MPI_COMM_WORLD);
        printf("Tasks done on proc %d: %d\n", pRank, localTasksCounter);

        MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, MAIN_PROC, MPI_COMM_WORLD);
        MPI_Reduce(&localTime, &minTime, 1, MPI_DOUBLE, MPI_MIN, MAIN_PROC, MPI_COMM_WORLD);
        MPI_Reduce(&localTasksCounter, &tasksCounter, 1, MPI_INT, MPI_SUM, MAIN_PROC, MPI_COMM_WORLD);

        if (pRank == MAIN_PROC) {
            printf("Tasks total: %d\n", tasksCounter);
            printf("Imbalance time: %.2f\n", maxTime - minTime);
            printf("Imbalance proportion: %.2f%%\n", (maxTime - minTime) / maxTime * 100);
            avaregeImbalance += (maxTime - minTime) / maxTime * 100;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    int sendMsg = WORK_DONE;
    MPI_Send(&sendMsg, 1, MPI_INT, pRank, REQUEST_TAG, MPI_COMM_WORLD);

    MPI_Reduce(&localRes, &globalRes, 1, MPI_DOUBLE, MPI_SUM, MAIN_PROC, MPI_COMM_WORLD);

    if (pRank == MAIN_PROC) {
        printf("----------RESULTS-----------\n");
        printf("Global result: %.2f\n", localRes);
        printf("Average imbalance: %.2f%%\n", avaregeImbalance / ITERS_TOTAL);
    }

    return NULL;
}

void* sendingTask(void*) {

    MPI_Status recvStatus;
    int sendMsg, recvMsg;

    while (true) {
        MPI_Recv(&recvMsg, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_TAG, MPI_COMM_WORLD, &recvStatus);
        int sender = recvStatus.MPI_SOURCE;
        bool otherProcNeedTasks = recvMsg == ASK_FOR_TASK && sender != pRank;
        pthread_mutex_lock(&ownIdxMutex);
        bool noTasksToSend = ownTaskIdx > (pRank + 1) * TASKS_ON_PROC - givenTasks;
        pthread_mutex_unlock(&ownIdxMutex);
        if (otherProcNeedTasks && noTasksToSend) {
            sendMsg = NO_TASKS;
            MPI_Send(&sendMsg, 1, MPI_INT, sender, ANSWER_TAG, MPI_COMM_WORLD);
        }
        else if (otherProcNeedTasks) {
            int sendingTasksCount = getTasksToGive();
            pthread_mutex_lock(&givenTasksMutex);
            givenTasks += sendingTasksCount;
            pthread_mutex_unlock(&givenTasksMutex);
            pthread_mutex_lock(&taskListMutex);
            int *sendingTasks = &taskList[(pRank + 1) * TASKS_ON_PROC - givenTasks];
            pthread_mutex_unlock(&taskListMutex);
            sendMsg = TASK_SENT;
            MPI_Send(&sendMsg, 1, MPI_INT, sender, ANSWER_TAG, MPI_COMM_WORLD);
            MPI_Send(&sendingTasksCount, 1, MPI_INT, sender, ANSWER_TAG, MPI_COMM_WORLD);
            MPI_Send(sendingTasks, sendingTasksCount, MPI_INT, sender, ANSWER_TAG, MPI_COMM_WORLD);
        }
        else if (recvMsg == WORK_DONE) {
            break;
        }
    }
    return NULL;
}

int main(int argc, char *argv[]) {

    int providedLevel;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &providedLevel);

    if (providedLevel != MPI_THREAD_MULTIPLE) {
        fprintf(stderr, "Error on pthread init");
        MPI_Finalize();
        return ERROR_STATUS;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &pRank);
    MPI_Comm_size(MPI_COMM_WORLD, &pSize);

    taskList = new int [TASKS_ON_PROC * pSize]();

    pthread_attr_t attrs;
    int attrsInitRes = pthread_attr_init(&attrs);
    if (attrsInitRes != SUCCESS_STATUS) {
        fprintf(stderr, "Error on initializing attrs\n");
        MPI_Finalize();
        return ERROR_STATUS;
    }

    int setStateRes = pthread_attr_setdetachstate(&attrs, PTHREAD_CREATE_JOINABLE);
    if (setStateRes != SUCCESS_STATUS) {
        fprintf(stderr, "Error on setting detach state\n");
        MPI_Finalize();
        return ERROR_STATUS;
    }

    for (int i = 0; i < MUTEX_TOTAL; ++i) {
        int mutexInitRes = pthread_mutex_init(&mutexList[i], NULL);
        if (mutexInitRes != SUCCESS_STATUS) {
            fprintf(stderr, "Error on initializing mutex\n");
            MPI_Finalize();
            return ERROR_STATUS;
        }
    }

    pthread_t threads[2];
    pthread_create(&threads[0], &attrs, sendingTask, NULL);
    pthread_create(&threads[1], &attrs, workingTask, NULL);
    pthread_join(threads[0], NULL);
    pthread_join(threads[1], NULL);

    for (int i = 0; i < MUTEX_TOTAL; ++i) {
        pthread_mutex_destroy(&mutexList[i]);
    }
    pthread_attr_destroy(&attrs);
    delete[] taskList;

    MPI_Finalize();
    return SUCCESS_STATUS;
}
