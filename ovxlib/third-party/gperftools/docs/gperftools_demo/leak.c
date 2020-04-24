#include "stdlib.h"
#include "heap-profiler.h"

int* func1()
{
    int a = 0;
    int *p = (int*)malloc(sizeof(int)*20);
    return p;
}

int* func2()
{
    int a = 0;
    int *p = (int*)malloc(sizeof(int)*30);
    return p;
}

int main()
{
    int *data1, *data2, *data3, *data4;

    HeapProfilerStart("check_fragment1");
    data1 =(int*) malloc(sizeof(int)*5);
    free(data1);
    data2 =(int*) malloc(sizeof(int)*10);
    HeapProfilerDump("check_fragment1");
    HeapProfilerStop();

    HeapProfilerStart("check_fragment2");
    data3 = func1();
    free(data3);
    data4 = func2();
    HeapProfilerDump("check_fragment2");
    HeapProfilerStop();

    return 0;
}
