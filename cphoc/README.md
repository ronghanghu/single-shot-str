A C program to build PHOC descriptors of the query strings. The cphoc library must be compiled as follows:

```
gcc -c -fPIC `python3-config --cflags` cphoc_python3.c

gcc -shared -o cphoc.so cphoc_python3.o `python3-config --ldflags`
```
