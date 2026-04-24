from numba import cuda
import numpy as np
from timeit import default_timer as timer

# gpu kernel function
@cuda.jit
def increment_by_one_gpu(an_array):
    #get the absolute position of the current thread in out 1 dimentional grid
    pos = cuda.grid(1) 

    #increment the entry in the array based on its thread position
    if pos < an_array.size:
        an_array[pos] += 1


# cpu function
def increment_by_one_nogpu(an_array):
    # increment each position using standard iterative approach
    pos = 0
    while pos < an_array.size:
        an_array[pos] += 1
        pos += 1

if __name__ == "__main__":

    # create numpy array of 10 million 1s
    n = 10_000_000
    arr = np.ones(n)

    # copy the array to gpu memory
    d_arr = cuda.to_device(arr)

    # print inital array values
    print("GPU Array: ", arr)
    print("NON-GPU Array: ", arr)

    #specify threads
    threadsperblock = 32
    blockspergrid = (len(arr) + (threadsperblock - 1)) // threadsperblock

    # start timer
    start = timer()
    # run gpu kernel
    increment_by_one_gpu[blockspergrid, threadsperblock](d_arr)
    # get time elapsed for gpu
    dt = timer() - start

    print("Time With GPU: ", dt)
    
    # restart timer
    start = timer()
    # run cpu function
    increment_by_one_nogpu(arr)
    # get time elapsed for cpu
    dt = timer() - start

    print("Time Without GPU: ", dt)

    # create empty array
    gpu_arr = np.empty(shape=d_arr.shape, dtype=d_arr.dtype)

    # move data back to host memory
    d_arr.copy_to_host(gpu_arr)

    print("GPU Array: ", gpu_arr)
    print("NON-GPU Array: ", arr)