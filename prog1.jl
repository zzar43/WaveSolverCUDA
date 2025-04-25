using CUDA

a = CUDA.rand(1024,1024,128);

sin.(a);

CUDA.@profile trace=true sin.(a)