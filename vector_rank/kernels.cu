// Independent kernels for row normalization and tiled cosine dot products.
extern "C" __global__ void normalize_rows(const float* input, float* output,
                                         int rows, int dims) {
    int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ double partial[256];
    double sum = 0.0;
    for (int col = threadIdx.x; col < dims; col += blockDim.x) {
        double value = input[(long long)row * dims + col];
        sum += value * value;
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    double norm = sqrt(partial[0]);
    for (int col = threadIdx.x; col < dims; col += blockDim.x)
        output[(long long)row * dims + col] = norm == 0.0 ? 0.0f :
            (float)(input[(long long)row * dims + col] / norm);
}

extern "C" __global__ void cosine_tiles(const float* queries, const float* documents,
                                       float* scores, int nq, int nd, int dims) {
    __shared__ float qtile[16][16];
    __shared__ float dtile[16][16];
    int qrow = blockIdx.y * 16 + threadIdx.y;
    int drow = blockIdx.x * 16 + threadIdx.x;
    float result = 0.0f;
    for (int base = 0; base < dims; base += 16) {
        int qcol = base + threadIdx.x;
        int dcol = base + threadIdx.y;
        qtile[threadIdx.y][threadIdx.x] = qrow < nq && qcol < dims ?
            queries[(long long)qrow * dims + qcol] : 0.0f;
        dtile[threadIdx.y][threadIdx.x] = drow < nd && dcol < dims ?
            documents[(long long)drow * dims + dcol] : 0.0f;
        __syncthreads();
        for (int k = 0; k < 16; ++k)
            result += qtile[threadIdx.y][k] * dtile[k][threadIdx.x];
        __syncthreads();
    }
    if (qrow < nq && drow < nd) scores[(long long)qrow * nd + drow] = result;
}
