#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(err) do { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Sequential median filter
void median_filter_seq(const float* input, float* output, int M, int N, int W) {
    int radius = (W - 1) / 2;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::vector<float> window;
            for (int x = std::max(0, i - radius); x <= std::min(M - 1, i + radius); ++x) {
                for (int y = std::max(0, j - radius); y <= std::min(N - 1, j + radius); ++y) {
                    window.push_back(input[x * N + y]);
                }
            }
            std::sort(window.begin(), window.end());
            output[i * N + j] = window[window.size() / 2];
        }
    }
}

// CUDA kernel without Texture Memory
__global__ void median_filter_kernel_global(const float* d_input, float* d_output, int M, int N, int W) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M && j < N) {
        const int MAX_WINDOW_SIZE = 49; // Supports W up to 7
        float window[MAX_WINDOW_SIZE];
        int radius = (W - 1) / 2;
        int row_start = max(0, i - radius);
        int row_end = min(M - 1, i + radius);
        int col_start = max(0, j - radius);
        int col_end = min(N - 1, j + radius);
        int count = 0;
        for (int x = row_start; x <= row_end; ++x) {
            for (int y = col_start; y <= col_end; ++y) {
                window[count++] = d_input[x * N + y];
            }
        }
        // Bubble sort
        for (int k = 0; k < count - 1; ++k) {
            for (int l = 0; l < count - 1 - k; ++l) {
                if (window[l] > window[l + 1]) {
                    float temp = window[l];
                    window[l] = window[l + 1];
                    window[l + 1] = temp;
                }
            }
        }
        d_output[i * N + j] = window[count / 2];
    }
}

// CUDA kernel with Texture Memory
__global__ void median_filter_kernel_texture(cudaTextureObject_t texObj, float* d_output, int M, int N, int W) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M && j < N) {
        const int MAX_WINDOW_SIZE = 49; // Supports W up to 7
        float window[MAX_WINDOW_SIZE];
        int radius = (W - 1) / 2;
        int row_start = max(0, i - radius);
        int row_end = min(M - 1, i + radius);
        int col_start = max(0, j - radius);
        int col_end = min(N - 1, j + radius);
        int count = 0;
        for (int x = row_start; x <= row_end; ++x) {
            for (int y = col_start; y <= col_end; ++y) {
                window[count++] = tex2D<float>(texObj, y, x);
            }
        }
        // Bubble sort
        for (int k = 0; k < count - 1; ++k) {
            for (int l = 0; l < count - 1 - k; ++l) {
                if (window[l] > window[l + 1]) {
                    float temp = window[l];
                    window[l] = window[l + 1];
                    window[l + 1] = temp;
                }
            }
        }
        d_output[i * N + j] = window[count / 2];
    }
}

// Function to generate synthetic seismic data
void generate_synthetic_data(float* data, int M, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            // Add horizontal events every 100 rows, else random noise
            data[i * N + j] = (i % 100 == 0) ? 1.0f : (rand() % 1000) / 1000.0f - 0.5f;
        }
    }
}

// Function to save data to binary file
void save_data(const float* data, int M, int N, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char*>(&M), sizeof(int));
    file.write(reinterpret_cast<const char*>(&N), sizeof(int));
    file.write(reinterpret_cast<const char*>(data), M * N * sizeof(float));
    file.close();
}

int main() {
    // Определяем размеры данных
    const int N = 4301;
    const int M = 600;
    const size_t num_elements = static_cast<size_t>(M) * N;

    // Создаем вектор для хранения данных
    std::vector<float> h_input(num_elements);

    // Открываем бинарный файл
    std::ifstream file("one_SP.bin", std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Ошибка: не удалось открыть файл one_SP.bin" << std::endl;
        return 1;
    }

    // Проверяем размер файла
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    size_t expected_size = num_elements * sizeof(double);
    if (file_size != expected_size) {
        std::cerr << "Ошибка: размер файла " << file_size
            << " не соответствует ожидаемому " << expected_size << std::endl;
        file.close();
        return 1;
    }

    // Создаем буфер для чтения данных типа double
    std::vector<double> buffer(num_elements);

    // Читаем данные из файла в буфер
    file.read(reinterpret_cast<char*>(buffer.data()), expected_size);
    if (!file) {
        std::cerr << "Ошибка: не удалось прочитать данные из файла" << std::endl;
        file.close();
        return 1;
    }

    // Закрываем файл
    file.close();

    // Преобразуем данные из double в float
    std::transform(buffer.begin(), buffer.end(), h_input.begin(),
        [](double d) { return static_cast<float>(d); });
    
//    const int M = 512; // Rows (timestep)
//    const int N = 512; // Columns (offset)
    //const int window_sizes[] = { 3, 5, 7};
    const int window_sizes[] = { 7 };
    const int block_sizes[] = { 16 };
    const int num_windows = 1;
    const int num_blocks = 1;

    // Allocate host memory
   // std::vector<float> h_input(M * N);
    std::vector<float> h_output_seq(M * N);
    std::vector<float> h_output_gpu(M * N);
    std::vector<float> h_output_tex(M * N);


    save_data(h_input.data(), M, N, "input_seismic.dat");

    // Task 1: Sequential Implementation
    std::cout << "Sequential Execution Times (seconds):\n";
    for (int w = 0; w < num_windows; ++w) {
        int W = window_sizes[w];
        auto start = std::chrono::high_resolution_clock::now();
        median_filter_seq(h_input.data(), h_output_seq.data(), M, N, W);
        auto end = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration<double>(end - start).count();
        std::cout << "W=" << W << ": " << time << "\n";
        if (W == 3) save_data(h_output_seq.data(), M, N, "output_seq_w3.dat");
    }
    std::cout << "\n";

    // Allocate device memory for GPU versions
    float* d_input, * d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, M * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, M * N * sizeof(float)));

    // Task 2: GPU Implementation without Texture Memory
    std::cout << "GPU Execution Times without Texture Memory (seconds):\n";
    for (int w = 0; w < num_windows; ++w) {
        int W = window_sizes[w];
        std::cout << "W=" << W << ":\n";
        for (int b = 0; b < num_blocks; ++b) {
            int block_size = block_sizes[b];
            dim3 blockDim(block_size, block_size);
            dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

            cudaEvent_t start, stop;
            CHECK_CUDA_ERROR(cudaEventCreate(&start));
            CHECK_CUDA_ERROR(cudaEventCreate(&stop));
            CHECK_CUDA_ERROR(cudaEventRecord(start));

            CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
            median_filter_kernel_global << <gridDim, blockDim >> > (d_input, d_output, M, N, W);
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu.data(), d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost));

            CHECK_CUDA_ERROR(cudaEventRecord(stop));
            CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
            float time_ms;
            CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_ms, start, stop));
            std::cout << "  Block=" << block_size << "x" << block_size << ": " << time_ms / 1000.0 << "\n";

            CHECK_CUDA_ERROR(cudaEventDestroy(start));
            CHECK_CUDA_ERROR(cudaEventDestroy(stop));

            if (W == 7 && block_size == 16) save_data(h_output_gpu.data(), M, N, "output_gpu_w7_b16.dat");
        }
    }
    std::cout << "\n";

    // Task 3: GPU Implementation with Texture Memory
    cudaArray* cuArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    CHECK_CUDA_ERROR(cudaMallocArray(&cuArray, &channelDesc, N, M));
    CHECK_CUDA_ERROR(cudaMemcpy2DToArray(cuArray, 0, 0, h_input.data(), N * sizeof(float), N * sizeof(float), M, cudaMemcpyHostToDevice));

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t texObj;
    CHECK_CUDA_ERROR(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    std::cout << "GPU Execution Times with Texture Memory (seconds):\n";
    for (int w = 0; w < num_windows; ++w) {
        int W = window_sizes[w];
        std::cout << "W=" << W << ":\n";
        for (int b = 0; b < num_blocks; ++b) {
            int block_size = block_sizes[b];
            dim3 blockDim(block_size, block_size);
            dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

            cudaEvent_t start, stop;
            CHECK_CUDA_ERROR(cudaEventCreate(&start));
            CHECK_CUDA_ERROR(cudaEventCreate(&stop));
            CHECK_CUDA_ERROR(cudaEventRecord(start));

            median_filter_kernel_texture << <gridDim, blockDim >> > (texObj, d_output, M, N, W);
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaMemcpy(h_output_tex.data(), d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost));

            CHECK_CUDA_ERROR(cudaEventRecord(stop));
            CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
            float time_ms;
            CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_ms, start, stop));
            std::cout << "  Block=" << block_size << "x" << block_size << ": " << time_ms / 1000.0 << "\n";

            CHECK_CUDA_ERROR(cudaEventDestroy(start));
            CHECK_CUDA_ERROR(cudaEventDestroy(stop));

            if (W == 3 && block_size == 16) save_data(h_output_tex.data(), M, N, "output_tex_w3_b16.dat");
        }
    }
    std::cout << "\n";

    // Task 4: Performance Analysis for W=5, Block=16
    int W = 5;
    int block_size = 16;
    dim3 blockDim(block_size, block_size);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));

    median_filter_kernel_texture << <gridDim, blockDim >> > (texObj, d_output, M, N, W);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_tex.data(), d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float time_ms;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_ms, start, stop));
    double time_s = time_ms / 1000.0;

    // Effective Memory Bandwidth
    size_t reads = M * N * W * W * sizeof(float); // Each thread reads W*W floats
    size_t writes = M * N * sizeof(float);        // Each thread writes 1 float
    size_t total_bytes = reads + writes;
    double bandwidth = total_bytes / time_s / 1e9; // GB/s
    std::cout << "Effective Memory Bandwidth (W=5, Block=16): " << bandwidth << " GB/s\n";
    // Note: Compare with GPU's theoretical bandwidth (e.g., for NVIDIA GTX 1080, ~320 GB/s)

    // Effective Computational Throughput (approximation based on bubble sort operations)
    size_t comparisons = M * N * (W * W) * (W * W - 1) / 2; // Approx. bubble sort comparisons
    double throughput = comparisons / time_s / 1e9; // GFLOPS
    std::cout << "Effective Computational Throughput (W=5, Block=16): " << throughput << " GFLOPS\n";
    // Note: Compare with GPU's theoretical FLOPS (e.g., for GTX 1080, ~9 TFLOPS)

    // Cleanup
    CHECK_CUDA_ERROR(cudaDestroyTextureObject(texObj));
    CHECK_CUDA_ERROR(cudaFreeArray(cuArray));
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));

    //// для проверки можно вывести первые несколько элементов (опционально)
    //for (int i = 0; i < 5 && i < num_elements; ++i) {
    //    std::cout << "h_input[" << i << "] = " << h_input[i] << std::endl;
    //}
    //// для проверки можно вывести первые несколько элементов (опционально)
    //for (int i = 0; i < 5 && i < num_elements; ++i) {
    //    std::cout << "h_output_seq[" << i << "] = " << h_output_seq[i] << std::endl;
    //}
    //// для проверки можно вывести первые несколько элементов (опционально)
    //for (int i = 0; i < 5 && i < num_elements; ++i) {
    //    std::cout << "h_output_gpu[" << i << "] = " << h_output_gpu[i] << std::endl;
    //}
    //// для проверки можно вывести первые несколько элементов (опционально)
    //for (int i = 0; i < 5 && i < num_elements; ++i) {
    //    std::cout << "h_output_tex[" << i << "] = " << h_output_tex[i] << std::endl;
    //}
    return 0;
}