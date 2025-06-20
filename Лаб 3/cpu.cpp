
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fstream>

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
    

    const int window_sizes[] = { 3, 5, 7};
    //const int window_sizes[] = { 7 };
    const int num_windows = 3;

    std::vector<float> h_output_seq(M * N);
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
    return 0;
}