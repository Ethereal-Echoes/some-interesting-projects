#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <atomic>
#include <string>
#include <sstream>

// Windows API
#include <windows.h>
#include <psapi.h>
#include <powerbase.h>
#include <powersetting.h>
#include <pdh.h> // 性能计数器核心库
#include <pdhmsg.h>
#include <immintrin.h> // AVX-512

// 链接库提示：必须链接 Pdh, PowrProf, Psapi
// g++ ... -lPdh -lPowrProf -lPsapi

// =================================================================
// 0. 全局配置
// =================================================================
double g_target_gb = 2.0;
size_t g_data_size = 0;
const int TEST_ROUNDS = 200;
const int WARMUP_SECONDS = 30;
const int THREAD_COUNT = 4;

// ANSI 颜色
#define C_RESET "\033[0m"
#define C_RED "\033[1;31m"
#define C_GREEN "\033[1;32m"
#define C_YELLOW "\033[1;33m"
#define C_BLUE "\033[1;34m"
#define C_CYAN "\033[1;36m"
#define C_WHITE "\033[1;37m"
#define C_GRAY "\033[1;90m"

std::atomic<double> g_current_speed(0.0);
std::atomic<double> g_progress_pct(0.0);
std::atomic<bool> g_running(true);
std::string g_stage_name = "初始化";

// =================================================================
// 1. 高级系统监控 (PDH 实现)
// =================================================================
class HardwareMonitor
{
private:
    PDH_HQUERY cpuQuery;
    PDH_HCOUNTER cpuTotalCounter;
    PDH_HCOUNTER cpuPerfCounter; // 用于计算睿频
    double baseFrequency = 0.0;
    int numProcessors = 0;

public:
    HardwareMonitor()
    {
        // 1. 获取 CPU 核心数
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        numProcessors = sysInfo.dwNumberOfProcessors;

        // 2. 获取基准频率 (Base Frequency)
        struct PROCESSOR_POWER_INFORMATION
        {
            ULONG Number;
            ULONG MaxMhz;
            ULONG CurrentMhz;
            ULONG MhzLimit;
            ULONG MaxIdleState;
            ULONG CurrentIdleState;
        };
        std::vector<PROCESSOR_POWER_INFORMATION> ppi(numProcessors);
        CallNtPowerInformation(ProcessorInformation, NULL, 0, &ppi[0], sizeof(PROCESSOR_POWER_INFORMATION) * numProcessors);
        // 取第一个核心的最大频率作为基准
        baseFrequency = (double)ppi[0].MaxMhz;

        // 3. 初始化 PDH 查询
        PdhOpenQuery(NULL, 0, &cpuQuery);

        // 添加计数器：CPU 总使用率
        // 注意：在中文系统上，可能需要使用英文计数器名称 (PdhAddEnglishCounter) 以保证兼容性
        PdhAddEnglishCounter(cpuQuery, "\\Processor(_Total)\\% Processor Time", 0, &cpuTotalCounter);

        // 添加计数器：处理器性能百分比 (用于计算睿频)
        // Real Frequency = Base Frequency * (% Processor Performance)
        PdhAddEnglishCounter(cpuQuery, "\\Processor Information(_Total)\\% Processor Performance", 0, &cpuPerfCounter);

        // 第一次收集，初始化数据
        PdhCollectQueryData(cpuQuery);
    }

    ~HardwareMonitor()
    {
        PdhCloseQuery(cpuQuery);
    }

    // 返回：当前频率 (GHz), CPU占用率 (%)
    std::pair<double, double> getCpuStats()
    {
        PdhCollectQueryData(cpuQuery);

        PDH_FMT_COUNTERVALUE counterVal;
        double cpuUsage = 0.0;
        double perfPercent = 0.0;

        // 读取 CPU 使用率
        if (PdhGetFormattedCounterValue(cpuTotalCounter, PDH_FMT_DOUBLE, NULL, &counterVal) == ERROR_SUCCESS)
        {
            cpuUsage = counterVal.doubleValue;
        }

        // 读取 性能百分比
        if (PdhGetFormattedCounterValue(cpuPerfCounter, PDH_FMT_DOUBLE, NULL, &counterVal) == ERROR_SUCCESS)
        {
            perfPercent = counterVal.doubleValue;
        }

        // 计算真实频率：基准 * (性能比 / 100)
        // 例如：2.8GHz * (150 / 100) = 4.2GHz
        double realFreq = (baseFrequency * perfPercent) / 100.0 / 1000.0; // 转换为 GHz

        // 修正：如果读不到性能比，回退到基准
        if (realFreq < 0.1)
            realFreq = baseFrequency / 1000.0;

        return {realFreq, cpuUsage};
    }

    // 返回：系统已用内存 (GB), 系统总内存 (GB)
    std::pair<double, double> getSystemMemory()
    {
        MEMORYSTATUSEX memInfo;
        memInfo.dwLength = sizeof(MEMORYSTATUSEX);
        GlobalMemoryStatusEx(&memInfo);

        double total = (double)memInfo.ullTotalPhys / (1024.0 * 1024.0 * 1024.0);
        double avail = (double)memInfo.ullAvailPhys / (1024.0 * 1024.0 * 1024.0);
        double used = total - avail;

        return {used, total};
    }
};

// =================================================================
// 2. UI 线程
// =================================================================
void ui_thread_func()
{
    HardwareMonitor monitor;

    while (g_running)
    {
        // 获取硬件数据
        auto cpuStats = monitor.getCpuStats();
        auto memStats = monitor.getSystemMemory();

        double cpu_freq_ghz = cpuStats.first;
        double cpu_usage = cpuStats.second;
        double sys_mem_used = memStats.first;
        double sys_mem_total = memStats.second;

        // 进度与速度
        double progress = g_progress_pct.load();
        double speed = g_current_speed.load();

        // 绘制界面
        std::cout << "\r                                                                                                    \r";

        // 阶段
        std::cout << C_CYAN << "[" << std::left << std::setw(8) << g_stage_name << "] " << C_RESET;

        // 进度条
        std::cout << C_WHITE << "[";
        int bar_width = 20;
        int pos = (int)(bar_width * progress);
        for (int i = 0; i < bar_width; ++i)
        {
            if (i < pos)
                std::cout << C_GREEN << "=";
            else if (i == pos)
                std::cout << C_GREEN << ">";
            else
                std::cout << C_GRAY << "-";
        }
        std::cout << C_WHITE << "] " << std::fixed << std::setprecision(1) << (progress * 100.0) << "% " << C_RESET;

        // 速度
        if (speed > 0)
            std::cout << C_YELLOW << std::setw(7) << (int)speed << " MB/s " << C_RESET;
        else
            std::cout << "            ";

        std::cout << C_GRAY << "| " << C_RESET;

        // CPU 信息 (频率 + 占用)
        std::cout << "CPU: ";
        if (cpu_usage > 95.0)
            std::cout << C_RED;
        else
            std::cout << C_GREEN;
        std::cout << std::fixed << std::setprecision(2) << cpu_freq_ghz << "GHz "
                  << std::setw(3) << (int)cpu_usage << "% " << C_RESET;

        // 内存信息 (本程序占用 / 系统总占用)
        // 注意：这里直接显示 g_target_gb，因为这是我们真实锁定的物理内存
        std::cout << "| RAM: " << C_BLUE << std::fixed << std::setprecision(2) << g_target_gb << "G" << C_RESET
                  << " (Sys: " << std::setprecision(1) << sys_mem_used << "/" << sys_mem_total << "G) ";

        std::cout << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(500)); // PDH 需要一点间隔来计算差值
    }
}

// =================================================================
// 3. 核心算法 (AVX-512 VAES)
// =================================================================
void avx512_vaes_rng(unsigned char *buffer, size_t size)
{
    unsigned long long s1, s2;
    _rdrand64_step(&s1);
    _rdrand64_step(&s2);
    __m512i key = _mm512_set1_epi64(s1);
    __m512i ctr = _mm512_set_epi64(0, 7, 0, 6, 0, 5, 0, 4);
    __m512i step = _mm512_set_epi64(0, 4, 0, 4, 0, 4, 0, 4);

    size_t chunks = size / 64;
    __m512i *ptr = (__m512i *)buffer;
    size_t loops = chunks / 4;

    for (size_t i = 0; i < loops; ++i)
    {
        __m512i c1 = ctr;
        __m512i c2 = _mm512_add_epi64(c1, step);
        __m512i c3 = _mm512_add_epi64(c2, step);
        __m512i c4 = _mm512_add_epi64(c3, step);
        ctr = _mm512_add_epi64(c4, step);

        __m512i r1 = _mm512_aesenc_epi128(c1, key);
        r1 = _mm512_aesenc_epi128(r1, key);
        __m512i r2 = _mm512_aesenc_epi128(c2, key);
        r2 = _mm512_aesenc_epi128(r2, key);
        __m512i r3 = _mm512_aesenc_epi128(c3, key);
        r3 = _mm512_aesenc_epi128(r3, key);
        __m512i r4 = _mm512_aesenc_epi128(c4, key);
        r4 = _mm512_aesenc_epi128(r4, key);

        _mm512_stream_si512(ptr + 0, r1);
        _mm512_stream_si512(ptr + 1, r2);
        _mm512_stream_si512(ptr + 2, r3);
        _mm512_stream_si512(ptr + 3, r4);
        ptr += 4;
    }
    _mm_sfence();
}

void worker_thread(unsigned char *buffer, size_t size)
{
    avx512_vaes_rng(buffer, size);
}

// =================================================================
// 4. 内存管理 (瀑布流)
// =================================================================
bool enable_large_pages()
{
    HANDLE hToken;
    TOKEN_PRIVILEGES tp;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken))
        return false;
    if (!LookupPrivilegeValue(NULL, SE_LOCK_MEMORY_NAME, &tp.Privileges[0].Luid))
        return false;
    tp.PrivilegeCount = 1;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    return AdjustTokenPrivileges(hToken, FALSE, &tp, 0, (PTOKEN_PRIVILEGES)NULL, 0);
}

unsigned char *try_alloc_large(double size_gb)
{
    size_t bytes = (size_t)(size_gb * 1024 * 1024 * 1024);
    std::cout << C_CYAN << "  -> 尝试 " << size_gb << " GB... " << C_RESET;
    void *ptr = VirtualAlloc(NULL, bytes, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
    if (ptr)
    {
        std::cout << C_GREEN << "成功!" << C_RESET << std::endl;
        g_target_gb = size_gb;
        g_data_size = bytes;
        return (unsigned char *)ptr;
    }
    std::cout << C_RED << "失败." << C_RESET << std::endl;
    return nullptr;
}

unsigned char *allocate_memory_cascade(bool &is_large_page)
{
    is_large_page = false;
    unsigned char *ptr = nullptr;

    if (enable_large_pages())
    {
        std::cout << C_YELLOW << "[内存] 启动大页内存分配 (瀑布流策略)..." << C_RESET << std::endl;
        ptr = try_alloc_large(2.0);
        if (ptr)
        {
            is_large_page = true;
            return ptr;
        }

        std::cout << C_YELLOW << "[内存] 2.0GB 失败，执行碎片整理..." << C_RESET << std::endl;
        size_t temp_size = 1024 * 1024 * 1024;
        void *temp = VirtualAlloc(NULL, temp_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (temp)
        {
            memset(temp, 0xCC, temp_size);
            VirtualFree(temp, 0, MEM_RELEASE);
        }

        ptr = try_alloc_large(2.0);
        if (ptr)
        {
            is_large_page = true;
            return ptr;
        }

        std::vector<double> fallbacks = {1.5, 1.0, 0.5};
        for (double size : fallbacks)
        {
            ptr = try_alloc_large(size);
            if (ptr)
            {
                is_large_page = true;
                return ptr;
            }
        }
    }

    std::cout << C_YELLOW << "[内存] 回退到标准内存 (4KB Pages)." << C_RESET << std::endl;
    g_target_gb = 2.0;
    g_data_size = (size_t)(g_target_gb * 1024 * 1024 * 1024);
    ptr = (unsigned char *)_aligned_malloc(g_data_size, 64);
    return ptr;
}

// =================================================================
// 5. 主程序
// =================================================================
int main()
{
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0;
    GetConsoleMode(hOut, &dwMode);
    dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    SetConsoleMode(hOut, dwMode);
    SetConsoleOutputCP(65001);

    SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);

    std::cout << C_CYAN << "==============================================================" << C_RESET << std::endl;
    std::cout << C_WHITE << "           TITAN RNG BENCHMARK (PDH 监控版)                   " << C_RESET << std::endl;
    std::cout << C_CYAN << "==============================================================" << C_RESET << std::endl;

    bool is_large_page = false;
    unsigned char *buffer = allocate_memory_cascade(is_large_page);
    if (!buffer)
        return -1;

    std::cout << C_GREEN << "[就绪] 锁定内存: " << g_target_gb << " GB" << C_RESET << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::thread ui_thread(ui_thread_func);

    // 预热
    g_stage_name = "系统预热";
    auto warm_start = std::chrono::high_resolution_clock::now();
    size_t chunk_size = g_data_size / THREAD_COUNT;
    while (true)
    {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = now - warm_start;
        if (elapsed.count() >= WARMUP_SECONDS)
            break;
        g_progress_pct = elapsed.count() / WARMUP_SECONDS;
        std::vector<std::thread> threads;
        for (int i = 0; i < THREAD_COUNT; i++)
            threads.emplace_back(worker_thread, buffer + i * chunk_size, chunk_size);
        for (auto &t : threads)
            t.join();
    }
    g_progress_pct = 1.0;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // 测试
    g_stage_name = "压力测试";
    g_progress_pct = 0.0;
    std::vector<double> results;
    results.reserve(TEST_ROUNDS);

    for (int round = 1; round <= TEST_ROUNDS; ++round)
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::thread> threads;
        for (int i = 0; i < THREAD_COUNT; i++)
            threads.emplace_back(worker_thread, buffer + i * chunk_size, chunk_size);
        for (auto &t : threads)
            t.join();
        volatile unsigned char check = buffer[g_data_size - 1];
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        double speed = (g_target_gb * 1024.0) / elapsed.count();
        results.push_back(speed);
        g_current_speed = speed;
        g_progress_pct = (double)round / TEST_ROUNDS;
    }

    g_running = false;
    ui_thread.join();
    std::cout << "\n"
              << std::endl;

    // 报告
    double sum = std::accumulate(results.begin(), results.end(), 0.0);
    double avg = sum / results.size();
    double max_speed = *std::max_element(results.begin(), results.end());
    double min_speed = *std::min_element(results.begin(), results.end());

    std::cout << C_CYAN << "==============================================================" << C_RESET << std::endl;
    std::cout << C_WHITE << "                       最终测试报告                           " << C_RESET << std::endl;
    std::cout << C_CYAN << "==============================================================" << C_RESET << std::endl;
    std::cout << " 内存模式 : " << (is_large_page ? C_GREEN "大页内存" : C_YELLOW "标准内存 (4KB)") << C_RESET << std::endl;
    std::cout << " 总吞吐量 : " << g_target_gb * TEST_ROUNDS << " GB" << std::endl;
    std::cout << C_GRAY << "--------------------------------------------------------------" << C_RESET << std::endl;
    std::cout << " 峰值速度 : " << C_GREEN << std::fixed << std::setprecision(2) << max_speed << " MB/s" << C_RESET << std::endl;
    std::cout << " 最低速度 : " << C_RED << min_speed << " MB/s" << C_RESET << std::endl;
    std::cout << " 平均速度 : " << C_YELLOW << avg << " MB/s" << C_RESET << std::endl;
    std::cout << C_CYAN << "==============================================================" << C_RESET << std::endl;

    if (is_large_page)
        VirtualFree(buffer, 0, MEM_RELEASE);
    else
        _aligned_free(buffer);
    return 0;
}