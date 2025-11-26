# ---广告位---
# 感谢插件*驼峰翻译助手*, 灰常好用的变量名起名工具

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from tqdm import tqdm
import multiprocessing
import time
import math

# 尝试导入numba
try:
    from numba import jit

    NUMBA_AVAILABLE = True
except ImportError:
    print("~杂鱼~ Numba库未安装. JIT加速功能将被禁用. ")
    print("可以通过 'pip install numba' 来安装. ")
    NUMBA_AVAILABLE = False

# ==============================================================================
# --- 用户配置区域 (USER CONFIGURATION) ---
# ==============================================================================
# 1. 搜索范围和步长
# 警告: 设置过大的值将导致极长的运行时间. 
#      此代码虽不会因内存崩溃,但计算时间是主要限制. 
SEARCH_MIN = 1.1  # 最小值
SEARCH_MAX = 5000000  # 最大值
NUM_STEPS = SEARCH_MAX*100  # 总步数

# 2. 精度设置
SOLVER_TOLERANCE = 1e-15
VERIFICATION_TOLERANCE = 1e-12

# 3. 核心功能开关
USE_JIT = True  #JIT加速开关
USE_ANALYTIC_DERIVATIVE = True  # 进行非线性优化

# 4. 输出与可视化
SHOW_TRIVIAL_SOLUTIONS = False  # 展示平凡解
SHOW_PLOT = False
PLOT_C_VALUE = 2.718

# 5. 并行处理与进度条优化
NUM_CORES = None  # 调用的处理器核心数, None表示使用所有可用核心
DYNAMIC_CHUNK_ADJUSTMENT = True  # True: 自动调整分块粒度; False: 使用下面的固定因子
CHUNK_FACTOR = 40960  # 分块粒度, 仅在 DYNAMIC_CHUNK_ADJUSTMENT 为 False 时生效


# ==============================================================================
# --- 函数定义区域 ---
# ==============================================================================

def generate_chunks(start_val, stop_val, total_steps, num_chunks):
    """
    一个内存高效(空间复杂度--)的生成器函数
    动态创建任务块,避免内存炸缸
    通过 'yield' 关键字, 逐个“产出”小块的numpy数组,供并行池使用
    """
    if total_steps == 0:
        return

    chunk_size = total_steps // num_chunks
    remainder = total_steps % num_chunks
    current_step_idx = 0

    total_range = stop_val - start_val
    step_increment = total_range / (total_steps - 1) if total_steps > 1 else 0

    for i in range(num_chunks):
        current_chunk_size = chunk_size + 1 if i < remainder else chunk_size
        if current_chunk_size == 0:
            continue

        start_step_idx = current_step_idx
        chunk_start_val = start_val + start_step_idx * step_increment
        chunk_end_val = start_val + (start_step_idx + current_chunk_size - 1) * step_increment

        yield np.linspace(chunk_start_val, chunk_end_val, current_chunk_size)

        current_step_idx += current_chunk_size


def H_log(x, y):
    if x <= 1 or y <= 1: return np.nan
    log_x = math.log(x)
    log_y = math.log(y)
    return (y * log_x - x * log_y - math.log(log_x) + math.log(log_y))


if NUMBA_AVAILABLE:
    @jit("float64(float64, float64)", nopython=True, fastmath=True)  # JIT万岁(doge)
    def H_log_numba(x, y):
        if x <= 1.0 or y <= 1.0: return np.nan
        log_x = math.log(x)
        log_y = math.log(y)
        return y * log_x - x * log_y - math.log(log_x) + math.log(log_y)

H = H_log_numba if USE_JIT and NUMBA_AVAILABLE else H_log


def g(x, c):
    return H(x, c)


def g_prime_log_python(x, c):
    if x <= 1 or c <= 1: return np.nan
    log_x = math.log(x)
    return c / x - math.log(c) - 1 / (x * log_x)


if NUMBA_AVAILABLE:
    @jit("float64(float64, float64)", nopython=True, fastmath=True)
    def g_prime_log_numba(x, c):
        if x <= 1.0 or c <= 1.0: return np.nan
        log_c = math.log(c)
        log_x = math.log(x)
        return c / x - log_c - 1.0 / (x * log_x)

g_prime_selected = None
if USE_ANALYTIC_DERIVATIVE:
    g_prime_selected = g_prime_log_numba if USE_JIT and NUMBA_AVAILABLE else g_prime_log_python


def solve_for_c(c):
    initial_guess = c * 1.01
    try:
        solution_x = newton(g, initial_guess, fprime=g_prime_selected, args=(c,), tol=SOLVER_TOLERANCE, maxiter=100)
        g_value = g(solution_x, c)

        if not np.isfinite(solution_x) or not np.isfinite(g_value):
            status = 'failed_nonfinite'
        elif abs(solution_x - c) < 1e-9:
            status = 'trivial'
        elif abs(g_value) > VERIFICATION_TOLERANCE:
            status = 'artifact'
        else:
            status = 'nontrivial'

        return {'c': c, 'x': solution_x, 'g_value': g_value, 'status': status}
    except (RuntimeError, ValueError):
        return None


def solve_for_chunk(c_chunk):
    results_list = []
    for c_val in c_chunk:
        result = solve_for_c(c_val)
        if result:
            results_list.append(result)
    return results_list


def print_summary(results, duration):
    print("\n\n" + "=" * 20 + " 狩猎结束 " + "=" * 20)
    print(f"总耗时: {duration:.4f} 秒")

    valid_results = [r for r in results if r is not None]

    nontrivial_finds = [r for r in valid_results if r['status'] == 'nontrivial']
    artifacts = [r for r in valid_results if r['status'] == 'artifact']
    trivial_finds = [r for r in valid_results if r['status'] == 'trivial']

    found_anything_interesting = False

    if nontrivial_finds:
        found_anything_interesting = True
        print(f"\n{'=' * 25} !!! 好厉害的说 !!! {'=' * 25}")  # 这行压根没用, 因为无非平凡解(确信)
        for res in nontrivial_finds:
            print(f"对于 y = {res['c']:.8f}, 找到一个经过验证的非平凡解:\n"
                  f"  x ≈ {res['x']:.12f}\n"
                  f"  验证 g(x) = {res['g_value']:.3e}")
        print(f"{'=' * 65}")

    if artifacts:
        found_anything_interesting = True
        print("\n--- ~杂鱼~ 发现的数值伪影 ---")
        for res in artifacts:
            print(f"[y = {res['c']:.4f}]: 伪影 x ≈ {res['x']:.12f} (警告: g(x) = {res['g_value']:.3e})")

    if SHOW_TRIVIAL_SOLUTIONS and trivial_finds:
        found_anything_interesting = True
        print("\n--- 平凡解统计 ---")
        print(f"~杂鱼~ 共发现 {len(trivial_finds)} 个平凡解(已隐藏详情). ")

    if not found_anything_interesting:
        print("\n 太棒了! 在指定的范围和精度下,未发现任何非平凡解或数值伪影. ")


def plot_g_function(c_val):
    # 懒得写可视化, 能注意到这句话是你我的缘分
    pass


# ==============================================================================
# --- 主程序入口 ---
# ==============================================================================
if __name__ == "__main__":
    try:
        assert SEARCH_MIN > 1.0, f"SEARCH_MIN 必须大于 1,当前为 {SEARCH_MIN}"
        assert SEARCH_MAX > SEARCH_MIN, f"SEARCH_MAX 必须大于 SEARCH_MIN"
        assert NUM_STEPS > 1, "NUM_STEPS 必须大于 1"
    except AssertionError as e:
        print(f"达咩, 配置不合法. {e}")
        exit(1)

    if SHOW_PLOT:
        plot_g_function(PLOT_C_VALUE)
        exit(0)

    print("\n" + "=" * 50)
    print("--- 喵! 程序跑起来了!  ---")
    print(f"搜索范围: y ∈ [{SEARCH_MIN}, {SEARCH_MAX}] in {NUM_STEPS} steps")
    print(f"JIT加速: {'启用' if USE_JIT and NUMBA_AVAILABLE else '杂鱼~ 禁用!'}")
    num_processes = multiprocessing.cpu_count() if NUM_CORES is None else NUM_CORES
    print(f"并行核心数: {num_processes}")

    # 动态分块逻辑, 粗糙的手动判断
    if DYNAMIC_CHUNK_ADJUSTMENT:
        if NUM_STEPS < 10000:
            factor = 2
        elif NUM_STEPS < 500000:
            factor = 8
        elif NUM_STEPS < 2000000:
            factor = 16
        elif NUM_STEPS < 20000000:
            factor = 32
        else:
            factor = 512  # 其实你用不到这么多分块
        num_chunks = num_processes * factor
    else:
        num_chunks = num_processes * CHUNK_FACTOR

    num_chunks = min(num_chunks, NUM_STEPS)

    print(f"任务分块数: {num_chunks} (动态调整: {'开' if DYNAMIC_CHUNK_ADJUSTMENT else '关'})")
    print("=" * 50 + "\n")

    start_time = time.time()

    # 1. 创建任务块生成器, 放心不会吃你内存啦
    chunk_generator = generate_chunks(SEARCH_MIN, SEARCH_MAX, NUM_STEPS, num_chunks)

    all_results = []
    with multiprocessing.Pool(processes=num_processes) as pool:
        # 2. 将生成器传递给 imap_unordered, 交由其按需从中提取数据块进行处理
        pbar = tqdm(pool.imap_unordered(solve_for_chunk, chunk_generator),
                    total=num_chunks, desc="Processing chunks")
        for result_chunk in pbar:
            all_results.extend(result_chunk)

    end_time = time.time()
    print_summary(all_results, end_time - start_time)