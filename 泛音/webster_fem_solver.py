import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import interpolate
import sys
import os

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 导入draw_tube.py中的函数
sys.path.append(os.path.dirname(__file__))
try:
    from draw_tube import S, radiation_impedance
except ImportError:
    print("警告：无法导入draw_tube.py中的函数，将使用本地实现")
    
    # 本地实现S函数 - 提供一个简单的默认管状形状
    def default_tube_shape(x):
        """
        默认管状形状函数
        返回一个简单的圆锥形：半径从0.1线性增加到0.2
        """
        # x在[0,1]范围内
        # 半径从0.1线性增加到0.2
        return 0.1 + 0.1 * x
    
    def S(x, tube_shape_func=None):
        """计算喇叭截面积随位置的变化，S = πr²"""
        # 使用默认的管状形状
        r = default_tube_shape(x)
        return np.pi * r * r


class WebsterFEMSolver:
    """Webster方程有限元求解器"""
    
    def __init__(self, n_elements=100, c=1.0, rho0=1.0):
        """
        初始化求解器
        
        参数:
        n_elements -- 有限元单元数量
        c -- 声速 (默认1.0)
        rho0 -- 密度 (默认1.0)
        """
        self.n_elements = n_elements
        self.c = c
        self.rho0 = rho0
        
        # 创建网格
        self.nodes = np.linspace(0, 1, n_elements + 1)
        self.h = 1.0 / n_elements  # 单元长度
        
        # 边界条件类型
        self.left_bc = 'closed'  # 默认左端闭口
        self.right_bc = 'closed'  # 默认右端闭口
        
        # 特征值和特征向量
        self.eigenvalues = None
        self.eigenvectors = None
        self.frequencies = None
        
    def set_boundary_conditions(self, left_bc='closed', right_bc='closed'):
        """
        设置边界条件
        
        参数:
        left_bc -- 左端边界条件: 'closed' (闭口) 或 'open' (开口)
        right_bc -- 右端边界条件: 'closed' (闭口) 或 'open' (开口)
        """
        if left_bc not in ['closed', 'open']:
            raise ValueError("左端边界条件必须是 'closed' 或 'open'")
        if right_bc not in ['closed', 'open']:
            raise ValueError("右端边界条件必须是 'closed' 或 'open'")
            
        self.left_bc = left_bc
        self.right_bc = right_bc
        
    def compute_S_at_points(self, points):
        """计算截面积S在给定点的值"""
        return np.array([S(x) for x in points])
    
    def assemble_matrices(self):
        """组装质量矩阵和刚度矩阵"""
        n_nodes = len(self.nodes)
        
        # 初始化矩阵
        K = sp.lil_matrix((n_nodes, n_nodes))  # 刚度矩阵
        M = sp.lil_matrix((n_nodes, n_nodes))  # 质量矩阵
        
        # 高斯积分点和权重 (2点高斯积分)
        gauss_points = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        gauss_weights = np.array([1.0, 1.0])
        
        for i in range(self.n_elements):
            # 单元节点
            node_left = i
            node_right = i + 1
            
            # 单元坐标
            x_left = self.nodes[node_left]
            x_right = self.nodes[node_right]
            
            # 单元长度
            h = x_right - x_left
            
            # 单元刚度矩阵和质量矩阵
            K_elem = np.zeros((2, 2))
            M_elem = np.zeros((2, 2))
            
            # 高斯积分
            for gp, weight in zip(gauss_points, gauss_weights):
                # 局部坐标到全局坐标的映射
                xi = gp  # 局部坐标 [-1, 1]
                x = 0.5 * (1 - xi) * x_left + 0.5 * (1 + xi) * x_right
                
                # 形函数及其导数
                N1 = 0.5 * (1 - xi)
                N2 = 0.5 * (1 + xi)
                dN1_dxi = -0.5
                dN2_dxi = 0.5
                
                # 雅可比行列式
                J = h / 2.0
                dx_dxi = h / 2.0
                
                # 形函数对x的导数
                dN1_dx = dN1_dxi / dx_dxi
                dN2_dx = dN2_dxi / dx_dxi
                
                # 计算S(x)和S'(x)（使用中心差分近似导数）
                S_val = S(x)
                
                # 刚度矩阵贡献
                K_elem[0, 0] += weight * S_val * dN1_dx * dN1_dx * J
                K_elem[0, 1] += weight * S_val * dN1_dx * dN2_dx * J
                K_elem[1, 0] += weight * S_val * dN2_dx * dN1_dx * J
                K_elem[1, 1] += weight * S_val * dN2_dx * dN2_dx * J
                
                # 质量矩阵贡献
                M_elem[0, 0] += weight * S_val * N1 * N1 * J
                M_elem[0, 1] += weight * S_val * N1 * N2 * J
                M_elem[1, 0] += weight * S_val * N2 * N1 * J
                M_elem[1, 1] += weight * S_val * N2 * N2 * J
            
            # 组装到全局矩阵
            K[node_left, node_left] += K_elem[0, 0]
            K[node_left, node_right] += K_elem[0, 1]
            K[node_right, node_left] += K_elem[1, 0]
            K[node_right, node_right] += K_elem[1, 1]
            
            M[node_left, node_left] += M_elem[0, 0]
            M[node_left, node_right] += M_elem[0, 1]
            M[node_right, node_left] += M_elem[1, 0]
            M[node_right, node_right] += M_elem[1, 1]
        
        # 转换为CSR格式以提高效率
        K = K.tocsr()
        M = M.tocsr()
        
        return K, M
    
    def apply_boundary_conditions(self, K, M):
        """应用边界条件"""
        n_nodes = len(self.nodes)
        
        # 复制矩阵以避免修改原始矩阵
        # 转换为LIL格式以便高效修改稀疏结构
        K_bc = K.tolil()
        M_bc = M.tolil()
        
        # 处理Dirichlet边界条件 (开口: p=0)
        dirichlet_nodes = []
        
        if self.left_bc == 'open':
            dirichlet_nodes.append(0)
        if self.right_bc == 'open':
            dirichlet_nodes.append(n_nodes - 1)
        
        # 应用Dirichlet边界条件
        for node in dirichlet_nodes:
            # 设置刚度矩阵行
            K_bc[node, :] = 0
            K_bc[node, node] = 1.0
            
            # 设置质量矩阵行
            M_bc[node, :] = 0
            M_bc[node, node] = 1.0 if self.left_bc == 'open' or self.right_bc == 'open' else 0
            
            # 设置刚度矩阵列 (除了对角线)
            for i in range(n_nodes):
                if i != node:
                    K_bc[i, node] = 0
                    M_bc[i, node] = 0
        
        # 转换回CSR格式以提高后续计算效率
        return K_bc.tocsr(), M_bc.tocsr()
    
    def solve_eigenproblem(self, num_modes=10):
        """
        求解特征值问题
        
        参数:
        num_modes -- 需要计算的特征模式数量
        
        返回:
        eigenvalues -- 特征值 (k²)
        eigenvectors -- 特征向量 (压力分布)
        """
        # 组装矩阵
        K, M = self.assemble_matrices()
        
        # 应用边界条件
        K_bc, M_bc = self.apply_boundary_conditions(K, M)
        
        # 求解广义特征值问题: K * p = λ * M * p
        # 其中 λ = k²
        
        # 首先尝试使用密集矩阵方法，因为它更稳定
        print("使用密集矩阵方法求解特征值问题...")
        try:
            # 转换为密集矩阵
            K_dense = K_bc.toarray()
            M_dense = M_bc.toarray()
            
            # 添加正则化项避免奇异矩阵
            reg = 1e-12 * np.eye(K_dense.shape[0])
            K_dense_reg = K_dense + reg
            M_dense_reg = M_dense + reg
            
            # 方法1: 使用广义特征值求解 (Kx = λMx)
            try:
                eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(M_dense_reg, K_dense_reg))
            except np.linalg.LinAlgError:
                # 方法2: 如果方法1失败，使用标准特征值求解
                print("广义特征值求解失败，尝试标准特征值求解...")
                eigenvalues, eigenvectors = np.linalg.eig(K_dense_reg)
            
            # 排序并取前num_modes个最小的特征值
            idx = np.argsort(np.abs(eigenvalues))
            eigenvalues = eigenvalues[idx[:num_modes]]
            eigenvectors = eigenvectors[:, idx[:num_modes]]
            
            print("密集矩阵方法求解成功")
            
        except Exception as dense_e:
            print(f"密集矩阵方法失败: {dense_e}")
            print("尝试使用稀疏矩阵方法...")
            
            # 稀疏矩阵方法作为备选
            try:
                # 尝试不同的sigma值
                sigma_values = [1e-6, 0.1, 1.0, 10.0]
                
                for sigma in sigma_values:
                    try:
                        eigenvalues, eigenvectors = spla.eigsh(
                            K_bc, 
                            k=num_modes, 
                            M=M_bc, 
                            which='SM',  # 最小的特征值
                            sigma=sigma,   # 尝试不同的sigma值
                            maxiter=5000,  # 进一步增加最大迭代次数
                            tol=1e-8      # 放宽收敛容差
                        )
                        print(f"稀疏矩阵方法成功 (sigma={sigma})")
                        break
                    except Exception as e:
                        print(f"稀疏矩阵方法失败 (sigma={sigma}): {e}")
                        continue
                else:
                    # 所有sigma值都失败，抛出异常
                    raise RuntimeError("所有稀疏矩阵方法尝试都失败")
                    
            except Exception as sparse_e:
                print(f"稀疏矩阵方法也失败: {sparse_e}")
                print("使用简单特征值求解作为最后手段...")
                
                # 最后手段：只求解K矩阵的特征值
                eigenvalues, eigenvectors = np.linalg.eig(K_bc.toarray())
                idx = np.argsort(np.abs(eigenvalues))
                eigenvalues = eigenvalues[idx[:num_modes]]
                eigenvectors = eigenvectors[:, idx[:num_modes]]
        
        # 确保特征值为正
        eigenvalues = np.abs(eigenvalues)
        
        # 排序特征值和特征向量
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 验证模式：检查二阶导数是否为零（无效模式）
        print("验证模式有效性...")
        valid_indices = []
        valid_eigenvalues = []
        valid_eigenvectors = []
        
        for i in range(len(eigenvalues)):
            mode_shape = eigenvectors[:, i]
            
            # 检查模式是否有效（二阶导数不全为零）
            if self._is_valid_mode(mode_shape):
                valid_indices.append(i)
                valid_eigenvalues.append(eigenvalues[i])
                valid_eigenvectors.append(mode_shape)
                print(f"  模式 {i+1}: 有效")
            else:
                print(f"  模式 {i+1}: 无效（二阶导数为零）")
        
        # 如果有效模式数量不足，尝试计算更多模式
        if len(valid_eigenvalues) < num_modes:
            print(f"有效模式不足 ({len(valid_eigenvalues)}/{num_modes})，尝试计算更多模式...")
            additional_needed = num_modes - len(valid_eigenvalues)
            
            # 这里可以扩展计算更多模式，但为了简化，我们暂时只使用有效的模式
            print(f"警告: 只找到 {len(valid_eigenvalues)} 个有效模式，但需要 {num_modes} 个")
        
        # 存储结果（只存储有效模式）
        self.eigenvalues = np.array(valid_eigenvalues)
        self.eigenvectors = np.column_stack(valid_eigenvectors) if valid_eigenvectors else np.array([])
        self.valid_mode_indices = valid_indices  # 存储有效模式的原始索引
        
        # 计算频率: ω = c * k, f = ω / (2π) = c * sqrt(λ) / (2π)
        self.frequencies = self.c * np.sqrt(self.eigenvalues) / (2 * np.pi)
        
        return self.eigenvalues, self.eigenvectors
    
    def _is_valid_mode(self, mode_shape, num_points=4, tolerance=1e-6):
        """
        验证模式是否有效：检查随机点的二阶导数是否为零
        
        参数:
        mode_shape -- 模式形状向量
        num_points -- 随机检查的点数
        tolerance -- 二阶导数为零的容差
        
        返回:
        True如果模式有效（二阶导数不全为零），否则False
        """
        n = len(mode_shape)
        if n < 3:
            return True  # 点数太少，无法计算二阶导数
        
        # 在[0,1]区间随机选择num_points个点（排除边界）
        # 使用当前时间作为随机种子，确保每次运行结果不同
        import time
        np.random.seed(int(time.time() * 1000) % 2**32)
        random_indices = np.random.choice(np.arange(1, n-1), size=min(num_points, n-2), replace=False)
        
        # 检查每个随机点的二阶导数
        zero_derivative_count = 0
        
        for idx in random_indices:
            # 使用中心差分计算二阶导数
            if idx > 0 and idx < n-1:
                h = self.nodes[idx] - self.nodes[idx-1] if idx > 0 else self.nodes[idx+1] - self.nodes[idx]
                if h > 0:
                    # 二阶导数: d²p/dx² ≈ (p(x+h) - 2p(x) + p(x-h)) / h²
                    second_deriv = (mode_shape[idx+1] - 2*mode_shape[idx] + mode_shape[idx-1]) / (h**2)
                    
                    # 检查二阶导数是否接近零
                    if abs(second_deriv) < tolerance:
                        zero_derivative_count += 1
        
        # 如果所有检查点的二阶导数都为零，则认为模式无效
        return zero_derivative_count < len(random_indices)
    
    def get_frequencies(self):
        """获取本征频率"""
        if self.frequencies is None:
            raise ValueError("请先调用 solve_eigenproblem() 方法")
        return self.frequencies
    
    def get_mode_shape(self, mode_index):
        """获取指定模式的波形"""
        if self.eigenvectors is None:
            raise ValueError("请先调用 solve_eigenproblem() 方法")
        
        if mode_index >= len(self.eigenvalues):
            raise ValueError(f"模式索引 {mode_index} 超出范围 (最大 {len(self.eigenvalues)-1})")
        
        return self.eigenvectors[:, mode_index]
    
    def plot_results(self, num_freq_plot=10, num_mode_plot=5):
        """
        绘制结果
        
        参数:
        num_freq_plot -- 在数轴上显示的本征频率数量
        num_mode_plot -- 绘制的波形数量
        """
        if self.frequencies is None or self.eigenvectors is None:
            raise ValueError("请先调用 solve_eigenproblem() 方法")
        
        # 获取有效模式的数量
        num_valid_modes = len(self.eigenvalues)
        if num_valid_modes == 0:
            print("警告: 没有有效的模式可绘制")
            return
        
        # 调整要绘制的模式数量，不超过有效模式数量
        actual_num_mode_plot = min(num_mode_plot, num_valid_modes)
        actual_num_freq_plot = min(num_freq_plot, num_valid_modes)
        
        if actual_num_mode_plot < num_mode_plot:
            print(f"注意: 只找到 {num_valid_modes} 个有效模式，将绘制 {actual_num_mode_plot} 个模式")
        
        # 计算第一个有效频率f0（用于相对频率计算）
        if num_valid_modes > 0:
            f0 = self.frequencies[0]
            print(f"第一个有效频率 f₀ = {f0:.6f}")
        else:
            f0 = 1.0  # 默认值，避免除零错误
        
        # 创建图形，第一个图高度较小，其他图高度正常
        # 使用gridspec控制子图高度比例
        from matplotlib import gridspec
        
        # 总高度：第一个图占1份，其他每个图占3份
        height_ratios = [1] + [3] * actual_num_mode_plot
        fig = plt.figure(figsize=(10, 2 + 3.5 * actual_num_mode_plot))  # 增加总高度
        
        # 增加子图之间的垂直间距，避免重叠
        gs = gridspec.GridSpec(actual_num_mode_plot + 1, 1, height_ratios=height_ratios)
        
        # 1. 绘制频率数轴（简化版）
        ax_freq = plt.subplot(gs[0])
        
        # 绘制数轴横线
        ax_freq.axhline(y=0, color='black', linewidth=1)
        
        # 在数轴上用短竖线标记频率位置（只标记有效频率）
        for i, freq in enumerate(self.frequencies[:actual_num_freq_plot]):
            ax_freq.plot([freq, freq], [0, 0.3], 'b-', linewidth=1.5)
        
        # 设置数轴属性（无量纲化，去掉Hz单位）
        ax_freq.set_xlabel('频率', fontsize=10)  # 去掉Hz单位
        ax_freq.set_yticks([])
        ax_freq.set_ylim(-0.1, 0.5)
        
        # 强制第一个数轴图最左边必须是原点
        if len(self.frequencies[:actual_num_freq_plot]) > 0:
            # 计算x轴范围：从0到最大频率的1.1倍，确保所有标记可见
            max_freq = max(self.frequencies[:actual_num_freq_plot])
            ax_freq.set_xlim(left=0, right=max(0.1, max_freq * 1.1))
        else:
            # 如果没有有效频率，设置默认范围
            ax_freq.set_xlim(0, 1.0)
        
        # 去掉边框，只保留底部边框
        ax_freq.spines['top'].set_visible(False)
        ax_freq.spines['right'].set_visible(False)
        ax_freq.spines['left'].set_visible(False)
        
        # 在上半边添加一个轴，显示f/f_0的值
        # 创建顶部轴（共享x轴）
        ax_top = ax_freq.twiny()
        
        # 计算相对频率 f/f_0
        if num_valid_modes > 0 and f0 > 0:
            # 设置顶部轴的范围与底部轴对应
            xlim_bottom = ax_freq.get_xlim()
            # 底部轴的频率范围对应的相对频率范围
            xlim_top = (xlim_bottom[0] / f0, xlim_bottom[1] / f0)
            ax_top.set_xlim(xlim_top)
            
            # 设置顶部轴的标签
            ax_top.set_xlabel('f_rel', fontsize=10)
            
            # 设置顶部轴的位置（上半部分）
            ax_top.spines['top'].set_position(('axes', 0.5))  # 将顶部轴移动到y=0.5位置（上半部分）
            ax_top.spines['bottom'].set_visible(False)
            ax_top.spines['right'].set_visible(False)
            ax_top.spines['left'].set_visible(False)
            ax_top.set_yticks([])
            ax_top.set_ylim(-0.1, 0.9)
            
            # 设置顶部轴刻度：以1为间隔画刻度线，以5为间隔标数字
            # 计算合适的刻度范围
            import math
            x_min, x_max = xlim_top
            # 确保x_min和x_max是合理的
            if x_max > x_min:
                # 计算主要刻度位置（每5个单位）
                major_ticks = np.arange(math.floor(x_min/5)*5, math.ceil(x_max/5)*5 + 1, 5)
                major_ticks = major_ticks[(major_ticks >= x_min) & (major_ticks <= x_max)]
                
                # 计算次要刻度位置（每1个单位）
                minor_ticks = np.arange(math.floor(x_min), math.ceil(x_max) + 1, 1)
                minor_ticks = minor_ticks[(minor_ticks >= x_min) & (minor_ticks <= x_max)]
                
                # 设置主要刻度和次要刻度
                ax_top.set_xticks(major_ticks)
                ax_top.set_xticks(minor_ticks, minor=True)
                
                # 设置主要刻度标签
                ax_top.set_xticklabels([f'{tick:.0f}' if tick % 5 == 0 else '' for tick in major_ticks])
            
            # 设置顶部轴刻度颜色为灰色
            ax_top.tick_params(axis='x', colors='gray', labelsize=9)
            ax_top.tick_params(axis='x', which='minor', length=3, color='gray')
            ax_top.tick_params(axis='x', which='major', length=6, color='gray')
            ax_top.xaxis.label.set_color('gray')
        else:
            # 如果没有有效频率，隐藏顶部轴
            ax_top.set_visible(False)
        
        # 2. 绘制波形（只绘制有效模式）
        for i in range(actual_num_mode_plot):
            ax_mode = plt.subplot(gs[i + 1])
            
            mode_shape = self.get_mode_shape(i)
            freq = self.frequencies[i]
            
            # 计算相对频率 f/f₀
            rel_freq = freq / f0 if f0 > 0 else 0
            
            # 归一化波形
            mode_shape_normalized = mode_shape / np.max(np.abs(mode_shape))
            
            # 修改标题：显示频率和相对频率，去掉Hz单位
            ax_mode.plot(self.nodes, mode_shape_normalized, 'r-', linewidth=2)
            ax_mode.set_title(f'模式 {i+1}: f = {freq:.3f}, f_rel = {rel_freq:.3f}', fontsize=12)
            ax_mode.set_xlabel('位置 x', fontsize=10)
            ax_mode.set_ylabel('压力 p', fontsize=10)  # 去掉"归一化"文字
            ax_mode.grid(True, alpha=0.3)
            ax_mode.set_xlim(0, 1)
            ax_mode.set_ylim(-1.2, 1.2)
        
        # 手动调整子图间距，避免使用tight_layout
        # 增加子图之间的垂直间距，防止重叠
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.05, top=0.95, hspace=1.0)
        plt.show()


def main():
    """主函数：用户界面和求解"""
    print("=" * 60)
    print("Webster方程有限元求解器")
    print("求解方程: d/dx [S(x) dp/dx] + k² S(x) p = 0")
    print("=" * 60)
    
    # 获取用户输入
    print("\n请选择边界条件类型:")
    print("1. 左端闭口, 右端闭口")
    print("2. 左端开口, 右端闭口")
    print("3. 左端闭口, 右端开口")
    print("4. 左端开口, 右端开口")
    
    try:
        choice = int(input("请选择 (1-4): "))
    except ValueError:
        print("输入无效，使用默认设置: 左端闭口, 右端闭口")
        choice = 1
    
    # 根据选择设置边界条件
    if choice == 1:
        left_bc, right_bc = 'closed', 'closed'
    elif choice == 2:
        left_bc, right_bc = 'open', 'closed'
    elif choice == 3:
        left_bc, right_bc = 'closed', 'open'
    elif choice == 4:
        left_bc, right_bc = 'open', 'open'
    else:
        print("选择无效，使用默认设置: 左端闭口, 右端闭口")
        left_bc, right_bc = 'closed', 'closed'
    
    print(f"\n边界条件: 左端={left_bc}, 右端={right_bc}")
    
    # 创建求解器
    print("\n创建有限元求解器...")
    solver = WebsterFEMSolver(n_elements=512, c=1.0, rho0=1.0)
    solver.set_boundary_conditions(left_bc, right_bc)
    
    # 求解特征值问题
    print("求解特征值问题...")
    eigenvalues, eigenvectors = solver.solve_eigenproblem(num_modes=10)
    
    # 显示结果
    frequencies = solver.get_frequencies()
    
    print(f"\n前10个本征频率 (Hz):")
    for i, freq in enumerate(frequencies):
        print(f"  模式 {i+1}: {freq:.6f} Hz")
    
    # 绘制结果
    print("\n绘制结果...")
    solver.plot_results(num_freq_plot=10, num_mode_plot=5)
    
    print("\n求解完成!")


if __name__ == "__main__":
    main()
