import os
import sys
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
import scipy.special

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def import_tube_function():
    """
    从相对路径导入tube_function.py中的tube_shape函数
    
    返回:
    tube_shape函数对象
    """
    # 相对路径：./temp/tube_function.py
    tube_function_path = os.path.join(os.path.dirname(__file__), "temp", "tube_function.py")
    
    if not os.path.exists(tube_function_path):
        raise FileNotFoundError(f"找不到tube_function.py文件: {tube_function_path}")
    
    # 尝试不同的编码方式读取文件
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
    
    for encoding in encodings:
        try:
            # 读取文件内容
            with open(tube_function_path, 'r', encoding=encoding) as f:
                file_content = f.read()
            
            # 创建一个命名空间来执行代码
            namespace = {}
            
            # 执行代码
            exec(file_content, namespace)
            
            # 检查是否有tube_shape函数
            if 'tube_shape' in namespace:
                return namespace['tube_shape']
            else:
                # 尝试从模块属性中获取
                # 查找包含tube_shape的对象
                for key, value in namespace.items():
                    if hasattr(value, '__name__') and value.__name__ == 'tube_shape':
                        return value
                
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    
    # 如果所有编码都失败，尝试使用二进制模式读取
    try:
        with open(tube_function_path, 'rb') as f:
            file_content = f.read().decode('utf-8', errors='ignore')
        
        namespace = {}
        exec(file_content, namespace)
        
        if 'tube_shape' in namespace:
            return namespace['tube_shape']
    except Exception as e:
        raise ImportError(f"导入tube_function.py失败，尝试了多种编码方式: {e}")
    
    raise AttributeError("tube_function.py模块中没有找到tube_shape函数")

def S(x, tube_shape_func=None):
    """
    计算喇叭截面积随位置的变化，S = πr²
    
    参数:
    x -- 归一化坐标，必须在[0,1]范围内
    tube_shape_func -- tube_shape函数对象，如果为None则自动导入
    
    返回:
    S -- 喇叭截面积
    
    注意:
    1. 假设tube_shape(x)返回的是喇叭半径r
    2. 函数是一阶可微的，因为tube_shape是一阶可微的
    """
    if tube_shape_func is None:
        tube_shape_func = import_tube_function()
    
    # 计算半径r
    r = tube_shape_func(x)
    
    # 计算截面积 S = πr²
    return np.pi * r * r

def plot_tube_shape_on_axis(tube_shape_func, ax, num_points=100):
    """
    在给定的axis上绘制tube_shape函数的图像
    
    参数:
    tube_shape_func -- tube_shape函数
    ax -- matplotlib的axis对象
    num_points -- 用于绘制的点数
    """
    # 生成x值（在[0,1]范围内）
    x_values = np.linspace(0, 1, num_points)
    y_values = []
    
    # 计算对应的y值
    for x in x_values:
        try:
            y = tube_shape_func(x)
            y_values.append(y)
        except ValueError:
            # 如果x超出范围，跳过
            y_values.append(np.nan)
    
    # 转换为numpy数组
    y_values = np.array(y_values)
    
    # 在给定的axis上绘制函数曲线
    ax.plot(x_values, y_values, 'b-', linewidth=2, label='tube_shape(x)')
    
    # 添加标题和标签
    ax.set_title('Tube Shape Function', fontsize=16)
    ax.set_xlabel('Normalized Coordinate (x)', fontsize=12)
    ax.set_ylabel('Tube Height (y)', fontsize=12)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 设置坐标轴范围
    ax.set_xlim(0, 1)
    # y轴范围自动调整，确保包含所有数据，并且y=0轴在图上可见
    # 同时保证图像不会顶到最上面，留一些余地
    y_min = np.nanmin(y_values)
    y_max = np.nanmax(y_values)
    
    # 由于函数是正定的（y>0），我们确保y=0轴在图上可见
    # 设置y轴下限为0或一个略小于0的值，确保y=0轴显示
    # 顶部留更多余地（20%而不是10%），避免图像顶到最上面
    margin_factor = 0.1  # 10%的边距
    y_lower = min(0, y_min - margin_factor * (y_max - y_min))
    y_upper = y_max + margin_factor * (y_max - y_lower)
    
    ax.set_ylim(y_lower, y_upper)
    
    # 添加图例
    ax.legend(fontsize=12)
    
    # 注意：根据用户要求，不绘制特征点，只绘制蓝线

def radiation_impedance(r, k):
    """
    计算圆口喇叭声辐射阻抗

    阻抗的定义：Z = v / P （在端口处）
    
    参数:
    r -- 喇叭口半径
    k -- 波数, k = ω/c = 2πf/c
    
    返回:
    Z -- 辐射阻抗 (复数), Z = R + i*X
    
    计算公式:
    R = 1 - J₁(2k·r) / (k·r)
    X = H₁(2k·r) / (k·r)
    
    其中:
    J₁ -- 第一类一阶贝塞尔函数 (scipy.special.j1)
    H₁ -- 第一类一阶斯特鲁弗函数 (scipy.special.struve)
    i -- 虚数单位
    """
    # 计算参数 u = 2k·r
    u = 2 * k * r
    
    # 避免除以零的情况
    if abs(k * r) < 1e-12:
        # 当 k·r 趋近于0时，使用极限值
        # J₁(u) ≈ u/2, H₁(u) ≈ 2/π * (u/2)
        # R ≈ 1 - (u/2)/(k·r) = 1 - (2k·r/2)/(k·r) = 1 - 1 = 0
        # X ≈ (2/π * u/2)/(k·r) = (2/π * k·r)/(k·r) = 2/π
        R = 0.0
        X = 2.0 / np.pi
    else:
        # 计算贝塞尔函数 J₁(u)
        J1 = scipy.special.j1(u)
        
        # 计算斯特鲁弗函数 H₁(u)
        H1 = scipy.special.struve(1, u)  # struve(v, x) 中 v=1 表示一阶斯特鲁弗函数
        
        # 计算实部 R
        R = 1.0 - J1 / (k * r)
        
        # 计算虚部 X
        X = H1 / (k * r)
    
    # 返回复数阻抗
    return R + 1j * X


def main():
    """
    主函数：绘制管状形状
    只需要plt.show，不需要保存
    """
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='绘制管状形状')
    parser.add_argument('--num-points', type=int, default=100,
                       help='用于绘制的点数 (默认: 100)')
    parser.add_argument('--title', type=str, default='管状形状函数',
                       help='图像标题 (默认: 管状形状函数)')
    parser.add_argument('--no-show', action='store_true',
                       help='不显示图像 (默认显示)')
    
    args = parser.parse_args()
    
    try:
        # 导入tube_shape函数
        tube_shape_func = import_tube_function()
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制管状形状
        plot_tube_shape_on_axis(tube_shape_func, ax, num_points=args.num_points)
        
        # 设置标题
        ax.set_title(args.title, fontsize=16, fontweight='bold')
        
        # 调整布局
        plt.tight_layout()
        
        # 显示图像（除非指定了--no-show）
        if not args.no_show:
            plt.show()
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请先运行 make_tube.py 创建管状形状并存储函数")
        return 1
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保 temp/tube_function.py 文件存在且格式正确")
        return 1
    except Exception as e:
        print(f"未知错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
