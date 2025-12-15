# 五度相生律 (Pythagorean Tuning) 音高比例计算

这个Python程序实现了五度相生律（毕达哥拉斯调音法）的音高比例计算和音高名称映射功能。

## 概述

五度相生律是一种基于纯五度（频率比3:2）的音律系统，通过连续叠加纯五度来生成所有音高。这个程序可以：
1. 生成五度相生律的音高频率表
2. 映射音高名称（包括Unicode音乐符号）
3. 找到最接近给定频率的音高
4. 显示音高频率在不同八度中的分布

## 文件结构

- `五度相生律的音高比例.py` - 主程序文件
- `README.md` - 本说明文件

## 主要功能

### 1. Pitch类
- `__init__()`: 初始化音高名称映射和频率列表
- `pitch_name2index(name)`: 将音高名称转换为索引（0-11）
- `index2pitch_name(index)`: 将索引转换为所有可能的音高名称
- `next_pure5(i, p_i, base)`: 纯五度生成方法
- `pythagorean_intonation(i=9, f_0=440.0)`: 生成五度相生律频率表

### 2. 辅助函数
- `print_pitch_names(pitch)`: 打印所有音高名称映射表
- `print_all_freq(pitch)`: 打印音高频率表（包含不同八度）
- `Prob22_Find_Nearest(freqs, base_name="C")`: 找到最接近给定频率的音高

### 3. 音高名称系统
程序支持多种音高表示方式：
- 字母表示：C, D, E, F, G, A, B
- Unicode音乐符号：♭（降号）, ♯（升号）, ♮（还原号）
- 音乐符号变体：𝄪（双升号）, 𝄫（双降号）等

## 使用方法

### 基本使用
```python
from 五度相生律的音高比例 import Pitch, print_pitch_names, print_all_freq

# 创建Pitch对象（默认以A=440Hz为基础）
pitch = Pitch()

# 打印音高名称映射表
print_pitch_names(pitch)

# 打印频率表
print_all_freq(pitch)
```

### 自定义基准音高
```python
# 以C=1.0为基础生成频率表
pitch_c = Pitch()
c_index = pitch_c.pitch_name2index('C')
pitch_c.pythagorean_intonation(i=c_index, f_0=1.0)
print_all_freq(pitch_c)
```

### 查找最接近的音高
```python
from 五度相生律的音高比例 import Prob22_Find_Nearest

# 测试频率列表
test_freqs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
Prob22_Find_Nearest(test_freqs, "C")
```

## 输出示例

### 音高名称映射表
```
  0     1     2     3     4       5     6     7     8     9    10    11       0   
  C    ♭D    ♮D    ♭E    ♮E      ♮F    ♭G    ♮G    ♭A    ♮A    ♭B    ♮B      C  
 ♮C    ♯C    ♯D    ♯E    ♯F      ♯F    ♯G    ♯A    ♯A    ♯B    ♯B    ♯C     ♮C 
 𝄫D    𝄪C    𝄪D    𝄪E    𝄪F      𝄪F    𝄪G    𝄪A    𝄪A    𝄪B    𝄪B    𝄪C     𝄫D 
```

### 频率表
```
    0       1       2       3       4         5       6       7       8       9      10      11     
 440.00         293.33         195.56         130.37          86.91          57.94   
 220.00  293.33  195.56  130.37   86.91       57.94   77.26  103.01  137.35  183.13  244.17  325.56 
        146.67         97.78          65.19          43.46          28.97   
```

## 数学原理

### 五度相生律
五度相生律基于纯五度的频率比3:2：
- 向上纯五度：频率乘以3/2
- 向下纯五度：频率乘以2/3
- 通过连续叠加纯五度生成所有12个半音

### 频率调整
程序会自动将频率调整到适当的八度范围内：
- 如果频率低于基准频率的一半，则乘以2
- 如果频率高于基准频率，则除以2
- 确保所有频率都在(f₀/2, f₀]范围内

## 依赖项
- Python 3.x
- 标准库：math

## 项目
北京大学《音乐与数学》课程项目
