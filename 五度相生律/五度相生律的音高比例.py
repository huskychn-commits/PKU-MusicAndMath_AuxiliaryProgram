class Pitch:
    def __init__(self):
        # 音高名称映射字典
        self.name_to_index = {
            # 第0个元素
            '\u266EC': 0, '\U0001D12BD': 0, '\u266FB': 0, 'C': 0,
            # 第1个元素
            '\U0001D12AB': 1, '\u266DD': 1, '\u266FC': 1,
            # 第2个元素
            '\u266ED': 2, '\U0001D12BE': 2, '\U0001D12AC': 2, 'D': 2,
            # 第3个元素
            '\U0001D12BF': 3, '\u266DE': 3, '\u266FD': 3,
            # 第4个元素
            '\u266EE': 4, '\U0001D12AD': 4, '\u266DF': 4, 'E': 4,
            # 第5个元素
            '\u266EF': 5, '\U0001D12BG': 5, '\u266FE': 5, 'F': 5,
            # 第6个元素
            '\U0001D12AE': 6, '\u266DG': 6, '\u266FF': 6,
            # 第7个元素
            '\u266EG': 7, '\U0001D12AF': 7, '\U0001D12BA': 7, 'G': 7,
            # 第8个元素
            '\u266DA': 8, '\u266FG': 8,
            # 第9个元素
            '\u266EA': 9, '\U0001D12AG': 9, '\U0001D12BB': 9, 'A': 9,
            # 第10个元素
            '\u266DB': 10, '\u266FA': 10, '\U0001D12BC': 10,
            # 第11个元素
            '\u266EB': 11, '\U0001D12AA': 11, '\u266DC': 11, 'B': 11
        }
        
        # 反向映射：从索引到名称列表
        self.index_to_names = {}
        for name, index in self.name_to_index.items():
            if index not in self.index_to_names:
                self.index_to_names[index] = []
            self.index_to_names[index].append(name)
        
        # 初始化频率列表
        self.pitch_freq = [0.0] * 12
        # 使用默认参数生成Pythagorean调音
        self.pythagorean_intonation()
    
    def pitch_name2index(self, name):
        """根据音高名称找到对应的索引"""
        return self.name_to_index.get(name, -1)
    
    def index2pitch_name(self, index):
        """根据索引返回所有对应的音高名称列表"""
        return self.index_to_names.get(index, [])
    
    def next_pure5(self, i, p_i, base=440.0):
        """纯五度生成方法"""
        j = (i + 5) % 12
        if p_i >= 3/4 * base:
            p_j = 2/3 * p_i
        else:
            p_j = 4/3 * p_i
        return j, p_j
    
    def pythagorean_intonation(self, i=9, f_0=440.0):
        base=f_0
        """Pythagorean调音系统"""
        # 清空频率列表
        self.pitch_freq = [0.0] * 12
        
        # 设置起始频率
        self.pitch_freq[i] = f_0
        current_i = i
        current_f = f_0
        
        # 循环直到填满所有音高
        while True:
            # 使用next_pure5生成下一个音高
            next_i, next_f = self.next_pure5(current_i, current_f, base)
            
            # 如果目标位置已经有值，说明填满了
            if self.pitch_freq[next_i] != 0:
                break
                
            # 存储新生成的频率
            self.pitch_freq[next_i] = next_f
            current_i = next_i
            current_f = next_f


def print_pitch_names(pitch):
    """打印所有音高名称，按列输出"""
    # 找出每个索引对应的最大名称数量
    max_names = 0
    for i in range(12):
        names = pitch.index2pitch_name(i)
        max_names = max(max_names, len(names))
    
    # 输出从索引0到11，最后再输出索引0
    indices = list(range(12)) + [0]
    
    # 第一行：输出序号（靠右对齐）
    for i, index in enumerate(indices):
        formatted_index = str(index)[:3].rjust(3)
        print(formatted_index, end=' ')
        # 在索引4和5之间、索引11和0之间增加额外的3位空格
        if i == 4 or i == 11:
            print('   ', end=' ')  # 额外的3个空格
    print()
    
    # 按行输出，每行对应一个名称位置
    for row in range(max_names):
        for i, index in enumerate(indices):
            names = pitch.index2pitch_name(index)
            # 如果当前行有名称，则输出，否则输出空格
            if row < len(names):
                name = names[row]
                # 固定宽度为3个字符，多则截断，少则补空格（靠右对齐）
                formatted_name = name[:3].rjust(3)
                print(formatted_name, end=' ')
            else:
                print('   ', end=' ')  # 3个空格
            # 在索引4和5之间、索引11和0之间增加额外的3位空格
            if i == 4 or i == 11:
                print('   ', end=' ')  # 额外的3个空格
        print()  # 换行


def print_all_freq(pitch):
    """打印三行音高频率"""
    pitch_freq = pitch.pitch_freq
    
    # 创建高八度列表
    high_octave = [freq * 2 for freq in pitch_freq]
    
    # 找到pitch_freq中最大值对应的索引
    max_freq = max(pitch_freq)
    M = pitch_freq.index(max_freq)
    
    # 第一行：输出index，右对齐但比下面数据的右端靠左两位
    for i in range(12):
        formatted_index = str(i).rjust(5)  # 5位宽度，比7位靠左2位
        print(formatted_index+'  ', end=' ')
    print()
    
    # 第二行：所有index<=M的high_octave值
    for i in range(12):
        if i <= M:
            value = float(high_octave[i])  # 强制转换为浮点数
            formatted_value = f"{value:.2f}".rjust(7)  # 两位小数
            print(formatted_value, end=' ')
        else:
            print('       ', end=' ')  # 7个空格
    print()
    
    # 第三行：index>M的high_octave值和index<=M的pitch_freq值
    for i in range(12):
        if i > M:
            value = float(high_octave[i])  # 强制转换为浮点数
        else:
            value = float(pitch_freq[i])   # 强制转换为浮点数
        formatted_value = f"{value:.2f}".rjust(7)  # 两位小数
        print(formatted_value, end=' ')
    print()
    
    # 第四行：index>M的pitch_freq值
    for i in range(12):
        if i > M:
            value = float(pitch_freq[i])   # 强制转换为浮点数
            formatted_value = f"{value:.2f}".rjust(7)  # 两位小数
            print(formatted_value, end=' ')
        else:
            print('       ', end=' ')  # 7个空格
    print()


def Prob22_Find_Nearest(freqs, base_name="C"):
    """找到最接近的音高名称"""
    # 创建Pitch对象
    pitch1 = Pitch()
    
    # 找到基准音高的索引
    base_index = pitch1.pitch_name2index(base_name)
    if base_index == -1:
        print(f"错误：找不到基准音高名称 '{base_name}'")
        return
    
    # 更新频率列表
    pitch1.pythagorean_intonation(i=base_index, f_0=freqs[0])
    
    # 将频率调整到区间(freqs[0]/2, freqs[0]]上
    def adjust_frequency(freq, base_freq):
        """将频率调整到指定区间"""
        lower_bound = base_freq / 2
        upper_bound = base_freq
        
        while freq <= lower_bound:
            freq *= 2
        while freq > upper_bound:
            freq /= 2
        return freq
    
    # 创建调整后的频率列表
    freqs_target = []
    for i in range(len(freqs)):
        if i == 0:
            freqs_target.append(freqs[0])  # 基准频率保持不变
        else:
            adjusted_freq = adjust_frequency(freqs[i], freqs[0])
            freqs_target.append(adjusted_freq)
    
    # 定义损失函数
    def loss_function(f1, f2):
        """计算两个频率之间的损失"""
        return abs(math.log(f1) - math.log(f2))
    
    # 找到最接近的音高
    def find_nearest_pitch(f_target):
        """找到最接近目标频率的音高索引"""
        min_loss = float('inf')
        best_index = -1
        
        for i in range(12):
            current_loss = loss_function(f_target, pitch1.pitch_freq[i])
            if current_loss < min_loss:
                min_loss = current_loss
                best_index = i
        
        return best_index
    
    # 输出结果
    print(f"基准音高: {base_name} = {freqs[0]:.2f} Hz")
    print("频率映射结果:")
    
    for i in range(len(freqs)):
        if i == 0:
            # 基准频率直接映射到基准音高
            pitch_names = pitch1.index2pitch_name(base_index)
            print(f"{freqs[i]:.2f} : {pitch_names}")
        else:
            # 其他频率找到最接近的音高
            nearest_index = find_nearest_pitch(freqs_target[i])
            pitch_names = pitch1.index2pitch_name(nearest_index)
            print(f"{freqs[i]:.2f} -> {freqs_target[i]:.2f} : {pitch_names}")
    
    return


# 测试代码
if __name__ == "__main__":
    import math
    
    print("音高名称映射表:")
    pitch1 = Pitch()
    print_pitch_names(pitch1)
    print("\n第3个音高的所有名称:", pitch1.index2pitch_name(3))
    print("\n音高频率表:")
    print_all_freq(pitch1)
    
    # 以名字为"C"的元素，pitch_freq=1为基础，更新音高频率表
    print("\n以C=1为基础的音高频率表:")
    pitch_c = Pitch()
    # 找到C对应的索引
    c_index = pitch_c.pitch_name2index('C')
    # 重新生成以C=1为基础的频率
    pitch_c.pythagorean_intonation(i=c_index, f_0=1.0)
    print_all_freq(pitch_c)
    
    # 测试Prob22_Find_Nearest函数
    print("\n" + "="*50)
    print("测试Prob22_Find_Nearest函数:")
    test_freqs = range(1,12)
    Prob22_Find_Nearest(test_freqs, "C")
