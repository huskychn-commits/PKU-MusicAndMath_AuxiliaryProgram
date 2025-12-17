import tkinter as tk
from tkinter import ttk
import numpy as np

class DraggablePoint:
    """可拖动的点类"""
    def __init__(self, canvas, x, y, radius=6, color="red", app=None):
        self.canvas = canvas
        self.x = x  # 数学坐标x
        self.y = y  # 数学坐标y
        self.radius = radius
        self.color = color
        self.app = app  # TubeApp实例的引用
        self.drag_data = {"x": 0, "y": 0, "item": None}
        self.label_id = None  # 坐标标签的ID
        self.label_visible = False
        self.is_dragging = False
        
        # 绘制点
        self.draw()
        
        # 绑定事件
        self.bind_events()
    
    # 注意：math_to_canvas和canvas_to_math方法将在创建点后被TubeApp替换
    # 这里只提供占位方法
    def math_to_canvas(self, x, y):
        """将数学坐标转换为画布坐标 - 将由TubeApp提供实现"""
        # 默认实现，稍后会被替换
        return x * 100, y * 100
    
    def canvas_to_math(self, canvas_x, canvas_y):
        """将画布坐标转换为数学坐标 - 将由TubeApp提供实现"""
        # 默认实现，稍后会被替换
        return canvas_x / 100, canvas_y / 100
    
    def draw(self):
        """绘制点"""
        canvas_x, canvas_y = self.math_to_canvas(self.x, self.y)
        self.point_id = self.canvas.create_oval(
            canvas_x - self.radius, canvas_y - self.radius,
            canvas_x + self.radius, canvas_y + self.radius,
            fill=self.color, outline="black", width=2,
            tags=("draggable_point", f"point_{id(self)}")
        )
    
    def bind_events(self):
        """绑定事件"""
        self.canvas.tag_bind(self.point_id, "<ButtonPress-1>", self.on_press)
        self.canvas.tag_bind(self.point_id, "<B1-Motion>", self.on_drag)
        self.canvas.tag_bind(self.point_id, "<ButtonRelease-1>", self.on_release)
        self.canvas.tag_bind(self.point_id, "<Enter>", self.on_enter)
        self.canvas.tag_bind(self.point_id, "<Leave>", self.on_leave)
    
    def on_press(self, event):
        """鼠标按下事件"""
        self.drag_data["item"] = self.point_id
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y
        
        # 隐藏标签
        self.hide_label()
    
    def on_drag(self, event):
        """拖动事件"""
        if self.drag_data["item"] is None:
            return
        
        # 计算鼠标位置对应的数学坐标
        mouse_x, mouse_y = self.canvas_to_math(event.x, event.y)
        
        # 限制数学坐标在[0,1]范围内
        mouse_x = max(0.0, min(1.0, mouse_x))
        mouse_y = max(0.0, min(1.0, mouse_y))
        
        # 更新点的数学坐标
        self.x, self.y = mouse_x, mouse_y
        
        # 将数学坐标转换回画布坐标
        canvas_x, canvas_y = self.math_to_canvas(self.x, self.y)
        
        # 移动点到正确位置
        self.canvas.coords(self.point_id,
                          canvas_x - self.radius, canvas_y - self.radius,
                          canvas_x + self.radius, canvas_y + self.radius)
        
        # 更新拖动数据（使用实际移动后的位置）
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y
        
        # 更新标签位置
        if self.label_visible:
            self.update_label_position()
        
        # 更新曲线
        if self.app:
            self.app.update_curves()
    
    def on_release(self, event):
        """鼠标释放事件"""
        self.drag_data["item"] = None
        self.drag_data["x"] = 0
        self.drag_data["y"] = 0
        
        # 不需要重新绘制点，因为拖动过程中已经更新了位置
        # 只需要确保点的位置与坐标轴对齐
        # 轻微调整位置以确保对齐
        self.adjust_position()
    
    def adjust_position(self):
        """调整点的位置以确保与坐标轴对齐"""
        # 获取当前画布坐标
        canvas_x, canvas_y = self.canvas.coords(self.point_id)[:2]
        canvas_x += self.radius
        canvas_y += self.radius
        
        # 转换为数学坐标
        new_x, new_y = self.canvas_to_math(canvas_x, canvas_y)
        
        # 如果坐标有变化，则更新
        if abs(new_x - self.x) > 0.001 or abs(new_y - self.y) > 0.001:
            self.x, self.y = new_x, new_y
            # 只更新点的位置，不重新创建
            self.update_display()
    
    def update_display(self):
        """更新点的显示位置"""
        canvas_x, canvas_y = self.math_to_canvas(self.x, self.y)
        self.canvas.coords(self.point_id,
                          canvas_x - self.radius, canvas_y - self.radius,
                          canvas_x + self.radius, canvas_y + self.radius)
        
        # 更新标签位置
        if self.label_visible:
            self.update_label_position()
    
    def on_enter(self, event):
        """鼠标进入事件 - 显示坐标标签"""
        self.show_label()
    
    def on_leave(self, event):
        """鼠标离开事件 - 隐藏坐标标签"""
        self.hide_label()
    
    def show_label(self):
        """显示坐标标签"""
        if self.label_id:
            self.canvas.delete(self.label_id)
        
        canvas_x, canvas_y = self.math_to_canvas(self.x, self.y)
        label_text = f"({self.x:.2f}, {self.y:.2f})"
        self.label_id = self.canvas.create_text(
            canvas_x, canvas_y - 20,
            text=label_text,
            fill="blue",
            font=("Arial", 10, "bold"),
            tags=("point_label", f"label_{id(self)}")
        )
        self.label_visible = True
    
    def hide_label(self):
        """隐藏坐标标签"""
        if self.label_id:
            self.canvas.delete(self.label_id)
            self.label_id = None
        self.label_visible = False
    
    def update_label_position(self):
        """更新标签位置"""
        if self.label_id and self.label_visible:
            canvas_x, canvas_y = self.math_to_canvas(self.x, self.y)
            label_text = f"({self.x:.2f}, {self.y:.2f})"
            self.canvas.coords(self.label_id, canvas_x, canvas_y - 20)
            self.canvas.itemconfig(self.label_id, text=label_text)
    
    def redraw(self):
        """重新绘制点"""
        # 删除旧的点
        self.canvas.delete(self.point_id)
        if self.label_id:
            self.canvas.delete(self.label_id)
            self.label_id = None
        
        # 绘制新的点
        self.draw()
        
        # 重新绑定事件
        self.bind_events()
        
        # 如果标签应该显示，则显示
        if self.label_visible:
            self.show_label()
    
    def update_position(self, x, y):
        """更新点的位置"""
        # 限制在X,Y都在[0,1]之间
        self.x = max(0.0, min(1.0, x))
        self.y = max(0.0, min(1.0, y))
        self.redraw()

class TubeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tube Maker")
        self.root.geometry("800x600")
        
        # 创建主框架
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重，使画布和按钮区域可以扩展
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=0)
        
        # 上半部分：画布区域
        canvas_frame = ttk.LabelFrame(main_frame, text="Display Area", padding="5")
        canvas_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        # 创建画布，设置背景为白色
        self.canvas = tk.Canvas(canvas_frame, bg="white", highlightthickness=1, highlightbackground="black")
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 设置坐标系：将(0,0)放在左下角
        self.setup_coordinate_system()
        
        # 下半部分：控制区域
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        control_frame.columnconfigure(0, weight=1)
        
        # 使用统一的网格布局，使三行像表格一样对齐，并靠左对齐
        # 配置控制区域的网格列 - 设置为靠左对齐
        control_frame.columnconfigure(0, weight=0)  # 第一列，不扩展
        control_frame.columnconfigure(1, weight=0)  # 第二列，不扩展
        control_frame.columnconfigure(2, weight=1)  # 空白列，占据剩余空间，使前面列靠左
        
        # 第一行：按钮
        row0_pady = (0, 10)
        
        # "Add Point" 按钮 - 第0行，第0列
        self.add_point_btn = ttk.Button(control_frame, text="Add Point")
        self.add_point_btn.grid(row=0, column=0, padx=(20, 40), pady=row0_pady, sticky=tk.W)
        
        # "Clear" 按钮 - 第0行，第1列
        self.clear_btn = ttk.Button(control_frame, text="Clear")
        self.clear_btn.grid(row=0, column=1, padx=(0, 20), pady=row0_pady, sticky=tk.W)
        
        # 第二行：边界条件复选框
        row1_pady = (0, 10)
        
        # "Closed Boundary (Left)" 复选框 - 第1行，第0列
        self.left_boundary_var = tk.BooleanVar(value=False)
        self.left_boundary_check = ttk.Checkbutton(control_frame, text="Closed Boundary (Left)", 
                                                  variable=self.left_boundary_var)
        self.left_boundary_check.grid(row=1, column=0, padx=(20, 40), pady=row1_pady, sticky=tk.W)
        
        # "Closed Boundary (Right)" 复选框 - 第1行，第1列
        self.right_boundary_var = tk.BooleanVar(value=False)
        self.right_boundary_check = ttk.Checkbutton(control_frame, text="Closed Boundary (Right)", 
                                                   variable=self.right_boundary_var)
        self.right_boundary_check.grid(row=1, column=1, padx=(0, 20), pady=row1_pady, sticky=tk.W)
        
        # 第三行：复选框和按钮
        row2_pady = (0, 0)
        
        # "Radiation Damping" 复选框 - 第2行，第0列
        self.damping_var = tk.BooleanVar(value=False)
        self.damping_check = ttk.Checkbutton(control_frame, text="Radiation Damping", 
                                            variable=self.damping_var)
        self.damping_check.grid(row=2, column=0, padx=(20, 40), pady=row2_pady, sticky=tk.W)
        
        # "Store" 按钮 - 第2行，第1列
        self.store_btn = ttk.Button(control_frame, text="Store")
        self.store_btn.grid(row=2, column=1, padx=(0, 20), pady=row2_pady, sticky=tk.W)
        
        # 确保所有元素靠左对齐
        # 通过设置第0列和第1列的weight=0，它们不会扩展
        # 第2列weight=1会占据所有剩余空间，从而将前面列推到左边
        
        # 存储点的列表
        self.points = []
        
        # 给Add Point按钮绑定功能
        self.add_point_btn.config(command=self.add_point_at_center)
        
        # 给Clear按钮绑定功能
        self.clear_btn.config(command=self.clear_all_points)
        
        # 给复选框绑定事件
        self.left_boundary_var.trace("w", self.on_boundary_changed)
        self.right_boundary_var.trace("w", self.on_boundary_changed)
        self.damping_var.trace("w", self.on_damping_changed)
        
        # 给Store按钮绑定功能
        self.store_btn.config(command=self.store_data)
        
        # 边界线ID存储
        self.left_boundary_line_id = None
        self.right_boundary_line_id = None
        self.left_damping_line_id = None
        self.right_damping_line_id = None
    
    def add_point_at_center(self):
        """在(0.5,0.5)位置添加一个新点"""
        new_point = DraggablePoint(self.canvas, 0.5, 0.5, app=self)
        # 修改DraggablePoint的math_to_canvas方法，使其使用TubeApp的方法
        new_point.math_to_canvas = self.math_to_canvas
        # 修改canvas_to_math方法
        new_point.canvas_to_math = self.canvas_to_math
        # 重新绘制点，使用正确的坐标转换
        new_point.redraw()
        self.points.append(new_point)
        print(f"添加了新点，当前共有 {len(self.points)} 个点")
        # 绘制曲线
        self.draw_curves()
    
    def get_sorted_points(self):
        """获取按X坐标排序的点"""
        if len(self.points) < 2:
            return []
        
        # 按X坐标排序
        sorted_points = sorted(self.points, key=lambda p: p.x)
        return sorted_points
    
    def draw_curves(self):
        """绘制曲线"""
        # 删除之前的曲线
        self.canvas.delete("curve")
        
        # 获取排序后的点
        sorted_points = self.get_sorted_points()
        if len(sorted_points) < 2:
            return
        
        # 提取点的坐标
        points = [(p.x, p.y) for p in sorted_points]
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        
        # 根据点数绘制不同的曲线
        n = len(points)
        
        if n == 2:
            # 两个点：画直线
            self.draw_line(points[0], points[1])
        elif n == 3:
            # 三个点：两条二次曲线
            self.draw_quadratic_spline(points)
        else:
            # 四个或更多点：混合样条
            self.draw_mixed_spline(points)
    
    def draw_line(self, p1, p2):
        """绘制直线"""
        x1, y1 = self.math_to_canvas(p1[0], p1[1])
        x2, y2 = self.math_to_canvas(p2[0], p2[1])
        self.canvas.create_line(x1, y1, x2, y2, fill="green", width=2, tags="curve")
    
    def draw_quadratic_spline(self, points):
        """绘制二次样条曲线（三个点）"""
        # 三个点：两条二次曲线
        # 我们需要在中间点处连续且一阶导数连续
        
        # 简化：使用二次贝塞尔曲线
        # 第一条曲线：点0到点1，控制点为点0和点1的中点
        # 第二条曲线：点1到点2，控制点为点1和点2的中点
        
        # 第一条二次曲线（点0到点1）
        x0, y0 = points[0]
        x1, y1 = points[1]
        x2, y2 = points[2]
        
        # 控制点：中点
        cx1 = (x0 + x1) / 2
        cy1 = (y0 + y1) / 2
        
        cx2 = (x1 + x2) / 2
        cy2 = (y1 + y2) / 2
        
        # 绘制第一条二次贝塞尔曲线
        self.draw_quadratic_bezier((x0, y0), (cx1, cy1), (x1, y1))
        
        # 绘制第二条二次贝塞尔曲线
        self.draw_quadratic_bezier((x1, y1), (cx2, cy2), (x2, y2))
    
    def draw_mixed_spline(self, points):
        """绘制混合样条曲线（四个或更多点）"""
        n = len(points)
        
        # 首尾两段是二次曲线，中间是三次曲线
        # 简化：使用样条插值
        
        # 提取x和y值
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        
        # 使用numpy的样条插值
        try:
            # 创建样条插值
            from scipy import interpolate
            # 检查是否有scipy
            tck = interpolate.splrep(x_vals, y_vals, s=0)
            
            # 生成平滑曲线
            x_smooth = np.linspace(min(x_vals), max(x_vals), 100)
            y_smooth = interpolate.splev(x_smooth, tck, der=0)
            
            # 绘制曲线
            self.draw_smooth_curve(x_smooth, y_smooth)
        except ImportError:
            # 如果没有scipy，使用简化方法：分段线性插值
            print("警告：未找到scipy，使用简化曲线")
            for i in range(n-1):
                self.draw_line(points[i], points[i+1])
    
    def draw_quadratic_bezier(self, p0, p1, p2, segments=20):
        """绘制二次贝塞尔曲线"""
        x0, y0 = p0
        x1, y1 = p1
        x2, y2 = p2
        
        # 生成贝塞尔曲线点
        curve_points = []
        for i in range(segments + 1):
            t = i / segments
            # 二次贝塞尔公式
            x = (1-t)**2 * x0 + 2*(1-t)*t * x1 + t**2 * x2
            y = (1-t)**2 * y0 + 2*(1-t)*t * y1 + t**2 * y2
            curve_points.append((x, y))
        
        # 绘制曲线
        self.draw_polyline(curve_points)
    
    def draw_smooth_curve(self, x_vals, y_vals):
        """绘制平滑曲线"""
        if len(x_vals) < 2:
            return
        
        # 将点转换为画布坐标
        canvas_points = []
        for x, y in zip(x_vals, y_vals):
            cx, cy = self.math_to_canvas(x, y)
            canvas_points.append((cx, cy))
        
        # 绘制折线
        self.canvas.create_line(canvas_points, fill="green", width=2, tags="curve", smooth=True)
    
    def draw_polyline(self, points):
        """绘制折线"""
        if len(points) < 2:
            return
        
        # 将点转换为画布坐标
        canvas_points = []
        for x, y in points:
            cx, cy = self.math_to_canvas(x, y)
            canvas_points.append((cx, cy))
        
        # 绘制折线
        self.canvas.create_line(canvas_points, fill="green", width=2, tags="curve")
    
    def update_curves(self):
        """更新曲线（当点移动时调用）"""
        self.draw_curves()
        self.update_boundary_lines()
    
    def on_boundary_changed(self, *args):
        """边界复选框改变时调用"""
        self.update_boundary_lines()
    
    def on_damping_changed(self, *args):
        """辐射阻抗复选框改变时调用"""
        self.update_boundary_lines()
    
    def update_boundary_lines(self):
        """更新边界线"""
        # 删除之前的边界线
        if self.left_boundary_line_id:
            self.canvas.delete(self.left_boundary_line_id)
            self.left_boundary_line_id = None
        if self.right_boundary_line_id:
            self.canvas.delete(self.right_boundary_line_id)
            self.right_boundary_line_id = None
        if self.left_damping_line_id:
            self.canvas.delete(self.left_damping_line_id)
            self.left_damping_line_id = None
        if self.right_damping_line_id:
            self.canvas.delete(self.right_damping_line_id)
            self.right_damping_line_id = None
        
        # 如果没有点，不绘制边界线
        if len(self.points) < 1:
            return
        
        # 获取排序后的点
        sorted_points = self.get_sorted_points()
        if len(sorted_points) < 1:
            return
        
        # 获取最左边和最右边的点
        left_point = sorted_points[0]
        right_point = sorted_points[-1]
        
        # 左边界闭合
        if self.left_boundary_var.get():
            self.draw_boundary_line(left_point, "left", "green")
        
        # 右边界闭合
        if self.right_boundary_var.get():
            self.draw_boundary_line(right_point, "right", "green")
        
        # 辐射阻抗
        if self.damping_var.get():
            # 在没有闭合的边界上绘制蓝色竖线
            if not self.left_boundary_var.get():
                self.draw_boundary_line(left_point, "left", "blue")
            if not self.right_boundary_var.get():
                self.draw_boundary_line(right_point, "right", "blue")
    
    def draw_boundary_line(self, point, side, color):
        """绘制边界线
        point: 点对象
        side: "left" 或 "right"
        color: 线条颜色
        """
        # 从点垂直向下到X轴 (y=0)
        x1, y1 = self.math_to_canvas(point.x, point.y)
        x2, y2 = self.math_to_canvas(point.x, 0)
        
        # 绘制竖线
        line_id = self.canvas.create_line(x1, y1, x2, y2, fill=color, width=2, tags="boundary_line")
        
        # 存储线条ID
        if side == "left":
            if color == "green":
                self.left_boundary_line_id = line_id
            else:  # blue
                self.left_damping_line_id = line_id
        else:  # right
            if color == "green":
                self.right_boundary_line_id = line_id
            else:  # blue
                self.right_damping_line_id = line_id
    
    def clear_all_points(self):
        """清除所有点，使图回到初始状态"""
        # 删除画布上的所有点和曲线
        self.canvas.delete("draggable_point")
        self.canvas.delete("point_label")
        self.canvas.delete("curve")
        
        # 清空点列表
        self.points.clear()
        
        print("已清除所有点，图已回到初始状态")
    
    def store_data(self):
        """存储数据到文件"""
        try:
            # 获取排序后的点
            sorted_points = self.get_sorted_points()
            
            if len(sorted_points) < 2:
                print("错误：至少需要2个点才能存储函数")
                return
            
            # 提取点的坐标
            points = [(p.x, p.y) for p in sorted_points]
            x_vals = [p[0] for p in points]
            y_vals = [p[1] for p in points]
            
            # 获取三个Bool值
            left_boundary = self.left_boundary_var.get()
            right_boundary = self.right_boundary_var.get()
            radiation_damping = self.damping_var.get()
            
            # 创建存储目录（如果不存在）
            import os
            # 获取当前文件所在目录（泛音目录）
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 在泛音目录中创建temp子目录
            output_dir = os.path.join(current_dir, "temp")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 1. 存储三个Bool值到文件
            bool_filename = os.path.join(output_dir, "tube_settings.txt")
            with open(bool_filename, 'w') as f:
                f.write(f"left_boundary_closed = {left_boundary}\n")
                f.write(f"right_boundary_closed = {right_boundary}\n")
                f.write(f"radiation_damping = {radiation_damping}\n")
            
            print(f"已存储设置到 {bool_filename}")
            
            # 2. 存储Python函数到文件
            func_filename = os.path.join(output_dir, "tube_function.py")
            
            # 生成函数代码
            func_code = self.generate_tube_function(x_vals, y_vals)
            
            with open(func_filename, 'w') as f:
                f.write(func_code)
            
            print(f"已存储函数到 {func_filename}")
            print("存储完成！")
            
        except Exception as e:
            print(f"存储数据时出错: {e}")
    
    def generate_tube_function(self, x_vals, y_vals):
        """Generate tube shape Python function in English (differentiable)"""
        # Calculate X coordinate range
        x_min = min(x_vals)
        x_max = max(x_vals)
        
        # Determine appropriate spline order based on number of points
        n_points = len(x_vals)
        if n_points < 2:
            raise ValueError("At least 2 points are required to define a function")
        elif n_points == 2:
            spline_order = 1  # Linear interpolation
        elif n_points == 3:
            spline_order = 2  # Quadratic spline
        else:
            spline_order = 3  # Cubic spline (default)
        
        # Generate function code in English
        code = f'''"""
Tube shape function (differentiable)
Input: x (a number in [0,1] range)
Output: corresponding Y coordinate
Raises exception if input is outside [0,1] range
The function is differentiable using spline interpolation (order={spline_order}).
"""

import numpy as np

def tube_shape(x):
    """
    Compute tube shape function (differentiable)
    
    Parameters:
    x -- normalized coordinate, must be in [0,1] range
    
    Returns:
    y -- corresponding Y coordinate
    
    Raises:
    ValueError -- if x is not in [0,1] range
    ImportError -- if scipy is not available (required for differentiable interpolation)
    """
    # Check input range
    if x < 0 or x > 1:
        raise ValueError(f"Input x={{x}} is outside [0,1] range")
    
    # Original data points
    x_vals = {x_vals}
    y_vals = {y_vals}
    
    # X coordinate range
    x_min = {x_min}
    x_max = {x_max}
    
    # Number of points and spline order
    n_points = {n_points}
    spline_order = {spline_order}
    
    # For differentiable interpolation, we require scipy
    try:
        from scipy import interpolate
    except ImportError:
        raise ImportError(
            "scipy is required for differentiable tube_shape function. "
            "Please install scipy: pip install scipy"
        )
    
    # Create spline interpolation with appropriate order
    # For linear interpolation (k=1), we need at least 2 points
    # For quadratic spline (k=2), we need at least 3 points
    # For cubic spline (k=3), we need at least 4 points
    tck = interpolate.splrep(x_vals, y_vals, s=0, k=spline_order)
    
    # Map input x from [0,1] to actual X coordinate range
    x_scaled = x_min + x * (x_max - x_min)
    
    # Ensure x_scaled is within the interpolation range
    # (splev can extrapolate, but we want to stay within bounds)
    x_scaled = max(min(x_scaled, x_vals[-1]), x_vals[0])
    
    # Compute interpolation using spline
    y = interpolate.splev(x_scaled, tck, der=0)
    return float(y)

def tube_shape_derivative(x, order=1):
    """
    Compute derivative of tube shape function
    
    Parameters:
    x -- normalized coordinate, must be in [0,1] range
    order -- order of derivative (1 for first derivative, 2 for second derivative)
    
    Returns:
    dy/dx -- derivative of tube shape function at x
    
    Raises:
    ValueError -- if x is not in [0,1] range or order is not 1 or 2
    ImportError -- if scipy is not available
    """
    if x < 0 or x > 1:
        raise ValueError(f"Input x={{x}} is outside [0,1] range")
    
    if order not in [1, 2]:
        raise ValueError(f"Derivative order must be 1 or 2, got {{order}}")
    
    # Original data points
    x_vals = {x_vals}
    y_vals = {y_vals}
    
    # X coordinate range
    x_min = {x_min}
    x_max = {x_max}
    
    # Number of points and spline order
    n_points = {n_points}
    spline_order = {spline_order}
    
    # Check if derivative is available for the given spline order
    if spline_order == 1 and order > 1:
        raise ValueError(f"Cannot compute order {{order}} derivative for linear interpolation (spline_order=1)")
    if spline_order == 2 and order > 2:
        raise ValueError(f"Cannot compute order {{order}} derivative for quadratic spline (spline_order=2)")
    
    # Require scipy for derivative computation
    try:
        from scipy import interpolate
    except ImportError:
        raise ImportError(
            "scipy is required for derivative computation. "
            "Please install scipy: pip install scipy"
        )
    
    # Create spline interpolation with appropriate order
    tck = interpolate.splrep(x_vals, y_vals, s=0, k=spline_order)
    
    # Map input x from [0,1] to actual X coordinate range
    x_scaled = x_min + x * (x_max - x_min)
    
    # Ensure x_scaled is within the interpolation range
    x_scaled = max(min(x_scaled, x_vals[-1]), x_vals[0])
    
    # Compute derivative using spline
    derivative = interpolate.splev(x_scaled, tck, der=order)
    return float(derivative)

# Example usage and testing
if __name__ == "__main__":
    # Test the function and its derivatives
    test_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    print("Testing tube_shape function (differentiable):")
    for x in test_points:
        try:
            y = tube_shape(x)
            print(f"  tube_shape({{x}}) = {{y:.6f}}")
        except (ValueError, ImportError) as e:
            print(f"  tube_shape({{x}}) error: {{e}}")
    
    print("\\nTesting first derivative:")
    for x in test_points:
        try:
            dy = tube_shape_derivative(x, order=1)
            print(f"  tube_shape'({{x}}) = {{dy:.6f}}")
        except (ValueError, ImportError) as e:
            print(f"  tube_shape'({{x}}) error: {{e}}")
    
    print("\\nTesting second derivative:")
    for x in test_points:
        try:
            d2y = tube_shape_derivative(x, order=2)
            print(f"  tube_shape''({{x}}) = {{d2y:.6f}}")
        except (ValueError, ImportError) as e:
            print(f"  tube_shape''({{x}}) error: {{e}}")
'''
        
        return code
    
    def canvas_to_math(self, canvas_x, canvas_y):
        """将画布坐标转换为数学坐标"""
        # 使用与math_to_canvas相同的逻辑但反向计算
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 0 or canvas_height <= 0:
            canvas_width = 600
            canvas_height = 400
        
        # 计算动态边距
        margin_x = max(20, min(60, int(canvas_width * 0.05)))
        margin_y = max(20, min(60, int(canvas_height * 0.05)))
        margin = max(margin_x, margin_y)
        
        # 计算可绘制区域
        draw_width = canvas_width - 2 * margin
        draw_height = canvas_height - 2 * margin
        
        if draw_width <= 0 or draw_height <= 0:
            return 0.5, 0.5  # 返回默认值
        
        # 转换坐标
        x = (canvas_x - margin) / draw_width
        y = (canvas_height - margin - canvas_y) / draw_height  # Y轴反转
        
        # 限制在X,Y都在[0,1]之间
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        
        return x, y
        
    def setup_coordinate_system(self):
        """设置坐标系，将(0,0)放在左下角"""
        # 绑定事件，在画布大小改变时重新绘制坐标轴
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        
    def on_canvas_configure(self, event):
        """当画布大小改变时，重新设置坐标系"""
        # 设置坐标系：将(0,0)放在左下角
        canvas_width = event.width
        canvas_height = event.height
        
        # 设置画布的可滚动区域
        self.canvas.config(scrollregion=(0, 0, canvas_width, canvas_height))
        
        # 反转Y轴：使Y轴向上为正
        self.canvas.yview_moveto(1.0)  # 将视图移动到顶部，这样(0,0)就在左下角
        
        # 绘制坐标轴作为参考
        self.draw_coordinate_axes(canvas_width, canvas_height)
    
    def math_to_canvas(self, x, y):
        """将数学坐标(x,y)转换为画布坐标"""
        # 数学坐标范围：x∈[0,1], y∈[0,1]
        # 画布坐标：x从margin到width-margin, y从height-margin到margin（因为Y轴反转）
        
        # 使用比例边距：画布宽度和高度的5%，但最小为20像素，最大为60像素
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 0 or canvas_height <= 0:
            canvas_width = 600
            canvas_height = 400
        
        # 计算动态边距
        margin_x = max(20, min(60, int(canvas_width * 0.05)))
        margin_y = max(20, min(60, int(canvas_height * 0.05)))
        margin = max(margin_x, margin_y)  # 使用较大的边距以确保对称
        
        # 计算可绘制区域
        draw_width = canvas_width - 2 * margin
        draw_height = canvas_height - 2 * margin
        
        # 确保可绘制区域为正
        if draw_width <= 0:
            draw_width = canvas_width
            margin = 0
        if draw_height <= 0:
            draw_height = canvas_height
            margin = 0
        
        # 转换坐标
        canvas_x = margin + x * draw_width
        canvas_y = canvas_height - margin - y * draw_height  # Y轴反转
        
        return canvas_x, canvas_y
    
    def draw_coordinate_axes(self, width, height):
        """绘制坐标轴作为参考，范围X∈[0,1], Y∈[0,1]"""
        # 清除之前的图形
        self.canvas.delete("axes")
        
        # 使用与math_to_canvas相同的边距计算逻辑
        # 计算动态边距
        margin_x = max(20, min(60, int(width * 0.05)))
        margin_y = max(20, min(60, int(height * 0.05)))
        margin = max(margin_x, margin_y)  # 使用较大的边距以确保对称
        
        # 计算可绘制区域
        draw_width = width - 2 * margin
        draw_height = height - 2 * margin
        
        # 如果画布太小，使用默认值
        if draw_width <= 10 or draw_height <= 10:
            # 画布太小，不绘制坐标轴
            return
        
        # 绘制坐标轴（带箭头的线）
        
        # X轴（从(0,0)到(1,0)）
        x1, y1 = self.math_to_canvas(0, 0)
        x2, y2 = self.math_to_canvas(1, 0)
        self.canvas.create_line(x1, y1, x2, y2, fill="black", tags="axes", width=2, arrow=tk.LAST)
        
        # Y轴（从(0,0)到(0,1)）
        x1, y1 = self.math_to_canvas(0, 0)
        x2, y2 = self.math_to_canvas(0, 1)
        self.canvas.create_line(x1, y1, x2, y2, fill="black", tags="axes", width=2, arrow=tk.LAST)
        
        # 添加原点标记 (0,0)
        ox, oy = self.math_to_canvas(0, 0)
        self.canvas.create_text(ox - 5, oy + 5, text="(0,0)", fill="blue", tags="axes", anchor="ne")
        
        # 注意：根据用户要求，不显示X和Y坐标的标题
        
        # 添加刻度标记
        # X轴刻度：0, 0.2, 0.4, 0.6, 0.8, 1.0
        for x_val in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            x_pos, y_pos = self.math_to_canvas(x_val, 0)
            # 刻度线
            self.canvas.create_line(x_pos, y_pos, x_pos, y_pos + 5, fill="black", tags="axes", width=1)
            # 刻度标签
            self.canvas.create_text(x_pos, y_pos + 15, text=f"{x_val:.1f}", fill="black", tags="axes",
                                   font=("Arial", 9))
        
        # Y轴刻度：0, 0.2, 0.4, 0.6, 0.8, 1.0
        for y_val in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            x_pos, y_pos = self.math_to_canvas(0, y_val)
            # 刻度线
            self.canvas.create_line(x_pos, y_pos, x_pos - 5, y_pos, fill="black", tags="axes", width=1)
            # 刻度标签
            self.canvas.create_text(x_pos - 15, y_pos, text=f"{y_val:.1f}", fill="black", tags="axes",
                                   font=("Arial", 9), anchor="e")
        
        # 添加网格线（浅灰色）
        for x_val in [0.2, 0.4, 0.6, 0.8]:
            x1, y1 = self.math_to_canvas(x_val, 0)
            x2, y2 = self.math_to_canvas(x_val, 1)
            self.canvas.create_line(x1, y1, x2, y2, fill="#e0e0e0", tags="axes", width=1, dash=(2, 2))
        
        for y_val in [0.2, 0.4, 0.6, 0.8]:
            x1, y1 = self.math_to_canvas(0, y_val)
            x2, y2 = self.math_to_canvas(1, y_val)
            self.canvas.create_line(x1, y1, x2, y2, fill="#e0e0e0", tags="axes", width=1, dash=(2, 2))
        
        # 添加范围标签
        range_x, range_y = self.math_to_canvas(0.5, -0.15)
        self.canvas.create_text(range_x, range_y, text="X ∈ [0, 1]", fill="darkblue", tags="axes",
                               font=("Arial", 10, "italic"))
    
        range_x, range_y = self.math_to_canvas(-0.15, 0.5)
        self.canvas.create_text(range_x, range_y, text="Y ∈ [0, 1]", fill="darkblue", tags="axes",
                               font=("Arial", 10, "italic"), angle=90)

def main():
    root = tk.Tk()
    app = TubeApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
