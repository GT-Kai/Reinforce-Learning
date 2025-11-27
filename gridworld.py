"""
优化的 GridWorld 环境

Credits: Intelligent Unmanned Systems Laboratory at Westlake University.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches          
from conf.arguments import args           


class GridWorld():
    """
    优化的网格世界环境
    
    主要改进：
    1. 使用元组形式的动作空间 (dx, dy)，更高效
    2. 添加完整的可视化功能（matplotlib）
    3. 支持轨迹记录和动画
    4. 支持策略可视化和状态值显示
    5. 更灵活的奖励系统
    6. 0-indexed 坐标系统，更符合编程习惯
    7. 详细的边界和碰撞处理
    
    使用示例：
        >>> env = GridWorld(env_size=(5,5), start_state=(0,0), target_state=(4,4))
        >>> state, info = env.reset()
        >>> next_state, reward, done, info = env.step((1, 0))  # 向右移动
        >>> env.render()
    """

    def __init__(self, 
                 env_size=args.env_size, 
                 start_state=args.start_state, 
                 target_state=args.target_state, 
                 forbidden_states=args.forbidden_states):
        """
        初始化 GridWorld 环境
        
        参数:
            env_size: tuple (rows, cols) 环境大小
            start_state: tuple (x, y) 起始状态，0-indexed
            target_state: tuple (x, y) 目标状态
            forbidden_states: list [(x, y), ...] 禁止状态列表
        """
        self.env_size = env_size
        self.num_states = env_size[0] * env_size[1]
        self.start_state = start_state
        self.target_state = target_state
        self.forbidden_states = forbidden_states

        self.agent_state = start_state
        self.action_space = args.action_space          
        self.reward_target = args.reward_target
        self.reward_forbidden = args.reward_forbidden
        self.reward_step = args.reward_step

        self.canvas = None
        self.animation_interval = args.animation_interval

        # 颜色配置
        self.color_forbid = (0.9290, 0.6940, 0.125)      # 黄色 - 禁止区域
        self.color_target = (0.3010, 0.7450, 0.9330)     # 蓝色 - 目标
        self.color_policy = (0.4660, 0.6740, 0.1880)     # 绿色 - 策略
        self.color_trajectory = (0, 1, 0)                 # 绿色 - 轨迹
        self.color_agent = (0, 0, 1)                      # 蓝色 - 智能体

    def reset(self):
        """
        重置环境到初始状态
        
        返回:
            state: tuple (x, y) 初始状态
            info: dict 额外信息（当前为空字典）
        """
        self.agent_state = self.start_state
        self.traj = [self.agent_state] 
        return self.agent_state, {}

    def step(self, action):
        """
        执行一个动作
        
        参数:
            action: tuple (dx, dy) 移动方向
                   可选值: (0,-1)上, (1,0)右, (0,1)下, (-1,0)左, (0,0)停留
            
        返回:
            next_state: tuple (x, y) 下一个状态
            reward: float 获得的奖励
            done: bool 是否到达终止状态
            info: dict 额外信息（当前为空字典）
        """
        assert action in self.action_space, f"Invalid action {action}"

        next_state, reward = self._get_next_state_and_reward(self.agent_state, action)
        done = self._is_done(next_state)

        # 添加轨迹记录（带随机偏移以显示路径）
        x_store = next_state[0] + 0.03 * np.random.randn()
        y_store = next_state[1] + 0.03 * np.random.randn()
        state_store = tuple(np.array((x_store, y_store)) + 0.2 * np.array(action))
        state_store_2 = (next_state[0], next_state[1])

        self.agent_state = next_state

        self.traj.append(state_store)   
        self.traj.append(state_store_2)
        return self.agent_state, reward, done, {}   
    
    def _get_next_state_and_reward(self, state, action):
        """
        计算下一个状态和奖励（内部方法）
        
        处理边界碰撞、禁止区域、目标到达等情况
        
        参数:
            state: tuple (x, y) 当前状态
            action: tuple (dx, dy) 动作
            
        返回:
            next_state: tuple (x, y) 下一个状态
            reward: float 奖励值
        """
        x, y = state
        new_state = tuple(np.array(state) + np.array(action))
        
        # 边界检测
        if y + 1 > self.env_size[1] - 1 and action == (0, 1):    # down
            y = self.env_size[1] - 1
            reward = self.reward_forbidden  
        elif x + 1 > self.env_size[0] - 1 and action == (1, 0):  # right
            x = self.env_size[0] - 1
            reward = self.reward_forbidden  
        elif y - 1 < 0 and action == (0, -1):   # up
            y = 0
            reward = self.reward_forbidden  
        elif x - 1 < 0 and action == (-1, 0):  # left
            x = 0
            reward = self.reward_forbidden 
        elif new_state == self.target_state:  # 到达目标
            x, y = self.target_state
            reward = self.reward_target
        elif new_state in self.forbidden_states:  # 进入禁止区域
            x, y = state  # 保持在原位
            reward = self.reward_forbidden        
        else:
            x, y = new_state
            reward = self.reward_step
            
        return (x, y), reward
        
    def _is_done(self, state):
        """
        检查是否到达终止状态（内部方法）
        
        参数:
            state: tuple (x, y) 状态
            
        返回:
            bool: 是否为终止状态
        """
        return state == self.target_state
    
    def render(self, animation_interval=None):
        """
        使用 matplotlib 渲染环境
        
        显示：
        - 网格
        - 禁止区域（黄色）
        - 目标区域（蓝色）
        - 智能体位置（蓝色星形）
        - 运动轨迹（绿色线条）
        
        参数:
            animation_interval: float 动画间隔时间（秒），None则使用默认值
        """
        if animation_interval is None:
            animation_interval = self.animation_interval
            
        if self.canvas is None:
            plt.ion()                             
            self.canvas, self.ax = plt.subplots()   
            self.ax.set_xlim(-0.5, self.env_size[0] - 0.5)
            self.ax.set_ylim(-0.5, self.env_size[1] - 0.5)
            self.ax.xaxis.set_ticks(np.arange(-0.5, self.env_size[0], 1))     
            self.ax.yaxis.set_ticks(np.arange(-0.5, self.env_size[1], 1))     
            self.ax.grid(True, linestyle="-", color="gray", linewidth="1", axis='both')          
            self.ax.set_aspect('equal')
            self.ax.invert_yaxis()                           
            self.ax.xaxis.set_ticks_position('top')           
            
            # 添加坐标标签
            idx_labels_x = [i for i in range(self.env_size[0])]
            idx_labels_y = [i for i in range(self.env_size[1])]
            for lb in idx_labels_x:
                self.ax.text(lb, -0.75, str(lb), size=10, ha='center', 
                           va='center', color='black')           
            for lb in idx_labels_y:
                self.ax.text(-0.75, lb, str(lb), size=10, ha='center', 
                           va='center', color='black')
            self.ax.tick_params(bottom=False, left=False, right=False, top=False, 
                              labelbottom=False, labelleft=False, labeltop=False)   

            # 绘制目标区域
            self.target_rect = patches.Rectangle(
                (self.target_state[0]-0.5, self.target_state[1]-0.5), 
                1, 1, linewidth=1, edgecolor=self.color_target, 
                facecolor=self.color_target)
            self.ax.add_patch(self.target_rect)     

            # 绘制禁止区域
            for forbidden_state in self.forbidden_states:
                rect = patches.Rectangle(
                    (forbidden_state[0]-0.5, forbidden_state[1]-0.5), 
                    1, 1, linewidth=1, edgecolor=self.color_forbid, 
                    facecolor=self.color_forbid)
                self.ax.add_patch(rect)

            # 初始化智能体和轨迹
            self.agent_star, = self.ax.plot([], [], marker='*', color=self.color_agent, 
                                           markersize=20, linewidth=0.5) 
            self.traj_obj, = self.ax.plot([], [], color=self.color_trajectory, linewidth=0.5)

        # 更新智能体位置和轨迹
        self.agent_star.set_data([self.agent_state[0]], [self.agent_state[1]])       
        traj_x, traj_y = zip(*self.traj)         
        self.traj_obj.set_data(traj_x, traj_y)

        plt.draw()
        plt.pause(animation_interval)
        if args.debug:
            input('press Enter to continue...')     
 
    def add_policy(self, policy_matrix):
        """
        在网格上可视化策略
        
        参数:
            policy_matrix: numpy.ndarray 策略矩阵，形状为 (num_states, num_actions)
                         每个元素表示在该状态下选择该动作的概率
                         
        示例:
            >>> policy = np.zeros((25, 5))  # 5x5网格，5个动作
            >>> policy[0, 1] = 1.0  # 状态0总是向右移动
            >>> env.add_policy(policy)
        """
        for state, state_action_group in enumerate(policy_matrix):    
            x = state % self.env_size[0]
            y = state // self.env_size[0]
            for i, action_probability in enumerate(state_action_group):
                if action_probability != 0:
                    dx, dy = self.action_space[i]
                    if (dx, dy) != (0, 0):
                        # 绘制动作箭头
                        self.ax.add_patch(patches.FancyArrow(
                            x, y, 
                            dx=(0.1+action_probability/2)*dx, 
                            dy=(0.1+action_probability/2)*dy, 
                            color=self.color_policy, 
                            width=0.001, 
                            head_width=0.05))
                    else:
                        # stay 动作用圆圈表示
                        self.ax.add_patch(patches.Circle(
                            (x, y), radius=0.07, 
                            facecolor=self.color_policy, 
                            edgecolor=self.color_policy, 
                            linewidth=1, fill=False))
    
    def add_state_values(self, values, precision=1):
        """
        在网格上显示状态值
        
        参数:
            values: numpy.ndarray 或 list 状态值数组，长度应为 num_states
            precision: int 小数精度
            
        示例:
            >>> values = np.random.randn(25)  # 5x5网格的随机值
            >>> env.add_state_values(values, precision=2)
        """
        values = np.round(values, precision)
        for i, value in enumerate(values):
            x = i % self.env_size[0]
            y = i // self.env_size[0]
            self.ax.text(x, y, str(value), ha='center', va='center', 
                        fontsize=10, color='black')


if __name__ == "__main__":
    # 简单测试
    print("GridWorld 环境测试")
    print("-" * 50)
    
    env = GridWorld(
        env_size=(5, 5),
        start_state=(0, 0),
        target_state=(4, 4),
        forbidden_states=[(1, 1), (2, 2)]
    )
    
    state, _ = env.reset()
    print(f"初始状态: {state}")
    
    # 执行几步
    actions = [(1, 0), (0, 1), (1, 0), (0, 1)]
    for i, action in enumerate(actions):
        next_state, reward, done, _ = env.step(action)
        print(f"步骤 {i+1}: 动作={action}, 状态={next_state}, "
              f"奖励={reward:.2f}, 完成={done}")
        if done:
            break
    
    print("\n测试完成！运行 GridWorld.ipynb 查看可视化示例。")
