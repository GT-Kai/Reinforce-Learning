"""
Grid World 环境配置参数
"""

class Arguments:
    def __init__(self):
        # 环境尺寸 (rows, cols)
        self.env_size = (5, 5)
        
        # 起始状态 (0-indexed)
        self.start_state = (0, 0)
        
        # 目标状态
        self.target_state = (4, 4)
        
        # 禁止状态列表
        self.forbidden_states = [(1, 1), (2, 2), (3, 1)]
        
        # 动作空间: (dx, dy) 元组形式
        self.action_space = [
            (0, -1),   # up
            (1, 0),    # right
            (0, 1),    # down
            (-1, 0),   # left
            (0, 0)     # stay
        ]
        
        # 奖励设置
        self.reward_target = 10.0      # 到达目标的奖励
        self.reward_forbidden = -5.0   # 撞墙或进入禁止区域的惩罚
        self.reward_step = -0.1        # 每步的小惩罚
        
        # 可视化设置
        self.animation_interval = 0.1  # 动画间隔（秒）
        self.debug = False             # 是否开启调试模式

# 创建全局配置实例
args = Arguments()
