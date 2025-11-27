"""
GridWorld 环境测试脚本
测试优化后的 GridWorld 功能
"""
from pathlib import Path
import sys
project_root = Path("./..")
sys.path.insert(0, str(project_root)) 
import numpy as np
from gridworld import GridWorld


def test_basic_functionality():
    """测试基本功能"""
    print("=" * 60)
    print("测试 1: 基本功能")
    print("=" * 60)
    
    env = GridWorld(
        env_size=(5, 5),
        start_state=(0, 0),
        target_state=(4, 4),
        forbidden_states=[(1, 1), (2, 2)]
    )
    
    # 测试 reset
    state, info = env.reset()
    print(f"✓ Reset 成功: 初始状态 = {state}")
    assert state == (0, 0), "初始状态错误"
    
    # 测试 step
    actions = [(1, 0), (0, 1), (1, 0), (0, 1)]
    print(f"\n执行动作序列: {actions}")
    
    for i, action in enumerate(actions):
        next_state, reward, done, info = env.step(action)
        print(f"  步骤 {i+1}: 动作={action}, 状态={next_state}, "
              f"奖励={reward:.2f}, 完成={done}")
        
    print("\n✓ 基本功能测试通过！\n")


def test_boundary_collision():
    """测试边界碰撞"""
    print("=" * 60)
    print("测试 2: 边界碰撞处理")
    print("=" * 60)
    
    env = GridWorld(
        env_size=(3, 3),
        start_state=(0, 0),
        target_state=(2, 2),
        forbidden_states=[]
    )
    
    state, _ = env.reset()
    
    # 测试向上碰撞
    state, reward, done, _ = env.step((0, -1))  # up
    print(f"向上移动（碰撞）: 状态={state}, 奖励={reward:.2f}")
    assert state == (0, 0), "边界碰撞处理错误"
    assert reward < 0, "碰撞应有负奖励"
    
    # 测试向左碰撞
    state, reward, done, _ = env.step((-1, 0))  # left
    print(f"向左移动（碰撞）: 状态={state}, 奖励={reward:.2f}")
    assert state == (0, 0), "边界碰撞处理错误"
    
    print("\n✓ 边界碰撞测试通过！\n")


def test_forbidden_states():
    """测试禁止区域"""
    print("=" * 60)
    print("测试 3: 禁止区域处理")
    print("=" * 60)
    
    env = GridWorld(
        env_size=(3, 3),
        start_state=(0, 0),
        target_state=(2, 2),
        forbidden_states=[(1, 0)]
    )
    
    state, _ = env.reset()
    
    # 尝试进入禁止区域
    state, reward, done, _ = env.step((1, 0))  # 向右进入 (1,0)
    print(f"尝试进入禁止区域 (1,0): 状态={state}, 奖励={reward:.2f}")
    assert state == (0, 0), "应该停留在原位"
    assert reward < 0, "进入禁止区域应有负奖励"
    
    print("\n✓ 禁止区域测试通过！\n")


def test_target_reaching():
    """测试到达目标"""
    print("=" * 60)
    print("测试 4: 到达目标")
    print("=" * 60)
    
    env = GridWorld(
        env_size=(3, 3),
        start_state=(1, 1),
        target_state=(2, 1),
        forbidden_states=[]
    )
    
    state, _ = env.reset()
    print(f"初始状态: {state}")
    
    # 向右移动到目标
    state, reward, done, _ = env.step((1, 0))
    print(f"到达目标: 状态={state}, 奖励={reward:.2f}, 完成={done}")
    
    assert state == (2, 1), "应该到达目标"
    assert reward > 0, "到达目标应有正奖励"
    assert done == True, "应该标记为完成"
    
    print("\n✓ 目标到达测试通过！\n")


def test_trajectory_recording():
    """测试轨迹记录"""
    print("=" * 60)
    print("测试 5: 轨迹记录")
    print("=" * 60)
    
    env = GridWorld(
        env_size=(3, 3),
        start_state=(0, 0),
        target_state=(2, 2),
        forbidden_states=[]
    )
    
    state, _ = env.reset()
    print(f"初始轨迹长度: {len(env.traj)}")
    assert len(env.traj) == 1, "初始轨迹应只有起点"
    
    # 执行几步
    for _ in range(3):
        env.step((1, 0))
    
    print(f"执行3步后轨迹长度: {len(env.traj)}")
    assert len(env.traj) > 1, "轨迹应该被记录"
    
    print("\n✓ 轨迹记录测试通过！\n")


def test_action_space():
    """测试动作空间"""
    print("=" * 60)
    print("测试 6: 动作空间")
    print("=" * 60)
    
    env = GridWorld()
    
    print(f"动作空间: {env.action_space}")
    assert len(env.action_space) == 5, "应该有5个动作"
    
    # 验证所有动作都是元组
    for action in env.action_space:
        assert isinstance(action, tuple), "动作应该是元组"
        assert len(action) == 2, "动作应该是2D元组"
    
    print("✓ 所有动作格式正确")
    print("\n✓ 动作空间测试通过！\n")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始 GridWorld 优化测试")
    print("=" * 60 + "\n")
    
    tests = [
        test_basic_functionality,
        test_boundary_collision,
        test_forbidden_states,
        test_target_reaching,
        test_trajectory_recording,
        test_action_space
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ 测试失败: {e}\n")
            failed += 1
        except Exception as e:
            print(f"✗ 测试出错: {e}\n")
            failed += 1
    
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"通过: {passed}/{len(tests)}")
    print(f"失败: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 所有测试通过！GridWorld 优化成功！")
    else:
        print(f"\n⚠️  有 {failed} 个测试失败，请检查代码。")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all_tests()
