# UAV Monocular Navigation 项目说明

本仓库提供一个 ROS Noetic 环境下的示例包 **uav_monocular_nav**，用于基于单目相机的安全区域分割与端到端末端状态规划。代码主要作为骨架，方便在 RViz 可视化和离线训练基础上快速扩展。

## 主要功能
- **数据采集**：`scripts/data_collector.py` 订阅相机与无人机状态，结合 `config/obstacles.json` 的虚拟圆柱体障碍物自动生成安全 mask，并保存图像、mask、状态和目标。
- **分割网络训练**：`training/train_segmentation.py` 使用 U-Net 结构和 BCE 损失训练安全区域二分类模型，权重保存为 `segmentation.pth`。
- **规划网络训练**：`training/train_planner.py` 将分割特征输入 PlannerNet，预测末端状态并通过五次多项式（`training/quintic.py`）生成轨迹，使用 `training/losses.py` 中的掩码碰撞、平滑和目标代价进行优化，权重保存为 `planner.pth`。
- **在线推理**：`scripts/inference_node.py` 运行分割与规划推理，生成短期轨迹并通过 `scripts/simple_uav_controller.py` 将期望点转为速度指令，同时在 RViz 发布轨迹与可视化标记。

## 目录结构
```
uav_monocular_nav/
  CMakeLists.txt
  package.xml
  launch/
    data_collection.launch
    inference.launch
  scripts/
    data_collector.py
    inference_node.py
    simple_uav_controller.py
  training/
    train_segmentation.py
    train_planner.py
    models.py
    quintic.py
    losses.py
    dataset_tools.py
  config/
    camera_info.yaml
    obstacles.json
  rviz/
    nav_visualization.rviz
```

## 快速使用
1. 将本仓库放入 catkin 工作空间的 `src` 目录，执行 `catkin_make` 构建。
2. 准备虚拟障碍物参数（编辑 `config/obstacles.json`），启动数据采集：
   ```bash
   roslaunch uav_monocular_nav data_collection.launch
   ```
3. 按 `training/dataset_tools.py` 约定整理数据集后运行训练脚本：
   ```bash
   python3 training/train_segmentation.py
   python3 training/train_planner.py
   ```
4. 加载训练好的权重进行在线推理并在 RViz 中查看轨迹：
   ```bash
   roslaunch uav_monocular_nav inference.launch
   ```

## 注意事项
- 所有脚本使用 Python 3 并基于 ROS Noetic，需确保 `rospy`、`sensor_msgs`、`nav_msgs` 等依赖已安装。
- 训练脚本依赖 PyTorch；根据需求补充数据加载、模型细节和评估逻辑。
- 当前控制器与地图为简化示例，适合算法原型验证，可根据实际无人机与仿真接口进行替换。
