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

## 详细流程（从数据采集到在线推理）
下面按时间顺序说明整条链路的用途、输入输出和关键命令，便于对照代码快速复现。

### 1. 数据采集（仅执行一次，离线）
- 启动采集节点：
  ```bash
  roslaunch uav_monocular_nav data_collection.launch
  ```
- 需要保证相机话题 `/camera/rgb/image_raw` 与状态话题（默认 `/odom`）在更新。
- `scripts/data_collector.py` 会定时读取图像和状态，根据 `config/obstacles.json` 中定义的柱体等障碍物几何生成真值安全掩码 `mask_safe_gt`，并保存到 `uav_data/images`、`uav_data/masks`、`uav_data/states`、`uav_data/goals`。
- 采集到设定数量（如几千张）后可手动 Ctrl+C 或等待节点自动结束。

### 2. 分割网络训练（安全区域二分类）
- 目的：使用环境真值掩码教网络从单目图预测安全区域。
- 运行示例：
  ```bash
  cd ~/catkin_ws/src/uav_monocular_nav/training
  python3 train_segmentation.py \
    --data_root ~/catkin_ws/src/uav_monocular_nav/uav_data \
    --out checkpoints/segmentation.pth
  ```
- 输入：RGB 图像；标签：采集阶段生成的 `mask_safe_gt`；损失：BCE/CE。
- 输出：分割模型权重 `segmentation.pth`。推理时仅依赖图像，环境几何不再参与。

### 3. 末端状态网络训练（PlannerNet + quintic）
- 目的：从「图像 + 当前状态」直接预测末端位置、速度、加速度和执行时间 T。
- 在 `training/train_planner.py` 中加载上一步 `segmentation.pth` 作为 backbone（可冻结，只取特征或预测掩码）。
- 运行示例：
  ```bash
  cd ~/catkin_ws/src/uav_monocular_nav/training
  python3 train_planner.py \
    --data_root ../uav_data \
    --seg_ckpt checkpoints/segmentation.pth \
    --out checkpoints/planner.pth
  ```
- 流程概要：
  1. 输入图像及当前状态 \(p_0, v_0, a_0\)，网络预测 \(\Delta p_{raw}, y_v, y_a\)，反归一化得到 \(p_T, v_T, a_T, T\)。
  2. 使用 `training/quintic.py` 以 \((p_0, v_0, a_0), (p_T, v_T, a_T, T)\) 求解三轴五次多项式，采样得到 `pos(t_k)`、`jerk(t_k)`、`dt`。
  3. 通过 `training/losses.py` 计算端到端损失：碰撞代价 \(J_c\)（将 `pos(t_k)` 投影到图像平面，用分割输出的危险概率采样累加）、平滑代价 \(J_s\)（`jerk` 二范数）、目标代价 \(J_g\)（末端与 goal 距离）。总损失 \(J = \lambda_s J_s + \lambda_c J_c + \lambda_g J_g\)。
- 输出：规划模型权重 `planner.pth`。

### 4. 在线推理 / 控制（仅用单目 + 状态）
- 启动推理：
  ```bash
  roslaunch uav_monocular_nav inference.launch
  ```
- `scripts/inference_node.py` 将订阅 `/camera/rgb/image_raw` 与 `/odom`，依次调用分割模型得到 `mask_safe_pred`，调用 PlannerNet 输出 \(p_T, v_T, a_T, T\)，再用五次多项式生成轨迹。
- 节点会选择轨迹前若干点作为参考，发布 `/cmd_vel`（或姿态/加速度）给 `scripts/simple_uav_controller.py`，并在 RViz 发布轨迹与掩码可视化。
- 在线阶段仅依赖单目图像和当前状态；环境障碍物几何只在训练阶段用于生成真值掩码和目标点。

## 注意事项
- 所有脚本使用 Python 3 并基于 ROS Noetic，需确保 `rospy`、`sensor_msgs`、`nav_msgs` 等依赖已安装。
- 训练脚本依赖 PyTorch；根据需求补充数据加载、模型细节和评估逻辑。
- 当前控制器与地图为简化示例，适合算法原型验证，可根据实际无人机与仿真接口进行替换。
