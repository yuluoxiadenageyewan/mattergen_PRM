# 残差强化学习在扩散模型微调中的应用

## 1. 什么是残差强化学习

### 1.1 经典定义

残差强化学习（Residual Reinforcement Learning）最早由 Silver et al. (2018) 和 Johannink et al. (2019, "Residual Reinforcement Learning for Robot Control") 提出。核心思想：

> 给定一个已有的基础策略 π_base（可以是手工控制器、预训练模型），不直接学习完整策略，而是学习一个**残差修正** δπ，最终策略为：
>
> **π_final(s) = π_base(s) + α · δπ(s)**

类比：你已经有了一个能走路的机器人控制器（π_base），残差 RL 不是从零学走路，而是学"怎么走得更好"——只学修正量。

### 1.2 与标准 RL 的区别

| 维度 | 标准 RL | 残差 RL |
|------|---------|---------|
| 学习目标 | 从零学习完整策略 π | 只学习修正量 δπ |
| 初始行为 | 随机策略（通常很差） | 等同于基础策略（已经能工作） |
| 搜索空间 | 完整策略空间 | 残差空间（小得多） |
| 训练稳定性 | 容易崩溃 | α 控制变化幅度，天然稳定 |
| 样本效率 | 低（需要大量探索） | 高（搜索空间压缩） |

### 1.3 与 KL 正则化的本质区别

这是一个容易混淆的点。两者都约束策略不偏离基础模型，但机制完全不同：

**KL 正则化**：
- 是一个**损失项**：L = L_task + σ · KL(π_agent || π_prior)
- 在计算出完整输出后，再惩罚偏离
- 是**事后约束**（soft constraint）——输出已经产生，只是梯度拉回来
- 如果 σ 设小了，agent 可以在某些 step 产生任意远的输出

**残差约束**：
- 是**结构性约束**（hard constraint）：输出 = π_prior + α · (π_agent - π_prior)
- α=0.1 时，无论 agent 的参数怎么变，有效输出最多偏离 prior 的 10%
- 是**事前约束**——输出在产生的那一刻就被结构性地限制住了
- 不存在"约束失效"的可能

**直观理解**：
- KL 正则化像给汽车装了弹性绳，拉着它不要开太远——但绳子可以被拉断
- 残差约束像把方向盘的转角物理限制在 ±10°——结构上不可能转更多

---

## 2. 为什么扩散模型微调适合用残差 RL

### 2.1 扩散模型的特殊结构

MatterGen 是一个去噪扩散模型。在每个时间步 t，模型预测三个场：
- **pos**：原子坐标噪声预测（连续值，ℝ³）
- **cell**：晶胞参数噪声预测（连续值，ℝ³ˣ³）
- **atomic_numbers**：原子类型 logits（分类值）

关键观察：**这些输出是可加的**。两个去噪预测可以直接做线性组合：

```
ε_eff = ε_prior + α · (ε_agent - ε_prior)
     = (1 - α) · ε_prior + α · ε_agent
```

这和机器人控制中"力矩可加"是一样的道理——残差 RL 最初就是为力矩可加的系统设计的。

### 2.2 预训练 prior 已经很强

MatterGen 在 Materials Project 上预训练，已经能生成有效的晶体结构。RL 微调的目标不是"学会生成晶体"，而是"生成满足特定性质约束的晶体"——这是一个**微调问题，不是从零学习的问题**。

残差 RL 恰好是为这种场景设计的：
- π_base（prior）已经能生成合理结构
- δπ 只需要学习"怎么调整去噪预测，使生成的晶体 formation energy 更低"
- 搜索空间从"所有可能的去噪函数"压缩到"prior 附近的小范围修正"

### 2.3 去噪预测的场独立性

MatterGen 的三个输出场（pos, cell, atomic_numbers）在扩散过程中是**独立腐蚀、独立预测**的：
- 每个场有自己的 corruption 过程
- 每个场有自己的损失函数
- 残差可以逐场施加

这意味着残差融合在每个场上都是合理的——不存在场间耦合导致残差失效的问题。

### 2.4 α 退火提供渐进式探索

```python
alpha = alpha_init + (1.0 - alpha_init) * step / warmup_steps
# alpha: 0.1 → 1.0 over 60 RL steps
```

训练早期（α=0.1）：
- agent 只能做微小修正，生成的晶体和 prior 几乎相同
- 探索安全，不会生成无效结构
- 奖励信号主要来自 prior 本身的质量差异

训练后期（α→1.0）：
- agent 获得完全控制权
- 此时 agent 已经通过小步修正积累了经验
- 可以做更大的结构性改变

这种渐进式放权在标准 RL 中没有自然的对应物。KL 正则化的 σ 不具备这种"结构性渐进"特性。

---

## 3. 残差策略适用于稀疏信号还是稠密信号？

### 3.1 简短回答

**残差策略在稀疏奖励下优势更大，但在稠密奖励下同样适用**。它不是专门为某种奖励设计的——它解决的是"搜索空间太大"的问题，而稀疏奖励让这个问题更严重。

### 3.2 详细分析

#### 稀疏奖励下的优势

稀疏奖励的核心困难是：大部分采样得到零或接近零的奖励，梯度信号极弱。

| 问题 | 标准 RL 的困境 | 残差 RL 的解决 |
|------|---------------|---------------|
| 探索效率低 | 随机策略采到有效样本的概率极低 | 从 prior 出发，初始就能生成合理样本 |
| 梯度消失 | reward≈0，adv·loss≈0，学不动 | 搜索空间小，少量有效信号足以引导 |
| 训练不稳定 | 偶尔的高奖励导致策略剧烈跳变 | α 限制变化幅度，不可能剧烈跳变 |
| 模式崩溃 | 策略可能坍缩到单一高奖励模式 | prior 的多样性通过 (1-α) 被保留 |

**关键洞察**：在稀疏奖励下，标准 RL 需要大量无效探索来"碰运气"找到高奖励区域。残差 RL 从 prior 出发，prior 本身就处于"合理区域"，只需要在合理区域的邻域内搜索。这相当于把搜索范围从"整个策略空间"缩小到"prior 附近的邻域"——在稀疏奖励下，这种搜索空间的压缩是决定性的。

#### 稠密奖励下的表现

在稠密奖励下，残差 RL 同样工作，但优势不如稀疏场景明显：
- 稠密奖励本身提供了足够的梯度信号，标准 RL 也能学
- 残差 RL 的初始优势（warm start from prior）在几轮后会被标准 RL 追上
- 但残差 RL 仍然提供更稳定的训练和更快的收敛

#### 量化对比（概念性）

```
                   稀疏奖励        稠密奖励
标准 RL 收敛速度:    很慢/不收敛     正常
残差 RL 收敛速度:    正常            略快于标准 RL

效率提升比:         >>>             >
```

### 3.3 MatInvent 中的具体情况

MatInvent 的奖励天然是**稀疏的**：
1. 大部分随机生成的晶体在过滤阶段就被淘汰（无效结构、不稳定等）
2. 通过过滤的结构，大部分 formation energy 不在目标范围 [-3.5, -1.0] 内
3. 加上组分约束（如必须含 Li），有效样本更少
4. 最终能获得 reward > 0.6 的样本可能只有个位数

在这种高度稀疏的奖励环境下，残差 RL 的搜索空间压缩效果最为显著。

---

## 4. MatInvent 中的具体实现

### 4.1 实现位置

`models/mattergen/pl_module.py` — `calc_residual_sample_loss` 方法

### 4.2 核心代码逻辑

```python
def calc_residual_sample_loss(self, noised_input, prior_model, alpha=1.0):
    noisy_batch, batch, t = noised_input

    # Agent 前向传播（梯度在此流动）
    agent_pred = self.diffusion_module.model(noisy_batch, t)

    # Prior 前向传播（无梯度）
    with torch.no_grad():
        prior_pred = prior_model.diffusion_module.model(noisy_batch, t)

    # 残差融合：ε_eff = ε_prior + α · (ε_agent - ε_prior)
    effective_output = agent_pred.replace(
        pos=prior_pred["pos"] + alpha * (agent_pred["pos"] - prior_pred["pos"]),
        cell=prior_pred["cell"] + alpha * (agent_pred["cell"] - prior_pred["cell"]),
        atomic_numbers=prior_pred["atomic_numbers"] + alpha * (
            agent_pred["atomic_numbers"] - prior_pred["atomic_numbers"]
        ),
    )

    # 在有效输出上计算去噪损失
    loss, metrics = self.sample_loss_fn(
        score_model_output=effective_output, ...
    )
    return loss, agent_pred, prior_pred
```

### 4.3 梯度流分析

```
effective_output = prior_pred + α · (agent_pred - prior_pred)

∂L/∂θ_agent = ∂L/∂ε_eff · ∂ε_eff/∂agent_pred · ∂agent_pred/∂θ_agent
            = ∂L/∂ε_eff · α · ∂agent_pred/∂θ_agent
```

关键：**梯度被 α 缩放**。α=0.1 时，梯度只有标准 RL 的 10%。这不是问题——这正是残差 RL 的设计意图：早期小步学习，避免策略跳变。随着 α 增大，梯度逐渐恢复正常。

### 4.4 与其他组件的协同

残差 RL 与本次其他改进互为补充：

| 组件 | 残差 RL 的协同作用 |
|------|--------------------|
| **EMA 优势归一化** | 残差限制了输出范围，使得优势估计更稳定（方差更小） |
| **自适应 KL 系数** | 残差已经提供结构性约束，KL 作为额外的软约束，σ 可以设得更小 |
| **优先经验回放** | 稀疏奖励下高奖励样本少，残差 RL + 优先回放双重缓解这一问题 |
| **时间步子采样** | 残差在每个时间步独立施加，与时间步采样正交、互不干扰 |

---

## 5. 局限性与注意事项

### 5.1 Prior 质量依赖

残差 RL 的前提是 prior 足够好。如果 prior 本身生成的结构质量很差，残差修正很难挽救。对于 MatterGen 这样在大规模数据上预训练的模型，这个前提成立。

### 5.2 α 退火策略的选择

线性退火是最简单的策略，但不一定是最优的：
- 如果 prior 已经很接近目标（如 formation energy 本身就是预训练目标之一），可以用更快的退火
- 如果目标和预训练差距大，应该用更慢的退火
- 也可以根据 reward 的变化率自适应调整 α

### 5.3 额外的前向传播开销

残差 RL 需要同时对 agent 和 prior 做前向传播。这意味着每个时间步的计算量翻倍。但配合时间步子采样（50/1000），总计算量反而降低了约 10 倍：
```
标准 RL:  1000 timesteps × 1 forward = 1000 次前向传播
残差 RL:  50 timesteps × 2 forward  = 100 次前向传播
```

### 5.4 不适用的场景

- 当 prior 和目标任务完全不相关时（如让语言模型 prior 去做图像生成）
- 当输出空间不支持线性组合时（如离散动作空间、图结构输出）
- 注意：atomic_numbers 是 logits（连续值），不是离散 one-hot，所以线性组合仍然合理

---

## 6. 参考文献

1. Johannink, T., et al. (2019). "Residual Reinforcement Learning for Robot Control." ICRA.
2. Silver, T., et al. (2018). "Residual Policy Learning." arXiv:1812.06298.
3. Ho, J., et al. (2020). "Denoising Diffusion Probabilistic Models." NeurIPS.
4. Ziebart, B. D. (2010). "Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy." PhD thesis, CMU. (KL 正则化的理论基础)
5. Black, K., et al. (2024). "Training Diffusion Models with Reinforcement Learning." ICLR.
