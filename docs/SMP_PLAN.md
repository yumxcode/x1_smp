# X1 29DOF SMP 训练方案（SMP_PLAN）

日期：2026-11-03 ｜ 目标：X1 29DOF 实现 **1 m/s、1-2 Hz/足低步频正常步幅行走**（walk10 档 1.0 m/s、walk05 档 0.5 m/s）
方法：SMP（Score-Matching Motion Priors, arXiv 2512.03028, SIGGRAPH 2026）｜平台：gradmotion 单卡 4090D 24G（¥5.76/h，硬顶 ¥150）

---

## 1. 开源代码核实结论（开工第一项，已核实）

**SMP 有完整开源实现，无需从头复现。** 证据链：

| 来源 | 内容 | 状态 |
|---|---|---|
| `github.com/xbpeng/MimicKit`（项目页 Code 按钮唯一指向，2413★，Apache-2.0） | 官方 SMP：`tools/diffusion_model/`（TinyMDM 先验训练）、`mimickit/learning/smp_agent.py / smp_model.py / diff_normalizer.py`、`mimickit/envs/smp_env.py`、`docs/README_SMP.md` | 活跃维护（2026-10 仍在 push） |
| `github.com/SUZ-tsinghua/smp` | **第三方 G1 29-DOF 端到端复现**：mjlab 1.6.0 ManagerBasedRlEnv + PPO + 乘性奖励 + GSI + 数据管线（CSV→窗口 NPZ→q01/q99 归一），四任务（Forward/Steering/Location/Getup） | 完整可参考，与 X1 同构 |
| 论文本身（v3, SIGGRAPH 2026） | 单卡 RTX 4090 口径：先验 400-800k iter ≈5h；单 clip 策略 ≈30min；HighKnees 任务 600M 样本 11.5h（AMP 6.2h） | 论文 §6.3 逐字 |

**可行性判定：可行**。论文算力口径（单卡 4090）与我们平台（4090D 同档）匹配；G1 复现证明 29-DOF 机器人构型路线成立（先验训练在重定向后的机器人关节空间数据上，非 SMPL 人形）。

## 2. 方法结构与论文口径（调研报告全文见 `.meta-agent/research/subtask-d8f7af11/report.md`）

两阶段结构：

**阶段一：运动先验预训练（TinyMDM，任务无关、冻结后复用）**
- 架构：2 层 transformer encoder（DiT 风格，adaLN 注入噪声级），4 头×64（隐维 256；G1 复现用 d_model 128×2 层），≈3M 参数
- 扩散：T=50 步、cosine schedule（squaredcos_cap_v2）、ε-预测、L1 loss、EMA 0.995（G1 复现关 EMA）
- 输入：H=10 帧运动窗口；特征 59 维/帧（G1）= root_pos(3)+root_rot(6D)+joint_pos(29)+ee_pos(5 末端×3)+root_lin_vel(3)+root_ang_vel(3)，锚定窗口末帧 pelvis 的 yaw-only 局部系
- 训练量：官方单 clip 50k iter / 多 clip 200k iter（batch 512，lr 1e-4~2e-4）；论文上限 400-800k iter ≈5h@4090
- 归一化：q01/q99 分位→[-1,1]（G1 复现口径，需宽分布数据计算统计量——窄分布会让 RL 探索期特征饱和 OOD，先验奖励恰在最需要处失效）

**阶段二：策略训练（PPO + SDS 奖励，先验冻结）**
- 奖励：`r_smp = exp(−(w_s/|K|)·Σ_{i∈K} MSE_i/μ_i)`，**K={22,15,8}**（T=50 时 {0.44N,0.30N,0.16N}），w_s=6，μ_i 为各扩散步误差的 running mean（adaptive normalization）
- 组合：论文加性 `r = 0.5·r_task + 0.5·r_smp`；G1 复现改**乘性** `r = task × r_smp`（免调权重比，防单边 farming）——本项目采用乘性（G1 29DOF 已验证），若失败回退论文加性
- GSI（生成式状态初始化）：冻结先验预采样窗口池，reset 时窗口末帧设为仿真初始状态、整窗回填在线特征缓冲（r_smp 从 step 0 有效；替代 RSI，训练不需原始数据）
- PPO：2×1024 MLP、SGD 1e-4（MimicKit 口径）/ rsl-rl（mjlab 口径）、4096 envs、仿真 50Hz（G1 复现）

**G1 实机实验（论文 §Real-World + 复现仓）**：29 关节、LaFAN1 重定向 G1 关节空间数据、非对称 actor-critic、域随机化、纯本体感知观测——证明"mocap→机器人构型→SMP"整条路线成立，X1 29DOF 直接同构。

## 3. 路线决策

**训练栈 = SUZ 复现仓移植 G1→X1（mjlab 1.6.0 栈），不是 super_mimic_F1 的 allmimic 底座。** 理由：
1. SMP 是标准 PPO + 冻结先验奖励；allmimic（SuperMimic）是世界模型+BPTT 栈，机制不匹配，硬插会扭曲两者
2. SUZ 仓已在 G1 29DOF 上端到端验证（先验+PPO+GSI+数据管线全套），移植面=资产+关节表+数据
3. X1 与 G1 同为 29 关节；X1 的 mjlab EntityCfg/PD 参数/关节表在 super_mimic_F1 `allmimic/robots/x1_29dof.py` 现成（1.2.0 API，需适配 1.6.0）
4. mjlab 1.6.0 依赖 torch>=2.7，gm 平台 IsaacSim image 自带 torch 2.7，兼容

**复用资产清单（只读源，拷贝/适配进本仓库）：**

| 资产 | 来源 | 用途 |
|---|---|---|
| `data/cmu_select_walk.py` | X1_29_AMP | AMASS 选段门（Froude 换算：cadence 95-140spm、0.7-1.9m/s、步长 0.50-0.85m、≥6s 直线稳态窗） |
| `tools/x1_arm_fix.py` + `x1_retarget_gates.py` | super_mimic_F1 | 重定向+臂修复（贴限位 67.2%→0.3%）+ 重定向验收门 |
| `motions/x1_phase2_train.npz`（281 段，50Hz，qpos 36 列=free7+joint29，名带 _walk/_turn 后缀）+ meta | super_mimic_F1 | **走类子集直接复用**（已过重定向验收门），qpos 与 SUZ 36 列 CSV 同构（注意 MuJoCo quat wxyz→xyzw） |
| `acceptance/rhythm_gates.py`（R1 步频 0.55-1.2Hz/足、R2 摆动≥0.28s、R3 占空≤0.75、R4 步长 0.35-0.65m@1m/s、L1 抬腿≥35mm） | X1_29_AMP | 策略步态验收（同款指标） |
| `scripts/gm_x1_train.py` + `scripts/gm/create_*.json`（PRO_20261102_002） | super_mimic_F1 | gm 平台接入模式（pip 装→验栈→训练→TB 通道→checkpoint 树外镜像） |
| SUZ smp 全仓 | third_party（已 clone） | 训练代码基座 |
| MimicKit | third_party（已 clone） | 官方实现对照（关键公式核查） |

## 4. 数据规模决策

- 论文口径：LaFAN1 locomotion 全集（段数论文未给）；数据效率实验 3 秒×3 段即可涌现速度调制 → 单步态先验对数据量不敏感，几十段远超充分
- **本项目决策：40-80 段过门走类段**（1 m/s 邻域为主、可含少量变速），全部来自：
  1. 首选：`x1_phase2_train.npz` 走类子集（已过重定向验收门）→ 新增**步态分布审计**（重定向后的步频/步速/步长须落在目标带：R1 邻域 0.55-1.2 Hz/足、速度 0.4-1.3 m/s、步长 0.35-0.65m@1m/s 自洽），越带剔除
     *修订记录（2026-11-03，drift 审查后对齐）*：数据审计速度带定稿 **0.40-1.30 m/s**（scripts/x1_data_lib.GAIT_BANDS 为准）——下限 0.4 支撑 walk05 档（0.5 m/s）以下的先验覆盖，上限 1.3 允许 1.0 档上方的少量快走段使分布不截尾；该带在见到 varied 受试者库分布之前确定，非见数据后放宽。phase2 走类子集实测仅 7/200 过门（速度 p50=0.39 m/s，63%<0.5），「直接复用」不成立，按预案转 AMASS 选段重定向补齐路线。
  2. 补齐：用 `cmu_select_walk.py` 选段门（人尺度门）+ GMR 重定向管线（arm_fix 含）+ **Froude XY 缩放 0.811**（√0.6578 骨盆缩放比；根因：ik config 只缩放根高未缩水平轨迹，X1 以人速行走超几何可行带导致 GMR 足部 IK 拖步，66 个 walk 家族 trial 全部被审计门拒绝——单变量根因修复，实测见 data/gait_audit_cmu_walk.json）从 AMASS（HF 镜像）下载 varied 受试者（111/132/136 等 20 族）重建
- held-out：~8-12 段走类不参与先验训练（用于先验重建质量抽查）
- 存储红线：原始 SMPL-H/中间 stageii 文件用完即删，仓库只留训练窗口 npz + norm_stats + meta（预计 <50MB）

## 5. 两阶段迭代数与预算表（4090D ¥5.76/h，硬顶 ¥150）

| 阶段 | 内容 | 迭代/时长口径 | 预算 |
|---|---|---|---|
| P0 平台栈 smoke | gm 任务：pip 装 mjlab 1.6.0 栈 + X1 Entity 加载 + G1 demo rollout + 4096 envs 显存验证 | 1h 内 | ≤¥6 |
| P1 数据管线（本地） | 走类子集筛选→步态分布审计→窗口化（X1 FK 末端）→q01/q99 归一统计 | 本地（0 平台成本）；若 FK 走平台 ≤1h | ≤¥6 |
| P2 先验预训练 | TinyMDM d_model 128×2 层、T=50、batch 512、stride 1 窗口；起步 200k 步，收敛判定 loss 平台+先验样本可视化 | 3-5h | ¥17-29 |
| P3 策略训练 v1 | Forward 任务（速度指令带采样训练），4096 envs，乘性奖励，GSI 开；5000-10000 iter | 3-4h | ¥17-23 |
| P4 策略迭代（最多 2 轮） | 按 4k-8k iter 日志/EVAL 检查点决策：续训/调参/回退加性奖励 | 6-8h | ¥35-46 |
| P5 sim2sim+渲染 | MuJoCo rollout 0.5/1.0 两档 + 指标表 + ≥2 段视频 | 1h | ≤¥6 |
| 缓冲 | 意外重跑 | — | ~¥20 |
| **合计** | | | **~¥100-130（硬顶 ¥150，要超先停下问）** |

每 4k-8k iter 核日志/EVAL，异常（reward 崩、v-err 发散、NaN）即停分析。

## 6. 预注册判定门（先行，不无限调参）

**成功门（walk10，1.0 m/s 指令）**：
- S-G1 存活：20s 无跌倒（4 场景）
- S-G2 节奏 R1∈[0.55,1.2] Hz/足（双足）
- S-G3 步长 R4∈[0.35,0.65]m 且与实际速度自洽（step_len ≈ v/f 自洽比 0.8-1.2）
- S-G4 速度跟踪：实测平均速度 ≥0.9 m/s（AMP 弧 v76 最好 0.83-0.87，此处刻意要求突破）
- S-G5 walk05（0.5 m/s 档）：R1 带内 + 不倒
- S-G6 sim2sim（MuJoCo 独立 rollout）复核 S-G1~S-G4 + ≥2 段渲染视频（1 m/s 目标步态 + walk05 档）

**部分成功**（与 AMP 弧同级）：速度 0.85-0.9 m/s 且 R 门全过 → 记录对照、不算失败。

**止损线**：
- L1：P4 预算耗尽前若 v-err 平台 <0.85 m/s 且两轮迭代无改善（±0.02 m/s 内）→ 停
- L2：R2/R3 卡死再现（AMP 弧同款失败模式：摆动<0.28s 或占空>0.75 在任何检查点连续出现且续训不修复）→ 停
- L3：先验训练不收敛（loss 不降）或 SDS 奖励在策略训练中恒 0/恒 1（失效）→ 修一次（检查归一化/特征对齐），不果 → 停
- 触发止损即写报告，不追加预算

**假设（预注册）**：H1：显式扩散先验（SMP）比 AMP 判别器更能锚定低步频步态的节奏与步幅几何，突破 AMP 弧 0.87 m/s 平台达到 ≥0.9 m/s @ R1∈[0.55,1.2]Hz。证否条件：L1/L2 触发。

## 7. 风险清单

| 风险 | 概率 | 缓解 |
|---|---|---|
| mjlab 1.6.0 在平台 image 兼容性（warp/CUDA 驱动） | 中 | P0 smoke 先行；失败则降级 mujoco 3.11→平台已验证组合或改用 1.5.x 兼容版 |
| X1 EntityCfg 1.2.0→1.6.0 API 移植 | 中 | SUZ 引用 mjlab 自带 G1 cfg 作模板；X1 xml+PD 表从 x1_29dof.py 平移 |
| SUZ 复现仓质量（course project） | 中 | 关键公式（K/w_s/归一化/exp 形式/GSI）与 MimicKit 官方逐项对照核查 |
| 4090D 24G 显存不足 4096 envs | 低 | G1 同构已跑通；不足则降 2048 envs（G1 复现仓同参数支持） |
| 低步频 1 m/s 数据不足 | 低 | 选段门正是为此设计（0.7-1.9 m/s 带）；x1_phase2 走类子集先行审计 |
| AMP 弧失败模式复现（R2/R3 卡死） | 中 | SMP 机制不同（显式先验 vs 对抗）；止损线 L2 预注册 |

## 8. 交付物

1. `docs/SMP_PLAN.md`（本文档）
2. 训练窗口 npz + norm_stats + meta（步态分布审计报告）
3. gm 平台训练任务与 checkpoint（先验 `prior_*.pt` + 策略 `policy_*.pt`）
4. sim2sim 指标表（0.5/1.0 两档 × R1-R4/L1/存活/速度）+ ≥2 段视频
5. `docs/SMP_REPORT.md`（方案、数据与存储账目、验收门结果、训练曲线、对照 AMP 弧 v76、限制与下一步）
