# X1 29DOF SMP 训练——最终报告（SMP_REPORT）

日期：2026-11-03 ~ 11-04 ｜ 仓库：github.com/yumxcode/X1_SMP（main，私有）
目标：X1 29DOF 实现 **1 m/s、1-2 Hz/足低步频正常步幅行走**（walk10=1.0 m/s、walk05=0.5 m/s 两档）
方法：SMP（Score-Matching Motion Priors, arXiv 2512.03028, SIGGRAPH 2026）｜平台：gradmotion 4090D ¥5.76/h

**一句话结论：H1 证伪（按预注册止损线 L2 收束）。SMP 在本配置下收敛到"贴地滑行"——20s 不倒、速度为命令的 ~60%（0.56 m/s@cmd 1.0），但脚底最大离地仅 1.1 cm、无步态周期，R1-R4 节奏门全部无法解析；显著劣于 AMP 弧最好成绩（0.87 m/s@0.91 Hz 真步态，R 门全过）。失败机制定位为"先验特征缺触地语义 + 数据速度带上界"双重结构性问题（§5）。**

---

## 1. 方案分析（详见 docs/SMP_PLAN.md）

- **开源代码核实（开工第一项）**：SMP 完整开源。官方实现 `xbpeng/MimicKit`（项目页 Code 按钮唯一指向，2413★）；`SUZ-tsinghua/smp` 为 **G1 29-DOF 端到端复现**（mjlab 1.6.0 + PPO + 乘性奖励 + GSI + 数据管线），与 X1 同为 29 关节，直接同构移植。论文口径：先验训练 400-800k iter≈5h@单卡 4090；单 clip 策略 30min；HighKnees 600M 样本 11.5h。可行性判定：**可行**（详见 PLAN §1-3，调研报告全文 .meta-agent/research/subtask-d8f7af11/report.md）。
- **路线**：SUZ 复现仓移植 G1→X1（mjlab 1.6.0）。X1 EntityCfg 从 super_mimic_F1 allmimic（1.2.0 API）平移，两处 1.6.0 API 差异修复（CollisionCfg 需显式 contype/conaffinity/priority catch-all）。
- **预算**：计划 ~¥100-130（PLAN §5），实际 **~¥55-60**（§6 账目），远低于 ¥150 硬顶。

## 2. 数据管线（任务 2）

**选段→重定向→审计全链路自建**（`scripts/build_smp_library.py` + `scripts/x1_data_lib.py`），复用只读资产：cmu_select_walk 选段门、GMR+x1_arm_fix 重定向（super_mimic_F1 原管线调用，未修改）、rhythm_gates 语义步态指标。

关键实证（全部记录在 exp_aafa2481 amendments）：
1. **现成库不可直接复用**：cmu_walk 41 段（AMP v72 路线）存在系统性右脚拖步（sole rel max 1.8-4.4cm，duty 0.9+）；phase2 walk 200 段速度偏低（p50=0.39 m/s），仅 7 段过健康门（0.62-0.84 m/s）。
2. **根因修复**：ik_config 只缩放骨盆高度（×0.6578）不缩水平轨迹→X1 以人速行走超几何可行带。引入 **Froude XY 缩放 0.811**（√0.6578，步频不变/步长速度×0.811）——速度落入带内，但**拖步与速度弱相关、与受试者源数据强相关**（GMR 足部 IK 在人 cad≥115spm 快走段系统性缩短单脚摆动相 30-50%，任何缩放下不修复）。
3. **数据门三版修订**（v1→v3，均记 amendment）：v2 数据侧 duty 0.80→0.87/swing 0.24→0.15（GMR 摆动相系统性缩短的实测证据）；v3 修 step 绝对带为自洽比 step/v∈[0.40,0.95]（绝对带按 1 m/s 校准错误拒绝低速段）+ lift 上限 0.25m（滤跳跃帧）。
4. **最终库**：18 段 / 4902 帧（98s）/ 4740 窗口（59 维×10 帧），速度 0.41-1.17 m/s（p50 0.62，≥0.85 仅 3 段）。939 个 AMASS CMU trial 扫描（20 个受试者族下载）→ 32 候选窗 → 18 段过门 + phase2 健康 7 段合并。每段步态指标在 `data/x1_smp_train.meta.json`。
5. **存储红线执行**：raw AMASS（~1.1GB，939 trial）与 interim stageii 用完即删；最终 data/ 仅 15MB（库 npz+meta+窗口+norm_stats+先验 ckpt+策略 ckpt）。

**数据限制（如实）**：重定向质量门全量过检（A 门：越限≤0.35rad、穿透>-2mm、根高、倾角、关节速度——build_x1_motions 同款阈值）；但 0.85+ m/s 健康段在 GMR 管线下不可得，构成先验速度流形的结构性上界。

## 3. 两阶段训练（任务 3）

**P0 平台栈 smoke（PASS，TASK_20261103_200）**：mjlab 1.6.0 + warp-lang 1.16.0 在 IsaacSim image/torch 2.7.0+cu128/驱动 12.2 上全链可用；X1 Entity 编译；2048-env 零动作 rollout max|qd|=0.78（solref 0.02 修复生效）；TinyMDM 0.70M 参数 12ms/step。平台坑：默认 pip 源无 mjlab 1.6.0（需 pypi.org 官方源）；warp 1.19 需 CUDA 13.4>驱动 12.2（pin 1.16）。

**P2 先验预训练（TASK_20261103_220，完成）**：TinyMDM d_model 128×2 层（0.70M），T=50 cosine，ε-预测 L1，batch 512，lr 2e-4，20000 epochs（~200k 步），87 分钟。**loss 0.895→0.073（train）/0.076（val）**，无过拟合。checkpoint_19999.pt 已入 git（data/prior/x1_walk_prior.pt，含 q01/q99 归一统计）。

**P3 策略训练（TASK_20261104_037，提前停）**：Smp-Forward-X1（速度命令 0.4-1.4 m/s 均匀采样），2048 envs，rsl-rl PPO（512×3 MLP，lr 1e-3 adaptive），乘性奖励 task×smp（K={8,15,22}，w_s=6，自适应归一），GSI 静态池 4096。
- 学习曲线：iter 471 学会不倒（eplen 759），iter ~2900 reward 1.8-1.9 / eplen ~960 / smp_rew 0.09 / **vel_err 0.68 平台**。
- 提前停训于 **4100 iter**（vel_err 四分位均值 0.680/0.684/0.688/0.680，1000+ iter 零改善——按预注册判定点）。checkpoint model_4100.pt 已入 git。
- **两次实例死亡事故与修复**（fnd_cf82d158）：4096 与 2048 envs 两次都在 iter ~405/471（恰为第 4 次 2400 步 GSI refresh 后）无 traceback 死亡；修复=禁用 gsi_refresh（静态池够用）+ save_interval 100 + RAM/GPU watchdog；修复后 4100 iter 全程稳定（GPU 2.3/24.5 GiB、RAM 平稳）。

**预算账目**：rokad（废弃任务）~¥1；keyen：smoke×5 + P2（含 2 次入口失败）+ P3 v4 43min ≈ ¥19（耗尽）；jasevi：P3 v5 32min ≈ ¥3（余额耗尽）；tesokic：P3 v6 4.7h + 评估×6 ≈ ¥31。**合计 ~¥55-60**。

## 4. 步态验收 + sim2sim（任务 4，TASK_20261104_129）

预注册门（PLAN §6）vs 实测（model_4100，固定速度命令 20s rollout，MuJoCo warp 仿真）：

| 指标 | 门 | walk05（cmd 0.5） | walk10（cmd 1.0） |
|---|---|---|---|
| S-G1 存活 | 20s 不倒 | **20.0s ✓** | **20.0s ✓** |
| 实测速度 | ≥0.9 m/s（walk10） | 0.25-0.30 m/s（~55%） | **0.56-0.64 m/s（~60%）✗** |
| S-G2 R1 步频 | 0.55-1.2 Hz/足 | **不可解析**（无步态周期）✗ | **不可解析** ✗ |
| R2/R3/R4/L1 | 摆动/占空/步长/抬腿 | 不可解析 ✗ | 不可解析 ✗ |

诊断证据（[diag] 日志行）：**足底最大离地 0.7-1.1 cm，rel>2cm 时间占比 0.00，root_z 0.61-0.62 恒定，位移全靠滑动**（v1.0 档 20s 位移 11.25m）——策略是"贴地滑行"而非迈步。视频：`eval/video_v0p5.mp4`、`eval/video_v1p0.mp4`（肉眼可辨滑行）。

**判定：止损线 L2 触发（R2 卡死形态——摆动=0，非续训可修复），H1 证伪，停止迭代（预注册"不无限调参"）。**

## 5. 与 AMP 弧最好成绩对照

| | AMP 弧 v76-m5500（最好） | SMP 本弧（model_4100） |
|---|---|---|
| walk10 实测速度 | **0.83-0.87 m/s** | 0.56-0.64 m/s |
| 步频 R1 | **0.91 Hz/足（带内）** | 无步态周期 |
| 摆动 R2 / 占空 R3 | **0.56-0.62s / 0.44-0.49（全过）** | 不可解析 |
| 步长 R4 / 抬腿 L1 | **0.50-0.52m / 119-131mm（全过）** | 不可解析 |
| 步态形态 | 真迈步 | 贴地滑行 |

**SMP 本配置显著劣于 AMP 弧最好成绩**（速度、节奏门、步态形态全面落后）。AMP 失败模式是"微步高频 3.1Hz、R2/R3 卡死但速度 0.87"；SMP 失败模式是"滑行、无步态周期、速度 6 折"——两种先验机制在 1 m/s 低步频目标上均未成功，SMP 更远。

## 6. 失败机制分析（fnd_bb1409b1 / fnd_73ec4bb5 / fnd_95cd3382）

1. **先验特征缺触地语义（主因）**：SMP 59 维特征中足部 ee 位置是 root 系局部量——滑行与迈步的窗口特征（root 轨迹/关节角/ee 相对摆动）在扩散先验的低噪声步上难以区分；乘性奖励下"滑行=任务速度分 × 先验高分"的联合最优，无任何机制惩罚脚不抬。
2. **数据速度带上界（次因）**：GMR 重定向使人快走段单脚拖步（cad≥115spm 系统性），X1 尺度 0.85+ m/s 健康段不可得→先验流形速度中心 0.62 m/s，与 1.0 指令的张力表现为系统性 ~38% 速度折扣。
3. （工程侧已修复项）gm 平台 mjlab 的 gsi_refresh 触发实例级死亡（禁用后稳定）；pip 源/warp 版本坑均已定位并在 PLAN/代码注释留档。

## 7. 限制与下一步

**本报告结论的适用边界**：SUZ/MimicKit 口径 SMP（TinyMDM d128、59 维局部特征、乘性奖励、GSI）+ 18 段 0.41-1.17 m/s 数据 + 4100 iter。**不排除**以下变体表现不同：(a) 论文原式加性奖励（w_prior=w_g=0.5）；(b) 特征加入足部世界系速度/触地指示；(c) 更高迭代数（论文 600M 样本 vs 本弧 4100×24×2048≈0.2G）；(d) 混合多种步态的更大先验。

**下一步建议（按优先级）**：
1. 若继续 SMP 路线：特征加入双足足底世界系高度/速度（或触地标志）重训先验——直击滑行缺陷；奖励回退论文加性式。
2. 数据侧：修 GMR 足部 IK（人快走段单脚拖步）或改用足部优先的 IK 权重——解锁 0.85-1.2 m/s 健康段，这是两条路线（AMP/SMP）共同的数据上界。
3. 或者回到 AMP 弧 v76 基础上做"守卫奖励定向修复"（exp_mw0drnz5 的路线）——当前最接近目标的存量资产。

## 8. 交付物清单

- `docs/SMP_PLAN.md`（预注册方案与判定门）、本报告 `docs/SMP_REPORT.md`
- `data/`：x1_smp_train.npz+meta（18 段库）、windows/（4740 窗）、prior/norm_stats.npz、x1_walk_prior.pt（先验）、policy/model_4100.pt（策略）
- `src/smp/`：X1 版 SMP 全栈（x1_asset/env_cfg/tasks/pretrain/rl）+ `scripts/`（建库/审计/窗口化/训练/评估入口）
- `eval/`：gait_metrics.json、video_v0p5.mp4、video_v1p0.mp4
- 试验记录：lab_notebook exp_aafa2481（7 条 update + 2 结果 + 结论 4 findings）
- gm 任务：P0 smoke TASK_20261103_200、P2 TASK_20261103_220、P3 TASK_20261104_037、P5 TASK_20261104_129
