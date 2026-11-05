# G1' 换数据源：候选库门控裁决与决策点（amendment rev2）

日期：2026-11-05 ｜ 实验：exp_2655ff02（预注册）｜ 触发条款：amendment rev2——"若所有可得候选都过不了门，停下带数字汇报，不降门不硬凑"。

## 0. 一句话裁定

**174 个候选段逐个实测（115 AMP + 59 GMR）：滑移主门 115/115 AMP rollout 全过（p50 0.0021-0.0216，余量 2.3-24 倍，GMR 对照 23 段全败 0.048-0.964）；但「双足离地帧>0」硬门 0/115——AMP 行走步态（duty 0.64-0.85）结构性无双摆相，且 0.4-1.0 m/s 的行走步态本无腾空（腾空属于跑态）。无候选过全套硬门，按修订停报；4 段过其余全部硬门仅差 flight。**

## 1. 候选与命令

```
# 门控脚本（新增，两种格式：rollout npz / 36 列库 npz）
~/code/X1_29_AMP/_gmr_local/venv/bin/python scripts/gate_candidate.py <npz...>
# 全量扫描（115 段）
~/code/X1_29_AMP/_gmr_local/venv/bin/python scripts/gate_candidate.py \
    ~/code/X1_29_AMP/acceptance/v83_eval/*_walk05.npz \
    ~/code/X1_29_AMP/acceptance/v83_eval/*_walk10.npz --out data/candidate_gates_sweep.json
# GMR 对照（23 段）
.../gate_candidate.py ~/code/super_mimic_F1/motions/x1_cmu_walk.npz \
    ~/code/super_mimic_F1/motions/x1_phase2_heldout.npz --out data/candidate_gates_gmr.json
```
候选来源：X1_29_AMP（只读）acceptance/v83_eval 的 AMP 策略 MuJoCo rollout（物理接触求解；含终局交付 DELIVERY_v95c_c9000、备选 c9196/h7499/j3000/x7398 全血统），50Hz、wxyz、关节序=X1_JOINT_ORDER、FK 交叉验证最大角点差 1.5-3.6mm；walkturn/back05 抽样同判。R 门中依赖人源参照的子门（wrist_err/chest_yaw_err）对 rollout 标 N/A（不默认通过）。

## 2. 门控汇总（逐段全量数字在 data/candidate_gates_sweep.json / _gmr.json）

| 门 | AMP rollout（115 段） | GMR 对照（23 段） |
|---|---|---|
| **滑移主门 p50<0.05** | **115/115 过**（0.0021-0.0216；n_strict 100-300 帧/足） | **0/59 段**（0.0477-1.188；1 段勉强过滑移门但 B 门挂） |
| **双足离地帧>0** | **0/115**（结构性；walkturn/back05 同 0） | 0/23 |
| 步频/步长 B 带 | walk10 系过（cad 1.04-1.06、step 0.45、比值带内）；walk05 多数差 lift 0.024 vs 0.025（1mm 边际） | 4/23 |
| 手/臂/躯干门 | **c9000/c9196 全血统零超限**（唯一超限=踝俯仰腿关节 0.013-0.034 rad；soup_ed75 肘 0.005 超限除外）；arm_at_limit/arm_qd/lumbar/twist 全过 | 与基线一致 |
| **全套** | **0/115** | 0/59 |

**过其余全部硬门、仅差 flight 的段（4 个，全 walk05）**：c9196_walk05（v0.48，slip 0.0026/0.0043，cad 0.78/0.78）、sx50_walk05、sx70_walk05、y8000_walk05（v0.46-0.48，slip 0.004-0.005）。
**覆盖 0.7 目标的段（walk10，v0.92-0.95）**：滑移 0.008-0.014、B 门过、手/臂/躯干零超限——唯一超限是踝俯仰 0.013-0.034 rad（腿关节；A 门 0.35 rad 口径内）。

## 3. 停报原因与决策点

`双足离地帧>0` 门的设计原意（AUDIT_FAILURE §6-2 + G1 契约）是防**重建路线的伪接触**（per-frame grounding 每帧把最低足压地 → 0/4902 离地帧 = 接触被伪造）。物理 rollout **没有 grounding 步骤可伪**——接触由求解器产生。且 0.4-1.0 m/s 行走步态在生物力学上本无腾空相（duty 0.6-0.85，腾空属跑态 ≥1.4 m/s）：**该门在本速度体制下对任何真行走数据都不可满足**，非质量缺陷。

按修订「不降门」条款，我无权自行豁免该门，停报待决：

- **(a) 豁免 flight>0 于物理求解数据**（等价保护已满足：无 grounding 步骤 + 滑移主门全过）→ 训练库可用段：4×walk05（0.46-0.48）+ c9000/c9196 walk10（0.92-0.95）覆盖 0.4-0.9；并可在本地用其 mujoco_rollout 生成 0.7 档命令 rollout 补带中心（同门控验收）→ 进 G2；
- **(b) 维持 flight>0 字面** → 无候选可用，SMP 线在 G1' 终止，归档第 5 轮负结果；
- **(c) 其它您指定的数据源**（外部公开 X1 行走库——本地未发现第三方 X1 库，GMR 系全部同病）。

预算：本轮全本地 ¥0（累计平台开销仍 ~¥0.7，¥150 未动）。G2-G5 按门序未启动。

证据：`data/candidate_gates_sweep.json`（115 段逐段）、`data/candidate_gates.json`、`data/candidate_gates_gmr.json`、`scripts/gate_candidate.py`；试验记录 exp_2655ff02（falsified + fnd_bd2f842e/fnd_f41d6dad）。
