#!/usr/bin/env bash
# scripts/tflayout/run_all.sh — TF-Layout 流水线入口：数据准备 → 模型自检 → 训练 → 导出 → 分析 → 论文表格
#
# 在任意目录都可以调用(脚本会先 cd 到仓库根目录，所以本文件必须放在 scripts/tflayout/ 下)。
# 每一步的产出文件已存在时默认跳过(--force 全部重跑)；每步日志写到 out/logs/<步骤>.log，同时打印到终端。
# 各步骤的具体配置(实验预设、seed、路径等)不在命令行里，写在对应脚本顶部的 CONFIG(16~29 号)。
#
# 默认(不加任何开关)：
#   01 → 02 → 03 → 04 → 08 → 11 → 13 → 09(真实数据 Dataset 核对) → 10/12/14/15 架构自检
#   只用 CPU，不训练。00 数据盘点和 05/06/07 探索性 QC 默认不跑(--legacy-qc)。
#
# 用法：./run_all.sh [开关...]
#
# 数据准备 / 自检
#   --skip-arch-selftest         跳过 10/12/14/15 号架构自检
#   --legacy-qc                  额外跑 00 数据盘点和 05/06/07 探索性 QC 图
#   --loop-selftest              16 号训练循环自检(假数据、CPU，几分钟)
#   --dense-check                20 号：稠密 log2FC 可用性诊断(只读)
#   --dense-target               21 号：构建 Head B 稠密辅助目标(--experiments 发现文件不存在时会先自动跑)
#
# 训练与实验
#   --bench                      16 号 bench：真实数据上只测量、不训练(分组前向对照、各配置吞吐/显存)
#   --smoke                      真实数据 1 epoch × 1 seed 的迷你训练，不存 checkpoint，用来确认跑得通并估算用时
#   --train                      16 号正式训练：5 个 seed(42/123/456/789/2024)，存到 out/checkpoints/cli_train/
#                                (固定的 run1 配置；其余实验配置用 --experiments)
#   --held-out-tfs TF1,TF2,...   --train 时额外做 leave-TF-out
#   --experiments                18 号：按 ACTIVE_PLAN 跑实验(冒烟 → 训练 → 导出)，配置在 18 号 CONFIG 里
#   --export                     17 号：用 out/checkpoints/ 的权重重新推理并导出逐样本预测、集成、指标、图
#   --compare                    19 号：多个实验在共有 seed 上的对比
#
# 分析
#   --pos-scan                   24 号：位置信息推理扫描(只推理，GPU 约 1 小时)
#   --head-a-all                 25 号：Head A 在全部基因上的评估
#   --probe                      28 号：孪生删位点的推理期探针(2×2)
#   --spec-control               29 号：删位点特异性对照(删错 TF 的位点)
#   --lto                        22 号：按 TF 分折的 leave-TF-out(GPU，约 3 小时)，跑完自动接 23 号诊断
#   --lto-diag                   23 号：leave-TF-out 诊断(只读，CPU)
#
# 论文
#   --baselines                  26 号：同口径基线表(CPU)
#   --tables                     27 号：论文表格和数字(CPU，读 19/25/26/28/29 号的产出)
#
# 训练 / 测速参数(--train、--smoke、--bench 使用；括号内是默认值)
#   --batch-size N (192)             --num-workers N (按 CPU 核数估计，上限 8)
#   --forward grouped|legacy (grouped)   --amp bf16|off (bf16)   --tfs-per-gene K (4)
#   --eval-batch-size N (512)        --layout-buckets N (1)
#   --compile                        torch.compile，默认关；配套 --compile-cache-limit N (16)、
#                                    --compile-dynamic auto|true (auto)、--compile-warmup-timeout 秒 (300)、
#                                    --compile-scope full|submodules (full)
#   --bench-steps N (30)             --bench-batch-sweep [--bench-batch-sweep-sizes 96,192,384]
#   --forward legacy --amp off --tfs-per-gene 1 等价于最初版本的逐样本 fp32 训练方式。
#
# 其它
#   --force                      忽略"产出已存在就跳过"，全部重跑
#   -h, --help                   打印本说明
#
# 同时给出多个开关时的执行顺序：
#   loop-selftest → bench → smoke → train → dense-target → export → pos-scan → experiments → head-a-all →
#   probe → spec-control → baselines → tables → compare → dense-check → lto → lto-diag
#
# 常用命令：
#   ./run_all.sh                                          数据准备 + 架构自检
#   ./run_all.sh --bench                                  先测一下吞吐和显存
#   ./run_all.sh --smoke                                  迷你训练，确认流程能跑通
#   ./run_all.sh --experiments --skip-arch-selftest       按 ACTIVE_PLAN 训练实验
#   ./run_all.sh --head-a-all --probe --spec-control --baselines --tables --skip-arch-selftest   论文分析与表格
set -euo pipefail
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"  # cd 之前记下，--help 要用
cd "$(dirname "$0")/../.."

# -------------------------------------------------------------------------
# 0. 参数解析
# -------------------------------------------------------------------------
# 回显收到的参数，方便从日志里确认这次到底传了哪些开关
echo "[信息] 收到的参数: $*"
DO_SMOKE=0
DO_TRAIN=0
DO_BENCH=0
DO_EXPORT=0        # 跑17号脚本导出结果
DO_EXPERIMENTS=0   # 跑18号脚本(按 ACTIVE_PLAN 跑实验)
DO_COMPARE=0       # 跑19号脚本(实验对比)
DO_DENSE=0         # 跑20号脚本(稠密 log2FC 诊断)
DO_DENSE_TARGET=0  # 跑21号脚本(构建稠密目标)
DO_LOOP_SELFTEST=0 # 跑16号自检
DO_LTO=0           # 跑22号脚本(leave-TF-out)
DO_LTO_DIAG=0      # 跑23号脚本(leave-TF-out 诊断，只用 CPU)
DO_POS_SCAN=0      # 跑24号脚本(位置信息推理扫描，只推理)
DO_HEAD_A_ALL=0    # 跑25号脚本(Head A 全部基因评估，只推理)
DO_BASELINES=0     # 跑26号脚本(论文基线对比，只读、只用CPU)
DO_TABLES=0        # 跑27号脚本(论文表格 + 训练随机性噪声底，只读、只用CPU)
DO_PROBE=0         # 跑28号脚本(孪生删位点的推理期探针，只推理)
DO_SPEC=0          # 跑29号脚本(删位点特异性对照：删错 TF 的位点，只推理)
DO_LEGACY_QC=0     # 跑 00 数据盘点 + 05/06/07 探索性 QC 图(默认不跑)
DENSE_TARGET_FILE=out/head_b_dense_target.parquet
TRAIN_SAVE_DIR=out/checkpoints/cli_train  # --train 的存放目录(不写 out/checkpoints/ 顶层，避免改动已有 checkpoint)
FORCE=0
SKIP_ARCH_SELFTEST=0
HELD_OUT_TFS=""
BATCH_SIZE=192  # 在 RTX 3090 上实测：192 比 96 吞吐高约 47%，显存仅用到约 4.5/23.6 GiB
NUM_WORKERS=""
# 训练提速开关
FORWARD=grouped
AMP=bf16
TFS_PER_GENE=4
EVAL_BATCH_SIZE=512
BENCH_STEPS=30
LAYOUT_BUCKETS=1   # layout 分支按长度分段 padding 的段数(1=不分段，实测最快)
COMPILE=0          # torch.compile，默认关
COMPILE_CACHE_LIMIT=16      # torch._dynamo
                            # 重编译次数上限，只在 --compile 时生效
COMPILE_DYNAMIC=auto        # auto|true，
                            # 只在 real/smoke(--train/--smoke --compile) 时生效
COMPILE_WARMUP_TIMEOUT=300  # 秒，
                            # 只在 --bench --compile 时生效
COMPILE_SCOPE=full          # full|submodules，
                            # 只在 real/smoke(--train/--smoke --compile) 时生效，bench
                            # 模式固定把两种都测一遍
BENCH_BATCH_SWEEP=0         # 只在 --bench 时生效
BENCH_BATCH_SWEEP_SIZES=""  # 逗号分隔候选batch_size，不传就自动用--batch-size的1x/2x/4x

while [[ $# -gt 0 ]]; do
  case "$1" in
    --legacy-qc) DO_LEGACY_QC=1; shift ;;
    --smoke) DO_SMOKE=1; shift ;;
    --train) DO_TRAIN=1; shift ;;
    --bench) DO_BENCH=1; shift ;;
    --export) DO_EXPORT=1; shift ;;
    --experiments) DO_EXPERIMENTS=1; shift ;;
    --compare) DO_COMPARE=1; shift ;;
    --dense-check) DO_DENSE=1; shift ;;
    --dense-target) DO_DENSE_TARGET=1; shift ;;
    --loop-selftest) DO_LOOP_SELFTEST=1; shift ;;
    --lto) DO_LTO=1; shift ;;
    --lto-diag) DO_LTO_DIAG=1; shift ;;
    --pos-scan) DO_POS_SCAN=1; shift ;;
    --head-a-all) DO_HEAD_A_ALL=1; shift ;;
    --baselines) DO_BASELINES=1; shift ;;
    --tables) DO_TABLES=1; shift ;;
    --probe) DO_PROBE=1; shift ;;
    --spec-control) DO_SPEC=1; shift ;;
    --forward) FORWARD="$2"; shift 2 ;;
    --forward=*) FORWARD="${1#*=}"; shift ;;
    --amp) AMP="$2"; shift 2 ;;
    --amp=*) AMP="${1#*=}"; shift ;;
    --tfs-per-gene) TFS_PER_GENE="$2"; shift 2 ;;
    --tfs-per-gene=*) TFS_PER_GENE="${1#*=}"; shift ;;
    --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
    --eval-batch-size=*) EVAL_BATCH_SIZE="${1#*=}"; shift ;;
    --layout-buckets) LAYOUT_BUCKETS="$2"; shift 2 ;;
    --layout-buckets=*) LAYOUT_BUCKETS="${1#*=}"; shift ;;
    --compile) COMPILE=1; shift ;;
    --compile-cache-limit) COMPILE_CACHE_LIMIT="$2"; shift 2 ;;
    --compile-cache-limit=*) COMPILE_CACHE_LIMIT="${1#*=}"; shift ;;
    --compile-dynamic) COMPILE_DYNAMIC="$2"; shift 2 ;;
    --compile-dynamic=*) COMPILE_DYNAMIC="${1#*=}"; shift ;;
    --compile-warmup-timeout) COMPILE_WARMUP_TIMEOUT="$2"; shift 2 ;;
    --compile-warmup-timeout=*) COMPILE_WARMUP_TIMEOUT="${1#*=}"; shift ;;
    --compile-scope) COMPILE_SCOPE="$2"; shift 2 ;;
    --compile-scope=*) COMPILE_SCOPE="${1#*=}"; shift ;;
    --bench-batch-sweep) BENCH_BATCH_SWEEP=1; shift ;;
    --bench-batch-sweep-sizes) BENCH_BATCH_SWEEP_SIZES="$2"; shift 2 ;;
    --bench-batch-sweep-sizes=*) BENCH_BATCH_SWEEP_SIZES="${1#*=}"; shift ;;
    --bench-steps) BENCH_STEPS="$2"; shift 2 ;;
    --bench-steps=*) BENCH_STEPS="${1#*=}"; shift ;;
    --force) FORCE=1; shift ;;
    --skip-arch-selftest) SKIP_ARCH_SELFTEST=1; shift ;;
    --held-out-tfs)
      HELD_OUT_TFS="$2"; shift 2 ;;
    --held-out-tfs=*)
      HELD_OUT_TFS="${1#*=}"; shift ;;
    --batch-size)
      BATCH_SIZE="$2"; shift 2 ;;
    --batch-size=*)
      BATCH_SIZE="${1#*=}"; shift ;;
    --num-workers)
      NUM_WORKERS="$2"; shift 2 ;;
    --num-workers=*)
      NUM_WORKERS="${1#*=}"; shift ;;
    -h|--help)
      # 打印文件头注释(第2行到 set -euo pipefail 之前)
      sed -n '2,/^set -euo pipefail/p' "$SCRIPT_PATH" | sed '$d'; exit 0 ;;
    *)
      echo "未知参数: $1 (用 --help 看用法)" >&2; exit 1 ;;
  esac
done

case "$FORWARD" in grouped|legacy) ;; *)
  echo "--forward 只能是 grouped 或 legacy，收到: $FORWARD" >&2; exit 1 ;; esac
case "$AMP" in bf16|off) ;; *)
  echo "--amp 只能是 bf16 或 off，收到: $AMP" >&2; exit 1 ;; esac
case "$COMPILE_SCOPE" in full|submodules) ;; *)
  echo "--compile-scope 只能是 full 或 submodules，收到: $COMPILE_SCOPE" >&2; exit 1 ;; esac
for v in BATCH_SIZE TFS_PER_GENE EVAL_BATCH_SIZE BENCH_STEPS LAYOUT_BUCKETS; do
  [[ "${!v}" =~ ^[1-9][0-9]*$ ]] || { echo "$v 必须是正整数，收到: ${!v}" >&2; exit 1; }
done
if [[ "$FORWARD" == "legacy" && "$TFS_PER_GENE" != "1" ]]; then
  echo "[信息] --forward legacy 时 --tfs-per-gene 不生效，按 1 处理"
  TFS_PER_GENE=1
fi

# --num-workers 不传时按 CPU 核数估计：留 1 个核给主进程，上限 8(经验值，未做过最优性实测)。
if [[ -z "$NUM_WORKERS" ]]; then
  CPU_CORES=$(nproc 2>/dev/null || echo 4)
  if [[ "$CPU_CORES" -gt 8 ]]; then
    NUM_WORKERS=8
  elif [[ "$CPU_CORES" -gt 1 ]]; then
    NUM_WORKERS=$((CPU_CORES - 1))
  else
    NUM_WORKERS=1
  fi
  echo "[信息] --num-workers 未指定，检测到 $CPU_CORES 个CPU核，自动设为 $NUM_WORKERS"
fi

mkdir -p out out/fig out/checkpoints out/logs out/results
PIPELINE_T0=$SECONDS

# -------------------------------------------------------------------------
# 1. 单步执行 helper：按产出文件判断是否跳过 + tee日志 + 计时
#    用法：run_step "描述" "产出文件1 产出文件2(空格分隔,可以是空字符串)" \
#                   out/logs/文件名.log  命令 参数...
# -------------------------------------------------------------------------
run_step() {
  local desc="$1" outcheck="$2" logfile="$3"; shift 3
  if [[ -n "$outcheck" && "$FORCE" != "1" ]]; then
    local all_exist=1
    for f in $outcheck; do
      [[ -e "$f" ]] || { all_exist=0; break; }
    done
    if [[ "$all_exist" == "1" ]]; then
      echo "[跳过] $desc  (产出已存在: $outcheck；--force 可强制重跑)"
      return 0
    fi
  fi
  echo "[开始] $desc"
  local t0=$SECONDS
  set +e
  "$@" 2>&1 | tee "$logfile"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[失败] $desc  (退出码=$rc，完整日志见 $logfile)" >&2
    exit "$rc"
  fi
  echo "[完成] $desc  用时$((SECONDS - t0))s  日志-> $logfile"
}

# -------------------------------------------------------------------------
# 2. 数据准备 00→08 (顺序跟原版一致，这部分没有依赖问题)
# -------------------------------------------------------------------------
if [[ "$DO_LEGACY_QC" == "1" ]]; then  # 探索性 QC，默认不跑
  run_step "00 数据盘点" "" out/logs/00_inventory.log \
    python scripts/tflayout/00_inventory.py --data data
fi

run_step "01 解析Nature补充表" \
  "out/binding_binary.parquet out/chec_occupancy.parquet out/log2fc_sig.parquet out/sig_mask.parquet" \
  out/logs/01_parse_supp.log \
  python scripts/tflayout/01_parse_supp.py \
    --xlsx data/41586_2025_8916_MOESM5_ESM.xlsx \
    --sgd data/SGD_features.tab --bwdir data/ChEC-seq --out out

run_step "02 bigWig调峰" "out/peaks.parquet" out/logs/02_call_peaks.log \
  python scripts/tflayout/02_call_peaks.py \
    --bwdir data/ChEC-seq --tss data/tss.bed \
    --fasta data/S288C.fsa --sgd data/SGD_features.tab

run_step "03 motif定位" "out/sites.parquet out/motif_choice.tsv" \
  out/logs/03_motif_anchor.log \
  python scripts/tflayout/03_motif_anchor.py \
    --jaspar data/motif/JASPAR2024_CORE_fungi_non-redundant_pfms_meme.txt \
    --yetfasco-dir data/motif/ALIGNED_ENOLOGO_FORMAT_PWMS

run_step "04 构建layout字典" "out/tf_layout.parquet out/pair_spacing.parquet" \
  out/logs/04_build_layout.log \
  python scripts/tflayout/04_build_layout.py

if [[ "$DO_LEGACY_QC" == "1" ]]; then  # 05/06/07 只产 PNG、没有脚本读它们，默认不跑
  run_step "05 Phase0统计图" "out/fig/phase0_layout_qc.png" \
    out/logs/05_phase0_stats.log \
    python scripts/tflayout/05_phase0_stats.py

  run_step "06 分层QC" "out/fig/rank_stratified_qc.png" \
    out/logs/06_stratified_qc.log \
    python scripts/tflayout/06_stratified_qc.py

  run_step "07 原始信号示例图" "out/fig/raw_signal_examples.png" \
    out/logs/07_inspect_examples.log \
    python scripts/tflayout/07_inspect_examples.py \
      --bwdir data/ChEC-seq --sgd data/SGD_features.tab
fi

run_step "08 组装Head A/B/C标签" \
  "out/head_a_baseline_logtpm.parquet out/head_bc_labels.parquet" \
  out/logs/08_build_model_inputs.log \
  python scripts/tflayout/08_build_model_inputs.py

# -------------------------------------------------------------------------
# 3. cis分支：promoter序列 + BPE分词(必须在 09 号之前：09 要读它们的产出)
# -------------------------------------------------------------------------
run_step "11 提取promoter序列" "out/promoter_seq.parquet" \
  out/logs/11_extract_promoter_seq.log \
  python scripts/tflayout/11_extract_promoter_seq.py

run_step "13 训练BPE分词器" "out/bpe_tokenizer.json out/promoter_token_ids.parquet" \
  out/logs/13_train_bpe.log \
  python scripts/tflayout/13_train_bpe.py

# 交叉核对：13号脚本实际训出的词表大小 vs 16号脚本real模式默认vocab_size(4000)。
# 词表比4000大会让token id越界；比4000小只是Embedding多留几行空位，不影响运行。
# 这里只读bpe_tokenizer.json里model.vocab的长度，读不出来时只警告、不阻塞流水线。
echo "[检查] 13号脚本词表大小 vs 16号脚本real模式默认vocab_size(4000)"
python3 - <<'PYEOF' || echo "  (词表大小交叉核对失败，非致命，跳过——不影响后续步骤)"
import json
try:
    d = json.load(open("out/bpe_tokenizer.json"))
    n = len(d["model"]["vocab"])
    print(f"  实际词表大小: {n}" +
          ("  ✓ 跟16号脚本默认vocab_size=4000一致" if n == 4000 else
           f"  ⚠ 跟16号脚本默认vocab_size=4000不一致"
           + ("：实际词表比4000大，token id 会越界，必须给16号脚本传 --vocab-size "
              f"{n}(16号脚本有这个参数；run_all.sh 目前不转发它，需要"
              "手动改本文件里调用16号脚本的那几行)" if n > 4000 else
              "：实际词表比4000小，不会出错(Embedding 只是多留几行空位)，不用管")))
except Exception as e:
    print(f"  没能读出词表大小({e})")
    raise
PYEOF

# -------------------------------------------------------------------------
# 4. 09号脚本：用真实数据构造一次TFLayoutDataset+取一个batch，快速核对形状
#    (必须放在11/13之后；本身不产出文件，每次都跑，很快)
# -------------------------------------------------------------------------
run_step "09 真实数据Dataset核对" "" out/logs/09_torch_dataset.log \
  python scripts/tflayout/09_torch_dataset.py

# -------------------------------------------------------------------------
# 5. 模型架构自检 10/12/14/15 (随机数据，不依赖真实文件，可选跳过)
# -------------------------------------------------------------------------
if [[ "$SKIP_ARCH_SELFTEST" == "1" ]]; then
  echo "[跳过] 10/12/14/15号架构自检 (--skip-arch-selftest)"
else
  run_step "10 LayoutTransformer自检" "" out/logs/10_layout_transformer.log \
    python scripts/tflayout/10_layout_transformer.py
  run_step "12 CisTransformer自检" "" out/logs/12_cis_transformer.log \
    python scripts/tflayout/12_cis_transformer.py
  run_step "14 条件融合自检" "" out/logs/14_condition_fusion.log \
    python scripts/tflayout/14_condition_fusion.py
  run_step "15 孪生头+损失自检" "" out/logs/15_siamese_heads.log \
    python scripts/tflayout/15_siamese_heads.py
fi

# 16号训练循环自检(假数据、CPU)，只在显式 --loop-selftest 时跑
if [[ "$DO_LOOP_SELFTEST" == "1" ]]; then
  run_step "16 训练循环自检(假数据，含稠密辅助目标)" "" out/logs/16_self_test.log \
    python scripts/tflayout/16_train_loop.py --mode self_test
fi

# -------------------------------------------------------------------------
# 6. GPU可用性提示 (信息性，不阻塞——万一真要在CPU上跑，至少让人是知情的)
# -------------------------------------------------------------------------
echo "[检查] GPU可用性"
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU可用: {torch.cuda.get_device_name(0)}')
else:
    print('  ⚠ 没检测到可用GPU，16号脚本会退回CPU；'
          '5-seed×585900样本×30epoch在CPU上大概率跑不动，建议先用--smoke'
          '看看单步/单epoch大概多久再决定要不要继续')
" || echo "  (GPU检查本身出错，非致命，跳过)"
echo "[信息] 本次--bench/--smoke/--train将使用 batch-size=$BATCH_SIZE  num-workers=$NUM_WORKERS"
echo "  提速开关: --forward $FORWARD  --amp $AMP  --tfs-per-gene $TFS_PER_GENE" \
     " --eval-batch-size $EVAL_BATCH_SIZE  --layout-buckets $LAYOUT_BUCKETS" \
     " --compile=$COMPILE (都可以在命令行覆盖，见 --help)"
if [[ "$COMPILE" == "1" ]]; then
  echo "  compile细节: --compile-cache-limit $COMPILE_CACHE_LIMIT" \
       " --compile-dynamic $COMPILE_DYNAMIC(real/smoke用)" \
       " --compile-scope $COMPILE_SCOPE(real/smoke用，bench固定full/submodules都测)" \
       " --compile-warmup-timeout ${COMPILE_WARMUP_TIMEOUT}s(bench用)"
fi
if [[ "$BENCH_BATCH_SWEEP" == "1" ]]; then
  echo "  bench batch_size扫描: 已开启" \
       "${BENCH_BATCH_SWEEP_SIZES:+(候选: $BENCH_BATCH_SWEEP_SIZES)}"
fi

# --compile 只是个开关，真正传给16号脚本时要么给 --compile(以及配套的
# --compile-cache-limit/--compile-dynamic/--compile-scope) 要么完全不给这些flag
# (16号脚本用 action="store_true"，没有 --compile=0 这种写法)
COMPILE_FLAG=()
if [[ "$COMPILE" == "1" ]]; then
  COMPILE_FLAG=(--compile --compile-cache-limit "$COMPILE_CACHE_LIMIT"
                --compile-dynamic "$COMPILE_DYNAMIC" --compile-scope "$COMPILE_SCOPE")
fi
# --bench-batch-sweep 只在 --bench 时有意义，只在 bench 调用里传
BENCH_SWEEP_FLAG=()
if [[ "$BENCH_BATCH_SWEEP" == "1" ]]; then
  BENCH_SWEEP_FLAG=(--bench-batch-sweep)
  if [[ -n "$BENCH_BATCH_SWEEP_SIZES" ]]; then
    BENCH_SWEEP_FLAG+=(--bench-batch-sweep-sizes "$BENCH_BATCH_SWEEP_SIZES")
  fi
fi

# 这块 GPU 上已有的计算进程(比如上一版还没停的训练)会跟接下来的步骤抢算力/显存，
# 只提醒、不替你杀
check_gpu_busy() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local apps
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
           --format=csv,noheader 2>/dev/null || true)
  if [[ -n "$apps" ]]; then
    echo "  ⚠ 这块 GPU 上已经有计算进程在跑(pid, 进程名, 显存)："
    echo "$apps" | sed 's/^/      /'
    echo "    如果是上一版还没停的训练，它会跟接下来的步骤抢算力和显存(测速偏慢、甚至"
    echo "    OOM)。确认后可以 kill <PID> 结束它；本脚本不会替你杀进程。"
  fi
}

# 16号脚本的代码版本号(写进 checkpoint，下面判断旧 checkpoint 能不能跳过要用)
CODE_VERSION=$(sed -n 's/^CODE_VERSION = "\([^"]*\)".*/\1/p' \
                 scripts/tflayout/16_train_loop.py | head -n 1)
echo "[信息] 16号脚本 CODE_VERSION=${CODE_VERSION:-<没读到>}"

# -------------------------------------------------------------------------
# 6b. --bench: 真实数据上只测量、不训练(分组前向对照、各配置样本/s 与显存)
# -------------------------------------------------------------------------
if [[ "$DO_BENCH" == "1" ]]; then
  check_gpu_busy
  run_step "16 实测(bench：等价性对照+各配置吞吐/显存)" "" out/logs/16_bench.log \
    python scripts/tflayout/16_train_loop.py --mode bench \
      --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
      --tfs-per-gene "$TFS_PER_GENE" --eval-batch-size "$EVAL_BATCH_SIZE" \
      --layout-buckets "$LAYOUT_BUCKETS" "${COMPILE_FLAG[@]}" \
      --compile-warmup-timeout "$COMPILE_WARMUP_TIMEOUT" \
      "${BENCH_SWEEP_FLAG[@]}" \
      --bench-steps "$BENCH_STEPS" --n-epochs 30 --seeds 42,123,456,789,2024
fi

# -------------------------------------------------------------------------
# 7. --smoke: 真实数据+正式架构规模，1 epoch/1 seed，不存checkpoint，只为了
#    确认"跑得通、不是NaN、显存够、大概多快"
# -------------------------------------------------------------------------
if [[ "$DO_SMOKE" == "1" ]]; then
  check_gpu_busy
  echo "[开始] 16 smoke test (真实数据，正式架构256/8/6/4，1 epoch×1 seed，"
  echo "  batch-size=$BATCH_SIZE num-workers=$NUM_WORKERS，"
  echo "  跟--train实际会用的这两个值一致，这样下面的用时外推才有参考意义)"
  t0=$SECONDS
  set +e
  python scripts/tflayout/16_train_loop.py --mode real \
    --n-epochs 1 --seeds 42 --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" --n-boot 100 --save-dir "" \
    --forward "$FORWARD" --amp "$AMP" --tfs-per-gene "$TFS_PER_GENE" \
    --eval-batch-size "$EVAL_BATCH_SIZE" --layout-buckets "$LAYOUT_BUCKETS" \
    "${COMPILE_FLAG[@]}" \
    2>&1 | tee out/logs/16_smoke.log
  rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[失败] 16 smoke test (退出码=$rc，日志-> out/logs/16_smoke.log)" >&2
    exit "$rc"
  fi
  dt=$((SECONDS - t0))
  echo "[完成] 16 smoke test  用时${dt}s  日志-> out/logs/16_smoke.log"
  echo "  (smoke 总用时 ${dt}s 含读数据、建 Dataset、1 个 epoch 训练+验证、test 评估+bootstrap)"
  # 按日志里 epoch0 那一行外推(不含读数据、test 评估、bootstrap 这些一次性开销)
  python3 - <<'PYEOF' || echo "  (没能从日志里解析出 epoch0 的用时，跳过外推)"
import re
txt = open("out/logs/16_smoke.log", encoding="utf-8", errors="ignore").read()
m = re.search(r"epoch0: .*?训练([0-9.]+)分钟.*?验证([0-9.]+)秒", txt)
if not m:
    raise SystemExit(1)
tr, va = float(m.group(1)), float(m.group(2)) / 60
per = tr + va
print(f"  按日志 epoch0 外推：每个 epoch ≈ 训练 {tr:.1f} + 验证 {va:.1f} = {per:.1f} 分钟；"
      f"5 seed × 30 epoch 上限 ≈ {per * 150 / 60:.1f} 小时")
print("  (另加每个 seed 一次 test 评估 + 两种口径的 bootstrap，约一分钟量级；有 patience=5 "
      "早停，大概率跑不满 30 epoch——这是上限，不是预测)")
PYEOF
fi

# -------------------------------------------------------------------------
# 8. --train: 正式5-seed训练，显式把关键超参打在命令行里，
#    按out/checkpoints/seed{42,123,456,789,2024}_best.pt是否都已存在做跳过判断
# -------------------------------------------------------------------------
if [[ "$DO_TRAIN" == "1" ]]; then
  check_gpu_busy
  # 整步跳过的条件：5 个 seed 的 checkpoint 都在，且代码版本号 +
  # amp/forward/K/batch/epoch/patience 都跟这次一致，且没有指定 --held-out-tfs。
  # 不满足就调用 16 号脚本，由它按 seed 断点续跑(一致的 seed 直接复用，不一致的旧文件
  # 改名备份后重训)。
  SEEDS_LIST=(42 123 456 789 2024)
  N_EPOCHS=30
  PATIENCE=5
  CKPT_STATUS="(没检查)"
  if [[ "$FORCE" != "1" ]]; then
    CKPT_STATUS=$(EXP_VERSION="$CODE_VERSION" EXP_AMP="$AMP" EXP_FORWARD="$FORWARD" \
      EXP_TFS="$TFS_PER_GENE" EXP_BS="$BATCH_SIZE" EXP_EPOCHS="$N_EPOCHS" \
      EXP_PATIENCE="$PATIENCE" SEEDS="${SEEDS_LIST[*]}" TRAIN_SAVE_DIR="$TRAIN_SAVE_DIR" \
      python3 - 2>/dev/null <<'PYEOF' \
      || echo "检查脚本本身出错"
import os
import sys
try:
    import torch
except Exception as ex:
    print(f"import torch 失败({type(ex).__name__})")
    sys.exit(0)
e = os.environ
exp = dict(amp=e["EXP_AMP"], forward_mode=e["EXP_FORWARD"],
           tfs_per_gene=1 if e["EXP_FORWARD"] == "legacy" else int(e["EXP_TFS"]),
           batch_size=int(e["EXP_BS"]), n_epochs=int(e["EXP_EPOCHS"]),
           patience=int(e["EXP_PATIENCE"]))
for s in e["SEEDS"].split():
    p = os.path.join(e["TRAIN_SAVE_DIR"], f"seed{s}_best.pt")
    if not os.path.exists(p):
        print(f"seed{s}_best.pt 不存在")
        sys.exit(0)
    try:
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
        except TypeError:
            ck = torch.load(p, map_location="cpu")
    except Exception as ex:
        print(f"seed{s}_best.pt 读不出来({type(ex).__name__})")
        sys.exit(0)
    if ck.get("code_version") != e["EXP_VERSION"]:
        print(f"seed{s}_best.pt 是旧代码版本({ck.get('code_version')})存的")
        sys.exit(0)
    cfg = ck.get("train_config") or {}
    bad = [k for k, v in exp.items() if cfg.get(k) != v]
    if bad:
        print(f"seed{s}_best.pt 的训练配置跟这次不同: {bad}")
        sys.exit(0)
print("ok")
PYEOF
    )
  fi
  if [[ "$CKPT_STATUS" == "ok" && -z "$HELD_OUT_TFS" ]]; then
    echo "[跳过] 16 正式训练 ($TRAIN_SAVE_DIR/seed{${SEEDS_LIST[*]}}_best.pt 都已存在，代码版本"
    echo "  $CODE_VERSION、训练配置也一致；--force 强制重跑)"
  else
    if [[ "$FORCE" != "1" ]]; then
      echo "[信息] 不能整步跳过: ${CKPT_STATUS}(16号脚本会按 seed 断点续跑)"
    fi
    TRAIN_ARGS=(--mode real --n-epochs "$N_EPOCHS" --batch-size "$BATCH_SIZE" --n-boot 1000
                --seeds 42,123,456,789,2024 --patience "$PATIENCE"
                --num-workers "$NUM_WORKERS" --save-dir "$TRAIN_SAVE_DIR"
                --forward "$FORWARD" --amp "$AMP" --tfs-per-gene "$TFS_PER_GENE"
                --eval-batch-size "$EVAL_BATCH_SIZE" --layout-buckets "$LAYOUT_BUCKETS"
                "${COMPILE_FLAG[@]}")
    if [[ "$FORCE" == "1" ]]; then
      TRAIN_ARGS+=(--no-resume)
    fi
    if [[ -n "$HELD_OUT_TFS" ]]; then
      TRAIN_ARGS+=(--held-out-tfs "$HELD_OUT_TFS")
      echo "[开始] 16 正式训练 (5-seed + leave-TF-out: $HELD_OUT_TFS)"
    else
      echo "[开始] 16 正式训练 (5-seed，未指定--held-out-tfs，本次不跑leave-TF-out；"
      echo "  想跑的话，"
      echo "  重新执行本脚本并加 --held-out-tfs TF1,TF2,...)"
    fi
    echo "  [信息] 存放目录 $TRAIN_SAVE_DIR/(不写 out/checkpoints/ 顶层，避免改动已有 checkpoint)；"
    echo "    配置是 --train 的固定参数(run1 配置)；其余实验配置请用 --experiments(18号)"
    echo "  实际调用: python scripts/tflayout/16_train_loop.py ${TRAIN_ARGS[*]}"
    t0=$SECONDS
    set +e
    python scripts/tflayout/16_train_loop.py "${TRAIN_ARGS[@]}" \
      2>&1 | tee out/logs/16_train_real.log
    rc=${PIPESTATUS[0]}
    set -e
    if [[ $rc -ne 0 ]]; then
      echo "[失败] 16 正式训练 (退出码=$rc，日志-> out/logs/16_train_real.log)" >&2
      exit "$rc"
    fi
    echo "[完成] 16 正式训练  用时$((SECONDS - t0))s  日志-> out/logs/16_train_real.log"
  fi
fi

# -------------------------------------------------------------------------
# 8a. --dense-target: 21号，构建 Head B 稠密目标。显式指定时每次都跑；--experiments 时
#     文件不存在就自动先跑(18号的 v3_*_dense 预设要用)。放在 --export 之前：17号导出会用它算 B_dr。
# -------------------------------------------------------------------------
if [[ "$DO_DENSE_TARGET" == "1" || ( "$DO_EXPERIMENTS" == "1" && ! -e "$DENSE_TARGET_FILE" ) ]]; then
  if [[ "$DO_DENSE_TARGET" != "1" ]]; then
    echo "[信息] --experiments 需要 $DENSE_TARGET_FILE，还不存在，先自动跑一次 21 号"
  fi
  run_step "21 构建Head B稠密log2FC目标" "" out/logs/21_build_dense_target.log \
    python scripts/tflayout/21_build_dense_target.py
  echo "  结果: $DENSE_TARGET_FILE ；诊断: out/results/_dense_target/summary.txt"
fi

# -------------------------------------------------------------------------
# 8b. --export: 17号脚本，用 out/checkpoints 的权重重新推理并导出全部结果
# -------------------------------------------------------------------------
if [[ "$DO_EXPORT" == "1" ]]; then
  if ! ls out/checkpoints/seed*_best.pt out/checkpoints/*/seed*_best.pt >/dev/null 2>&1; then
    echo "[失败] 17 导出结果：out/checkpoints/ 下(含子目录)没有 seed*_best.pt，先训练" >&2
    exit 1
  fi
  check_gpu_busy
  run_step "17 导出训练结果(自动发现各实验；已导出且比checkpoint新的跳过)" "" \
    out/logs/17_export_results.log \
    python scripts/tflayout/17_export_results.py
  echo "  结果目录: out/results/<实验名>/  (run1=out/checkpoints/ 顶层那5个 seed)"
fi

# -------------------------------------------------------------------------
# 8b2. --pos-scan: 24号，位置信息推理扫描(只推理、不训练，GPU 约1小时)。放在 --experiments 之前：新代码先跑
# -------------------------------------------------------------------------
if [[ "$DO_POS_SCAN" == "1" ]]; then
  if [[ ! -e out/results/v3_ce_marker_dense/predictions_test.parquet ]]; then
    echo "[失败] 24 位置扫描：找不到主线导出 out/results/v3_ce_marker_dense/predictions_test.parquet(先 --export)" >&2
    exit 1
  fi
  check_gpu_busy
  run_step "24 位置信息扫描(只推理；配置见 24_position_scan.py 顶部 CONFIG)" "" out/logs/24_position_scan.log \
    python scripts/tflayout/24_position_scan.py
  echo "  结果: out/results/_pos_scan/summary.txt；图: out/results/_pos_scan/fig/pos_scan.png"
fi

# -------------------------------------------------------------------------
# 8c. --experiments: 18号，按 ACTIVE_PLAN 跑实验(冒烟→训练→导出，配置在18号 CONFIG 里)
# -------------------------------------------------------------------------
if [[ "$DO_EXPERIMENTS" == "1" ]]; then
  check_gpu_busy
  run_step "18 对照实验(计划见 18_run_experiments.py 顶部 PLANS/ACTIVE_PLAN)" "" \
    out/logs/18_run_experiments.log \
    python scripts/tflayout/18_run_experiments.py
  echo "  逐实验结果: out/results/<实验名>/summary.txt；18 号 CONFIG 里 auto_compare=False(不自动重写对比结果)，要对比另跑 ./run_all.sh --compare"
fi

# -------------------------------------------------------------------------
# 8c2. --head-a-all: 25号，Head A 在全部 Head A 基因上的评估与配对比较(只推理，GPU 几分钟)。放在 --experiments 之后
# -------------------------------------------------------------------------
if [[ "$DO_HEAD_A_ALL" == "1" ]]; then
  if ! ls out/checkpoints/v8_headA_all/seed*_best.pt >/dev/null 2>&1; then  # 参照 checkpoint：v8_headA_all
    echo "[失败] 25 Head A 全基因评估：找不到参照 checkpoint out/checkpoints/v8_headA_all/seed*_best.pt" >&2
    exit 1
  fi
  check_gpu_busy
  run_step "25 Head A 全部基因评估(只推理；配置见 25_head_a_all_genes.py 顶部 CONFIG)" "" out/logs/25_head_a_all_genes.log \
    python scripts/tflayout/25_head_a_all_genes.py
  echo "  结果: 25 号 CONFIG[\"outdir\"] 下的 summary.txt(默认 out/results/_head_a_all_b11/；其它批次的 _head_a_all*/ 不覆盖)"
fi

# -------------------------------------------------------------------------
# 8c2b. --probe: 28号，孪生删位点的推理期探针(只推理，GPU 估10~20分钟)。放在 --baselines/--tables 之前：27 号会并入它的表
# -------------------------------------------------------------------------
if [[ "$DO_PROBE" == "1" ]]; then
  for _r in v8_headA_all v10_abl_no_knockout; do
    if ! ls "out/checkpoints/$_r"/seed*_best.pt >/dev/null 2>&1; then
      echo "[失败] 28 删位点探针：找不到 out/checkpoints/$_r/seed*_best.pt" >&2
      exit 1
    fi
  done
  for _f in out/results/v8_headA_all/predictions_test.parquet out/results/v10_abl_no_knockout/predictions_test.parquet \
            out/results/_paper/null_seed_variability.csv out/results/_compare_b11/strata_delta.csv; do
    [[ -e "$_f" ]] || echo "  提示: 没有 $_f，28 号对应的自检/合成区间会跳过(其余照常)"
  done
  check_gpu_busy
  run_step "28 删位点推理期探针 2×2(只推理；配置见 28_siamese_probe.py 顶部 CONFIG)" "" out/logs/28_siamese_probe.log \
    python scripts/tflayout/28_siamese_probe.py
  echo "  结果: out/results/_siamese_probe/summary.txt；表: tables_probe.tex(--tables 时并入 out/results/_paper/tables.tex)"
fi

# -------------------------------------------------------------------------
# 8c2c. --spec-control: 29号，删位点特异性对照(只推理，GPU 估5~10分钟)。放在 --probe 之后、--tables 之前：27 号会并入它的表
# -------------------------------------------------------------------------
if [[ "$DO_SPEC" == "1" ]]; then
  if ! ls "out/checkpoints/v8_headA_all"/seed*_best.pt >/dev/null 2>&1; then
    echo "[失败] 29 删位点特异性对照：找不到 out/checkpoints/v8_headA_all/seed*_best.pt" >&2
    exit 1
  fi
  [[ -e out/results/v8_headA_all/predictions_test.parquet ]] || \
    echo "  提示: 没有 out/results/v8_headA_all/predictions_test.parquet，29 号的 (s1) 复现核对会跳过(其余照常；建议先 --export)"
  check_gpu_busy
  run_step "29 删位点特异性对照(只推理；配置见 29_deletion_specificity.py 顶部 CONFIG)" "" out/logs/29_deletion_specificity.log \
    python scripts/tflayout/29_deletion_specificity.py
  echo "  结果: out/results/_spec_control/summary.txt；表: tables_spec.tex(--tables 时并入 out/results/_paper/tables.tex)"
fi

# -------------------------------------------------------------------------
# 8c3. --baselines: 26号，论文用的同口径强基线表(只读、只用 CPU、几分钟)。放在 --head-a-all 之后：要读 17/25 号的产出
# -------------------------------------------------------------------------
if [[ "$DO_BASELINES" == "1" ]]; then
  _BL_RUN=""
  for _r in v8_headA_all v3_ce_marker_dense; do
    if [[ -e "out/results/$_r/predictions_test.parquet" ]]; then _BL_RUN="$_r"; break; fi
  done
  if [[ -z "$_BL_RUN" ]]; then
    echo "[失败] 26 基线对比：找不到 out/results/{v8_headA_all,v3_ce_marker_dense}/predictions_test.parquet(先 --export)" >&2
    exit 1
  fi
  echo "  模型侧用 $_BL_RUN 的导出(26 号 CONFIG 里可改 model_run/fallback_run)"
  if [[ ! -e out/results/_head_a_all/per_gene_head_a.csv ]]; then
    echo "  提示: 没有 25 号的 per_gene_head_a.csv，Head A 只会在 820 个网格基因上比(先 --head-a-all 才能比全部 1108 个)"
  fi
  run_step "26 论文基线对比 v6(只读、CPU、含 B7 平铺 layout GBDT + bagging 对等集成 + [2d] 公平集成对照与训练随机性区间，估35~50分钟；配置见 26_paper_baselines.py 顶部 CONFIG)" "" out/logs/26_paper_baselines.log \
    python scripts/tflayout/26_paper_baselines.py
  echo "  结果: 26 号 CONFIG[\"outdir\"] 下的 summary.txt(论文冻结结果是 out/results/_baselines/)；表: table1_head_a.csv ~ table6_fair_ensemble.csv；B7/bag 在 [2]/[2b]/[2c]/[3] 里，公平集成对照在 [2d]"
  echo "  检查点: [2d] 的 seed 列表应是 [42, 123, 456, 789, 2024]；B7 的轮数若带 ⚠(顶在网格边界)，说明候选范围需要再放宽"
  echo "          [2d] 末尾应有\"含训练随机性的区间\"一段和 (i‡)(j‡) 两行判读；bag 那几行应写着\"另训练 5 个成员只用来量重训噪声\""
fi

# -------------------------------------------------------------------------
# 8c4. --tables: 27号，论文表格 + 训练随机性噪声底(只读、只用 CPU)。放在 --baselines 之后：要读 19/25/26 号的产出
# -------------------------------------------------------------------------
if [[ "$DO_TABLES" == "1" ]]; then
  if [[ ! -e out/results/v8_headA_all/predictions_test.parquet ]]; then
    echo "[失败] 27 论文表格：找不到主模型导出 out/results/v8_headA_all/predictions_test.parquet(先 --export)" >&2
    exit 1
  fi
  for _f in out/results/_compare/headline_all_seeds.csv out/results/_baselines/table2_head_bc.csv \
            out/results/_compare_b10/delta_vs_reference.csv out/results/_head_a_all_b10/delta_vs_reference.csv; do
    [[ -e "$_f" ]] || echo "  提示: 没有 $_f，27 号里对应那张表会跳过(其余照常)"
  done
  if [[ -e out/results/_baselines/table2_head_bc.csv ]] && ! grep -q "B7a" out/results/_baselines/table2_head_bc.csv; then
    echo "  提示: out/results/_baselines/table2_head_bc.csv 里没有 B7a(还是 26 号 v3 的结果)；先 --baselines 再 --tables，Table 2 才有平铺 layout 基线"
  fi
  if [[ -e out/results/_baselines/table2_head_bc.csv ]] && ! grep -q "B7a_bag" out/results/_baselines/table2_head_bc.csv; then
    echo "  提示: table2_head_bc.csv 里没有 B7a_bag(还是 26 号 v4 的结果)；先 --baselines(26 号 v5)再 --tables，Table 2/Table 6 才有对等集成的对照"
  fi
  [[ -e out/results/_baselines/table6_fair_ensemble.csv ]] || \
    echo "  提示: 没有 out/results/_baselines/table6_fair_ensemble.csv(26 号 v5 [2d] 的产出)；27 号会提示并跳过 Table 6"
  if [[ -e out/results/_baselines/table6_fair_ensemble.csv ]] && ! head -1 out/results/_baselines/table6_fair_ensemble.csv | grep -q "tot_sig"; then
    echo "  提示: table6_fair_ensemble.csv 没有 tot_sig 列(还是 26 号 v5 的结果)；先 --baselines(26 号 v6)再 --tables，Table 6 才有 ‡/§ 和 (i‡)(m) 判读"
  fi
  [[ -e out/results/_spec_control/rule_checks_spec.csv ]] || \
    echo "  提示: 没有 out/results/_spec_control/rule_checks_spec.csv(29 号还没跑)；27 号会提示一行、其余照常，之后再 --tables 一次即可"
  run_step "27 论文表格 + 噪声底(只读、CPU；首次约10~15分钟，含 19 号重算位置机制；配置见 27_paper_tables.py 顶部 CONFIG)" "" \
    out/logs/27_paper_tables.log \
    python scripts/tflayout/27_paper_tables.py
  echo "  结果: out/results/_paper/summary.txt；论文用 tables.tex、paper_numbers.md、fig/*.png；短文用 tables_short.tex/.docx 和 abstract_draft.md"
fi

# -------------------------------------------------------------------------
# 8d. --compare: 19号，只重跑实验对比
# -------------------------------------------------------------------------
if [[ "$DO_COMPARE" == "1" ]]; then
  run_step "19 实验对比(共有seed上重算集成 + 配对整群bootstrap)" "" \
    out/logs/19_compare_runs.log \
    python scripts/tflayout/19_compare_runs.py
  echo "  结果: 19 号 CONFIG[\"outdir\"] 下的 summary.txt"
fi

# -------------------------------------------------------------------------
# 8e. --dense-check: 20号，稠密 log2FC 可用性诊断(只读)
# -------------------------------------------------------------------------
if [[ "$DO_DENSE" == "1" ]]; then
  run_step "20 稠密log2FC诊断(只读)" "" out/logs/20_dense_log2fc_check.log \
    python scripts/tflayout/20_dense_log2fc_check.py
  echo "  结果: out/results/_dense_check/summary.txt"
fi

# -------------------------------------------------------------------------
# 8f. --lto: 22号，按 TF 分折的 leave-TF-out(需要 GPU，约3小时)
# -------------------------------------------------------------------------
if [[ "$DO_LTO" == "1" ]]; then
  check_gpu_busy
  run_step "22 leave-TF-out v2(配置见 22_leave_tf_out.py 顶部 CONFIG)" "" out/logs/22_leave_tf_out.log \
    python scripts/tflayout/22_leave_tf_out.py
  echo "  结果: out/results/_lto2/summary.txt；每折权重在 out/checkpoints/_lto2/，同配置重跑只重新评估"
  run_step "23 leave-TF-out 诊断(只读，CPU)" "" out/logs/23_lto_diagnose.log \
    python scripts/tflayout/23_lto_diagnose.py
  echo "  结果: out/results/_lto_diag/summary.txt"
fi

# -------------------------------------------------------------------------
# 8g. --lto-diag: 23号，leave-TF-out 诊断(只读、只用 CPU，几分钟)。--lto 时已经跑过就不重复
# -------------------------------------------------------------------------
if [[ "$DO_LTO_DIAG" == "1" && "$DO_LTO" != "1" ]]; then
  run_step "23 leave-TF-out 诊断(只读，CPU)" "" out/logs/23_lto_diagnose.log \
    python scripts/tflayout/23_lto_diagnose.py
  echo "  结果: out/results/_lto_diag/summary.txt"
fi

# -------------------------------------------------------------------------
# 9. 收尾
# -------------------------------------------------------------------------
echo ""
echo "===== 流水线结束，总用时$((SECONDS - PIPELINE_T0))s ====="
if [[ "$DO_TRAIN" != "1" && "$DO_EXPERIMENTS" != "1" ]]; then
  echo ""
  echo "还没跑训练(需要显式 --train 或 --experiments)。"
  echo "建议顺序："
  echo "  0) ./run_all.sh --bench   真实数据上只测量不训练：分组前向对照、各配置的样本/s、"
  echo "     显存，几分钟量级(先确认 nvidia-smi 里没有别的训练进程在跑)"
  echo "  1) ./run_all.sh --smoke   先用真实数据跑1-epoch/1-seed看看多快、"
  echo "     loss是不是有限值、这台机器的显存够不够d_model=256的正式架构"
  echo "  2) 确认没问题后： ./run_all.sh --experiments --skip-arch-selftest"
  echo "     (按 18 号 ACTIVE_PLAN 训练实验；或 ./run_all.sh --train：固定的 run1 配置，存到 $TRAIN_SAVE_DIR/)"
  echo "  3) 训练完：./run_all.sh --export --skip-arch-selftest  导出逐样本预测/集成/基线/"
  echo "     诊断/图到 out/results/<实验名>/"
  echo "  (这些步骤都会自动跳过已经跑过的数据准备步骤，不用担心重复计算)"
fi