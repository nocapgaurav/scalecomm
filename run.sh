#!/usr/bin/env bash
# ============================================================
# run.sh — ScaleComm pipeline runner
# Usage:
#   chmod +x run.sh
#   ./run.sh                       # Run with default settings (Cora)
#   ./run.sh --dataset citeseer    # Run on CiteSeer
#   ./run.sh --dataset cora --epochs 300 --cluster dpgmm
# ============================================================

set -e  # Exit on error

# ---- Colors ----
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║  ScaleComm — Self-Supervised Community Detection     ║"
echo "  ║  Deep Learning for Graph Mining                      ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ---- Default configuration ----
DATASET="${DATASET:-cora}"
EPOCHS="${EPOCHS:-200}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
OUT_DIM="${OUT_DIM:-128}"
NUM_HEADS="${NUM_HEADS:-8}"
NUM_LAYERS="${NUM_LAYERS:-3}"
LR="${LR:-0.001}"
DROPOUT="${DROPOUT:-0.3}"
TEMPERATURE="${TEMPERATURE:-0.5}"
P_EDGE="${P_EDGE:-0.2}"
P_FEAT="${P_FEAT:-0.2}"
CLUSTER="${CLUSTER:-kmeans}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"

# Override with command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)   DATASET="$2";   shift 2 ;;
        --epochs)    EPOCHS="$2";    shift 2 ;;
        --cluster)   CLUSTER="$2";   shift 2 ;;
        --device)    DEVICE="$2";    shift 2 ;;
        --lr)        LR="$2";        shift 2 ;;
        --seed)      SEED="$2";      shift 2 ;;
        *) shift ;;
    esac
done

echo -e "${GREEN}[Config]${NC}"
echo "  Dataset    : $DATASET"
echo "  Epochs     : $EPOCHS"
echo "  Cluster    : $CLUSTER"
echo "  Device     : $DEVICE"
echo "  Seed       : $SEED"
echo ""

# ---- Step 1: Check Python ----
echo -e "${YELLOW}[Step 1] Checking environment...${NC}"
python --version
echo ""

# ---- Step 2: Create output dirs ----
echo -e "${YELLOW}[Step 2] Creating directories...${NC}"
mkdir -p checkpoints outputs data/raw
echo "  checkpoints/  outputs/  data/raw/  created"
echo ""

# ---- Step 3: Run training ----
echo -e "${YELLOW}[Step 3] Running ScaleComm training...${NC}"
python training/train.py \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --hidden_dim "$HIDDEN_DIM" \
    --out_dim "$OUT_DIM" \
    --num_heads "$NUM_HEADS" \
    --num_layers "$NUM_LAYERS" \
    --lr "$LR" \
    --dropout "$DROPOUT" \
    --temperature "$TEMPERATURE" \
    --p_edge "$P_EDGE" \
    --p_feat "$P_FEAT" \
    --cluster "$CLUSTER" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --log_every 10 \
    --eval_every 50

echo ""

# ---- Step 4: Evaluate ----
echo -e "${YELLOW}[Step 4] Running evaluation...${NC}"
python training/evaluate.py \
    --dataset "$DATASET" \
    --checkpoint "checkpoints/scalecomm_${DATASET}_best.pt" \
    --hidden_dim "$HIDDEN_DIM" \
    --out_dim "$OUT_DIM" \
    --num_heads "$NUM_HEADS" \
    --num_layers "$NUM_LAYERS" \
    --cluster "$CLUSTER" \
    --device "$DEVICE" \
    --visualize

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Pipeline complete! Check outputs/ dir   ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
