set -e

log_dir="./log"
if [ ! -d $log_dir ]; then
  mkdir $log_dir
fi

ckpt_dir="./model"
if [ ! -d $ckpt_dir ]; then
  mkdir $ckpt_dir
fi

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# run benchmark
seed=${1:-${start}}

export COMPLIANCE_FILE="log/resnet_compliance_${seed}.log"
CONSOLE_LOG="log/resnet_run_${seed}.log"

# Register the model as a source root
export PYTHONPATH="$(pwd):${PYTHONPATH}"

#MODEL_DIR="/models/resnet_imagenet_${RANDOM_SEED}"

CUDA_VISIBLE_DEVICES=4,5,6,7 python imagenet_main.py $seed \
  --model_dir "./model" \
  --data_dir "/data/ImageNet_TFRecorder" \
  -gt -ec --learn_rate 0.128 --label_smoothing 0.1 --gap_lambda 0.0001 \
  --num_gpus 4 --train_epochs 90 --stop_threshold 0.7646 --batch_size 256 \
  --version 1 --resnet_size 50 --epochs_between_evals 10 | tee "log/resnet_gap_${seed}.log"
