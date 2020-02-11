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

CUDA_VISIBLE_DEVICES=0 python imagenet_main.py $seed \
  --model_dir "./model" \
  --data_dir "/home/admin/workspace/data/ImageNet_TFRecorder" \
  --num_gpus 1 --train_epochs 200 --stop_threshold 0.82 --batch_size 128 \
  -ec --cos_alpha 0.0001 --learn_rate 0.18 --label_smoothing 0.1 \
  -ek --version_t 14 --kd_size 101 --temp_dst 2. --w_dst 2. -mu --mx_mode 1 \
  --version 34 --resnet_size 26 --epochs_between_evals 5 | tee "log/resnet_submission_${seed}.log"
