set -e

log_dir="./log"
if [ ! -d $log_dir ]; then
  mkdir $log_dir
fi

ckpt_dir="./ft_model"
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
  --model_dir "./ft_model" \
  --pickle_model "./gap_save/res26_s4.pkl" \
  --data_dir "/home/admin/workspace/data/ImageNet_TFRecorder" \
  -ec --cos_alpha 0.001 --learn_rate 0.018 --label_smoothing 0.1 -gft -eqz --q_bits 8 --copy_num 1 \
  -ek --version_t 14 --kd_size 101 --temp_dst 2. --w_dst 2. \
  --num_gpus 1 --train_epochs 90 --stop_threshold 0.7746 --batch_size 128 \
  --version 34 --resnet_size 26 --epochs_between_evals 5 | tee "log/resnet_pruned_${seed}.log"
