
log_dir="./log"
if [ ! -d $log_dir ]; then
  mkdir $log_dir
fi

CUDA_VISIBLE_DEVICES=0 python main_eval.py --model ../export/ --bits 8 --log-name test
