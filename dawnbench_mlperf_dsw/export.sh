
export_dir="./export"
if [ ! -d $export_dir ]; then
  mkdir $export_dir
fi

CUDA_VISIBLE_DEVICES=0 python imagenet_export.py --model-dir "./model" \
                                                 --data-dir "/home/admin/workspace/data/ImageNet_TFRecorder"
