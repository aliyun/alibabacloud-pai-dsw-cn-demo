# DAWNBench and MLPerf Training Setup on DSW terminal mode(jupyter version coming soon)
## GPU Instance
pai.medium.1xv100

## Docker Image Version
py27_cuda90_tf1.12_ubuntu

## Train Steps:
Before training, the parameters of teacher model (unpruned_teacher.pkl) should be firstly download and put into a local path `./gap_save`. The OSS URL is: http://230388yyw.oss-cn-hangzhou-zmf.aliyuncs.com/data/unpruned_teacher.pkl

Run below command to start training:
```
sh train.sh
```

ImageNet dataset is default loaded from a local path by the option `--data_dir`. If the option `-osl` is enabled, dataset will be directly loaded from OSS.

The model version is set with the option `--version 34` for ResNet26-s4 (DAWNBench model), and `--version 24` for ResNet26-s5 (MLPerf Open model).

The trained model is default saved to `./model`, which can be changed with the option `--model_dir`.

## INT8-aware Training Steps:

Before INT8-aware training, the parameters of trained model should be generated and saved to `./gap_save`:
```
python gap_prune.py --model model/model.ckpt-250200 -fs
```

The parameter file (ResNet26-s4 as example) can also be download at: http://230388yyw.oss-cn-hangzhou-zmf.aliyuncs.com/data/res26_s4.pkl

And off-line calibrated scaling factors should be put into a local path `./calib`. For ResNet26-s4, the OSS URL is: http://230388yyw.oss-cn-hangzhou-zmf.aliyuncs.com/data/act_dict_8.npy

Then run below command to start INT8-aware training:
```
sh ft_int.sh
```

## Export Saved Model
Run command:
```
sh export.sh
```

## OSS Setting
If the option `-osl` is enabled, the OSS access should be correctly setup, which can be found in: `./imagenet_main.py` and `./imagenet_export.py`

_ACCESS_ID = ""

_ACCESS_KEY = ""

_HOST = ""

_BUCKET = ""

## OSS URLs
ImageNet dataset:

http://230388yyw.oss-cn-hangzhou-zmf.aliyuncs.com/data/train-00000-of-00008

http://230388yyw.oss-cn-hangzhou-zmf.aliyuncs.com/data/train-00001-of-00008

http://230388yyw.oss-cn-hangzhou-zmf.aliyuncs.com/data/train-00002-of-00008

http://230388yyw.oss-cn-hangzhou-zmf.aliyuncs.com/data/train-00003-of-00008

http://230388yyw.oss-cn-hangzhou-zmf.aliyuncs.com/data/train-00004-of-00008

http://230388yyw.oss-cn-hangzhou-zmf.aliyuncs.com/data/train-00005-of-00008

http://230388yyw.oss-cn-hangzhou-zmf.aliyuncs.com/data/train-00006-of-00008

http://230388yyw.oss-cn-hangzhou-zmf.aliyuncs.com/data/train-00007-of-00008

http://230388yyw.oss-cn-hangzhou-zmf.aliyuncs.com/data/validation-00000-of-00001
