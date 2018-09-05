#!/bin/sh
echo $HOSTNAME
## MODIFY PATH for YOUR SETTING
ROOT_DIR=/home/m62tang/rloss

CAFFE_DIR=${ROOT_DIR}/deeplab/
CAFFE_BIN=${CAFFE_DIR}/.build_release/tools/caffe.bin

EXP=pascal_scribble

if [ "${EXP}" = "pascal_scribble" ]; then
    NUM_LABELS=21
    DATA_ROOT=${ROOT_DIR}/data/pascal_scribble/
else
    return
fi

## Specify which model to train
########### voc12 ################
#NET_ID=deeplab_largeFOV
#NET_ID=deeplab_msc_largeFOV
#NET_ID=deeplab_vgg16
NET_ID=resnet-101



## Variables used for weakly or semi-supervisedly training
TRAIN_SET_SUFFIX=
#TRAIN_SET_SUFFIX=_aug

DEV_ID=0

#####

## Create dirs

CONFIG_DIR=${EXP}/config/${NET_ID}
MODEL_DIR=${EXP}/model/${NET_ID}
mkdir -p ${MODEL_DIR}
LOG_DIR=${EXP}/log/${NET_ID}
mkdir -p ${LOG_DIR}
export GLOG_log_dir=${LOG_DIR}

## Run

RUN_TRAIN=1
RUN_TRAINWITHDENSECRFLOSS=1
RUN_TEST=1

DENSECRF_LOSS_WEIGHT=1e-8


## Training #1 (on train_aug)

if [ ${RUN_TRAIN} -eq 1 ]; then
    #
    LIST_DIR=${EXP}/list
    TRAIN_SET=train${TRAIN_SET_SUFFIX}
    #
    if [ "${NET_ID}" = "deeplab_largeFOV" ]; then
      # download from http://liangchiehchen.com/projects/Init%20Models.html
	  MODEL=${EXP}/model/${NET_ID}/vgg16_20M.caffemodel
	elif [ "${NET_ID}" = "deeplab_vgg16" ]; then
	  # download from http://liangchiehchen.com/projects/DeepLabv2_vgg.html
	  MODEL=${EXP}/model/${NET_ID}/init.caffemodel
	elif [ "${NET_ID}" = "resnet-101" ]; then
	  # download from http://liangchiehchen.com/projects/DeepLabv2_resnet.html
	  MODEL=${EXP}/model/${NET_ID}/init.caffemodel
	fi
    #
    echo Training net ${EXP}/${NET_ID}
    for pname in train solver; do
				sed "$(eval echo $(cat sub.sed))" \
						${CONFIG_DIR}/${pname}.prototxt > ${CONFIG_DIR}/${pname}_${TRAIN_SET}.prototxt
    done
        CMD="${CAFFE_BIN} train \
         --solver=${CONFIG_DIR}/solver_${TRAIN_SET}.prototxt \
         --weights=${MODEL} \
         --gpu=${DEV_ID}"
		echo $CMD
		echo Running ${CMD} && ${CMD}
		echo $CMD
fi

if [ ${RUN_TRAINWITHDENSECRFLOSS} -eq 1 ]; then
    #
    LIST_DIR=${EXP}/list
    TRAIN_SET=train${TRAIN_SET_SUFFIX}
    if [ "${NET_ID}" = "deeplab_largeFOV" ]; then
	  MODEL=${EXP}/model/${NET_ID}/train_iter_9000.caffemodel
	elif [ "${NET_ID}" = "deeplab_msc_largeFOV" ]; then
	  MODEL=${EXP}/model/deeplab_largeFOV/trainwithdensecrfloss_iter_9000.caffemodel
	elif [ "${NET_ID}" = "deeplab_vgg16" ]; then
	  MODEL=${EXP}/model/${NET_ID}/train_iter_20000.caffemodel
	elif [ "${NET_ID}" = "resnet-101" ]; then
	  MODEL=${EXP}/model/${NET_ID}/train_iter_20000.caffemodel
	fi
    #
    echo Training net ${EXP}/${NET_ID}
    for pname in trainwithdensecrfloss solverwithdensecrfloss; do
				sed "$(eval echo $(cat sub.sed))" \
						${CONFIG_DIR}/${pname}.prototxt > ${CONFIG_DIR}/${pname}_${TRAIN_SET}.prototxt
    done
        CMD="${CAFFE_BIN} train \
         --solver=${CONFIG_DIR}/solverwithdensecrfloss_${TRAIN_SET}.prototxt \
         --weights=${MODEL} \
         --gpu=${DEV_ID}"
		echo $CMD
		echo Running ${CMD} && ${CMD}
		echo $CMD
fi

## Test #1 specification (on val or test)

if [ ${RUN_TEST} -eq 1 ]; then
    #
    for TEST_SET in val; do
				TEST_ITER=`cat ${EXP}/list/${TEST_SET}.txt | wc -l`
				# for deeplab_vgg16
				#MODEL=${EXP}/model/${NET_ID}/train_iter_20000.caffemodel
				#MODEL=${EXP}/model/${NET_ID}/trainwithdensecrfloss_iter_10000.caffemodel
				# for deeplab_largeFOV
				#MODEL=${EXP}/model/${NET_ID}/train_iter_9000.caffemodel
				#MODEL=${EXP}/model/${NET_ID}/trainwithdensecrfloss_iter_9000.caffemodel
				# for deeplab_msc_largeFOV
				#MODEL=${EXP}/model/${NET_ID}/trainwithdensecrfloss_iter_9000.caffemodel
				# for resnet-101
				MODEL=${EXP}/model/${NET_ID}/train_iter_20000.caffemodel
				echo $MODEL
				if [ ! -f ${MODEL} ]; then
						return
				fi
				#
				echo Testing net ${EXP}/${NET_ID}
				FEATURE_DIR=${EXP}/features/${NET_ID}
				sed "$(eval echo $(cat sub.sed))" \
						${CONFIG_DIR}/test.prototxt > ${CONFIG_DIR}/test_${TEST_SET}.prototxt
				CMD="${CAFFE_BIN} test \
             --model=${CONFIG_DIR}/test_${TEST_SET}.prototxt \
             --weights=${MODEL} \
             --gpu=${DEV_ID} \
             --iterations=${TEST_ITER}"
				echo Running ${CMD} && ${CMD}
				echo $CMD
    done
fi


