#!/bin/sh

# download scribble annotations from http://cs.uwaterloo.ca/~m62tang/rloss/
# annotation is given in the format of pixelwise labeling
if [ ! -f "pascal_2012_scribble.zip" ];
then
	wget http://cs.uwaterloo.ca/~m62tang/rloss/pascal_2012_scribble.zip
	unzip pascal_2012_scribble.zip
fi

# Alternatively, scribbles can be obtained from http://www.jifengdai.org/downloads/scribble_sup/
# which is in the format of XY coordinates of scribble skeleton.
# Then obtain pixel-wise labeling by running convertscribbles.m
if [ ! -f "scribble_annotation.zip" ];
then
	wget https://www.dropbox.com/s/9vh3kvtd742red8/scribble_annotation.zip --no-check-certificate
	unzip scribble_annotation.zip
fi

# JPEG images
ln -s ../VOC2012/VOCdevkit/VOC2012/JPEGImages JPEGImages

# ground truth
ln -s ../VOC2012/SegmentationClassAug SegmentationClassAug

# list of train set and val set
wget http://cs.uwaterloo.ca/~m62tang/rloss/SegmentationAug.zip
unzip SegmentationAug.zip -d ImageSets
