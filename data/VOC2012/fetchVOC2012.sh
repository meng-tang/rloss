#!/bin/sh

# download training/validation data of VOC2012
if [ ! -f "VOCtrainval_11-May-2012.tar" ];
then
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
	tar -xvf VOCtrainval_11-May-2012.tar
	ln -s VOCdevkit/VOC2012/JPEGImages JPEGImages
fi

# augmented VOC12 (originally from http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz)
if [ ! -f "SegmentationClassAug.zip" ];
then
	wget https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip --no-check-certificate
	unzip SegmentationClassAug.zip
fi

