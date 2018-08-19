#!/bin/sh

# download scribble annotations from http://www.jifengdai.org/downloads/scribble_sup/
if [ ! -f "scribble_annotation.zip" ];
then
	wget https://www.dropbox.com/s/9vh3kvtd742red8/scribble_annotation.zip --no-check-certificate
	unzip scribble_annotation.zip
fi

# JPEG images
ln -s ../VOC2012/VOCdevkit/VOC2012/JPEGImages JPEGImages

# ground truth
ln -s ../VOC2012/SegmentationClassAug SegmentationClassAug
