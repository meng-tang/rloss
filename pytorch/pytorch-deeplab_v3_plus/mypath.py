class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            # folder that contains pascal/. It should have three subdirectories 
            # called "JPEGImages", "SegmentationClassAug", and "pascal_2012_scribble" 
            # containing RGB images, groundtruth, and scribbles respectively.
            return '/path/to/datasets/VOCdevkit/VOC2012/'  
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
