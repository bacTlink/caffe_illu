import os, sys
caffe_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../python/')
sys.path.append(caffe_path)
caffe_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../../../caffe_illu/python/')
sys.path.append(caffe_path)

import caffe
import numpy as np
import lmdb
import shutil
import random

src_dir = '/data3/lzh/10000x224x224_ring_images_diff/'
dst_dir = '/data3/lzh/10000x10x224x224_ring_images_diff/'
filelist = os.path.join(src_dir, 'filelist.txt')
img_count = 10

count = 0
path = os.path.join(dst_dir, 'label,data')
shutil.rmtree(path, ignore_errors = True)
if not os.path.exists(path):
    os.makedirs(path)
env = lmdb.open(path, map_size=int(1e12))

with env.begin(write=True) as txn:
    for line in open(filelist):
        label_filename = line[:-1]
        print label_filename

        # process label
        label_img = caffe.io.load_image(os.path.join(src_dir, label_filename))
        shape = label_img.shape
        label = []
        for i in xrange(3):
            label.append(label_img[:, :, i].reshape(1, shape[0], shape[1]))

        # process data
        base_filename = label_filename[:-9]
        imgs = []
        for i in xrange(1, img_count + 1):
            filename = base_filename + '_' + str(i) + '.png'
            imgs.append(caffe.io.load_image(os.path.join(src_dir, filename)))
        data = []
        for img in imgs:
            assert (shape == img.shape)
            for i in xrange(3):
                tmp_data = img[:, :, i].reshape(1, shape[0], shape[1])
                if len(data) < 3:
                    data.append(tmp_data)
                else:
                    data[i] = np.append(data[i], tmp_data, axis = 0)

        for i in xrange(2):
            datum = caffe.io.array_to_datum(np.append(label[i], data[i], axis = 0))
            txn.put(str(random.randint(0, int(1e10))) + '-' + base_filename + 'c' + str(i), datum.SerializeToString())

