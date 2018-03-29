import os, sys
caffe_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../python/')
sys.path.append(caffe_path)

import caffe
import numpy as np
import lmdb
import shutil

base_dir = '/data3/lzh/1000x224x224_ring_images/'
filelist = os.path.join(base_dir, 'filelist_shuf.txt')
img_count = 10

count = 0
label_path = os.path.join(base_dir, 'label')
data_path = os.path.join(base_dir, 'data')
shutil.rmtree(label_path, ignore_errors = True)
shutil.rmtree(data_path, ignore_errors = True)
env_label = lmdb.open(label_path, map_size=int(1e12))
env_data = lmdb.open(data_path, map_size=int(1e12))

with env_label.begin(write=True) as txn_label, env_data.begin(write=True) as txn_data:
    for line in open(filelist):
        label_filename = line[:-1]
        print label_filename

        # process label
        label_img = caffe.io.load_image(os.path.join(base_dir, label_filename))
        shape = label_img.shape
        label = []
        for i in xrange(3):
            label.append(label_img[:, :, i].reshape(1, shape[0], shape[1]))

        # process data
        base_filename = label_filename[:-9]
        imgs = []
        for i in xrange(1, img_count + 1):
            filename = base_filename + '_' + str(i) + '.png'
            imgs.append(caffe.io.load_image(os.path.join(base_dir, filename)))
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
            datum_label = caffe.io.array_to_datum(label[i])
            datum_data = caffe.io.array_to_datum(data[i])
            txn_label.put(base_filename + str(i), datum_label.SerializeToString())
            txn_data.put(base_filename + str(i), datum_data.SerializeToString())
