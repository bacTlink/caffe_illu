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

src_dir = '/data3/lzh/10000x672x672_box_diff/'
dst_dir = '/data3/lzh/10000x10x224x224_box_diff/'
filelist = os.path.join(src_dir, 'filelist.txt')
img_count = 10

if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

def sec_into_224(pics):
    assert pics.ndim == 3
    if pics.shape[1] == 224:
        return pics.reshape(1, pics.shape[0], pics.shape[1], pics.shape[2])
    assert pics.shape[1] == 672
    for i in xrange(3):
        for j in xrange(3):
            tmp = pics[:, i*224:i*224+224, j*224:j*224+224].reshape(1, pics.shape[0], 224, 224)
            if i == 0 and j == 0:
                res = tmp
            else:
                res = np.append(res, tmp, axis = 0)
    return res

def mse(pic1, pic2):
    assert pic1.ndim == 2
    assert pic2.ndim == 2
    assert pic1.shape == pic2.shape
    pic1 = pic1 * 255
    pic2 = pic2 * 255
    return np.mean((pic1 - pic2) ** 2)

def large_mse(label, data):
    mse_res = mse(label[0,:,:], np.mean(data, axis = 0))
    if (mse_res > 20):
        print mse_res
    return mse_res > 20

count = 0
train_path = os.path.join(dst_dir, 'train-label,data')
test_path = os.path.join(dst_dir, 'test-label,data')
shutil.rmtree(train_path, ignore_errors = True)
shutil.rmtree(test_path, ignore_errors = True)
train_env = lmdb.open(train_path, map_size=int(1e12))
test_env = lmdb.open(test_path, map_size=int(1e12))

train_txn = train_env.begin(write=True)
test_txn = test_env.begin(write=True)
cnt = 0
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

    for i in xrange(3):
        label[i] = sec_into_224(label[i])
        data[i] = sec_into_224(data[i])

    for i in xrange(3):
        for j in xrange(data[i].shape[0]):
            datum = caffe.io.array_to_datum(np.append(label[i][j,:,:,:], data[i][j,:,:,:], axis = 0))
            if random.randint(0, 100) == 0:
                classification = 'test'
            else:
                classification = 'train'
            if classification == 'train':
                img_id = str(random.randint(0, int(1e10))) + '-' + base_filename + 'c' + str(i)
                train_txn.put(img_id, datum.SerializeToString())
            else:
                img_id = base_filename + 'c' + str(i)
                test_txn.put(img_id, datum.SerializeToString())

    cnt = cnt + 1
    if (cnt == 100):
        train_txn.commit()
        test_txn.commit()
        cnt = 0
        train_txn = train_env.begin(write=True)
        test_txn = test_env.begin(write=True)

if cnt != 0:
    train_txn.commit()
    test_txn.commit()
    cnt = 0
    train_txn = train_env.begin(write=True)
    test_txn = test_env.begin(write=True)
train_env.close()
test_env.close()
