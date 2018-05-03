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

src_dir = '/data3/lzh/10000x672x672_Diamond_diff/'
dst_dir = '/data3/lzh/10000x10x224x224_Diamond_diff/'
filelist = os.path.join(src_dir, 'filelist.txt')
img_count = 10

if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

def sec_into_224(pic1, pic2):
    assert pic1.ndim == 4
    assert pic2.ndim == 4
    if pic1.shape[2] == 224:
        return pic1, pic2
    assert pic1.shape[2] == 672
    x = random.randint(0,672-224)
    y = random.randint(0,672-224)
    return pic1[:, :, x:x+224, y:y+224], pic2[:, :, x:x+224, y:y+224]

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
tot_cnt = 0
for line in open(filelist):
    label_filename = line[:-1]
    print label_filename

    # process label
    label_img = caffe.io.load_image(os.path.join(src_dir, label_filename))
    shape = label_img.shape
    label = []
    for i in xrange(3):
        if i == 0:
            label = label_img[:, :, i].reshape(1, 1, shape[0], shape[1])
        else:
            label = np.append(label, label_img[:, :, i].reshape(1, 1, shape[0], shape[1]), axis = 0)

    # process data
    base_filename = label_filename[:-9]
    imgs = []
    for i in xrange(1, img_count + 1):
        filename = base_filename + '_' + str(i) + '.png'
        imgs.append(caffe.io.load_image(os.path.join(src_dir, filename)))
    data = []
    for i in xrange(3):
        img_data = np.array(0)
        for img in imgs:
            tmp_data = img[:, :, i].reshape(1, shape[0], shape[1])
            if img_data.size == 1:
                img_data = tmp_data
            else:
                img_data = np.append(img_data, tmp_data, axis = 0)
        img_data = img_data.reshape(1, img_count, shape[0], shape[1])
        if i == 0:
            data = img_data
        else:
            data = np.append(data, img_data, axis = 0)

    print label.shape
    print data.shape
    label, data = sec_into_224(label, data)

    if random.randint(1, 100) == 1:
        classification = 'test'
    else:
        classification = 'train'
    for i in xrange(3):
        datum = caffe.io.array_to_datum(np.append(label[i], data[i], axis = 0))
        img_id = str(tot_cnt)
        while len(img_id) < 6:
            img_id = '0' + img_id
        img_id += 'c' + str(i)
        if classification == 'train':
            train_txn.put(img_id, datum.SerializeToString())
        else:
            test_txn.put(img_id, datum.SerializeToString())

    cnt = cnt + 1
    tot_cnt = tot_cnt + 1
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

