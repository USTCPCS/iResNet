#!/usr/bin/env python
import os, sys
import subprocess
import time
from math import ceil

my_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(my_dir)

caffe_bin = '/home/leo/caffe_lzf/build/tools/caffe.bin'
img_size_bin = '/home/leo/caffe_lzf/build/tools/get_image_size'

template = 'model/deploy_iresnet.tpl.prototxt'

# =========================================================


def get_image_size(filename):
    global img_size_bin
    dim_list = [int(dimstr) for dimstr in str(subprocess.check_output([img_size_bin, filename])).split(',')]
    if not len(dim_list) == 2:
        print('Could not determine size of image %s' % filename)
        sys.exit(1)
    return dim_list


def sizes_equal(size1, size2):
    return size1[0] == size2[0] and size1[1] == size2[1]


if not (os.path.isfile(caffe_bin) and os.path.isfile(img_size_bin)):
    print('Caffe tool binaries not found. Did you compile caffe with tools (make all tools)?')
    sys.exit(1)


subprocess.call('mkdir -p tmp', shell=True)

for idx in range(200):
  
    im0 = '/home/leo/DataSet/data_scene_flow/testing/image_2/%06d_10.png' % idx
    im1 = '/home/leo/DataSet/data_scene_flow/testing/image_3/%06d_10.png' % idx
        
    if not os.path.isfile(im0):
        print('Image %s not found' % im0)
        sys.exit(1)

    if not os.path.isfile(im1):
        print('Image %s not found' % im1)
        sys.exit(1)

    im0_size = get_image_size(im0)
    im1_size = get_image_size(im1)

    if not (sizes_equal(im0_size, im1_size)):
        print('The images do not have the same size. (Images: %s and %s )\n Please use the pair-mode.' %  (im0, im1))
        sys.exit(1)
    
    with open('tmp/img1.txt', "w") as tfile:
        tfile.write("%s\n" % im0)

    with open('tmp/img2.txt', "w") as tfile:
        tfile.write("%s\n" % im1)
    
    
    width  = im0_size[0]
    height = im0_size[1]

    divisor = 64.
    adapted_width = ceil(width/divisor) * divisor
    adapted_height = ceil(height/divisor) * divisor
    rescale_coeff_x = width / adapted_width

    replacement_list = {
        '$ADAPTED_WIDTH': ('%d' % adapted_width),
        '$ADAPTED_HEIGHT': ('%d' % adapted_height),
        '$TARGET_WIDTH': ('%d' % width),
        '$TARGET_HEIGHT': ('%d' % height),
        '$SCALE_WIDTH': ('%.8f' % rescale_coeff_x),
        '$IMG_NUM':('%04d' % idx)
    }

    proto = ''
    with open(template, "r") as tfile:
        proto = tfile.read()

    for r in replacement_list:
        proto = proto.replace(r, replacement_list[r])

    with open('tmp/deploy.prototxt', "w") as tfile:
        tfile.write(proto)

    # Run caffe

    args = [caffe_bin, 'test', '-model', 'tmp/deploy.prototxt',
            '-weights', './model/iresnet_kitti2015.caffemodel',
            '-iterations', '1',
            '-gpu', '0']

    cmd = str.join(' ', args)
    print('Executing %s' % cmd)

    subprocess.call(args)

    print('\nThe resulting disparity is stored in ./results_itr2/kitti-iresnet-%07d.pfm' % idx)

