'''
Goal:
Implement the relative location probability map proposed in
Gould, 2008, IJCV: Multi-Class Segmentation with Relative Location Prior
'''
import numpy as np
import os
import shutil
import random

import sys
sys.path.append('../util/src-py')

from skimage.util import img_as_float
from skimage.segmentation import slic
from skimage import io
from skimage.segmentation import mark_boundaries
from skimage.filter import gaussian_filter

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import scipy
import msrc

def init_prob_map(cprime_labels, c_labels, size):
    prob_map = dict.fromkeys(cprime_labels)
    for key in prob_map:
        sub_prob_map = dict.fromkeys(c_labels)
        for subkey in sub_prob_map:
            sub_prob_map[subkey] = np.zeros(size)
        prob_map[key] = sub_prob_map

    return prob_map

def get_segment(img):
    segmentation = slic(img, n_segments=140, compactness=13, sigma=4, enforce_connectivity=True)
    segmentation_img = mark_boundaries(img, segmentation)
    # io.imsave('slic.jpg', segmentation_img)

    return segmentation

def get_segment_list(segmentation):
    n_segment = len(set(segmentation.flatten()))
    segment_list = [[] for i in range(n_segment)]

    for row in range(segmentation.shape[0]):
        for col in range(segmentation.shape[1]):
            segment_list[ int(segmentation[row][col]) ].append( (row,col) )

    return segment_list

def get_centroid(segment):
    c_row = sum([p[0] for p in segment])/float(len(segment))
    c_col = sum([p[1] for p in segment])/float(len(segment))
    
    return ( int(c_row), int(c_col) )

def get_label(centroid, annotation):
    label_id = annotation[centroid]
    label_name = msrc.class_id2name_map[label_id]

    return {'id': label_id, 'name': label_name}

def get_pixel_of_label(label, annotation):
    pixels = []
    for row in range(annotation.shape[0]):
        for col in range(annotation.shape[1]):
            if annotation[row][col] == label['id']:
                pixels.append( (row,col) )

    return pixels

def get_offset(pixel_1, pixel_2):
    '''
    Notice the used coordinate system:
    x+
    y+
    Origin
    '''
    drow = -1.0 * (pixel_2[0] - pixel_1[0])
    dcol = pixel_2[1] - pixel_1[1]

    return (drow,dcol)

def add_diriclet_noise(offset, alpha, relative_location_matrix_shape):
    '''
    Because of the sparsity of training examples in the multi-class scenario, 
    we apply a Dirichlet prior with parameter $\alpha= 5$ to the relative offset count. 
    '''
    noisy = np.random.dirichlet(alpha=alpha, size=1)
    noisy_drow = noisy[0][0] * relative_location_matrix_shape[0]/2 # divided by two as the offset is measured from the centroid of a segment that is alwaus in the center of the relative location mat
    noisy_dcol = noisy[0][1] * relative_location_matrix_shape[1]/2 # same as above

    noisy_offset = ( offset[0]+int(noisy_drow), offset[1]+int(noisy_dcol) )
    return noisy_offset

def normalize_offset(offset, relative_location_matrix_shape, img_shape):
    '''
    we normalize the offsets by the image img_width and img_height.
    the prob map is defined over the range [-1, 1] in normalized image coordinates
    the x axis is horizontal axis, associated with columns
    the y axis is horizontal axis, associated with rows
    divided by 2, because the segment centroid, from which the offset is calculated, is always positioned in the center of prob map
    '''
    normalizer_x = 1.0
    normalizer_y = normalizer_x

    # 
    norm_drow = float(offset[0])/img_shape[0] * normalizer_y * (relative_location_matrix_shape[0]/2)
    norm_dcol = float(offset[1])/img_shape[1] * normalizer_x * (relative_location_matrix_shape[1]/2)

    return ( int(norm_drow),int(norm_dcol) )

def get_weight(segment):
    '''
    weight counts by the number of pixels in each superpixel
    TODO consider to weight counts by the number of pixels in each superpixel whose label is the same as the centroid_label
    '''
    return len(segment)

def normalize_prob_map(prob_map, relative_location_matrix_shape):
    '''
    We also have $sum_{c=1}^K M_{c|c'} (\hat{u}, \hat{v}) = 1$ , 
    so that $M_{c|c'}$ represents a proper conditional probability distribution over labels, c.
    '''
    total_count_mat_dict = dict.fromkeys(prob_map.keys())
    for key in total_count_mat_dict:
        total_count_mat_dict[key] = np.zeros(relative_location_matrix_shape)

    print('calculating total_count_mat_dict ...')
    for cprime, prob_map_c_given_cprime in prob_map.iteritems():
        for row in range(relative_location_matrix_shape[0]):
            for col in range(relative_location_matrix_shape[1]):
                for c,relative_location_mat in prob_map_c_given_cprime.iteritems():
                    # if c is not 'grass': # TODO remove me
                    #     continue
                    total_old = total_count_mat_dict[cprime][row][col]
                    add = relative_location_mat[row][col]
                    total_count_mat_dict[cprime][row][col] =  total_old + add

    print('normalizing to sum up to 1.0 ...')
    cprime_labels = prob_map.keys()
    c_labels = next(prob_map.itervalues()).keys()
    norm_prob_map = init_prob_map(cprime_labels, c_labels, relative_location_matrix_shape)

    for cprime, prob_map_c_given_cprime in prob_map.iteritems():
        for c,relative_location_mat in prob_map_c_given_cprime.iteritems():
            # if c is not 'grass': # TODO remove me
            #     continue
            # print('processing prob_map: c= %s, given cprime= %s' % (c,cprime))
            for row in range(relative_location_matrix_shape[0]):
                for col in range(relative_location_matrix_shape[1]):
                    val = relative_location_mat[row][col]
                    total = total_count_mat_dict[cprime][row][col]

                    if total > 0.0:
                        norm_prob_map[cprime][c][row][col] = val / total

    return norm_prob_map            

def apply_gaussian_filter(sigma, prob_map, relative_location_matrix_shape):
    cprime_labels = prob_map.keys()
    c_labels = next(prob_map.itervalues()).keys()
    norm_prob_map = init_prob_map(cprime_labels, c_labels, relative_location_matrix_shape)

    filtered_prob_map = init_prob_map(cprime_labels, c_labels, relative_location_matrix_shape)
    for cprime, prob_map_c_given_cprime in prob_map.iteritems():
        for c,relative_location_mat in prob_map_c_given_cprime.iteritems():
            filtered_relative_location_mat = gaussian_filter(relative_location_mat, sigma=sigma, multichannel=True)
            filtered_prob_map[cprime][c] = filtered_relative_location_mat
    return filtered_prob_map

def write_prob_map(prob_map, out_dir):
    for cprime, prob_map_c_given_cprime in prob_map.iteritems():
        cprime_dir = out_dir + '/' + cprime
        if os.path.exists(cprime_dir):
            shutil.rmtree(cprime_dir)
        os.makedirs(cprime_dir)

        for c,relative_location_mat in prob_map_c_given_cprime.iteritems():
            # if c is not 'grass': # TODO remove me
            #     continue

            mat_filepath = cprime_dir + '/' + c + '.csv'
            np.savetxt(mat_filepath, relative_location_mat, delimiter=",")

            mat_img_filepath = cprime_dir + '/' + c + '_wrt_' + cprime + '.png'
            plt.imshow(relative_location_mat, cmap = cm.Greys_r)
            plt.axis('off') # clear x- and y-axes
            plt.savefig(mat_img_filepath)
            # scipy.misc.imsave(mat_img_filepath, relative_location_mat)

def main(argv):
    #
    assert len(argv)==6, 'INSUFFICIENT NUMBER OF ARGVs'
    chosen_cprime = argv[1]
    img_list_filepath = argv[2]
    gt_csv_dir = argv[3]
    img_dir = argv[4]
    prob_map_out_dir = argv[5]

    #
    relative_location_matrix_shape = (200,200) # following [Gould, 2008]
    variance_factor = 0.10 # following [Gould, 2008]
    diriclet_noise_alpha = (5.0,5.0) # following [Gould, 2008]

    #
    c_labels = [{'id':key, 'name':val} for key,val in msrc.class_id2name_map.iteritems() if val is not 'void']
    cprime_labels = c_labels
    if chosen_cprime is not 'all':
        cprime_labels = [{'id':key, 'name':val} for key,val in msrc.class_id2name_map.iteritems() if val==chosen_cprime]

    prob_map = init_prob_map([i['name'] for i in cprime_labels], [i['name'] for i in c_labels], relative_location_matrix_shape)
    print('Processing for c_prime= %s') % (prob_map.keys())

    with open(img_list_filepath) as f:
        img_ids = f.readlines()
    img_ids = [x.strip('\n') for x in img_ids]

    for i, img_id in enumerate(img_ids):
        img_filepath = img_dir + '/' + img_id +'.bmp'
        img = img_as_float(io.imread(img_filepath))
        img_height = img.shape[0] 
        img_width = img.shape[1]

        segmentation = get_segment(img)
        segment_list = get_segment_list(segmentation)

        gt_ann_filepath = gt_csv_dir + '/' + img_id + '.csv'
        gt_annotation = np.genfromtxt(gt_ann_filepath, delimiter=',')
        # print('gt_ann_filepath= %s' % (gt_ann_filepath))
        assert img_height==gt_annotation.shape[0] 
        assert img_width==gt_annotation.shape[1]

        for j, segment in enumerate(segment_list):
            centroid = get_centroid(segment)
            centroid_label = get_label(centroid, gt_annotation)
            centroid_weight = get_weight(segment)

            if centroid_label['name'] is 'void':
                continue

            if centroid_label['name'] not in prob_map.keys():
                continue

            for label in c_labels:
                # if label['name'] is not 'grass':# TODO remove me
                #     continue                

                pixels = get_pixel_of_label(label, gt_annotation)
                print ('Processing img_id=%s (%i/%i): segment_id=%i (%i/%i): centroid_label=%s: pair_label=%s (n_pixel=%i)' % (img_id,i+1,len(img_ids),j,j+1,len(segment_list),centroid_label['name'], label['name'],len(pixels)))

                for pixel in pixels:
                    #
                    offset = get_offset(centroid,pixel)
                    diriclet_offset = add_diriclet_noise(offset, diriclet_noise_alpha, relative_location_matrix_shape)
                    norm_offset = normalize_offset(diriclet_offset, relative_location_matrix_shape, gt_annotation.shape)

                    #
                    count = prob_map[centroid_label['name']][label['name']][norm_offset[0]][norm_offset[1]]
                    count = count + centroid_weight
                    # print 'count=', count
                    # print ('local_prob_map[%i][%i] has count= %i' % (norm_offset[0],norm_offset[1],count))

                    prob_map[centroid_label['name']][label['name']][norm_offset[0]][norm_offset[1]] = count

    #
    print('normalize_prob_map()...')
    norm_prob_map = normalize_prob_map(prob_map, relative_location_matrix_shape)
    
    #
    print('apply_gaussian_filter() ...')
    sigma = (np.sqrt(variance_factor*img_height),np.sqrt(variance_factor*img_width))
    filtered_norm_prob_map = apply_gaussian_filter(sigma, norm_prob_map, relative_location_matrix_shape)

    #
    print('write_prob_map()...')
    write_prob_map(filtered_norm_prob_map, prob_map_out_dir)

if __name__ == '__main__':
    main(sys.argv)