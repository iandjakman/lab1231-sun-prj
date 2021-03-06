#!/usr/bin/python

import numpy as np 
from lxml import etree
from knowledge_loader import load_knowledge
from pascal_voc_2012 import translate
from pascal_voc_2012 import class_name_map

def get_spatial(i, height, param):
    top_height = int( round(param['ratio_top']*height) )
    center_height = int( round(param['ratio_center']*height) )
 
    top_center_th = top_height
    center_bottom_th = top_height + center_height

    if i < top_center_th:
        return 'top'
    elif i < center_bottom_th:
        return 'center'
    else:
        return 'bottom'

def get_scene_class(filename):
    '''
    This acts as an oracle that returns an absolutely correct scene class of a given image.
    \param filename 
    \return scene class
    '''
    annotation_dir = '/home/tor/sun3/dataset/pascal/VOC2012/VOCdevkit/VOC2012/ScenePropertyAnnotations/cikupastar-20141111/ScenePropertyAnnotations'
    filepath = annotation_dir + '/' + filename + '.xml'

    tree = etree.parse(filepath)
    root = tree.getroot()
    sub = root[0]
    scene_class = sub.attrib['name']

    return scene_class

def extract_fea_c(ann, knowledge):
    numeric_classes = list( set(ann.flatten()) )
    raw_classes = translate(numeric_classes)
    classes = [i for i in raw_classes if i!='background' and i!='void']

    probs = [knowledge[classes[i]][classes[j]] for i in range(len(classes)-1) for j in range(i+1,len(classes))]
    probs = np.asarray(probs)

    n_element = 4
    c = [0.0] * n_element
    if len(probs) != 0:
        c = [np.mean(probs), 
             np.percentile(probs, 25), 
             np.percentile(probs, 50), 
             np.percentile(probs, 75)]

    return c

def extract_fea_s(ann, knowledge):
    '''
    @brief extract spatial compatibility score
    For each annotated pixel (excluding 'void' and 'background', get the probability of such annotation given pixel spatial location.

    \return the average probility of p_s from all annotated pixel.
    '''
    n_row = ann.shape[0]
    n_col = ann.shape[1]

    probs = []
    for i in range(n_row):
        for j in range(n_col):
            ann_ij = class_name_map[ ann[i][j] ]
            spatial = get_spatial(i, n_row, knowledge['param'])

            if ann_ij!='background':
                prob = knowledge[ann_ij][spatial]
                probs.append(prob)

    n_element = 4
    s = [0.0] * n_element
    if len(probs) != 0:
        s = [np.mean(probs), 
             np.percentile(probs, 25), 
             np.percentile(probs, 50), 
             np.percentile(probs, 75)]
    
    return s

def extract_fea_r(ann, knowledge):
    r = [0.0]
    return r

def extract_fea_p(ann, knowledge, filename):
    '''
    Calculate the probabibility of each present object given the scene class.

    For now, the scene class is given by an Oracle
    '''
    scene_class = get_scene_class(filename).lower()
    numeric_present_objects = list( set(ann.flatten()) )
    raw_present_objects = translate(numeric_present_objects)
    present_objects = [i for i in raw_present_objects if i!='background' and i!='void']

    probs = [knowledge[scene_class][obj] for obj in present_objects if scene_class in knowledge]
    
    n_element = 4
    p = [0.0] * n_element
    if len(probs) != 0:
        p = [np.mean(probs), 
             np.percentile(probs, 25), 
             np.percentile(probs, 50), 
             np.percentile(probs, 75)]

    return p

def extract_fea(ann_filepaths, knowledge_dir):
    '''
    fea = [c, s, r, p]
    c: cooccurence
    s: spatial layout
    r: relative position
    p: scene property
    '''
    #
    knowledge = load_knowledge(knowledge_dir)

    #
    fea_list = []

    for i, ann_filepath in enumerate(ann_filepaths):
        print 'Extracting feature:', str(i+1), 'of', str(len(ann_filepaths))

        ann = np.loadtxt(ann_filepath, delimiter=',')

        c = extract_fea_c(ann, knowledge['c'])
        s = extract_fea_s(ann, knowledge['s'])
        r = extract_fea_r(ann, knowledge['r'])
        p = extract_fea_p(ann, knowledge['p'], ann_filepath.split('/')[-1][0:-4])

        fea = c + s + p #TODO include r
        fea_list.append(fea)

    return fea_list

def main():
    print 'main()'
    
if __name__ == '__main__':
    main()