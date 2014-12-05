#!/usr/bin/python

import numpy as np
from lxml import etree

def load_cooccurence_knowledge(filepath):
    knowledge = {}

    tree = etree.parse(filepath)
    root = tree.getroot()
    
    for sub in root:
        local_knowledge = {}
        for sub_2 in sub:
            for sub_3 in sub_2:
                local_knowledge[sub_3.tag] = float(sub_3.attrib['norm_freq'])

        knowledge[sub.attrib['name']] = local_knowledge

    return knowledge

def load_spatial_knowledge(filepath):
    knowledge = {}

    tree = etree.parse(filepath)
    root = tree.getroot()

    #
    for sub in root:
        local_knowledge = {}
        for sub_2 in sub:
                local_knowledge[sub_2.tag] = float(sub_2.attrib['norm_freq'])

        knowledge[sub.tag] = local_knowledge

    #
    ratio_top = float(root.attrib['top'])
    ratio_center = float(root.attrib['center'])
    ratio_bottom = float(root.attrib['bottom'])
    param = {'ratio_top': ratio_top, 'ratio_center': ratio_center, 'ratio_bottom': ratio_bottom}
    knowledge['param'] = param

    return knowledge

def load_relativeposition_knowledge(filepath):
	knowledge = {}
	
	tree = etree.parse(filepath)
	root = tree.getroot()
	for sub in root:
		local_knowledge = {}
		for sub_2 in sub:
			local_position_knowledge={}
			for sub_3 in sub_2:
				local_position_knowledge[sub_3.attrib['name']] = float(sub_3.attrib['norm_freq'])
			local_knowledge[sub_2.tag]=local_position_knowledge
		knowledge[sub.attrib['name']] = local_knowledge	
	return knowledge

def load_sceneproperty_knowledge(filepath):
    knowledge = {}

    tree = etree.parse(filepath)
    root = tree.getroot()

    for sub in root:
        scene_class = sub.attrib['name']
        obj_freq_map = {}
        for sub_2 in sub:
            for sub_3 in sub_2:
                obj_freq_map[sub_3.tag] = float(sub_3.attrib['norm_freq'])
        knowledge[scene_class] = obj_freq_map

    return knowledge

def load_knowledge(knowledge_dir):
    knowledge = {}
    knowledge['c'] = load_cooccurence_knowledge(knowledge_dir+'/cooccurence_knowledge.xml')
    knowledge['s'] = load_spatial_knowledge(knowledge_dir+'/spatial_knowledge.xml')
    knowledge['r'] = load_relativeposition_knowledge(knowledge_dir+'/relativeposition_knowledge.xml')
    knowledge['p'] = load_sceneproperty_knowledge(knowledge_dir+'/scene_properties_knowledge.xml')

    return knowledge

def main():
    knowledge_dir = '/home/ian-djakman/Documents/data/output/knowledge-compatibility-benchmarker/knowledge'
    knowledge = load_knowledge(knowledge_dir)
if __name__ == '__main__':
    main()
