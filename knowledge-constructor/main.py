# custom py file import
from knowledgeConstructor import KnowledgeConstructor

# directory for training data and xml
datasetVOC="/home/ian-djakman/Documents/data/voc_dataset/VOCdevkit/VOC2012/SegmentationClass"
xmlVOC="/home/ian-djakman/Documents/data/voc_dataset/VOCdevkit/VOC2012/mergedxml"
datasetTest="img/"
datasetTest2="img2/"
				
#Work to be done by the program is listed here
vocKC = KnowledgeConstructor("VocOutput")
vocKC.set_training_data_dir(datasetVOC)
vocKC.set_xml_data_dir(xmlVOC)
#vocKC.cooccurence_knowledge()
#vocKC.cooccurence_knowledge_from_xml()
#vocKC.spatial_knowledge(33.33 , 33.33 , 33.33)
vocKC.scene_properties_knowledge ()
#vocKC.relative_knowledge()

