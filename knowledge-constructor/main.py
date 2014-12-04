# custom py file import
from knowledgeConstructor import KnowledgeConstructor

# directory for training data and xml
annotated_img_voc="/home/ian-djakman/Documents/data/input/voc_dataset_2012/SegmentationClass"
sceneprop_xml_voc="/home/ian-djakman/Documents/data/input/voc_dataset_2012/ckpstar_modified_annotation"
relation_xml_voc ="/home/ian-djakman/Documents/data/input/voc_dataset_2012/voc_gt_xml"
output_path = "/home/ian-djakman/Documents/data/output/knowledge-compatibility-benchmarker/knowledge"
				
#Construct an Instance of Knowledge Constructor, providing its output name
vocKC = KnowledgeConstructor(output_path)
vocKC.set_training_data_dir(annotated_img_voc)
vocKC.set_general_xml_data_dir(sceneprop_xml_voc)
vocKC.set_relative_position_xml_data_dir(sceneprop_xml_voc)

#Construct Knowledge
vocKC.cooccurence_knowledge_from_xml()
vocKC.spatial_knowledge(33.33 , 33.33 , 33.33)
vocKC.scene_properties_knowledge ()
vocKC.relative_knowledge(20.0 , 60.0)
vocKC.gould_relative_knowledge()

