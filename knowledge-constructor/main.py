# custom py file import
#import metaConstructor
from knowledgeConstructor import KnowledgeConstructor
#from knowledgeLoader import KnowledgeLoader

# directory for training data and xml
datasetVOC="/home/ian-djakman/Documents/data/voc_dataset/VOCdevkit/VOC2012/SegmentationClass"
xmlVOC="/home/ian-djakman/Documents/data/voc_dataset/VOCdevkit/VOC2012/mergedxml"
datasetTest="img/"
datasetTest2="img2/"
				
#Work to be done by the program is listed here
vocKC = KnowledgeConstructor("VocOutput")
vocKC.set_training_data_dir(datasetVOC)
vocKC.set_xml_data_dir(xmlVOC)
#vocKC.c()
#vocKC.s(100)
#vocKC.p()
vocKC.r()

#kl = KnowledgeLoader()
#kl.c()
#kc.c(bla, bla)
