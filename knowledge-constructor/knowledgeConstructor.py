import os
import glob
import math
import voc_obj_intolbl
import voc_obj_lbltoin
import voc_spa_intopos
import datetime
import cv2
import numpy as np
from PIL import Image
from lxml import etree
from io import StringIO, BytesIO
from skimage.segmentation import slic
from positionSearcher import PositionSearcher

#dictionary
indexToNameDict=voc_obj_intolbl.LABELS
nameToIndexDict=voc_obj_lbltoin.LABELS
indexToPosDict=voc_spa_intopos.POS

class KnowledgeConstructor:	
	
	output_path=None
	traindat=None
	xmldat=None
	rxmlpath=None
	cOutFile=None
	cArraySize=None
	cooccurArray= None
	spatialSize=None
	spatialFile=None
	spatialArray = None
	spatial_top_portion=None
	spatial_center_portion=None
	spatial_bottom_portion=None
	placeList=None
	activityList=None
	propertyArray=None
	pArraySize=None
	rArray=None
	relative_is_around_threshold = None
	relative_is_below_threshold = None
	generatedTime=None
    
	def __init__(self, path):
		
		cName="csv/cooccur.csv"
		sName="csv/spatial.csv"
		
		self.output_path = path
		self.cOutFile=open(cName,"w")
		self.spatialFile=open(sName,"w")
		self.spatialSize=len(indexToPosDict)+1
		self.cArraySize=len(indexToNameDict)+1
		self.cooccurArray= [[0 for x in range(0,self.cArraySize)] for x in range(0,self.cArraySize)]
		self.spatialArray = [[0 for x in range (0,self.spatialSize)] for x in range(0,self.cArraySize)]
		self.rArray = [[[0 for x in range (0,4)] for x in range (0,self.cArraySize)] for x in range(0,self.cArraySize)]
		self.placeList=[]
		self.activityList=[]
		self.generatedTime=datetime.datetime.now()
		
	def set_training_data_dir(self, dataset_dir):
		self.traindat=dataset_dir
	
	def set_general_xml_data_dir(self, sceneprop_dir):
		self.xmldat=sceneprop_dir
		
	def set_relative_position_xml_data_dir(self, relative_dir):
		self.rxmlpath=relative_dir

	def writePrintS(self):
		print("")
		print("")
		print("Spatial Frequency from your array :")
		print("")
		# column label
		print("%14s"%"", end=" ") # for leftmost part, let it be blank for crossing label
		self.spatialFile.write(" ,")
		for x in range (1,self.spatialSize):
			lbl = self.spatialArray[0][x]
			print("%14s"%lbl, end=" ")
			self.spatialFile.write("%s,"%lbl)
		print("")
		self.spatialFile.write("\n")
		# row label and value
		for x in range (1,self.cArraySize):
			lbl = self.spatialArray[x][0]
			print("%14s"%lbl, end=" ")
			self.spatialFile.write("%s,"%lbl)
			for y in range (1,self.spatialSize):
				val = self.spatialArray[x][y]
				print(("%14d"%val),end=" ")
				self.spatialFile.write("%d,"%val)
			print("")
			self.spatialFile.write("\n")
		self.spatialFile.close()
		print("")
	
	def writePrintC(self):
		# Print cooccurrence frequency info into system and file
		print("")
		print("")
		print("Cooccurrence Frequency from your array :")
		print("")
		# column label
		print("%11s"%"", end=" ")
		self.cOutFile.write(" ,")
		for x in range (1,self.cArraySize):
			lbl = self.cooccurArray[0][x]
			print("%11s"%lbl, end=" ")
			self.cOutFile.write("%s,"%lbl)	
		print("")
		self.cOutFile.write("\n")
		# row label and value
		for x in range (1,self.cArraySize):
			lbl = self.cooccurArray[x][0]
			print("%11s"%lbl, end=" ")
			self.cOutFile.write("%s,"%lbl)
			for y in range (1,self.cArraySize):
				val = self.cooccurArray[x][y]
				print(("%11d"%val),end=" ")
				self.cOutFile.write("%d,"%val)
			print("")
			self.cOutFile.write("\n")
		self.cOutFile.close()
		print("")
	
	def writePrintP(self):
		print()
		
	def sXML(self):
		s_tree_writer = etree.Element("spatial_knowledge")
		s_tree_writer.set("dataset","pascal_voc_2010")
		s_tree_writer.set('timestamp', str(self.generatedTime))
		s_tree_writer.set("top", str(self.spatial_top_portion))
		s_tree_writer.set("center", str(self.spatial_center_portion))
		s_tree_writer.set("bottom", str(self.spatial_bottom_portion))
		for x in range (1,self.cArraySize):
			lbl = self.spatialArray[x][0]
			annoClass = etree.SubElement(s_tree_writer, lbl)
			top_value    = self.spatialArray[x][1]
			center_value = self.spatialArray[x][2]
			bottom_value = self.spatialArray[x][3]
			overall_value= self.spatialArray[x][4]
			
			annoT = etree.SubElement(annoClass, "top")
			annoT.set("freq", str(top_value))
			if overall_value!=0:
				annoT.set("norm_freq", str(top_value/overall_value))
			else:
				annoT.set("norm_freq", "0.00")
				
			annoC = etree.SubElement(annoClass, "center")
			annoC.set("freq", str(center_value))
			if overall_value!=0:
				annoC.set("norm_freq", str(center_value/overall_value))
			else:
				annoC.set("norm_freq", "0.00")
				
			annoB = etree.SubElement(annoClass, "bottom")
			annoB.set("freq", str(bottom_value))
			if overall_value!=0:
				annoB.set("norm_freq", str(bottom_value/overall_value))
			else:
				annoB.set("norm_freq", "0.00")
			
		writer = etree.ElementTree(s_tree_writer)
		xmlFilename= self.output_path + '/spatial_knowledge.xml'
		writer.write(xmlFilename, pretty_print=True)
	
	def cXML(self):
		c_knowledge_dictionary={}
		for x in range (1, self.cArraySize):
			key=self.cooccurArray[x][0]
			c_knowledge_dictionary[key]={}
			for y in range (1, self.cArraySize):
				valueName=self.cooccurArray[0][y]
				valueVal=self.cooccurArray[x][y]
				content={valueName:valueVal}
				c_knowledge_dictionary[key].update(content)
			
		root = etree.Element("cooccurence_knowledge")
		root.set("dataset","pascal_voc_2010")
		root.set('timestamp', str(self.generatedTime))
		
		for key, val in c_knowledge_dictionary.items():
			sub = etree.SubElement(root, 'class')
			sub.set('name', key)
			total=0.0
			for subkey,subval in val.items():
				total = total + subval
			subsub = etree.SubElement(sub, 'cooccur_with')
			for subkey, subval in val.items():
				subsubsub = etree.SubElement(subsub, subkey)
				subsubsub.set('freq', str(subval))
				if total!=0.0:
					subsubsub.set('norm_freq', str(subval/total))
				else:
					subsubsub.set('norm_freq', "0.0")
		
		xml_filepath = self.output_path + '/cooccurence_knowledge.xml'
		writer = etree.ElementTree(root)
		print(xml_filepath)
		writer.write(xml_filepath, pretty_print=True)
	
	def pXML(self):
		p_knowledge_dictionary={}
		for x in range (1, self.pArraySize):
			key=self.propertyArray[x][0]
			p_knowledge_dictionary[key]={}
			for y in range (1, self.cArraySize):
				valueName=self.propertyArray[0][y]
				valueVal=self.propertyArray[x][y]
				content={valueName:valueVal}
				p_knowledge_dictionary[key].update(content)
		
		root = etree.Element("scene_properties_knowledge")
		root.set("dataset", "pascal_voc_2010")
		root.set("timestamp", str(self.generatedTime))
		root.set("number_of_place_class", str(self.pArraySize))
		
		for key, val in p_knowledge_dictionary.items():
			sub = etree.SubElement(root, "place")
			sub.set("name", key)
			total=0.0
			for subkey, subval in val.items():
				total = total + subval
			subsub = etree.SubElement(sub, "objects")
			for subkey, subval in val.items():
				subsubsub = etree.SubElement(subsub, str(subkey))
				subsubsub.set("freq", str(subval))
				if total != 0.0:
					subsubsub.set("norm_freq", str(subval/total))
				else:
					subsubsub.set("norm_freq", "0.0")
		xml_filepath = self.output_path + "/scene_properties_knowledge.xml"
		writer = etree.ElementTree(root)
		writer.write(xml_filepath, pretty_print=True)
		
	def rXML(self):
		print()
		
		root = etree.Element("relative_knowledge")
		root.set("dataset","pascal_voc_2010")
		root.set("timestamp", str(self.generatedTime))
		root.set("below_threshold", str(self.relative_is_below_threshold))
		root.set("around_threshold", str(self.relative_is_around_threshold))
		
		for class_name in range(1, self.cArraySize):
			sub = etree.SubElement(root, "class")
			sub.set("name", self.rArray[class_name][0])
			subsub_below = etree.SubElement(sub, "is_below")
			subsub_beside = etree.SubElement(sub, "is_beside")
			subsub_around = etree.SubElement(sub, "is_around")
			below_total = 0
			beside_total = 0
			around_total = 0
			for x in range (1, self.cArraySize):
				below_total = below_total + self.rArray[class_name][x][0]
				beside_total = beside_total + self.rArray[class_name][x][1]
				around_total = around_total + self.rArray[class_name][x][2]
				
			for its_pair in range(1, self.cArraySize):
				subsubsub_below = etree.SubElement(subsub_below, "class")
				subsubsub_below.set("name", self.rArray[0][its_pair])
				subsubsub_below.set("freq", str(self.rArray[class_name][its_pair][0]))
				if below_total != 0.0:
					subsubsub_below.set("norm_freq", str(self.rArray[class_name][its_pair][0] / below_total))
				else:
					subsubsub_below.set("norm_freq", "0.0")
				
				subsubsub_beside = etree.SubElement(subsub_beside, "class")
				subsubsub_beside.set("name", self.rArray[0][its_pair])
				subsubsub_beside.set("freq", str(self.rArray[class_name][its_pair][1]))
				if beside_total != 0.0:
					subsubsub_beside.set("norm_freq", str(self.rArray[class_name][its_pair][1] / beside_total))
				else:
					subsubsub_beside.set("norm_freq", "0.0")
				
				subsubsub_around = etree.SubElement(subsub_around, "class")
				subsubsub_around.set("name", self.rArray[0][its_pair])
				subsubsub_around.set("freq", str(self.rArray[class_name][its_pair][2]))
				if around_total != 0.0:
					subsubsub_around.set("norm_freq", str(self.rArray[class_name][its_pair][2] / around_total))
				else:
					subsubsub_around.set("norm_freq", "0.0")
				
		xml_filepath = self.output_path + '/relativeposition_knowledge.xml'
		writer = etree.ElementTree(root)
		writer.write(xml_filepath, pretty_print=True)
			
	def spatial_knowledge(self, top_portion, center_portion, bottom_portion):
		
		fileQuant = glob.glob(os.path.join(self.traindat, '*.png'))
		total=len(fileQuant)
		 										
		#preparing label name in the array
		for label in range (1,self.spatialSize):
			self.spatialArray[0][label]=indexToPosDict[str(label)]
		for label in range (1,self.cArraySize):
			self.spatialArray[label][0]=indexToNameDict[str(label)]
		
		if (top_portion + center_portion + bottom_portion) < 99.9 and (top_portion + center_portion + bottom_portion) > 100:
			return(print ("The amount of the portion is not 100%"))
			
		else:
			self.spatial_top_portion = top_portion
			self.spatial_center_portion = center_portion
			self.spatial_bottom_portion = bottom_portion
			
		for (num, infile) in enumerate(fileQuant):
			datasetImage = Image.open(infile)
			pixel        = datasetImage.load()
			imageSize    = datasetImage.size
			(imageWidth, imageHeight)  = imageSize
		
			top_end_point = math.floor(imageHeight * (self.spatial_top_portion / 100))
			center_end_point = top_end_point + math.floor(imageHeight * (self.spatial_center_portion / 100))
			bottom_end_point = center_end_point + math.ceil(imageHeight * (self.spatial_bottom_portion / 100))
		
			print("Processing " + str(num+1) +"/" + str(total) + " files for Spatial Relation", end="\r") # Proccessing Progress Feedback
		
		
			#TOP
			pixelLabelListPart = [pixel[i,j] for i in range(0,imageWidth) for j in range (0,top_end_point) if pixel[i, j] != 0 and pixel[i, j] != 255] #read value
			setListPart = pixelLabelListPart
			setSize= len(setListPart)
			for x in range(0, setSize): #compute frequency
				self.spatialArray[setListPart[x]][1] = self.spatialArray[setListPart[x]][1] + 1
			
			#BOTTOM
			pixelLabelListPart = [pixel[i,j] for i in range(0,imageWidth) for j in range (top_end_point, center_end_point) if pixel[i, j] != 0 and pixel[i, j] != 255]
			setListPart = list((pixelLabelListPart))
			setSize= len(setListPart)
			for x in range(0, setSize):
				self.spatialArray[setListPart[x]][2] = self.spatialArray[setListPart[x]][2] + 1		
			
			#CENTER
			pixelLabelListPart = [pixel[i,j] for i in range(0,imageWidth) for j in range (center_end_point, bottom_end_point) if pixel[i, j] != 0 and pixel[i, j] != 255]
			setListPart = list((pixelLabelListPart))
			setSize= len(setListPart)
			for x in range(0, setSize):
				self.spatialArray[setListPart[x]][3] = self.spatialArray[setListPart[x]][3] + 1
				
			#OVERALL
			pixelLabelListPart = [pixel[i,j] for i in range(0,imageWidth) for j in range (0,imageHeight) if pixel[i, j] != 0 and pixel[i, j] != 255]
			setListPart = list((pixelLabelListPart))
			setSize= len(setListPart)
			for x in range(0, setSize):
				self.spatialArray[setListPart[x]][4] = self.spatialArray[setListPart[x]][4] + 1
				
		#self.writePrintS()
		self.sXML()
			
	def cooccurrence_knowledge(self):
		fileQuant = glob.glob(os.path.join(self.traindat, '*.png'))
		total=len(fileQuant)
		
		for label in range (1,self.cArraySize):
			self.cooccurArray[0][label]=indexToNameDict[str(label)]
			self.cooccurArray[label][0]=indexToNameDict[str(label)]
		
		for (num, infile) in enumerate(fileQuant):
			datasetImage = Image.open(infile)
			pixel        = datasetImage.load()
			imageSize    = datasetImage.size
			(imageWidth, imageHeight)  = imageSize
			
			print("Processing " + str(num+1) +"/" + str(total) + " files for Co-Occurrence Relation", end="\r") # Progress Feedback
			
			#read every pixel value
			pixelLabelList = [pixel[i,j] for i in range(imageWidth) for j in range (imageHeight) if pixel[i, j] != 0 and pixel[i, j] != 255]
			setList = list(set(pixelLabelList))
			setSize= len(setList)
	
			# Frequency List, including mutual appearance
			for x in range(0, setSize):
				for y in range (0, setSize):
					self.cooccurArray[setList[x]][setList[y]] = self.cooccurArray[setList[x]][setList[y]] + 1
		
		#self.writePrintC()
		self.cXML()
	
	def cooccurence_knowledge_from_xml(self):
		files_to_computed = glob.glob(os.path.join(self.rxmlpath, "*.xml"))
		total = len(files_to_computed)
		
		for label in range (1,self.cArraySize):
			self.cooccurArray[0][label]=indexToNameDict[str(label)]
			self.cooccurArray[label][0]=indexToNameDict[str(label)]
			
		for (num, infile) in enumerate (files_to_computed):
			#print("Processing " + str(num+1) +"/" + str(total) + " files for Co-Occurrence Relation Knowledge", end="\r") # Progress Feedback
			
			#collect objects in xml files
			objects_index = []
			cooccurence_in_file = []
			rtree = etree.parse(infile)
			tree = rtree.getroot()
			for objname in tree.findall("object/name"):
				object_name = objname.text
				objects_index.append(nameToIndexDict[object_name])
			index_length = len(objects_index)
			for x in range (0, index_length):
				for y in range (0, index_length):
					if x == y:
						continue
					else:
						index_a = str(objects_index[x])
						index_b = str(objects_index[y])
						if int(index_a) < 10:
							index_a = "0" + index_a
						if int(index_b) < 10:
							index_b = "0" + index_b
						cooccurence_in_file.append(index_a + index_b)
			cooccurence_set = list(set(cooccurence_in_file))
			
			#read the data and storing in array
			for x in range (0, len(cooccurence_set)):
				object_a = cooccurence_set[x][:2]
				object_b = cooccurence_set[x][2:]
				if object_a[:1] == "0":
					checked_a = object_a[1:]
				else:
					checked_a = object_a
				if object_b[:1] == "0":
					checked_b = object_b[1:]
				else:
					checked_b = object_b
				index_a = int(checked_a)
				index_b = int(checked_b)
				self.cooccurArray[index_a][index_b] = self.cooccurArray[index_a][index_b] + 1
		
		#self.writePrintC()
		self.cXML()
		
	def scene_properties_knowledge(self):
		xmlFilePath=glob.glob(os.path.join(self.xmldat, '*.xml'))
		xmlTotal=len(xmlFilePath)
		
		#searching property array element for place
		for (num, infile) in enumerate (xmlFilePath):	
			xmlTree = etree.parse(infile)
			tree = xmlTree.getroot()
			for place in tree.findall("place"):
				placeName=place.get("name")
				self.placeList.append(placeName)
				
		self.placeList=list(set(self.placeList))
		self.placeList.sort()
		self.pArraySize = len(self.placeList)+1
		
		self.propertyArray= [[0 for x in range(0,self.cArraySize)] for x in range(0,(self.pArraySize))]
		for label in range (1,self.pArraySize):
			self.propertyArray[label][0]=self.placeList[label-1].lower()
		for label in range (1,self.cArraySize):
			self.propertyArray[0][label]=indexToNameDict[str(label)]
		
		#entering data
		for (num, infile) in enumerate (xmlFilePath):
			xmlTree = etree.parse(infile)
			tree = xmlTree.getroot()
			for place in tree.findall("place"):
				placeName=place.get("name")
				for objects in place.findall("object"):
					objectName = objects.get("name")
					arrayPos = 1
					while placeName.lower() != self.propertyArray[arrayPos][0]:
						arrayPos = arrayPos + 1 # to search the position to place the counter until it's found
					self.propertyArray[arrayPos][int(nameToIndexDict[objectName])] = self.propertyArray[arrayPos][int(nameToIndexDict[objectName])] + 1
		
		#for x in range (0, self.pArraySize):
		#	print (self.propertyArray[x])
		self.pXML()	
				
	def relative_knowledge(self, below_beside_th, around_th):
		
		for label in range (1,self.cArraySize):
			self.rArray[0][label]=indexToNameDict[str(label)]
			self.rArray[label][0]=indexToNameDict[str(label)]
		self.rArray[0][0]=""
			
		relative_xml_file_group = glob.glob(os.path.join(self.rxmlpath, "*.xml"))
		xmlTotal=len(relative_xml_file_group)
		
		self.relative_is_around_threshold = around_th
		self.relative_is_below_threshold = below_beside_th
		
		for (num, infile) in enumerate (relative_xml_file_group):
			
			#print("Processing " + str(num+1) +"/" + str(xmlTotal) + " files for Relative Knowledge", end="\r") # Progress Feedback
			
			imgList = []
			rtree = etree.parse(infile)
			tree = rtree.getroot()
			for obj in tree.findall("object"):
				lclrelList = []
				name = obj.findall("name")
				objname = name[0].text
				lclrelList.append(objname)
				box = obj.findall("bndbox")
				for coord in box[0].getchildren():
					c=coord.text
					lclrelList.append(c)
				imgList.append(lclrelList)
			
			listL = len(imgList)
			for x in range (0,listL):
				for y in range (x,listL):
					if(x==y):
						continue
					else:
						obj1=imgList[x]
						obj2=imgList[y]
						obj1_classname = obj1[0]
						obj2_classname = obj2[0]
						obj1_arrayindex = int(nameToIndexDict[obj1_classname])
						obj2_arrayindex = int(nameToIndexDict[obj2_classname])
						
						#below
						if PositionSearcher.isBelow(obj1, obj2, self.relative_is_below_threshold):
							self.rArray[obj1_arrayindex][obj2_arrayindex][0] = self.rArray[obj1_arrayindex][obj2_arrayindex][0] + 1

						
						#beside, doesn't handle intersection
						# if some object located on the leftside or rightside of the other object and obj2 does not located to far from obj1 vertically
						if PositionSearcher.isBeside(obj1, obj2):
							self.rArray[obj1_arrayindex][obj2_arrayindex][1] = self.rArray[obj1_arrayindex][obj2_arrayindex][1] + 1
						
						#around
						
						#objek 1 ada di dalam objek 2
						if PositionSearcher.isInside(obj1, obj2):
							self.rArray[obj1_arrayindex][obj2_arrayindex][2] = self.rArray[obj1_arrayindex][obj2_arrayindex][2] + 1
						#ada di sekitar, dimana threshold terpenuhi
						elif PositionSearcher.isAround(obj1, obj2, self.relative_is_around_threshold):
							self.rArray[obj1_arrayindex][obj2_arrayindex][2] = self.rArray[obj1_arrayindex][obj2_arrayindex][2] + 1
						
		self.rXML()
		
	def gould_relative_location_probability_map(self):
		
		files_to_read = glob.glob(os.path.join(self.traindat, '*.png'))
		total = len(files_to_read)
		prob_map = None
		
		for (num, imgfile) in enumerate(files_to_read):
			
			image = Image.open(imgfile)
			img = img_as_float(image)
			segment_list = slic(img, n_segments=200, compactness=25, sigma=2)
			segment_amount = len(np.unique(segmentation))
			
			for each_class in range (0,len(indexToNameDict)):
				given_class = indexToNameDict[str(each_class + 1)]
				segment_containing_given_class = []
				
			
			
		print()

