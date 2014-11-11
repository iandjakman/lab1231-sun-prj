import os
import glob
import math
import voc_obj_intolbl
import voc_obj_lbltoin
import voc_spa_intopos
import datetime
from PIL import Image
from lxml import etree
from io import StringIO, BytesIO
 
#dictionary
indexToNameDict=voc_obj_intolbl.LABELS
nameToIndexDict=voc_obj_lbltoin.LABELS
indexToPosDict=voc_spa_intopos.POS


class KnowledgeConstructor:	
	
	traindat=None
	xmldat=None
	rxmlpath=None
	cOutFile=None
	cArraySize=None
	cooccurArray= None
	spatialSize=None
	spatialFile=None
	spatialArray = None
	spatialThreshold=None
	sXMLName=None
	placeList=None
	activityList=None
	propertyArray=None
	pArraySize=None
	rArray=None
	generatedTime=None
    
	def __init__(self, fileOutputName):
		
		cName="csv/cooccur" + fileOutputName +".csv"
		sName="csv/spatial" + fileOutputName +".csv"
		self.sXMLName=fileOutputName
		
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
	
	def set_xml_data_dir(self, xml_dir):
		self.xmldat=xml_dir
		self.rxmlpath="/home/ian-djakman/Documents/data/voc_dataset/VOCdevkit/VOC2012/voc_xml"
		
	def set_dictionary_name(self, dict_name):
		print()
	
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
		s_tree_writer.set("threshold", str(self.spatialThreshold))
		for x in range (1,self.cArraySize):
			lbl = self.spatialArray[x][0]
			annoClass = etree.SubElement(s_tree_writer, lbl)
			tv= self.spatialArray[x][1]
			cv= self.spatialArray[x][2]
			bv= self.spatialArray[x][3]
			ov= self.spatialArray[x][4]
			annoT = etree.SubElement(annoClass, "top")
			annoT.set("freq", str(tv))
			if ov!=0:
				annoT.set("norm_freq", str(tv/ov))
			else:
				annoT.set("norm_freq", "0.00")
			annoC = etree.SubElement(annoClass, "center")
			annoC.set("freq", str(cv))
			if ov!=0:
				annoC.set("norm_freq", str(cv/ov))
			else:
				annoC.set("norm_freq", "0.00")
			annoB = etree.SubElement(annoClass, "bottom")
			annoB.set("freq", str(bv))
			if ov!=0:
				annoB.set("norm_freq", str(bv/ov))
			else:
				annoB.set("norm_freq", "0.00")
			
		writer = etree.ElementTree(s_tree_writer)
		xmlFilename='xmlout/spatial_knowledge.xml'
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
				if(x!=y):
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
		
		xml_filepath = 'xmlout/cooccurence_knowledge.xml'
		writer = etree.ElementTree(root)
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
		root.set("amount_of_place", str(self.pArraySize))
		
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
		xml_filepath = "xmlout/scene_properties_knowledge.xml"
		writer = etree.ElementTree(root)
		writer.write(xml_filepath, pretty_print=True)
		
	def rXML(self):
		print()
					
	def s(self, threshold):
		
		fileQuant = glob.glob(os.path.join(self.traindat, '*.png'))
		total=len(fileQuant)
		 										
		#preparing label name in the array
		for label in range (1,self.spatialSize):
			self.spatialArray[0][label]=indexToPosDict[str(label)]
		for label in range (1,self.cArraySize):
			self.spatialArray[label][0]=indexToNameDict[str(label)]

		for (num, infile) in enumerate(fileQuant):
			datasetImage = Image.open(infile)
			pixel        = datasetImage.load()
			imageSize    = datasetImage.size
			(imageWidth, imageHeight)  = imageSize
		
			self.spatialThreshold = threshold
			centerPortion = ((imageHeight/3)/2)*((100-self.spatialThreshold)/100) 
			centerHStart = (math.floor(imageHeight/3))+(math.ceil(centerPortion))
			centerHLimit = math.floor(imageHeight*2/3)-(math.floor(centerPortion))
			topLimit = centerHStart
		
			print("Processing " + str(num+1) +"/" + str(total) + " files for Spatial Relation", end="\r") # Proccessing Progress Feedback
		
		
			#TOP
			pixelLabelListPart = [pixel[i,j] for i in range(0,imageWidth) for j in range (0,centerHStart) if pixel[i, j] != 0 and pixel[i, j] != 255] #read value
			setListPart = pixelLabelListPart
			setSize= len(setListPart)
			for x in range(0, setSize): #compute frequency
				self.spatialArray[setListPart[x]][1] = self.spatialArray[setListPart[x]][1] + 1
			
			#BOTTOM
			pixelLabelListPart = [pixel[i,j] for i in range(0,imageWidth) for j in range (centerHLimit,imageHeight) if pixel[i, j] != 0 and pixel[i, j] != 255]
			setListPart = list((pixelLabelListPart))
			setSize= len(setListPart)
			for x in range(0, setSize):
				self.spatialArray[setListPart[x]][2] = self.spatialArray[setListPart[x]][2] + 1		
			
			#CENTER
			pixelLabelListPart = [pixel[i,j] for i in range(0,imageWidth) for j in range (centerHStart,centerHLimit) if pixel[i, j] != 0 and pixel[i, j] != 255]
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
				
		self.writePrintS()
		self.sXML()
			
	def c(self):
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
		
		self.writePrintC()
		self.cXML()
	
	def p(self):
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
						arrayPos = arrayPos + 1
					self.propertyArray[arrayPos][int(nameToIndexDict[objectName])] = self.propertyArray[arrayPos][int(nameToIndexDict[objectName])] + 1
		
		self.pXML()	
				
	def r(self):
		rxmlfpath=glob.glob(os.path.join(self.rxmlpath, "*.xml"))
		xmlTotal=len(rxmlfpath)
		
		for (num, infile) in enumerate (rxmlfpath):
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
						#below
						obj1=imgList[x]
						obj2=imgList[y]
						if obj1[4] < obj2[2]:
							#objek 2 ada di bawah objek 1
							self.rArray[int(nameToIndexDict[obj2[0]])][int(nameToIndexDict[obj1[0]])][0] = self.rArray[int(nameToIndexDict[obj2[0]])][int(nameToIndexDict[obj1[0]])][0] + 1
						elif obj2[4] < obj1[2]:
							#onjek 1 ada di bawah objek 2
							self.rArray[int(nameToIndexDict[obj1[0]])][int(nameToIndexDict[obj2[0]])][0] = self.rArray[int(nameToIndexDict[obj1[0]])][int(nameToIndexDict[obj2[0]])][0] + 1
						
						#beside
						# if obj2 is in the left side or right side of obj1 and obj2 does not located to far from obj1 vertically
						if (obj1[1] > obj2[3] or obj1[3] < obj2[1]) and ((obj1[2] <= obj2[4] <= obj1[4]) or (obj1[2] <= obj2[2] <= obj1[4])):
							self.rArray[int(nameToIndexDict[obj1[0]])][int(nameToIndexDict[obj2[0]])][1] = self.rArray[int(nameToIndexDict[obj1[0]])][int(nameToIndexDict[obj2[0]])][1] + 1
							self.rArray[int(nameToIndexDict[obj2[0]])][int(nameToIndexDict[obj1[0]])][1] = self.rArray[int(nameToIndexDict[obj2[0]])][int(nameToIndexDict[obj1[0]])][1] + 1
						#inside
		for x in range (0,20):
			print(self.rArray[x])
			
	def c_by_metadata(self):
		
		fileQuant = glob.glob(os.path.join(self.traindat, '*.png'))
		total=len(fileQuant)
		
		#preparing label name in the array
		for label in range (1,cArraySize):
			cooccurArray[0][label]=indexToNameDict[str(label)]
			cooccurArray[label][0]=indexToNameDict[str(label)]


		metaToRead=open("metadata/cooccurrenceMeta.txt","rU")
		print("")
		for line in metaToRead.readlines():
			setList = line.split(",")
			setSize= len(setList)
		
			# Frequency List, including mutual appearance
			for x in range(0, setSize):
				for y in range (0, setSize):
					cooccurArray[int(setList[x])][int(setList[y])] = cooccurArray[int(setList[x])][int(setList[y])] + 1
		metaToRead.close()
	
		KnowledgeConstructor.writePrintC()

