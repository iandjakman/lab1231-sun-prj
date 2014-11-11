import os
import glob
from PIL import Image

cooccurMeta=open("metadata/cooccurrenceMeta.txt","w")	

class Metadata:
	def constructOccurrenceMetadata(datasetDirectory) :
	
		path=datasetDirectory
		fileQuant=glob.glob(os.path.join(path, '*.png'))
		total=len(fileQuant)
	
		for (num, infile) in enumerate(fileQuant):
			datasetImage = Image.open(infile)
			pixel        = datasetImage.load()
			imageSize    = datasetImage.size
			(imageWidth, imageHeight)  = imageSize
		
			print("Processing " + str(num+1) +"/" + str(total) + " for metadata", end="\r") # Proccessing Progress Feedback
		
			#read every pixel value
			pixelLabelList = [pixel[i,j] for i in range(imageWidth) for j in range (imageHeight) if pixel[i, j] != 0 and pixel[i, j] != 255]
			setList = list(set(pixelLabelList))
			setSize= len(setList)
			for i in range(0,setSize-1):
				cooccurMeta.write("%d,"%setList[i])
			cooccurMeta.write("%d"%setList[setSize-1])
			cooccurMeta.write("\n")
		cooccurMeta.close()

