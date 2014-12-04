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
from positionSearcher import PositionSearcher

import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from skimage.util import img_as_float
from scipy import ndimage

#dictionary
indexToNameDict=voc_obj_intolbl.LABELS
nameToIndexDict=voc_obj_lbltoin.LABELS
indexToPosDict=voc_spa_intopos.POS

def gould_relative_location_probability_map():
		files_to_read = glob.glob(os.path.join(img_dir, '*.jpg'))
		total = len(files_to_read)
		probability_map = [[[[0 for x in range (0,200)] for x in range (0,200)] for x in range (0,21)] for x in range (0,21)]
		
		for (num, imgfile) in enumerate(files_to_read):
			print("Processing " + str(num+1) +"/" + str(total) + " image for Gould relative map") # Progress Feedback
			filename = imgfile[len(img_dir):][:12]
			image = Image.open(imgfile)
			annpath = '/home/ian-djakman/Documents/data/input/voc_dataset_2012/SegmentationClass' + filename + '.png'
			ann_image = Image.open(annpath)
			img = img_as_float(image)
			segmentation = slic(img, n_segments=200, compactness=25, sigma=2)
			#print("Slic number of segments: %d" % len(np.unique(segmentation)))
			
			for each_class in range (0,len(indexToNameDict)):
				given_class = indexToNameDict[str(each_class+1)]
				segment_list_for_given_class = []
				pixel = ann_image.load()
				imageSize = ann_image.size
				(imageWidth, imageHeight)  = imageSize
				
				#GET SEGMENT LIST
				for x in range (0,imageWidth):
					for y in range (0, imageHeight):
						if pixel[x,y] != 0 and pixel[x,y] != 255:
							if indexToNameDict[str(pixel[x,y])] == given_class:
								segment_list_for_given_class.append(segmentation[y][x])
				
				segment_list_for_given_class = list(set(segment_list_for_given_class))
				#print("untuk kelas " + given_class + " :")
				#rint(segment_list_for_given_class)
				
				#GET CENTROID
				for segments in range (0, len(segment_list_for_given_class)):
					segment_to_process = segment_list_for_given_class[segments]
					pixel_coordinate_x = []
					pixel_coordinate_y = []
					for i in range (0, imageHeight):
						for j in range (0, imageWidth):
							if segmentation[i][j] == segment_to_process:
								pixel_coordinate_x.append(i)
								pixel_coordinate_y.append(j)
								
					centroid = ((sum(pixel_coordinate_x) /len(pixel_coordinate_x)),(sum(pixel_coordinate_y)/len(pixel_coordinate_y)))
					#print ("center of mass superpiksel " + str(segment_to_process) + " :")
					#print centroid
														
				#PROCESSING EVERY PIXEL TO MAKE GOULD RELATIVE MAP
					offset = 0
					for x in range(0,imageWidth):
						for y in range(0,imageHeight):
							if pixel[x,y] != 0 and pixel[x,y] != 255:
								pixel_class = pixel[x,y]
								cen_h = centroid[0]
								cen_w = centroid[1]
								height_diff = math.fabs(cen_h - y)
								norm_height = (height_diff / imageHeight ) * 200
								width_diff =  math.fabs(cen_w - x)
								norm_width = (width_diff / imageWidth ) * 200
								offset =  math.hypot(height_diff, width_diff)
								norm_offset =  math.hypot(norm_height, norm_width)
								weight = len(pixel_coordinate_x) # this part compute all pixels in that segment, can be seen from the length of x or y
								norm_weight = weight * (40000 / float(imageWidth*imageHeight))
								probability_map[(each_class+1)][pixel_class][int(norm_height)][int(norm_width)] = weight * norm_offset
					
					'''			
					print ("offset dari superpixel : " + str(offset))
					print ("Setelah dinormalized : " + str(norm_offset))
					print ("banyak superpixel di koordinat x : " + str(len(pixel_coordinate_x)))
					print ("banyak superpixel di koordinat y : " + str(len(pixel_coordinate_y)))
					print weight
					print norm_weight
					'''
						
			#Proses gambar menggambar
			'''
			fig, ax = plt.subplots(1, 3)
			fig.set_size_inches(8, 1, forward=True)
			fig.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.05, 0.05)
			ax[0].imshow(img)
			ax[0].set_title("Original")
			ax[1].imshow(ann_image)
			ax[1].set_title("Annotated")
			ax[2].imshow(mark_boundaries(img, segmentation))
			ax[2].set_title("SLIC")
			for a in ax:
				a.set_xticks(())
				a.set_yticks(())
			plt.show()
			'''
		tes = (probability_map[int(nameToIndexDict['aeroplane'])] [int(nameToIndexDict['aeroplane'])])
		plt.imshow(tes)
		plt.savefig('tes.jpg')
		outfile = "/home/ian-djakman/Desktop/gould_relative_position"
		np.save(outfile,probability_map)
		plt.show(tes)
		
def main():
    global img_dir
    img_dir = '/home/ian-djakman/online/Github/lab1231-sun-prj/knowledge-constructor/test'
    gould_relative_location_probability_map()

if __name__ == '__main__':
    main()
