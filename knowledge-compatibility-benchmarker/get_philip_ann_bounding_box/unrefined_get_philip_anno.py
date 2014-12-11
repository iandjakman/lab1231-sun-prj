import os
import glob
import Image
import cv2
import msrc
import numpy as np
import matplotlib.pyplot as plot

from skimage import data
from skimage import feature

color2class = msrc.color_class_map

def get_objects_from_annotation():
	files_to_read = glob.glob(os.path.join(ann_image_to_get_contour, '*.bmp'))
	total = len(files_to_read)

	for (num, imgfile) in enumerate(files_to_read):
		
		#print("Processing " + str(num+1) +"/" + str(total) + " image for Gould relative map")
		
		filename = imgfile[len(ann_image_to_get_contour):][:-4]
		outpath = "/home/ian-djakman/Desktop/philip_getsegment/out"
		
		philip_image = cv2.imread(imgfile)
		clean_philip_image = cv2.imread(imgfile)
		philip_gray = cv2.cvtColor(philip_image,cv2.COLOR_BGR2GRAY)
		gray_filename = outpath + filename + "_step1_g.bmp"
		cv2.imwrite(gray_filename, philip_gray)
		
		c_edges = cv2.Canny(philip_gray,1,25)
		bw_filename = outpath + filename +"_step2_bw.bmp"
		cv2.imwrite(bw_filename, c_edges)
		
		e_kernel = np.ones((3,3),np.uint8)
		e2_kernel = np.ones((2,2),np.uint8)
		d_kernel = np.ones((7,7),np.uint8)
		eroded_img = cv2.erode(philip_gray,e_kernel,iterations = 1)
		eroded_img = cv2.erode(eroded_img,e2_kernel,iterations = 2)
		dilated_img = cv2.dilate(eroded_img,d_kernel,iterations = 1)
		c_edges_alt = cv2.Canny(dilated_img,1,25)
		ed_filename = outpath + filename + "_step3_g.bmp"
		cv2.imwrite(ed_filename, c_edges_alt)
		contours, hierarchy = cv2.findContours(c_edges_alt,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		
				
		# Gambar menggambar
		rectangle=[]
		#cv2.drawContours(philip_image, contours, -1, (255,255,255), 1)
		for x in range (0, len(contours)):
			cnt = contours[x]
			area = cv2.contourArea(cnt)
			x,y,w,h = cv2.boundingRect(contours[x])
			
			
			#tambahan pengecekan rect void
			coordinate = [x,y,x+w,y+h]
			xcenter = (coordinate[0] + coordinate[2]) / 2
			ycenter = (coordinate[1] + coordinate[3]) / 2
			center = [xcenter, ycenter]
			pixel_val = clean_philip_image[ycenter][xcenter]
			r = pixel_val[0]
			g = pixel_val[1]
			b = pixel_val[2]
			pixel_val = (b,g,r)
			
			if (r!=0 or g!=0 or b!=0) and (r!=255 or g!=255 or b!=255) and ((w*h)>3000):
				class_val = color2class[pixel_val]
				print(class_val)
				rectangle.append([x,y,x+w,y+h])
				cv2.rectangle(philip_image,(x,y),(x+w,y+h),(0,255,0),1)
				
		
		#tampilkan
		cv2.imshow('image',philip_image)
		ed_filename = outpath + filename + "_step4_bound.bmp"
		cv2.imwrite(ed_filename, philip_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		
		#informasi
		print ("banyak contour pada gambar " + str(num) + " : " + str(len(contours)))
		print ("banyak rectangle pada gambar " + str(num) + " : " + str(len(rectangle)))
		#for x in range (o, len(rectangle)):
			
			#print ()
				
def main():
	global ann_image_to_get_contour
	ann_image_to_get_contour =  '/home/ian-djakman/Desktop/philip_getsegment/philan'
	get_objects_from_annotation()

if __name__ == '__main__':
    main()
