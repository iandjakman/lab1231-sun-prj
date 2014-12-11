import os
import glob
import cv2
import csv
import json
import numpy as np
import matplotlib.pyplot as plot

def get_objects_from_annotation():
	ann_image_to_get_contour =  '/home/ian-djakman/Desktop/philip_getsegment/philan'
	files_to_read = glob.glob(os.path.join(ann_image_to_get_contour, '*.bmp'))
	total = len(files_to_read)

	for (num, imgfile) in enumerate(files_to_read):
		
		print('Processing ' + str(num+1) +'/' + str(total) + ' image for Gould relative map')
		
		filename = imgfile[len(ann_image_to_get_contour):][:-4]
		outpath = '/home/ian-djakman/Desktop/philip_getsegment/rect_bbox'
		csvfile = '/home/ian-djakman/Documents/data/output/knowledge-compatibility-benchmarker/annotation/annotation-philippunary' + filename + '.csv'

		philip_csv = np.array(list(csv.reader(open(csvfile,'r'),delimiter=','))).astype('int')
		clean_philip_image = cv2.imread(imgfile)
		draw_philip_image = cv2.imread(imgfile)
		imh = len(clean_philip_image)
		imw = len(clean_philip_image[0])
		
		rectangle=[]
		class_in_img = list(np.unique(philip_csv)) #search every contained class in img, for a faster loop
		
		for classes in range (0,len(class_in_img)):
			ctp = class_in_img[classes]
			if ctp != 0 :
				philip_image = cv2.imread(imgfile)
				for h in range (0, imh):
					for w in range (0, imw):
						pixel_class = philip_csv[h][w]
						if pixel_class == ctp:
							philip_image[h][w] = (255,255,255)
						else:
							philip_image[h][w] = (0,0,0)
							
				e_kernel = np.ones((13,13),np.uint8)
				d_kernel = np.ones((15,15),np.uint8)
				eroded_img = cv2.erode(philip_image,e_kernel,iterations = 1)
				dilated_img = cv2.dilate(eroded_img,d_kernel,iterations = 1)
				canny_edgedetect = cv2.Canny(dilated_img,1,25)
				contours, hierarchy = cv2.findContours(canny_edgedetect,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
				for x in range (0, len(contours)):
					cnt = contours[x]
					area = cv2.contourArea(cnt)
					x,y,w,h = cv2.boundingRect(contours[x])
					coordinate = [x,y,x+w,y+h]
					xcenter = (coordinate[0] + coordinate[2]) / 2
					ycenter = (coordinate[1] + coordinate[3]) / 2
					center = [xcenter, ycenter]
					center_rect_class = philip_csv[ycenter][xcenter]
					if (center_rect_class == class_in_img[classes]) and ((w*h)>1500):
						class_val = center_rect_class
						rectangle.append([x,y,x+w,y+h,center_rect_class])
		
		out_filename = outpath + filename + ".json"
		keyval_for_json = {'pa_bbox' : rectangle}
		with open(out_filename, 'w') as f:
			json.dump(keyval_for_json, f)
		
		'''
		#for showing image purpose
		for x in range (0, len(rectangle)):
			rtp = rectangle[x]
			cv2.rectangle(draw_philip_image,(rtp[0],rtp[1]),(rtp[2],rtp[3]),(0,255,0),1)
			
		cv2.imshow('image', draw_philip_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		'''
		
def main():
	get_objects_from_annotation()

if __name__ == '__main__':
    main()
