import os
import glob
import json


def load_bbox():
	
	ann_image_to_get_contour =  '/home/ian-djakman/Desktop/philip_getsegment/philan'
	files_to_read = glob.glob(os.path.join(ann_image_to_get_contour, '*.bmp'))
	total = len(files_to_read)
	
	for (num, imgfile) in enumerate(files_to_read):
		filename = imgfile[len(ann_image_to_get_contour):][:-4]
		outpath = '/home/ian-djakman/Desktop/philip_getsegment/rect_bbox'
		out_filename = outpath + filename + ".json"
		with open(out_filename, 'r') as fp:
			kv = json.load(fp)
		bbox = kv['pa_bbox']
		print(bbox)
		
def main():
	load_bbox()
	

if __name__ == '__main__':
    main()
