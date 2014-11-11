
import glob
import csv

scsv= open("csv/spaTest.csv","rb")

class KnowledgeLoader:
	
	def makeXML(self, type):
		
		if type is 'p':
			reader = csv.reader(scsv, delimiter=',')
			for row in reader:
				print (', ').join(row)
