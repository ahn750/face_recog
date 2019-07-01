from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import cv2
import os


#command line args
ap=argparse.ArgumentParser()
ap.add_argument("-i","--dataset",required=True,help="path to dataset")
ap.add_argument("-e","--encodings",required=True,help="path to encodings")
ap.add_argument("-d","--detection-methods",type=str,default='hog',help="path to detection-methods")
args=vars(ap.parse_args())


print('fetching images from:')
imagepaths=list(paths.list_images(args['dataset']))
print(imagepaths)
print('success')



encodings=[]
names=[]
error=0
for i,path in enumerate(imagepaths):

	print('loading image....',end='')
	image=cv2.imread(path)
	name=os.path.basename(os.path.dirname(path))
	rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	print('done\n')

	print('detecting face {}/{}....'.format(i+1,len(imagepaths)),end='')
	boxes=face_recognition.face_locations(rgb,model=args['detection_methods'])
	print('done')

	no_faces=len(boxes)
	
	if no_faces==1:
		print('encoding face {}/{}....'.format(i+1,len(imagepaths)),end='')
		encoding=face_recognition.face_encodings(rgb,boxes)[0]
		encodings.append(encoding)
		names.append(name)
		print('done\n')
		
		
	elif no_faces==0:
		print('Error: dataset image {} contains multiple or no face'.format(path))
		cv2.imshow('img',image)
		cv2.waitKey(0)
		error=1
		break
		
	else :
			image_dummy=image.copy()

			
			for i,(top,right,bottom,left) in enumerate(boxes):
				cv2.rectangle(image_dummy,(left,top),(right,bottom),(0,255,0),2)
				y = top - 15 if top - 15 > 15 else top + 15
				cv2.putText(image_dummy,str(i),(left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)

			image_dummy=cv2.resize(image_dummy,(int(image_dummy.shape[1]/4),int(image_dummy.shape[1]/4)))	
			cv2.imshow('choose face',image_dummy)
			key=cv2.waitKey(0)
			key=chr(key)
			cv2.destroyAllWindows()
			print(key)
			print('encoding face {}/{}....'.format(i+1,len(imagepaths)),end='')
			boxes=[boxes[int(key)]]
			encoding=face_recognition.face_encodings(rgb,boxes)[0]
			
			encodings.append(encoding)
			names.append(name)
			print('done\n')






if(not error):
	print('success')
	data={'encodings':encodings,'names':names}

	print("storing data file")
	f=open(args['encodings'],'wb')
	f.write(pickle.dumps(data))
	f.close()
	print('success')
	



