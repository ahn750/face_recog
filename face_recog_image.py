import face_recognition
import argparse
import pickle 
import cv2
import imutils

ap=argparse.ArgumentParser()
ap.add_argument('-i','--image',required=True,help="path to input image")
ap.add_argument('-e','--encodings',required=True,help="path to encodings")
ap.add_argument('-d','--detection_model',type=str,default='cnn',help="detection model- hog or cnn")
args=vars(ap.parse_args())

face_cascade=cv2.CascadeClassifier('D:/Atharva/python/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
data = pickle.loads(open(args["encodings"], "rb").read())

image=cv2.imread(args['image'])

rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

print('detecting faces....',end='')

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

boxes=face_recognition.face_locations(rgb,model='hog')

print(boxes)

print('faces detected: {} \n'.format(len(boxes)))

print('encoding faces....',end='')
encodings=face_recognition.face_encodings(image,boxes)
print('done \n')

persons=[]
for i,encoding in enumerate(encodings):

	print('checking face {}/{}'.format(i+1,len(encodings)))
	matches=face_recognition.compare_faces(data['encodings'],encoding,0.5)

	
	if True in matches:
		matchedIdxs=[i for(i,b) in enumerate(matches) if b]
		
		count={}

		for indx in matchedIdxs:
			name=data['names'][indx]
			count[name]=count.get(name,0)+1

		name=max(count,key=count.get)
		print('face{} recognized: {} \n'.format(i+1,name))
		persons.append(name)

	else:
		print('face{} not recognized \n'.format(i+1))
		persons.append('Unknown')


for ((starty,endx,endy,startx),name) in zip(boxes,persons):
	cv2.rectangle(image,(startx,starty),(endx,endy),(0,255,0),2)

	y = starty - 15 if starty - 15 > 15 else starty + 15
	cv2.putText(image, name, (startx, y), cv2.FONT_HERSHEY_SIMPLEX,
		0.75, (0, 255, 0), 2)

cv2.imshow('face_recog',image)
cv2.waitKey(0)


