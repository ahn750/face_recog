import face_recognition
import argparse
import pickle 
import cv2
import imutils

ap=argparse.ArgumentParser()
ap.add_argument('-e','--encodings',required=True,help="path to encodings")
ap.add_argument('-d','--detection_model',type=str,default='hog',help="detection model- hog or cnn")
args=vars(ap.parse_args())

cap=cv2.VideoCapture(0)

face_cascade=cv2.CascadeClassifier('D:/Atharva/python/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
data = pickle.loads(open(args["encodings"], "rb").read())



while True:
	
	ret,image=cap.read()
	image = imutils.resize(image, width=500)

	rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

	frame_no=int(cap.get(cv2.CAP_PROP_POS_FRAMES))

	boxes=face_recognition.face_locations(rgb,model='hog')
	print('faces detected:{}'.format(len(boxes)))

	
	if len(boxes)>0:
		
		
		encodings=face_recognition.face_encodings(image,boxes)

		persons=[]


		for i,encoding in enumerate(encodings):


			matches=face_recognition.compare_faces(data['encodings'],encoding,0.5)

			
			if True in matches:
				matchedIdxs=[i for(i,b) in enumerate(matches) if b]
				
				count={}

				for indx in matchedIdxs:
					name=data['names'][indx]
					count[name]=count.get(name,0)+1

				name=max(count,key=count.get)
			
				persons.append(name)

			else:
			
				persons.append('Unknown')


		for ((starty,endx,endy,startx),name) in zip(boxes,persons):
			cv2.rectangle(image,(startx,starty),(endx,endy),(0,255,0),2)

			y = starty - 15 if starty - 15 > 15 else starty + 15
			cv2.putText(image, name, (startx, y), cv2.FONT_HERSHEY_SIMPLEX,
				0.75, (0, 255, 0), 2)
	
	cv2.imshow('face_recog',image)
	if (cv2.waitKey(1)&0xFF)==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()










	

	
	
	


	