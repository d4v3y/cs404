# COMMAND FOR EXECUTION
# python detect_age_video.py --face face_detector --age age_detector

# libraries
from imutils.video import VideoStream
import numpy as np
import cv2 as cv
import argparse
import imutils
import time
import os

def age_prediction(frame, faceNet, ageNet, minConf=0.5):

	# list of ranges for age 
	AGE_RANGES = ["(0-2)", "(4-6)", "(8-12)", "(14-18)", "(21-25)", "(27-32)",
                "(38-43)", "(48-53)", "(60-100)"]

	results = []
	(h, w) = frame.shape[:2]
	blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# retrieve the probability percentage
		confidence = detections[0, 0, i, 2]

		# filter out weak facial detections
		if confidence > minConf:
			# bounding box for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the ROI of the face
			face = frame[startY:endY, startX:endX]

			# check if the face ROI is big enough
			if face.shape[0] < 20 or face.shape[1] < 20:
				continue

			# construct a blob from only the face ROI
			faceBlob = cv.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

			# make predictions
			ageNet.setInput(faceBlob)
			preds = ageNet.forward()
			i = preds[0].argmax()
			age = AGE_RANGES[i]
			ageConfidence = preds[0][i]

			# construct a dictionary consisting of both the face bounding box location along
			# and the age prediction,
			d = {
				"loc": (startX, startY, endX, endY),
				"age": (age, ageConfidence)
			}
			results.append(d)

	return results

# create command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required=True, help="path to face detector model directory")
ap.add_argument("-a", "--age", required=True, help="path to age detector model directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load face detector model
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv.dnn.readNet(prototxtPath, weightsPath)

# load age detector model
print("[INFO] loading age detector model...")
prototxtPath = os.path.sep.join([args["age"], "age_deploy.prototxt"])
weightsPath = os.path.sep.join([args["age"], "age_net.caffemodel"])
ageNet = cv.dnn.readNet(prototxtPath, weightsPath)

# start camera
print("[INFO] starting camera...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# resize frame to 900 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=900)

	# detect faces in the frame
	results = age_prediction(frame, faceNet, ageNet, minConf=args["confidence"])

	# for each face in the frame, predict the age
	for person in results:
		# draw the bounding box of the face and predicted age
		text = "{}: {:.2f}%".format(person["age"][0], person["age"][1] * 100)
		(startX, startY, endX, endY) = person["loc"]
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
		cv.putText(frame, text, (startX, y), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# output
	cv.imshow("Age Guesser", frame)
	key = cv.waitKey(1) & 0xFF

	if key == ord("q"):
		break

cv.destroyAllWindows()
vs.stop()
