from __future__ import annotations
import numpy as np
import argparse
import cv2
import imutils

def validateInput(prompt):
	while True:
		userInput = input(prompt)
		if userInput.lower() == "y" or userInput.lower() == "n" or userInput.lower() == "yes" or userInput.lower() == "no":
			return userInput.lower()
		else:
			print("Please input either 'y', 'n', 'yes', or 'no'.\n")

def loadImage():
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True, help="Path to Image")
	args = vars(ap.parse_args())
	image = cv2.imread(args["image"])
	imgPath = args["image"]

	IS_PNG = False
	if imgPath[len(imgPath)-4:] != ".png":
		# splitPath = imgPath.split("/")
		# imgName = splitPath[len(splitPath)-1]
		# imgName = imgName[:len(imgName)-4]
		if imgPath[len(imgPath)-5:] == "webp":
			imgName = imgPath[len(imgPath)-5:]
			imgPath = imgPath[len(imgPath)-4:] + "png"
		else:
			imgName = imgPath[:len(imgPath)-4]
			imgPath = imgPath[:len(imgPath)-3] + "png"
		cv2.imwrite(f"{imgName}.png", image)
		print(imgPath)
		image = cv2.imread(imgPath)

	return image

def detectEdge(image) -> None:
	# rectangle and square structuring overlays
	rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
	sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

	image = imutils.resize(image, height=700)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (11, 11), 0)

	# edged = cv2.Canny(blurred, 30, 150)

	# convert to high-contrast black and white
	blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
	# cv2.imshow("blackhat", blackhat)

	# compute gradient magnitude
	gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
	gradX = np.absolute(gradX)
	(minVal, maxVal) = (np.min(gradX), np.max(gradX))
	gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

	# compute and threshold actual MRZ's (Machine Readable Zones)
	# rectangular kernel closes gaps between letters to make rectangles
	# then threshold the image
	gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
	thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) [1]

	# merge the smaller rectangles into one large rectangle
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
	thresh = cv2.erode(thresh, None, iterations=4)

	(cnts, _) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(thresh, cnts, -1, (0, 255, 0), 2)
	# cv2.imshow("Threshheld", thresh)

	# sort thresholds by size
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	
	# draw the rectangle
	(x, y, w, h) = cv2.boundingRect(cnts[0])
	box = image[y - 20:y + h + 20, x - 20:x + w + 20].copy()
	# box = image.copy()
	cv2.rectangle(image, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 2)
	cv2.imshow("Identified", image)

	cv2.imshow("rect", box)

	return [[x-20, y-20], [x + w + 20, y + h + 20]]

def blurImage(image, boxCoords):
	BLUR = validateInput("Would you like to blur? (y/n) ")

	BLUR = True
	if BLUR:
		image[boxCoords[0][1]: boxCoords[1][1], boxCoords[0][0]: boxCoords[1][0]] = cv2.blur(image[boxCoords[0][1]: boxCoords[1][1], boxCoords[0][0]: boxCoords[1][0]], (20, 20))
		cv2.imshow("Blurred", image)

def featureMapping(image1, image2):

	minHessian = 400
	detector = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
	keypoints1, descriptors1 = detector.detectandCompute(image1, None)
	keypoints2, descriptors2 = detector.detectandCompute(image2, None)


	matcher = cv2.DescriptionMatcher_create(cv2.DescriptionMatcher_FLANBASED)
	knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)

	ratio_thresh = 0.7
	goodm = []

	for m, n in knn_matches:
		if m.distance < ratio_thresh * n.distance:
			goodm.append(m)
	img_matches = np.empty((max(image1.shape[0], image2.shape[0]), image1.shape[1]+image2.shape[1], 3), dtype=np.uint8)
	cv2.drawMatches(image1, keypoints1, image2, keypoints2, goodm, img_matches, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

	cv2.imshow("Good Matches", img_matches)
	


def loadImageFeatureMap():
	ap = argparse.Arguementparser()
	ap.add_arguement("-i", "--image1", required=True, help="Path to Image" )
	ap.add_arguement("-i", "--image2", required=True, help="Path to Image" )
	args = vars(ap.parse_args())
	image1 = cv2.imread(args["image1"], cv2.IMREAD_GRAYSCALE)
	image2 = cv2.imread(args["image2"], cv2.IMREAD_GRAYSCALE)

	featureMapping(image1, image2)
	


def confidenceFactor():
	pass
	#use the len list of keypoints of image 1 and image 2
	# then take the len of the list at good mathces
	#take the ratio


def main():

	image = loadImage()
	image = imutils.resize(image, height=700)
	# cv2.imshow("Original Image", image)

	boxCoords = detectEdge(image)

	blurImage(image, boxCoords)

	cv2.waitKey(0)



main()