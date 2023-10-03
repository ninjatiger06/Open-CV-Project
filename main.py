from __future__ import annotations
import numpy as np
import argparse
import cv2
import imutils

"""
	Description: Program identifies car license plates from images and can blur
				plates or swap the plates in different images.
	Authors: Jonas Pfefferman '24 and Kenan Begovic '24
	Date: 9/15/23
"""

def checkPNG(image, imgPath):
	"""
	Purpose: Checks to see if the image is a png. If not, saves the image as a png.
	Parameters: The image as interpreted by cv2, the path to the image
	Returns: The image object (in png form if not already)
	"""
	# Checks to see image isn't already png
	if imgPath[len(imgPath)-4:] != ".png":
		if imgPath[len(imgPath)-4:] == "webp":
			# imgName = imgPath[:len(imgPath)-4]
			imgPath = imgPath[:len(imgPath)-4] + "png"
		else:
			# imgName = imgPath[:len(imgPath)-3]
			imgPath = imgPath[:len(imgPath)-3] + "png"
		
		# Re-saves image as png
		cv2.imwrite(imgPath, image)
		pngImage = cv2.imread(imgPath)
	else:
		pngImage = image

	return pngImage

def loadImage():
	"""
	Purpose: Loads images from command line arguments
	Parameters: None
	Returns: The image objects
	"""
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True, help="Path to Image")
	ap.add_argument("-im2", "--im2", required=True, help="Path to image 2")
	args = vars(ap.parse_args())

	image = cv2.imread(args["image"])
	imgPath = args["image"]
	image = checkPNG(image, imgPath)

	im2 = cv2.imread(args["im2"])
	im2Path = args["im2"]
	im2 = checkPNG(im2, im2Path)

	return image, im2

def detectEdge(image) -> None:
	"""
	Purpose: Detects the rectangle of characters that forms a license plate and
			 identifies it for the user
	Parameters: The image object
	Returns: The coordinates of the license plate, as a list of lists (I know
			 better data structures exist, but this one just was easy and made sense.)
	"""
	# rectangle and square structuring overlays
	rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
	sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

	image = imutils.resize(image, height=700)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# blurred = cv2.GaussianBlur(gray, (11, 11), 0)

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
	box = image[y - 10:y + h + 10, x - 10:x + w + 10].copy()
	# box = image.copy()
	cv2.rectangle(image, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 2)
	cv2.imshow("Identified", image)

	cv2.imshow("Plate", box)

	return [[x-10, y-10], [x + w + 10, y + h + 10]]

def blurImage(image, boxCoords):
	"""
	Purpose: Blurs out the license plate beyond recognition
	Parameters: The image of the car and the coordinates of the plate
	Returns: The blurred plate image
	"""
	# Blurs only the part of the image identified as the license plate
	image[boxCoords[0][1]: boxCoords[1][1], boxCoords[0][0]: boxCoords[1][0]] = cv2.blur(image[boxCoords[0][1]: boxCoords[1][1], boxCoords[0][0]: boxCoords[1][0]], (30, 30))

	cv2.imshow("Blurred", image)

	return image

def swapPlates(image, im2, boxCoords, boxCoords2):
	"""
	Purpose: Creates a new image in which the license plate of the second car is
			 overlaid onto where the license plate of the first car is
	Parameters: The images of both cars, the coordiantes of both plates
	Returns: The new image
	"""
	# Grabs just the second license plate
	plate2 = im2[boxCoords2[0][1]: boxCoords2[1][1], boxCoords2[0][0]: boxCoords2[1][0]]

	# Determines the space needed to be filled on the original image
	plateBoxX = boxCoords[1][0] - boxCoords[0][0]
	plateBoxY = boxCoords[1][1] - boxCoords[0][1]

	# Resizes second license plate to fit the hole where the first would be
	plate2 = cv2.resize(plate2, (plateBoxX, plateBoxY))

	image[boxCoords[0][1]: boxCoords[1][1], boxCoords[0][0]: boxCoords[1][0]] = plate2
	cv2.imshow("Swapped Plates", image)

	return image

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

	image, im2 = loadImage()
	image = imutils.resize(image, height=700)
	im2 = imutils.resize(im2, height=700)

	boxCoords = detectEdge(image)
	cv2.waitKey(0)

	boxCoords2 = detectEdge(im2)

	cv2.waitKey(0)

	blurred = blurImage(image, boxCoords)
	# cv2.imwrite("./images/blurredPlate.png", blurred)

	cv2.waitKey(0)

	swappedPlates = swapPlates(image, im2, boxCoords, boxCoords2)
	# cv2.imwrite("./images/swappedPlates.png", swappedPlates)

	cv2.waitKey(0)


main()