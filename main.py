from __future__ import annotations
import numpy as np
import argparse
import cv2
import imutils
import matplotlib.pyplot as plt

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
		if imgPath[len(imgPath)-4:] == "webp" or imgPath[len(imgPath)-4:] == "jpeg":
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
	ap.add_argument("-im3", "--im3", required=True, help="Path to Image 3")
	ap.add_argument("-im4", "--im4", required=True, help="Path to Image 4")
	args = vars(ap.parse_args())

	image = cv2.imread(args["image"])
	imgPath = args["image"]
	image = checkPNG(image, imgPath)

	im2 = cv2.imread(args["im2"])
	im2Path = args["im2"]
	im2 = checkPNG(im2, im2Path)

	im3 = cv2.imread(args["im3"], )
	im3Path = args["im3"]
	im3 = checkPNG(im3, im3Path)

	im4 = cv2.imread(args["im4"])
	im4Path = args["im4"]
	im4 = checkPNG(im4, im4Path)

	return image, im2, im3, im4


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
	cv2.imshow("blackhat", blackhat)

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
	cv2.imshow("Threshheld", thresh)
	cv2.waitKey(0)

	# sort thresholds by size
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	
	# draw the rectangle
	(x, y, w, h) = cv2.boundingRect(cnts[0])
	box = image[y - 10:y + h + 10, x - 10:x + w + 10].copy()
	# box = image.copy()
	cv2.rectangle(image, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 2)
	cv2.imshow("Identified", image)
	cv2.waitKey(0)

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

def featureMapping(img1, img2) -> None:
	"""
	Note: Deprecated
	Description: Feature mapping via the patented SURF function
	Parameters: Two images to match
	Returns: None
	"""

	#-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
	minHessian = 400
	detector = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
	keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
	keypoints2, descriptors2 = detector.detectAndCompute(img2, None)
	#-- Step 2: Matching descriptor vectors with a FLANN based matcher
	# Since SURF is a floating-point descriptor NORM_L2 is used
	matcher = cv2.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
	knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)
	#-- Filter matches using the Lowe's ratio test
	ratio_thresh = 0.7
	good_matches = []
	for m,n in knn_matches:
 		if m.distance < ratio_thresh * n.distance:
 			good_matches.append(m)
	
	# Draw matches
	img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
	cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
	#-- Show detected matches
	cv2.imshow('Good Matches', img_matches)

def siftMatching(img1, img2) -> None:
	"""
	Description: Feature mapping via the (not deprecated) SIFT function
	Parameters: Two images to match to each other
	Returns: None
	"""
	# Initiate SIFT detector
	sift = cv2.SIFT_create()
	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)
	# FLANN parameters
	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50) # or pass empty dictionary
	flann = cv2.FlannBasedMatcher(index_params,search_params)
	matches = flann.knnMatch(des1,des2,k=2)
	# Need to draw only good matches, so create a mask
	matchesMask = [[0,0] for i in range(len(matches))]
	# ratio test as per Lowe's paper
	for i,(m,n) in enumerate(matches):
		if m.distance < 0.7*n.distance:
			matchesMask[i] = [1,0]
	draw_params = dict(matchColor = (0,255,0),
	singlePointColor = (255,0,0),
	matchesMask = matchesMask,
	flags = cv2.DrawMatchesFlags_DEFAULT)
	img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
	plt.imshow(img3,),plt.show()

def loadImageFeatureMap():

	ap = argparse.ArgumentParser()
	ap.add_argument("-image1", "--i1", required=True, help="Path to Image" )
	ap.add_argument("-image2", "--i2", required=True, help="Path to Image" )
	args = vars(ap.parse_args())

	i1 = cv2.imread(args["image1"], cv2.IMREAD_GRAYSCALE)
	im1Path = args["i1"]
	i1 = checkPNG(i1, im1Path)

	i2 = cv2.imread(args["image2"], cv2.IMREAD_GRAYSCALE)
	im2Path = args["i2"]
	i2 = checkPNG(i2, im2Path)

	return i1, i2
	


def confidenceFactor():
	pass
	#use the len list of keypoints of image 1 and image 2
	# then take the len of the list at good mathces
	#take the ratio


def main():
	
	image, im2, im3, im4 = loadImage()
	image = imutils.resize(image, height=700)
	im2 = imutils.resize(im2, height=700)
	im3 = imutils.resize(im3, height=700)
	im4 = imutils.resize(im4, height=700)

	cv2.imshow("Original", image)
	cv2.waitKey(0)

	boxCoords = detectEdge(image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	cv2.imshow("Original", im2)
	cv2.waitKey(0)

	boxCoords2 = detectEdge(im2)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	blurred = blurImage(image, boxCoords)
	cv2.imwrite("./output/blurredPlate.png", blurred)

	cv2.waitKey(0)

	swappedPlates = swapPlates(image, im2, boxCoords, boxCoords2)
	cv2.imwrite("./output/swappedPlates.png", swappedPlates)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

	siftMatching(im3, im4)



main()