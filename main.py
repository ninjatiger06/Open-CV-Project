import numpy as np
import argparse
import cv2
import imutils


def loadImage():
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True, help="Path to Image")
	args = vars(ap.parse_args())
	image = cv2.imread(args["image"])

	IS_PNG = False
	if args["image"][:len-5]

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
	img_matches = np.empty((max(image1.shape[0], image2.shape[0]) img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
	cv2.drawMatches(image1, keypoints1, image2, keypoints2, goodm, img_matches, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

	cv2.imshow("Good Matches", img_matches)
	cv2.waitKey()
	


def loadImageFeatureMap():
	ap = argparse.Arguementparser()
	ap.add_arguement("-i", "--image1", required=True, help="Path to Image" )
	ap.add_arguement("-i", "--image2", required=True, help="Path to Image" )
	args = vars(ap.parse_args())
	image1 = cv2.imread(args["image1"], cv2.IMREAD_GRAYSCALE)
	image2 = cv2.imread(args["image2"], cv2.IMREAD_GRAYSCALE)

	featureMapping(image1, image2)
	


def confidenceFactor():
	
	#use the len list of keypoints of image 1 and image 2
	# then take the len of the list at good mathces
	#take the ratio





def main():

	image = loadImage()
	



main()