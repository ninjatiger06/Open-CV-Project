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

def featureMapping():
	pass

def main():

	image = loadImage()
	



main()