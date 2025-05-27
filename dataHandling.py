"""
File to build training, Validation and testing datasets
For tree Detection using deep learning
"""
from imageUtils import sliding_window,read_Color_Image,read_Binary_Mask, sliceAndBox,boxCoordsToFile,boxesFromMask,color_to_gray
import os
import cv2
import sys
from random import randint,uniform
import shutil
from pathlib import Path
#from patch_classification import loadModelReadClassDict,testAndOutputForAnnotations
import re
import numpy as np


def isSpeciesMask(file,sn):
    """
        Function that uses regular expressions to
        check if a file contains a species mask
    """
    match = re.search(sn+"S\d*"+".jpg", file)
    return match is not None

def buildGT(folder):
    """
        Function to build label image, receive folder
        as input and traverse all files
        accummulate those that contains mask into a
        label image
    """

    def processFile(f):
        """
            Inner function to add binary mask
            to the labelImage
        """
        # If it is a species mask,
        # retrieve the species number
        # And add the species label to the label image
        if isSpeciesMask(f,siteName):
            sp_num = int(f[-6:-4])
            mask[read_Binary_Mask(os.path.join(folder,f)) <150] = sp_num

    # Get the list of files in the folder.
    fileList=os.listdir(folder)

    siteName = os.path.basename(os.path.normpath(folder))

    #read ROI
    roi = read_Binary_Mask( os.path.join(folder,siteName+"ROI.jpg") )
    # create black mask of the same size
    mask = np.zeros((roi.shape[0],roi.shape[1]),dtype=np.uint8)

    #now go over the list of sites and add the code of the classes present
    list(map(processFile,fileList))

    # in the end, put everything outside the ROI to background
    mask[roi>150]=0
    return mask

def readBB(file_path):
    """
        Reads a text file where each line contains:
        px, py, w, h, cat
        and returns a list of bounding box dictionaries.
    """
    with open(file_path) as f:
        return list(map(lambda l: tuple(map(int, l.split())), f))


def filterBoxesWindow(boxes, ymin, ymax, xmin, xmax):
    """
        See what boxes are in the current tile given its limits
    """
    return [
        (px - xmin, py - ymin, w, h, cat)
        for (px, py, w, h, cat) in boxes
        if xmin <= px and px + w <= xmax and
           ymin <= py and py + h <= ymax
    ]

def boxCoordsToFile(file,boxC):
    """
        Receive a list of tuples with bounding boxes and write it to file
    """
    def writeTuple(tup):
        px,py,w,h,sP = tup
        f.write(str(px)+" "+str(py)+" "+str(w)+" "+str(h)+" "+str(sP)+"\n")

    with open(file, 'a') as f:
        list(map( writeTuple, boxC))

def sliceFolder(dataFolder,siteNumber,outFolder, slice, verbose = False):
    """
        Receive a folder in Sarah's data format (dataFolder/S"siteNumber")
        create Label Image
        run a sliding window over the mosaic
        divide the mosaic into tiles of size "slice"
        do the same to the label image
        do the same to the list of boxes
    """
    # create label image, read mosaic and list of boxes
    labelIm = buildGT(os.path.join(dataFolder,"S"+siteNumber))
    mosaic = read_Color_Image(os.path.join(dataFolder,"S"+siteNumber,"S"+siteNumber+".jpg"))
    boxes = readBB(os.path.join(dataFolder,"S"+siteNumber,"S"+siteNumber+"BBoxes.txt"))
    #cv2.imwrite("ou.png",labelIm)
    #cv2.imwrite("mosou.png",mosaic)

    # slice the three things and output
    wSize = (slice,slice)
    count = 0
    for (x, y, window) in sliding_window(mosaic, stepSize = int(slice*0.8), windowSize = wSize ):
        # get mask window
        if window.shape[:2] == (slice,slice) :
            labelW = labelIm[y:y + wSize[1], x:x + wSize[0]]
            boxesW = filterBoxesWindow(boxes,y,y + wSize[1], x,x + wSize[0])

            if verbose: print(boxesW)

            # here we should probably add cleanUpMaskBlackPixels and maybe do it for YOLO too (in buildtrainvalidation?)
            if len(boxesW) > 0:
                # store them both
                if verbose: print("writing to "+str(os.path.join(outFolder,"Tile"+str(count)+"S"+siteNumber+".png")))
                cv2.imwrite(os.path.join(outFolder,"Tile"+str(count)+"S"+siteNumber+".png"),window)
                cv2.imwrite(os.path.join(outFolder,"Tile"+str(count)+"S"+siteNumber+"Labels.tif"),labelW)
                boxCoordsToFile(os.path.join(outFolder,"Tile"+str(count)+"S"+siteNumber+"Boxes.txt"),boxesW)
                count+=1
            else:
                if verbose: print("no boxes here")
        else:
            if verbose:  print("sliceFolder, non full window, ignoring"+str(window.shape))

def prepareDataFolder(dataFolder,sites,outFolder,slice):
    """
        Receives the root of a data folder in Sarah's format
        and a list of sieNumbers
        calls sliceFolder for all the sites in the list
    """
    for site in sites:
        print("Starting site "+str(site))
        sliceFolder(dataFolder,site,outputFolder, slice)



if __name__ == '__main__':
    dataFolder = sys.argv[1]
    outputFolder = sys.argv[2]
    listOfSites = sys.argv[3:]
    slice = 1000
    prepareDataFolder(dataFolder,listOfSites,outputFolder, slice)
