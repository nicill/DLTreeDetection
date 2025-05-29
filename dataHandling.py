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
from scipy import ndimage


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


#def filterBoxesWindow(boxes, ymin, ymax, xmin, xmax):
#    """
#        See what boxes are in the current tile given its limits
#    """
#    return [
#        (px - xmin, py - ymin, w, h, cat)
#        for (px, py, w, h, cat) in boxes
#        if xmin <= px and px + w <= xmax and
#           ymin <= py and py + h <= ymax
#    ]

def boxIntersectionArea(px, py, w, h, xmin, ymin, xmax, ymax):
    inter_w = max(0, min(px + w, xmax) - max(px, xmin))
    inter_h = max(0, min(py + h, ymax) - max(py, ymin))
    return inter_w * inter_h, max(0, min(px + w, xmax) - max(px, xmin)), max(0, min(py + h, ymax) - max(py, ymin)), max(px, xmin), max(py, ymin)

def filterBoxesWindow(boxes, ymin, ymax, xmin, xmax):
    """
        See what boxes are in the current tile given its limits
    """
    filtered = []
    for px, py, w, h, cat in boxes:
        inter_area, inter_w, inter_h, inter_xmin, inter_ymin = boxIntersectionArea(px, py, w, h, xmin, ymin, xmax, ymax)
        if w * h > 0 and inter_area >= 0.5 * w * h:
            filtered.append((inter_xmin - xmin, inter_ymin - ymin, inter_w, inter_h, cat))
    return filtered



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
    print("reading "+str(os.path.join(dataFolder,"S"+siteNumber,"S"+siteNumber+"ROI.jpg")))
    ROI = read_Binary_Mask(os.path.join(dataFolder,"S"+siteNumber,"S"+siteNumber+"ROI.jpg"))
    # apply ROI (black out everything outside of the ROI the ROI is black on white so 0 is inside)
    mosaic[ROI!=0] = (0,0,0)

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
    # make output folder if it did not exist
    Path(outFolder).mkdir(parents=True, exist_ok=True)

    for site in sites:
        print("Starting site "+str(site))
        sliceFolder(dataFolder,site,outputFolder, slice)


def computeBBfromLI(labelIM):
    """
        Compute bounding boxes 
        from the label image    
    """
    int_labelIM = labelIM.astype(np.int32)
    slices = ndimage.find_objects(int_labelIM)
    boxes = []

    for label, slc in enumerate(slices, start=1):
        if slc is None:
            continue  # Label not present

        y_start, y_stop = slc[0].start, slc[0].stop
        x_start, x_stop = slc[1].start, slc[1].stop

        px = x_start
        py = y_start
        w = x_stop - x_start
        h = y_stop - y_start

        boxes.append((px, py, w, h, 1)) # in this case, all labels are just "tree"

    return boxes

def prepareDataKoi(lIMfile, mosFile, outFolderRoot, trainPerc, slice, verbose = True):
    """
        Given a mosaic and 
    """
    labelIM = cv2.imread(lIMfile,cv2.IMREAD_UNCHANGED)
    mosaic = read_Color_Image(mosFile)
    boxes = computeBBfromLI(labelIM)

    # because all labels are just "tree", we change the label image so it only contains 0 and 1
    labelIM[labelIM != 0] = 1     

    # create output folder if it does not exist
    Path(outFolderRoot).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(outFolderRoot,"train")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(outFolderRoot,"test")).mkdir(parents=True, exist_ok=True)

    # slice the three things and output
    wSize = (slice,slice)
    count = 0
    for (x, y, window) in sliding_window(mosaic, stepSize = int(slice*0.8), windowSize = wSize ):
        # get mask window
        if window.shape[:2] == (slice,slice) :
            labelW = labelIM[y:y + wSize[1], x:x + wSize[0]]
            boxesW = filterBoxesWindow(boxes,y,y + wSize[1], x,x + wSize[0])

            if verbose: print(boxesW)

            # here we should probably add cleanUpMaskBlackPixels and maybe do it for YOLO too (in buildtrainvalidation?)
            if len(boxesW) > 0:
                # store them both, doing a randomDraw to see if they go to training or testing
                outFolder = os.path.join(outFolderRoot,"train") if randint(1,100) < trainPerc else os.path.join(outFolderRoot,"test")
                if verbose: print("writing to "+str(os.path.join(outFolder,"Tile"+str(count)+".png")))
                cv2.imwrite(os.path.join(outFolder,"Tile"+str(count)+".png"),window)
                cv2.imwrite(os.path.join(outFolder,"Tile"+str(count)+"Labels.tif"),labelW)
                boxCoordsToFile(os.path.join(outFolder,"Tile"+str(count)+"Boxes.txt"),boxesW)
                count+=1
            else:
                if verbose: print("no boxes here")
        else:
            if verbose:  print("sliceFolder, non full window, ignoring"+str(window.shape))


if __name__ == '__main__':
    
    prepare = "Sarah"
    slice = 1000

    if prepare == "Sarah":
        print("Preparing with Sarah's format, change parameter to do it for Koiwainojo" \
        "")
        dataFolder = sys.argv[1]
        outputFolder = sys.argv[2]
        listOfSites = sys.argv[3:]
        prepareDataFolder(dataFolder, listOfSites, outputFolder, slice)
    elif prepare == "koi":
        print("Preparing data for koiwainojo, change parameter to do it for Sarah's data")
        labelImFile = sys.argv[1]
        mosaicFile = sys.argv[2]
        outputFolder = sys.argv[3]
        trainPerc = int(sys.argv[4])
        prepareDataKoi(labelImFile, mosaicFile, outputFolder, trainPerc, slice )
