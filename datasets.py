import numpy as np
import os
import sys
import re
from random import sample
import cv2

import torch
from torch.utils.data.dataset import Dataset

from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

from imageUtils import sliding_window,read_Color_Image,read_Binary_Mask, cleanUpMask, cleanUpMaskBlackPixels, boxesFromMask
from pathlib import Path
from collections import defaultdict

from dataHandling import readBB

from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.transforms.functional import to_pil_image
from torchvision.tv_tensors import Image, BoundingBoxes, Mask
import matplotlib.pyplot as plt


from PIL import Image

class CPDataset(Dataset):
    # Given a folder containing files stored following a certain regular expression,
    # Load all image files from the folder, put them into a list
    # At the same time, load all the labels from the folder names, put them into a list too!

    def __init__(self,dataFolder=None,transForm=None,listOfClasses=None,verbose = False):

        # Data Structures:
        self.classesDict = {} #a dictionary to store class codes
        self.imageList = [] # a list to store our images
        self.labelList = [] # a list to store our labels
        self.transform = transForm

        # Use this if so we can also build "empty DataSets"
        if dataFolder is not None:
            # Use os.walk to obtain all the files in our folder
            for root, dirs, files in os.walk(dataFolder):
                #Traverse all files
                for f in files:
                    # For every file, get its category
                    currentClass = root.split(os.sep)[-1]

                    # Read the file as a grayscale opencv image
                    currentImage = cv2.imread(os.path.join(root,f),0)
                    if currentImage is None: raise Exception("CPDataset Constructor, problems reading file "+f)

                    # now binarize strictly
                    currentImage[currentImage<=100] = 0
                    currentImage[currentImage>100] = 255

                    # Now be careful, pytorch needs the pixel dimension at the start,
                    # so we have to change the way the images are stored (moveaxis)
                    # We also store the image in the image list
                    self.imageList.append(currentImage)

                    # Finally, maintain a class dictionary, the codes of the
                    # class are assigned in the order in which we encounter them
                    # also, store the label in the label list
                    if currentClass not in self.classesDict: self.classesDict[currentClass] = len(self.classesDict)
                    self.labelList.append(self.classesDict[currentClass])
        if verbose:
            self.classDictToFile("classDict.txt")

    def __getitem__(self, index):

            # When we call getitem, we will generally be passing a piece of data to pytorch
            # First, simply retrieve the proper image from the list of images
            currentImage = cv2.cvtColor(self.imageList[index],cv2.COLOR_GRAY2RGB)
            currentImage = np.moveaxis(currentImage,-1,0)
            #currentImage = currentImage[:,:,::-1] #change from BGR to RGB

            # We will need to transform our images to torch tensors
            currentImage = torch.from_numpy(currentImage.astype(np.float32)) # transform to torch tensor with floats
            if self.transform :
                currentImage = self.transform(currentImage) # apply transforms that may be necessary to

            inputs = currentImage

            # in this case, and to make things simple, the target is the code of the class the patch belongs to
            target = self.labelList[index]

            return inputs, target

    def __len__(self):
        return len(self.imageList)

    def numClasses(self):return len(np.unique(list(self.classesDict.keys())))

    #Create two dataset (training and validation) from an existing one, do a random split
    def breakTrainValid(self,proportion):
        train=CPDataset(None)
        valid=CPDataset(None)
        train.classesDict = self.classesDict
        valid.classesDict = self.classesDict

        train.transform = self.transform
        valid.transform = self.transform

        #randomly shuffle the data
        toDivide = sample(list(zip(self.imageList,self.labelList)),len(self.imageList))

        for i in range(int(len(self)*proportion)):
            valid.imageList.append(toDivide[i][0].copy())
            valid.labelList.append(toDivide[i][1])

        for i in range(int(len(self)*proportion),len(self)):
            train.imageList.append(toDivide[i][0].copy())
            train.labelList.append(toDivide[i][1])

        return train,valid

    def classDictToFile(self,outFile):
        """
        write the classDictionary to file

        """
        if (len(self.classesDict.items())>0):
            with open(outFile,"w") as f:
                for cod,unicode in self.classesDict.items():
                    f.write(str(unicode)+","+str(cod)+"\n")


class tDataset(CPDataset):
    # Given a list of images, create a simple dataset with empty labels
    # this is for testing purposes, to have the dataset format
    # for lists of binarized grayscale images

    def __init__(self,imageList,transform = None):
        # Data Structures:
        self.imageList = imageList
        self.labelList = ["nolabel"]*len(imageList) # a list to store our labels
        self.transform = transform
        #print(self.transform)


def extractMask(label_image, box):
    """
    Returns a boolean mask of the same shape as label_image,
    where only pixels inside the bounding box (px, py, w, h)
    and equal to the given cat are True.
    """
    px, py, w, h, cat = box
    mask = np.zeros_like(label_image, dtype=bool)

    # Compute bounding box limits and clip to image bounds
    x_end = min(px + w, label_image.shape[1])
    y_end = min(py + h, label_image.shape[0])

    # Create the mask in the bounding box area
    mask[py:y_end, px:x_end] = (label_image[py:y_end, px:x_end] == cat)

    return mask

def readClassDict(file):
    """
        Read a file containing a class
        dictionary as a pair of keys in every
        line
    """
    retDict = {0:0}
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            retDict[int(line.strip().split(",")[0])] = int(line.strip().split(",")[1])
    return retDict


# sources
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# https://pytorch.org/vision/main/models.html
class TDDataset(Dataset):
    """
        Dataset for object detection with
        pytorch predefined networks
    """
    def __init__(self, dataFolder = None, slice = 1000 , transform = None, classDictFile = "", verbose = False):
        """
            Receive a folder, PRE SLICED!:
            Each piece of data is made up of three parts
            1) an image of png format (the name of the image follows the convention SAAxXyY.png) where SA is the site is the original image the tile comes from and x,y are the position of the tile in the image
            2) a label image of tif format (with the same name as the image but with Label at the end and tif format)
            3) a text file with the boxes in the image (same name as image but with Boxes at the end and txt format)

            Create list of names for the three types of files and a dictionary to reconstruct the slices
        """
        # Data Structures:
        self.imageNameList = []
        self.labelNameList = []
        self.boxNameList = []
        self.transform = transform

        self.slicesToImages = defaultdict(lambda:[])

        # create output Folder if it does not exist
        #self.outFolder = os.path.join(dataFolder,"forOD")
        #Path(self.outFolder).mkdir(parents=True, exist_ok=True)

        for dirpath, dnames, fnames in os.walk(dataFolder):
            for f in fnames:
                # read only the images
                if f[-4:] == ".png":
                    imName = os.path.join(dataFolder,f)
                    labelName = os.path.join(dataFolder,f[:-4]+"Labels.tif")
                    boxFileName = os.path.join(dataFolder,f[:-4]+"Boxes.txt")
                    siteName = f[f.rfind("S"):f.find("x")]
                    if verbose: print([imName,labelName,boxFileName,siteName])
                    self.imageNameList.append(imName)
                    self.labelNameList.append(labelName)
                    self.boxNameList.append(boxFileName)
                    self.slicesToImages[siteName].append((imName,labelName,boxFileName))

        # class dictionary 
        self.classDict = {} if classDictFile == "" else readClassDict(classDictFile)
        self.numClasses = max(self.classDict.values()) + 1 if len(self.classDict) > 0 else self.findMaxClass() + 1 # the +1 are there to account for the background class

        if verbose:
            print("\n\n\n")
            print(self.classDict)
            print("\n\n\n")
            print(self.imageNameList)
            print("\n\n\n")
            print(self.labelNameList)
            print("\n\n\n")
            print(self.boxNameList)
            print("\n\n\n")
            print(self.slicesToImages)
            print("\n\n\n")
            print(self.getNumClasses())

    def __getitem__(self, idx):

        # load images and masks
        img_path = self.imageNameList[idx]
        label_path = self.labelNameList[idx]
        img = Image.open(img_path)
 
        #print(label_path)
        labelIm = cv2.imread(label_path,cv2.IMREAD_UNCHANGED)
 
        boxesRaw = readBB(self.boxNameList[idx]) # raw boxes are px,py,w,h,cat

        #print(boxes)
        #sys.exit()
        # convert the PIL Image into a numpy array
        #mask = np.array(mask)
        #numLabels, labelIm,  stats, centroids = cv2.connectedComponentsWithStats(255-mask)

        # need to Create labels, masks ,boxes
        num_objs = len(boxesRaw)
        boxes = []
        labels = []
        masks = []
        for px,py,w,h, cat in boxesRaw:
            l = self.classDict[cat] if len(self.classDict) > 0 else cat # look up the label corresponding to this category if necessary 
            boxes.append([px, py,px + w, py + h])
            labels.append(l)
            masks.append(extractMask(labelIm,(px,py,w,h,cat)))
        # transform labels and masks to torch tensor
        torch.tensor(labels, dtype=torch.int64)
        #masks = torch.as_tensor(masks, dtype=torch.uint8)
        masks = torch.from_numpy(np.stack(masks)).to(torch.bool)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(torch.as_tensor(boxes), format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask( masks )
        target["labels"] = torch.as_tensor(labels)

        if self.transform is not None:
            img, target = self.transform(img, target)

        #print(target)
        return img, target

    def findMaxClass(self):
        """
            When we are not give a
            class dictionary, 
            find the maximum class by going over 
            the list of boxes 
        """
        maxCat = 0
        for idx in range(len(self)):
            boxesRaw = readBB(self.boxNameList[idx]) # raw boxes are px,py,w,h,cat
            for px,py,w,h, cat in boxesRaw:
                if cat > maxCat: maxCat = cat
        return maxCat


    def __len__(self):
        return len(self.imageNameList)

    def getSliceFileInfo(self):
        """
        return the information about how everything was sliced
        """
        return self.slicesToImages

    def getNumClasses(self):
        """
            Go over all boxes to see how many different 
            categories we have
            take into account class dictionary
        """
        return self.numClasses

    def saveVisualizations(self, output_dir, label_colors=None):
        """
        Save images with masks and bounding boxes overlaid to `output_dir`.
        Each label is visualized with a different color.
        """
        os.makedirs(output_dir, exist_ok=True)

        for idx in range(len(self)):
            img, target = self[idx]

            # Convert to uint8 if needed
            img_tensor = img if img.dtype == torch.uint8 else (img * 255).to(torch.uint8)

            labels = target["labels"]
            boxes = target["boxes"]
            masks = target["masks"]

            # Determine number of classes and generate colors if not given
            num_classes = self.getNumClasses()
            if label_colors is None:
                colors = [tuple(torch.randint(0, 256, (3,), dtype=torch.uint8).tolist()) for _ in range(num_classes)]
            else:
                colors = label_colors

            # Draw masks
            if masks.ndim == 3 and masks.shape[0] > 0:
                img_with_masks = draw_segmentation_masks(
                    img_tensor, masks.bool(), alpha=0.4, colors=[colors[l] for l in labels]
                )
            else:
                img_with_masks = img_tensor

            # Draw bounding boxes and labels
            img_with_boxes = draw_bounding_boxes(
                img_with_masks,
                boxes,
                labels=[str(l.item()) for l in labels],
                colors=[colors[l] for l in labels],
                width=2,
                font_size=16
            )

            # Save to disk
            to_pil_image(img_with_boxes).save(os.path.join(output_dir, f"img_{idx:04d}.png"))


if __name__ == '__main__':

    inData = sys.argv[1]
    if os.path.exists(sys.argv[2]):
        aDataset = TDDataset( dataFolder = inData, slice = 1000 , transform = None, classDictFile = sys.argv[2], verbose = True )
    else:
        print("dictionary file does not exist")    
        aDataset = TDDataset( dataFolder = inData, slice = 1000 , transform = None, verbose = True )
    if len(sys.argv) > 3: aDataset.saveVisualizations(sys.argv[3])
