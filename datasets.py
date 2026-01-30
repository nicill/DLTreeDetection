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

def extractMask(labelIm, b):
    px, py, w, h, cat = b
    H, W = labelIm.shape[:2]
    m = np.zeros((H, W), dtype=np.uint8)
    m[py:py+h, px:px+w] = (labelIm[py:py+h, px:px+w] == cat).astype(np.uint8)
    return m
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

        #for dirpath, dnames, fnames in os.walk(dataFolder):
        for f in os.listdir(dataFolder): # ignore subfolders
            # read only the images
            if f[-4:] == ".png":
                imName = os.path.join(dataFolder,f)
                labelName = os.path.join(dataFolder,f[:-4]+"Labels.tif")
                boxFileName = os.path.join(dataFolder,f[:-4]+"Boxes.txt")
                #siteName = f[f.rfind("S"):f.find("x")]
                siteName = f[:f.find("x")]
                if verbose: print([imName,labelName,boxFileName,siteName])
                self.imageNameList.append(imName)
                self.labelNameList.append(labelName)
                self.boxNameList.append(boxFileName)
                self.slicesToImages[siteName].append((imName,labelName,boxFileName))

        # class dictionary
        self.classDict = {} if classDictFile == "" else readClassDict(classDictFile)
        self.numClasses = max(self.classDict.values()) + 1 if len(self.classDict) > 0 else self.findMaxClass() + 1 # the +1 is there to account for the background class
        self.numClasses = self.findMaxClass() + 1 # the +1 is there to account for the background class

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
        for  cat,px,py,w,h in boxesRaw:
            l = self.classDict[cat] if len(self.classDict) > 0 else cat # look up the label corresponding to this category if necessary
            boxes.append([px, py,px + w, py + h])
            labels.append(l)
            masks.append(extractMask(labelIm,(px,py,w,h,cat)))
        # transform labels and masks to torch tensor
        labels = torch.tensor(labels, dtype=torch.int64)
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

class TDDETRDataset(TDDataset):
    """
    Extended version that also includes segmentation masks in COCO format.
    Use this if your DETR model supports instance segmentation.

    Can safely be used for detection-only tasks - the segmentation field
    will simply be ignored by detection models.
    """

    def __init__(self, dataFolder=None, slice=1000, transform=None,
                 classDictFile="", verbose=False, include_masks=False):
        """
        Args:
            include_masks (bool): If False, behaves like TDDETRDataset (no masks).
                                 If True, includes segmentation masks.
                                 Default True for flexibility.
        """
        super().__init__(dataFolder, slice, transform, classDictFile, verbose)
        self.include_masks = include_masks

    def __getitem__(self, idx):
        """
        Returns image and annotations in COCO format with masks.
        """
        # Load image
        img_path = self.imageNameList[idx]
        img = Image.open(img_path).convert("RGB")

        # Load label image
        label_path = self.labelNameList[idx]
        labelIm = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

        # Read box information from text file
        boxesRaw = readBB(self.boxNameList[idx])

       # DEBUG: Print first 5 samples
        if idx < 5:
            print(f"\n=== DEBUG Sample {idx} ===")
            print(f"Image: {os.path.basename(img_path)}, Size: {img.size}")
            print(f"Number of raw boxes: {len(boxesRaw)}")
            if len(boxesRaw) > 0:
                print(f"First raw box: {boxesRaw[0]}")
                print(f"All raw boxes: {boxesRaw}")

        annotations = []

        # FIXED: boxes are in format cat, px, py, w, h (NOT xyxy!)
        for cat, px, py, w, h in boxesRaw:
            # Skip degenerate boxes
            if w <= 1 or h <= 1:
                continue

            # Map category using class dictionary if available
            category_id = (self.classDict.get(cat, cat) if len(self.classDict) > 0 else cat) - 1

            # COCO format bbox: [x, y, width, height]
            bbox = [float(px), float(py), float(w), float(h)]

            annotation = {
                "bbox": bbox,
                "category_id": int(category_id),
                "iscrowd": 0
            }

            # Only extract masks if requested
            if self.include_masks:
                mask = extractMask(labelIm, (px, py, w, h, cat))
                area = float(np.sum(mask > 0))  # More accurate from mask
                annotation["segmentation"] = mask
                annotation["area"] = area
            else:
                # Calculate area from bbox
                annotation["area"] = float(w * h)

            annotations.append(annotation)

        # Return as numpy array
        image = np.array(img)
        target = {"annotations": annotations}

        return image, target



if __name__ == '__main__':

    inData = sys.argv[1]
    if os.path.exists(sys.argv[2]):
        aDataset = TDDataset( dataFolder = inData, slice = 1000 , transform = None, classDictFile = sys.argv[2], verbose = True )
    else:
        print("dictionary file does not exist")
        aDataset = TDDataset( dataFolder = inData, slice = 1000 , transform = None, verbose = True )
    if len(sys.argv) > 3: aDataset.saveVisualizations(sys.argv[3])
