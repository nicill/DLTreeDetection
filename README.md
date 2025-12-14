# DLTreeDetection

Repository fo Detection of trees and weeds in different Datasets using Deep Learning networks

## DataSets:

1) Old Enshurin Data can be found at: https://www.dropbox.com/scl/fi/e159ijeku5uytqdek1hkh/Data_SarahWithBoxes.zip?rlkey=2d8qist1s5fz47ckg8toy7t2h&st=1mi8othj&dl=0

2) Nart weed detection data can be found at:
https://www.dropbox.com/scl/fi/b53k3eojxdrf0g0q7q305/NART.zip?rlkey=o8v9fwuqtle8l14w58awjqjmm&st=wutkzzys&dl=0


# Current usage 14th December 2025

The code to detect weeds (and trees) using deep learning networks has finally reached a stage when we can start testing it:

- At the moment only pytorch models are fully operational and only maskrcnn has been properly tested. I recommend starting testing there.
- Yolo is also more or less operational but it does not produce all the pretty outputs that Maskrcnn does.
- The evaluation is detection only at this moment and does not consider classes, so weeds that are predicted in the right place but belonging to the wrong species will be considered correct at the moment.
- All of the above things will be improved as soon as possible (they are not very difficult to fix) but the priority now is to get the code running in your computers. While I have done my best to make this code as easy to run as possible, Problems are likely to appear, do not despair. Also having someone with experience using jupyter notebooks and python code would be very helpful if you can achieve it.

Installation:

You need to do two things 1) Install the code repository 2) Copy the Data files to a folder that the code can see.

You need a computer with a GPU installed. The bigger (in terms of GPU RAM and number of cores) the GPU the better. The GPU RAM will determine the batch size which will significantly alter the time it takes to run the experiments.

How to install the repository:

1) Go to https://github.com/nicill/DLTreeDetection and download the code.

This is a github repository, so if you can use github the easiest way to do this is to clone the repository. I recommend doing this as this would make it easier to update the code when new improvements are added.

If you cannot use github, simply go to the page and click the green button labeled "code" and then "Dowload Zip".

2) Once your code has been downloaded, you need to add the data in a place that the code can access. I recommend creating a `Data` folder inside the folder containing the code and then uncompressing the `Nart.zip` file inside of `Data`.

3) Open the `DLWeedDetectionNart.ipynb` file, for example using Anaconda and the jupyter notebook app inside of it. Follow the instructions there on how to process the data so Python can read it (you only need to do this once) and then see how to set up experiments.

The next step is to make sure that you can run experiments on your own. After that we should decide whether or not the results obtained are good enough to publish a paper. The code includes 8 different types of models that can be tested with YOLO and MaskRcnn being the more relevant. If the results are good enough it is pretty easy to make a comparision between different models.

I would recommend setting up a slack channel including all the people interested in running this type of networks so that problems can be ironed out and we can create a knowledge base at the forestry group for efficient use of this code.





# Legacy, may not work at the moment

## How to run

Sarah's Data

**Before you start** The dataHandling file performs two tasks at the moment, 1) simplifies the problem given a list of classes (see the file for details) and 2) prepares the data to be read by the deep learning networks. The file should only perform **one** of the two functions at any given time. Which one it performs is decided by two boolean variables after the "ifmain" clause. Make sure to have the appropiate one on.

1) Download the Data File, uncompress (from now on we suppose the uncompressed folder is ./Data/Data_Sarah/

1.5) If you need to simplify the problem so not all classes are considered and pixels from classes not considered are put to black in the mosaic, make sure that the dataHandling.py shows the following configuration:
    simplifyData = True
    prepareData = False

example call, to only consider classes 2,3,5 and 11

    python dataHandling.py ./Data/Data_Sarah/ ./Data/Data23511/ 2 3 5 11

take into account that the classes will be renamed according to the order they are given in. in the new dataset class 2 will be "1", class 3 will be "2", class 5 will be "3" and class 11 will be "4"

2) Transform into training and testing folders with the appropriate format (tiles cut with mosaic tile, image label tile and tiles boxes text file)


	IMPORTANT: Open the datahandling.py file and check that, after the "ifmain" clause, the following variables are set as follows:
	simplifyData = False
   	prepareData = True
    	prepare = "Sarah"

	Choose training/validation (ex: 1 2 3 4 5) and testing (ex 6 7) folders
	Build training folder: python dataHandling.py ./Data/Data_Sarah/ ./Data/SarahPrepared1to5/  1 2 3 4 5
	Build testing folder: python dataHandling.py ./Data/Data_Sarah/ ./Data/SarahPrepared6to7/  6 7

	With this,  ./Data/SarahPrepared1to5/  will be used as training folder and ./Data/SarahPrepared6to7/ will be used as testing folder

3) (optional but recommended) To check if datasets are being created correctly, use

	python datasets.py PATH DICTFile visualOutputFolder

	for example

	python datasets.py ./Data/SarahPreparedAll/ ./classDicttreenotree.txt ./outVisualAllTNT

	- Where dict file is a text file used to reduce the number of classes considered. In these files every line should contain two numbers and a comma.

		For example, the line

		3,1

		means that the class originally annotated as "3" will now be codified as "1". If we then also have 4,1 then it means that both classes 3 and 4 will now be part of a "gathering class" called "1".

		If the dictionary file does not exist, the classes are left as in the original

	- The final parameter is a folder where the tiles with the masks and the bounding boxes will be stored for visual checking. This is useful to check that the code is working properly


4) Training and testing,

	prepare a config file following configTDWS.ini, add your own paths, at this moment, basically you only need to adapt the "train" part


Koiwainojo Data:

	At this moment I have been using the files I mentioned in slack, we should set this up properly with a dropbox link, but let's wait until we have the correct files

1) Download, uncompress data

2) Transform into training and testing folders with the appropriate format (tiles cut with mosaic tile, image label tile and tiles boxes text file)

	IMPORTANT: Open the datahandling.py file and check that, after the "ifmain" clause, the following variables are set as follows:
	simplifyData = False
   	prepareData = True
    	prepare = "koi"

	python dataHandling.py ./Data/Sergi/Label_image_ROI.tif ./Data/Sergi/ROI_KoiwainoujoMosaic.tif ./Data/koiPrepared/ 90

	this will create two subfolders, train and test inside of the output folder (./DatakoiPrepared) divided randomly into training and testing

steps 3, 4 are the same as before but the config file needs to include a name for the dictionary file that does not reference any existing file (working example commmented out)
