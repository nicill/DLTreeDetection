# DLTreeDetection

Repository fo Detection of trees in different Datasets using Deep Learning networks

## DataSets:

1) Enshurin Data can be found at: https://www.dropbox.com/scl/fi/e159ijeku5uytqdek1hkh/Data_SarahWithBoxes.zip?rlkey=2d8qist1s5fz47ckg8toy7t2h&st=1mi8othj&dl=0

## Code Structure



## How to run

Sarah's Data

**Before you start** The dataHandling file performs two tasks at the moment, 1) simplifies the problem given a list of classes (see the file for details) and 2) prepares the data to be read by the deep learning networks. The file should only perform **one** of hte two functions at any given time. Which one it performs is decided by two boolean variables after the "ifmain" clause. Make sure to have the appropiate one on.


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

