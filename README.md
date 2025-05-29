# DLTreeDetection

Repository fo Detection of trees in different Datasets using Deep Learning networks

## DataSets:

1) Enshurin Data can be found at: https://www.dropbox.com/scl/fi/e159ijeku5uytqdek1hkh/Data_SarahWithBoxes.zip?rlkey=2d8qist1s5fz47ckg8toy7t2h&st=1mi8othj&dl=0

## Code Structure



## How to run

Sarah's Data

1) Download the Data File, uncompress (from now on we suppose the uncompressed folder is ./Data/Data_Sarah/
2) Transform into training and testing folders with the appropriate format (tiles cut with mosaic tile, image label tile and tiles boxes text file)

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
	
