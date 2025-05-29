"""
   File to do experiments in Tree
   Detection methods
"""

import configparser
import sys
import os
import time
import cv2
import torch

from pathlib import Path
from itertools import product

from datasets import TDDataset

from config import read_config
#from imageUtils import boxesFound,read_Binary_Mask,recoupMasks,color_to_gray
from train import train_YOLO,makeTrainYAML, get_transform, train_pytorchModel

#from dataHandling import buildTRVT,buildNewDataTesting,separateTrainTest, forPytorchFromYOLO, buildTestingFromSingleFolderSakuma2
from predict import predict_yolo, predict_pytorch, predict_pytorch_maskRCNN

def makeParamDicts(pars,vals):
    """
        Receives a list with parameter names
        and a list of list with values
        for each parameter

        Creates a list of dictionaries
        with the combinations of parameters
    """
    prod = list(product(*vals))
    res = [dict(zip(pars,tup)) for tup in prod]
    return res

def paramsDictToString(aDict, sep = ""):
    """
    Function to create a string from a params dict
    """
    ret = ""
    for k,v in aDict.items():
        ret+=str(k)+sep+str(v)+sep
    return ret[:-1] if sep != "" else ret


def DLExperiment(conf, doYolo = False, doPytorchModels = False):
    """
        Experiment to compare different values of DL networks
    """
    # use the GPU or the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # the pre processing may be added here instead of doign it separately

    # at the moment yolo is NOT WORKING WITH THIS CODE
    f = open(conf["outTEXT"][:-4]+"YOLO"+conf["outTEXT"][-4:],"w+")
    print("consider YOLO? "+str(doYolo))
    # start YOLO experiment
    # Yolo Params is a list of dictionaries with all possible parameters
    yoloParams = makeParamDicts(["scale", "mosaic"],
                                [[0.5,0.9],[0.0,1.0]]) if doYolo else []
    # Print first line of results file
    if yoloParams != []:
        for k in yoloParams[0].keys():
            f.write(str(k)+",")
        f.write("PRECISION"+","+"RECALL"+"TrainT"+"TestT"+"\n")

    for params in yoloParams:
        raise Exception("YOLO HAS NOT BEEN ADAPTED TO THIS DATA YET")
        # Train this version of the YOLO NETWORK
        yamlTrainFile = "trainEXP.yaml"
        prefix = "exp"+paramsDictToString(params)
        makeTrainYAML(conf,yamlTrainFile,params)

        start = time.time()
        if conf["Train"]:
            train_YOLO(conf, yamlTrainFile, prefix)
        end = time.time()
        trainTime = end - start

        # Test this version of the YOLO Network
        print("TESTING YOLO!!!!!!!!!!!!!!!!!")
        start = time.time()
        prec,rec = predict_yolo(conf,prefix+"epochs"+str(conf["ep"])+'ex' )
        end = time.time()
        testTime = end - start
        for k,v in params.items():
            f.write(str(v)+",")
        f.write(str(prec)+","+str(rec)+","+str(trainTime)+","+str(testTime)+"\n")
        f.flush()

    f.close()

    doPytorchModels = True
    print("consider pytorch models? "+str(doPytorchModels))
    f = open(conf["outTEXT"][:-4]+"FRCNN"+conf["outTEXT"][-4:],"w+")

    bs = 64 # should probably be a parameter
    proportion = conf["Train_Perc"]/100

    print("creating dataset in experiment")
    # add dictionary file
    if os.path.exists(conf["dict_file"]):
        dataset = TDDataset(conf["Train_dir"], conf["slice"], get_transform(), classDictFile = conf["dict_file"], verbose = False)
        dataset_test = TDDataset(conf["Test_dir"], conf["slice"], get_transform(), classDictFile = conf["dict_file"],verbose = False)
    else:
        print("no dictionary file!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")    
        dataset = TDDataset(conf["Train_dir"], conf["slice"], get_transform(), verbose = False)
        dataset_test = TDDataset(conf["Test_dir"], conf["slice"], get_transform(), verbose = False)

    num_classes = max(dataset.getNumClasses(),dataset_test.getNumClasses()) # should check between test and training datasets in case one class is not present in some of the two

    print("Experiments, train dataset length "+str(len(dataset) ))

    frcnnParams = makeParamDicts(["modelType","score", "nms", "predconf"],
                                [["maskrcnn","fasterrcnn","ssd","fcos","retinanet"],[0.05, 0.25, 0.5],[0,1,0.25,0,3,0.5],[0.7,0.9,0.95]]) if doPytorchModels else []
    # score: Increase to filter out low-confidence boxes (default ~0.05)
    # nms: Reduce to suppress more overlapping boxes (default ~0.5)
    # predconf prediction confidence in testing

    if frcnnParams != []:
        for k in frcnnParams[0].keys():
            f.write(str(k)+",")
        f.write("PRECC"+","+"RECC"+","+"PRECO"+","+"reco"+","+"TrainT"+","+"TestT"+"\n")

    # this should be for faster rcnn mask rcnn
    for tParams in frcnnParams:
        filePath = "exp"+paramsDictToString(tParams)+"fasterrcnn_resnet50_fpn.pth"

        my_file = Path("/path/to/file")
        trainAgain = not Path(filePath).is_file()
        start = time.time()
        if conf["Train"]:
            pmodel = train_pytorchModel(dataset = dataset, device = device, num_classes = num_classes, file_path = filePath,
                                        num_epochs = conf["ep"], trainAgain=trainAgain, proportion = proportion, mType = tParams["modelType"], trainParams = tParams)
        end = time.time()
        trainTime = end - start

        predConf = tParams["predconf"]
        start = time.time()
        prec,rec, oprec, orec = predict_pytorch(dataset_test = dataset_test, model = pmodel, device = device, predConfidence = predConf, predFolder = os.path.join(conf["Pred_dir"], "exp"+paramsDictToString(tParams))  )
        #prec,rec, oprec, orec = predict_pytorch_maskRCNN(dataset_test = dataset_test, model = pmodel, device = device, predConfidence = predConf) #debugging purposes
        end = time.time()
        testTime = end - start

        for k,v in tParams.items():
            f.write(str(v)+",")
        f.write(str(prec)+","+str(rec)+","+str(oprec)+","+str(orec)+","+str(trainTime)+","+str(testTime)+"\n")
        f.flush()

    # there should be another loop for retina fcos

    f.close()


if __name__ == "__main__":

    # Configuration file name, can be entered in the command line
    configFile = "config.ini" if len(sys.argv) < 2 else sys.argv[1]

    #computeAndCombineMasks(configFile)
    #classicalDescriptorExperiment(configFile)
    #BEST
    #DOG 77.5665178571429	 {'over': 0.5;min_s': 20;max_s': 100}
    # MSER 72.6126785714286	 {'delta': 5;minA': 500;maxA': 25000}


    # DL experiment
    conf = read_config(configFile)
    print(conf)

    DLExperiment(conf,doYolo=False,doPytorchModels=True)
