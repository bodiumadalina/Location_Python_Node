from __future__ import print_function
from __future__ import division

import torch
import warnings
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import json
import time
import os

from sklearn.metrics.pairwise import cosine_similarity
import csv
from PIL import Image
import sys
from numpy import genfromtxt
from collections import Counter

num_classes = 16
feature_extract = True
warnings.filterwarnings("ignore")

def Average(lst):
    return sum(lst) / len(lst)


class Img2Vec():

    def __init__(self, model, model_name="resnetplaces365", layer_output_size=2048, cuda=False, layer='default'):
        """ Img2Vec
        :param cuda: If set to True, will run forward pass on GPU
        :param model: String name of requested model
        :param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git
        :param layer_output_size: Int depicting the output size of the requested layer
        """

        self.device = torch.device("cuda" if cuda else "cpu")
        self.model_name = model_name

        if model_name == "resnet" or model_name == "resnetplaces365":
            self.model, self.extraction_layer = self._get_model_and_layer("resnet", layer, model)

        self.model = self.model.to(self.device)

        self.model.eval()
        self.layer_output_size = layer_output_size

        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def get_vec(self, img, tensor=False):
        """ Get vector embedding from PIL image
        :param img: PIL Image or list of PIL Images
        :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
        """
        if type(img) == list:

            a = [self.normalize(self.to_tensor(self.scaler(im))) for im in img]
            images = torch.stack(a).to(self.device)
            my_embedding = torch.zeros(len(img), self.layer_output_size, 1, 1)  # 11

            def copy_data(m, i, o):
                my_embedding.copy_(o.data)

            h = self.extraction_layer.register_forward_hook(copy_data)
            h_x = self.model(images)
            h.remove()

            if tensor:
                return my_embedding
            else:
                print(my_embedding.numpy()[:, :, 0, 0].shape)
                return my_embedding.numpy()[:, :, 0, 0]
        else:
            image = self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0).to(self.device)

            my_embedding = torch.zeros(1, self.layer_output_size, 1, 1)  # 11

            def copy_data(m, i, o):
                my_embedding.copy_(o.data)

            h = self.extraction_layer.register_forward_hook(copy_data)
            h_x = self.model(image)
            h.remove()

            if tensor:
                return my_embedding
            else:
                return my_embedding.numpy()[0, :, 0, 0]

    def _get_model_and_layer(self, model_name, layer, model_ft):
        """ Internal method for getting layer from model
        :param model_name: model name such as 'resnet-18'
        :param layer: layer as a string for resnet-18 or int for alexnet
        :returns: pytorch model, selected layer
        """

        if model_name == 'resnet':
            model = model_ft
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 2048
            else:
                layer = model._modules.get(layer)

            return model, layer


def main():
    room_names = []
    room_list = []
    path = 'vectors/'

    # birou0=np.ndarray((1,1))
    with open(path + 'birou0_file.csv') as birou0_file:
        reader = csv.reader(birou0_file)
        # for row in reader:
        #  np.append(birou0,row)
        birou0 = genfromtxt(birou0_file, delimiter=',')

        room_list.append(birou0)
        room_names.append("birou0")

    # bucatarie0=np.ndarray((1,1))
    with open(path + 'bucatarie0_file.csv') as bucatarie0_file:
        reader = csv.reader(bucatarie0_file)
        bucatarie0 = genfromtxt(bucatarie0_file, delimiter=',')
        # for row in reader:
        #  np.append(bucatarie0,row)
        room_list.append(bucatarie0)
        room_names.append("bucatarie0")

    with open(path + 'cameramica1_file.csv') as cameramica1_file:
        reader = csv.reader(cameramica1_file)
        cameramica1 = genfromtxt(cameramica1_file, delimiter=',')
        room_list.append(cameramica1)
        room_names.append("cameramica1")

    with open(path + 'cameramica2_file.csv') as cameramica2_file:
        reader = csv.reader(cameramica2_file)
        cameramica2 = genfromtxt(cameramica2_file, delimiter=',')
        room_list.append(cameramica2)
        room_names.append("cameramica2")

    with open(path + 'dreapta1_file.csv') as dreapta1_file:
        reader = csv.reader(dreapta1_file)
        dreapta1 = genfromtxt(dreapta1_file, delimiter=',')
        room_list.append(dreapta1)
        room_names.append("dreapta1")

    with open(path + 'hol0_file.csv') as hol0_file:
        reader = csv.reader(hol0_file)
        hol0 = genfromtxt(hol0_file, delimiter=',')
        room_list.append(hol0)
        room_names.append("hol0")

    with open(path + 'salaconferinta2_file.csv') as salaconf2_file:
        reader = csv.reader(salaconf2_file)
        salaconf2 = genfromtxt(salaconf2_file, delimiter=',')
        room_list.append(salaconf2)
        room_names.append("salaconf2")

    with open(path + 'salaintalniri2_file.csv') as salaint2_file:
        reader = csv.reader(salaint2_file)
        salaint2 = genfromtxt(salaint2_file, delimiter=',')
        room_list.append(salaint2)
        room_names.append("salaint2")

    with open(path + 'scamera1_file.csv') as sc1_file:
        reader = csv.reader(sc1_file)
        sc1 = genfromtxt(sc1_file, delimiter=',')
        room_list.append(sc1)
        room_names.append("sc1")

    with open(path + 'scamera2_file.csv') as sc2_file:
        reader = csv.reader(sc2_file)
        sc2 = genfromtxt(sc2_file, delimiter=',')
        room_list.append(sc2)
        room_names.append("sc2")

    with open(path + 'scamera3_file.csv') as sc3_file:
        reader = csv.reader(sc3_file)
        sc3 = genfromtxt(sc3_file, delimiter=',')
        room_list.append(sc3)
        room_names.append("sc3")

    with open(path + 'sjocuri_file.csv') as sjocuri_file:
        reader = csv.reader(sjocuri_file)
        sjocuri = genfromtxt(sjocuri_file, delimiter=',')
        room_list.append(sjocuri)
        room_names.append("sjocuri")

    with open(path + 'shol_file.csv') as shol_file:
        reader = csv.reader(shol_file)
        shol = genfromtxt(shol_file, delimiter=',')
        room_list.append(shol)
        room_names.append("shol")

    with open(path + 'terasa_file.csv') as terasa_file:
        reader = csv.reader(terasa_file)
        terasa = genfromtxt(terasa_file, delimiter=',')
        room_list.append(terasa)
        room_names.append("terasa")
    ##################################

    model_ft = torch.load('resnet_places365.h5')
    #model_ft.fc = nn.Linear(2048, num_classes)
    img2vec = Img2Vec(model_ft)

    start_time = time.time()

    U = sys.argv[1]    #str(input("Which filename would you like similarities for?\n"))

    img = Image.open(os.path.join(U))
    U = img2vec.get_vec(img)

    room_values = {}
    count = 0
    for room in room_list:
        scores = []
        for picture in room:
            scores.append(cosine_similarity(U.reshape((1, -1)), picture.reshape((1, -1)))[0][0])

        scores = sorted(scores, reverse=True)
        scores = scores[:1]
        room_values[room_names[count]] = Average(scores)
        count += 1

    d_view = [(v, k) for k, v in room_values.items()]
    d_view.sort(reverse=True)
    v, k = d_view[0]
    print(k)
    with open('data.json',"r+") as json_file:

        json_file.seek(0)  # rewind
        json.dump(k, json_file)
        json_file.truncate()




main()
