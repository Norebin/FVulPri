#-*- coding: utf-8 -*-

# 全局取消证书验证
import csv
import datetime
import os
import ssl
from datetime import time

import pandas as pd

ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange
import argparse
from jsonparse import saveAllDataToRam
from jsonparse import getCodePairDataList
from sublayers.Focal_loss import FocalLoss
import random
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from myModels.RNN.rnn_vul_detection import VulnerabilityDetection
from myModels.RNN.GAT_Edgepool_rnn import rnn_detect


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
# parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--hidden', type=int, default=768)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--nheads', type=int, default=16)
parser.add_argument('--dropout', type=int, default=0.1)
parser.add_argument('--alpha', type=int, default=0.2)
parser.add_argument("--threshold", default=0)
args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

# device = "cpu"

def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print('{:02d}/{:03d}: Val Loss: {:.4f}, Test Accuracy: {:.3f}'.format(fold, epoch, val_loss, test_acc))


#----------------------------------------------------------------
indexdir='../BERT/'
id = 'my'
jsonVecPath = "../BERT/data/"
sourceCodePath = "../BERT/data/"
# print(jsonVecPath, " ", id)

trainfile = open(indexdir + 'trainlist.txt')
testfile = open(indexdir + 'testlist.txt')

trainlist=trainfile.readlines()
testlist=testfile.readlines()

print("trainlist",len(trainlist))
print("testlist",len(testlist))


def getBatch(line_list, batch_size, batch_index, device):
    with torch.no_grad():
        start_line = batch_size*batch_index
        end_line = start_line+batch_size
        dataList = getCodePairDataList(ramData,line_list[start_line:end_line])
    return dataList
#----------------------------------------------------------------
'''
def create_batches(data):
    #random.shuffle(data)
    batches = [data[graph:graph+args.batch_size] for graph in range(0, len(data), args.batch_size)]
    return batches
'''



#损失函数使用交叉熵
#criterion = nn.CrossEntropyLoss().to(device)
#criterion = nn.CosineEmbeddingLoss().to(device)
#criterion=nn.MSELoss()
#criterion = nn.BCEWithLogitsLoss().to(device)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, size_average=None, reduce=None)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, size_average=None, reduce=None)

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

criterion=FocalLoss().to(device)

def graph_emb(data,epoch):
    # model = graphEmb(args.num_layers, args.hidden, args.nheads, args.num_classes, args.dropout, args.alpha, False).to(device)
    # saveModel = torch.load('./saveModels/epoch'+str(epoch)+'.pkl')
    # model_dict = model.state_dict()
    # state_dict = {k:v for k,v in saveModel.items() if k in model_dict.keys()}
    # #print(state_dict.keys())
    # model_dict.update(state_dict)
    # model.load_state_dict(model_dict)
    # #print("loaded "+ 'epoch'+str(epoch)+'.pkl')
    # model.eval()
    desc_features, mess_features = data
    data = mess_features
    # h = model(data)
    return data
def bi_lstm_detection(data,epoch):
    model = rnn_detect(args.num_layers, args.hidden, args.nheads, args.num_classes, args.dropout, args.alpha, False).to(device)
    saveModel = torch.load('./saveModels/rnn/epoch'+str(epoch)+'.pkl')
    model_dict = model.state_dict()
    state_dict = {k:v for k,v in saveModel.items() if k in model_dict.keys()}
    #print(state_dict.keys())
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    #print("loaded "+ 'epoch'+str(epoch)+'.pkl')
    model.eval()
    # h1, h2 = data
    # out = model(h1, h2)
    h1= data
    out = model(h1)
    return out

def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list

def test(testlist, model_index, ramData, batch_size):
    graphEmbDict = {}
    print("save graphEmbDict...")
    for codeID in tqdm(ramData):
        data = ramData[codeID]
        h1,h2 = data
        # graphEmbDict[codeID] = graph_emb(data, model_index).tolist()
        graphEmbDict[codeID] = h1
    notFound = 0
    testCount = 0
    y_preds = []
    y_trues = []

    data_df = []

    batches = split_batch(testlist, batch_size)
    Test_data_batches = trange(len(batches), leave=True, desc = "Test")
    for i in Test_data_batches:
        h1_batch = []
        # h2_batch = []
        label_batch = []
        predicted = []
        for codepair in batches[i]:
            try:
                test_data = codepair.split()
                graphEmbDict[test_data[0]]
                # graphEmbDict[test_data[1]]
                testCount+=1
            except:
                notFound+=1
                continue

            h1 = torch.as_tensor(graphEmbDict[test_data[0]]).to(device)
            # h2 = torch.as_tensor(graphEmbDict[test_data[1]]).to(device)
            label = int(test_data[1])-1
            try:
                label = int(test_data[1])-1
            except:
                print(test_data)

            output = bi_lstm_detection(h1, model_index)
            print(output)
            predicted.append(output.tolist())

            h1_batch.append(h1)
            # h2_batch.append(h2)
            label_batch.append(label)

        # h1_batch_t = torch.stack(h1_batch, dim=1).squeeze(0)
        # h2_batch_t = torch.stack(h2_batch, dim=1).squeeze(0)
        #print("h1_batch",h1_batch.shape)
        # data = h1_batch_t, h2_batch_t
        # data = h1_batch_t
        # outputs = bi_lstm_detection(data, model_index)
        predicted = torch.as_tensor(predicted).squeeze(1).to(device)
        # _, predicted = torch.max(outputs.data, 1)
        _, predicted = torch.max(predicted, 1)
        # print('predicted', predicted)
        y_preds += predicted.tolist()
        y_trues += label_batch
        print("y_preds", y_preds)
        print("y_trues", y_trues)

        h1_batch = []
        h2_batch = []
        label_batch = []

        acc = accuracy_score(y_trues, y_preds)
        r_a=recall_score(y_trues, y_preds, average='macro')
        p_a=precision_score(y_trues, y_preds, average='macro')
        f_a=f1_score(y_trues, y_preds, average='macro')
        mae = mean_absolute_error(y_trues,y_preds)
        mse = mean_squared_error(y_trues,y_preds)
        rmse = np.sqrt(mean_squared_error(y_trues,y_preds))
        mape = (abs(np.array(y_preds) -np.array(y_trues)) / np.array(y_trues)).mean()
        r_2 = r2_score(y_trues, y_preds)

        Test_data_batches.set_description("Test (acc=%.4g,p_a=%.4g,r_a=%.4g,f_a=%.4g,mae=%.4g,mse=%.4g,rmse=%.4g,mape=%.4g,r_2=%.4g)" % (acc,p_a, r_a, f_a, mae,mse,rmse,mape,r_2))
    print("testCount",testCount)
    print("notFound",notFound)
    return acc, p_a, r_a, f_a, mae,mse,rmse,mape,r_2

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def train():
    #start_train_model_index = 7
    addNum = 0
    model = VulnerabilityDetection(args.num_layers, args.hidden, args.nheads, args.num_classes, args.dropout, args.alpha, True).to(device)

    #model.load_state_dict(torch.load('./saveModel/epoch'+str(start_train_model_index)+'.pkl'))
    #优化器使用Adam
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    random.shuffle(trainlist)
    epochs = trange(args.epochs, leave=True, desc = "Epoch")
    iterations = 0
    acc_loss = open('acc_loss_rnn.csv', mode='a', encoding="utf=8", newline='')
    fieldnames = ["Accuracy", "loss"]
    writer = csv.DictWriter(acc_loss, fieldnames=fieldnames)
    writer.writeheader()
    test_rec = open('test_rec_rnn.csv', mode='a', encoding="utf=8", newline='')
    fieldnames1 = ["acc", "p","r","f1", "mae","mse","rmse","mape","r_2"]
    writer1 = csv.DictWriter(test_rec, fieldnames=fieldnames1)
    writer1.writeheader()
    for epoch in epochs:
        # print(epoch)
        totalloss=0.0
        main_index=0.0
        #batches = create_batches(trainDataList)
        #for index, batch in tqdm(enumerate(batches), total=len(batches), desc = "Batches"):
        count = 0
        right = 0

        for batch_index in tqdm(range(int(len(trainlist)/args.batch_size))):
        # for batch_index in range(int(len(trainlist)/args.batch_size)):
            batch = getBatch(trainlist, args.batch_size, batch_index, device)
            # print('batch', batch)
            optimizer.zero_grad()
            batchloss= 0
            recoreds = open("./recoreds.txt", 'a')
            recoreds.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            for data in batch:

                # features1, features2, edge_index1, edge_index2, edgesAttr1, edgesAttr2, adjacency1, adjacency2, node2node_features1, node2node_features2, label = data
                desc_features, mess_features,label = data

                # data = features1, features2, edge_index1, edge_index2, edgesAttr1, edgesAttr2, adjacency1, adjacency2, node2node_features1, node2node_features2
                data = mess_features
                labely = torch.zeros(1,10).to(device)
                labely[0][int(label)-1] = 1
                # label=torch.Tensor([[0,1]]).to(device) if label==10 else torch.Tensor([[1,0]]).to(device)
                # print("label ",label.device," ",label)
                output = model(data)
                # print("criterion",criterion(prediction,label))
                #cossim=F.cosine_similarity(h1,h2)
                # print("output",output,torch.argmax(output, dim=1))
                # print("label",label,torch.argmax(label, dim=1))
                batchloss = batchloss + criterion(output, labely)
                count += 1
                right += torch.sum(torch.eq(torch.argmax(output, dim=1), torch.argmax(labely, dim=1)))

            # print("batchloss",batchloss)
            # print(right, count)
            acc = right*1.0/count

            batchloss.backward(retain_graph = True)
            optimizer.step()
            loss = batchloss.item()
            totalloss += loss
            main_index = main_index + len(batch)
            loss = totalloss/main_index
            # epochs.set_description("Epoch (Loss=%g) (Acc = %g)" % (round(loss,5) , acc))
            iterations += 1
            # recoreds.write(str(iterations+addNum*14078) +" "+ str(acc.item()) +" "+ str(loss)+"\n")
            writer.writerow({'Accuracy': str(acc.item()), 'loss': str(loss)})
            recoreds.write(str(iterations) +" acc:"+ str(acc.item()) +" loss:"+ str(loss)+"\n")
            recoreds.close()

        #if(epoch%10==0 and epoch>0):
        print("save epoch")
        torch.save(model.state_dict(), './saveModels/rnn/epoch'+str(epoch+addNum)+'.pkl')

        test_recoreds = open("test_recoreds.txt", 'a')
        acc, p,r,f1, mae,mse,rmse,mape,r_2 = test(testlist,epoch+addNum, ramData, 15000)
        test_recoreds.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + 'epoch'+str(epoch+addNum) +" acc="+ str(acc)+'p='+ str(p) +" r="+ str(r) +" f1="+ str(f1)+" mae="+ str(mae)+" mse="+ str(mse)+" rmse="+ str(rmse)+" mape="+ str(mape)+"r_2="+ str(r_2)+"\n")
        writer1.writerow({'acc':acc,'p':p,'r':r,'f1':f1, 'mae':mae,'mse':mse,'rmse':rmse,'mape':mape,'r_2':r_2})
        test_recoreds.close()


print("add all data to ram...")
# ramData = saveAllDataToRam(sourceCodePath, jsonVecPath)
ramData = np.load('./ramData.npy', allow_pickle=True).item()
# print(ramData['../json/jsonData/id1/CWE78_OS_Command_Injection__char_environment_w32_spawnlp_07_good_20772.json'])
# print(len(ramData))

print("start train...")
train()

