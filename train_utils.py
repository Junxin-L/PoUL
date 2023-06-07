import torch
torch.backends.cudnn.benchmark=True
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import argparse
import time
import numpy as np
import subprocess
from numpy import random
import copy
from Model import *
from data_utils import *
from Param import *
import time

def incremental_train(outfile, class_map, map_reverse, num_iters, train_data, test_data):
    with open(outfile, 'w') as file:
        print("Start training!\n", file=file)
        t1 = time.time()
        model = Model(1, class_map, class_num, num_samples_poi, init_lr, epoch, batch_size)
        model.cuda()
        # acc_matr = np.zeros((int(total_classes/num_classes), num_iters))
        for s in range(0, num_iters, num_classes):
            print('Iteration: ', s)
            print('Iteration: ', s, file=file)
            if(s < clientNum): 
                print (f"Now training clean data for  {clients[s]}")
                print (f"Now training clean data for  {clients[s]}", file=file)
            else:
                print (f"Now training poi data {(s - clientNum) // clientNum} for client {clients[(s + 1 - clientNum) % clientNum]}")
                print (f"Now training poi data {(s - clientNum) // clientNum} for client {clients[(s + 1 - clientNum) % clientNum]}", file=file)

            # Set up the parameters
            if(s > clientNum): 
                model.num_epochs = num_epoch(s)
                model.init_lr = lr(s)
            print(f'Learning rate: {model.init_lr} ', file=file)
            print(f'Learning rate: {model.init_lr} ')
            # Load Datasets and initialize test loader
            train_set = train_data[s]
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                shuffle=True, num_workers=0)
    
            clean_test_loader = torch.utils.data.DataLoader(clean_test(test_data), batch_size=batch_size,
                                                shuffle=True, num_workers=0)

            if(s <= clientNum):
                poi_test_loader = []
                pre_poi_data = []
            else:
                pre_poi_data = train_data[clientNum : s]
                poi_test_set = test_data[clientNum : s + 1]
                poi_test_loader = []
                for set in poi_test_set:
                    poi_test_loader.append(torch.utils.data.DataLoader(set, batch_size=batch_size,
                                                shuffle=True, num_workers=0))
            poi_now_test_loader = torch.utils.data.DataLoader(test_data[s], batch_size=batch_size,
                                                shuffle=True, num_workers=0)

            # Update representation via BackProp
            if s < clientNum : extract = False
            else: extract = True
            model.update(s, train_set, class_map, train_data[0:clientNum], pre_poi_data, True)
            model.eval()

            model.n_known = model.n_classes
    
            total = 0.0
            correct = 0.0
            for images, labels in train_loader:
                images = Variable(images).cuda()
                preds = model.classify(images)
                preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
                total += labels.size(0)
                correct += (preds == labels.numpy()).sum()

            # Train Accuracy
            print ('Train Accuracy : %.2f ,' % (100.0 * correct / total))
            print ('Train Accuracy : %.2f ,' % (100.0 * correct / total), file=file)

            total = 0.0
            correct = 0.0
            test_i = 0
            for images, labels in clean_test_loader:
                images = Variable(images).cuda()
                preds = model.classify(images)
                preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
                total += labels.size(0)
                correct += (preds == labels.numpy()).sum()
                #test_i += 1
                #if test_i == testDataNum: break

            # Clean Test Accuracy
            #print ('%.2f' % (100.0 * correct / total), file=file)
            print ('Clean Test Accuracy : %.2f' % (100.0 * correct / total))
            print ('Clean Test Accuracy : %.2f' % (100.0 * correct / total), file = file)
            
            total = 0.0
            correct = 0.0
            test_i = 0
            for images, labels in poi_now_test_loader:
                images = Variable(images).cuda()
                preds = model.classify(images)
                preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
                total += labels.size(0)
                correct += (preds == labels.numpy()).sum()
                #test_i += 1
                #if test_i == testDataNum: break

            # Poi now Test Accuracy
            #print ('%.2f' % (100.0 * correct / total), file=file)
            print ('Data Now Training Test Accuracy : %.2f' % (100.0 * correct / total))
            print ('Data Now Training Test Accuracy : %.2f' % (100.0 * correct / total), file = file)
    
            total = 0.0
            correct = 0.0
            if(s > clientNum):
                for loader in poi_test_loader:
                    test_i = 0
                    for images, labels in loader:
                        images = Variable(images).cuda()
                        preds = model.classify(images)
                        preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
                        total += labels.size(0)
                        correct += (preds == labels.numpy()).sum()
                # Poison Test Accuracy
                print ('Poison Test Accuracy : %.2f' % (100.0 * correct / total))
                print ('Poison Test Accuracy : %.2f' % (100.0 * correct / total), file=file)
            
            
            model.train()
            t2 = time.time()
            print (f'Time : {t2 - t1}', file=file)
            print ('\n', file=file)
            print ('\n')
            if(s % model_batch == 0 and (s == clientNum*0 or s == clientNum*7 or s == clientNum*5 or s == clientNum*2)): torch.save(model, f".\\Models\\Incremental\\{s//model_batch}.pth")
    return model
            

def joint_train(outfile, class_map, map_reverse, num_iters, train_data, test_data):
    with open(outfile, 'w') as file:
        print("Start training!\n", file=file)
        t1 = time.time()
        model = Model(1, class_map, class_num, 100, joint_init_lr, joint_epoch, batch_size)
        model.cuda()
        acc_matr = np.zeros((int(total_classes/num_classes), num_iters))
    
        # Set up the parameters
        model.num_epochs = joint_epoch
        model.init_lr = joint_init_lr

        # Load Datasets and initialize test loader
        train_loader = []
        clean_test_loader = []
        train_set_joint = []
        for set in train_data:
            train_set_joint += set
        train_loader += torch.utils.data.DataLoader(train_set_joint, batch_size=batch_size,
                                            shuffle=True, num_workers=0)
        for set in test_data[0 : clientNum]:
            clean_test_loader.append(torch.utils.data.DataLoader(set, batch_size=batch_size,
                                                shuffle=True, num_workers=0))
        poi_test_set = test_data[clientNum : ]
        poi_test_loader = []
        for set in poi_test_set:
            poi_test_loader.append(torch.utils.data.DataLoader(set, batch_size=batch_size,
                                                shuffle=True, num_workers=0))
            
        # Update representation via BackProp
        model.update(0, train_set_joint, class_map, [], [], False)
        model.eval()
        model.n_known = model.n_classes
    
        total = 0.0
        correct = 0.0
        for images, labels in train_loader:
            images = Variable(images).cuda()
            preds = model.classify(images)
            preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
            total += labels.size(0)
            correct += (preds == labels.numpy()).sum()

        # Train Accuracy
        print ('Train Accuracy : %.2f ,' % (100.0 * correct / total))
        print ('Train Accuracy : %.2f ,' % (100.0 * correct / total), file=file)
        total = 0.0
        correct = 0.0
        for loader in clean_test_loader:
            for images, labels in loader:
                images = Variable(images).cuda()
                preds = model.classify(images)
                preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
                total += labels.size(0)
                correct += (preds == labels.numpy()).sum()

        # Clean Test Accuracy
        #print ('%.2f' % (100.0 * correct / total), file=file)
        print ('Clean Test Accuracy : %.2f' % (100.0 * correct / total))
        print ('Clean Test Accuracy : %.2f' % (100.0 * correct / total), file = file)

        total = 0.0
        correct = 0.0


        for loader in poi_test_loader:
            for images, labels in loader:
                images = Variable(images).cuda()
                preds = model.classify(images)
                preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
                total += labels.size(0)
                correct += (preds == labels.numpy()).sum()

        # Poison Test Accuracy
        print ('Poison Test Accuracy : %.2f' % (100.0 * correct / total))
        print ('Poison Test Accuracy : %.2f' % (100.0 * correct / total), file=file)
        
        
        model.train()
        t2 = time.time()
        print (f'Time : {t2 - t1}', file=file)
        torch.save(model, f".\\Models\\Joint\\Joint.pth")
    return model

            

def re_incremental_train(start, start_model, outfile, class_map, map_reverse, num_iters, train_data, test_data):
    with open(outfile, 'a') as file:
        print("---------Start Retraining!\n-----------", file=file)
        print("---------Start Retraining!\n-----------")
        t1 = time.time()
        model = model = torch.load(f".\\Models\\Incremental\{start_model}.pth")
        model.cuda()
        acc_matr = np.zeros((int(total_classes/num_classes), num_iters))
        for s in range(start, num_iters, num_classes):
            print('Iteration: ', s)
            print('Iteration: ', s, file=file)
            if(s < clientNum): 
                print (f"Now training clean data for  {clients[s]}")
                print (f"Now training clean data for  {clients[s]}", file=file)
            else:
                print (f"Now training poi data for client {clients[(s + 1 - clientNum) % clientNum]}")
                print (f"Now training poi data for client {clients[(s + 1 - clientNum) % clientNum]}", file=file)

            # Set up the parameters
            if(s > clientNum): 
                model.num_epochs = num_epoch(s)
                model.init_lr = lr(s)
            print(f'Learning rate: {model.init_lr} ', file=file)
            print(f'Learning rate: {model.init_lr} ')
            # Load Datasets and initialize test loader
            train_set = train_data[s]
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                shuffle=True, num_workers=0)
    
            clean_test_loader = torch.utils.data.DataLoader(clean_test(test_data), batch_size=batch_size,
                                                shuffle=True, num_workers=0)

            if(s <= clientNum):
                poi_test_loader = []
                pre_poi_data = []
            else:
                pre_poi_data = train_data[clientNum : s]
                poi_test_set = test_data[clientNum : s + 1]
                poi_test_loader = []
                for set in poi_test_set:
                    poi_test_loader.append(torch.utils.data.DataLoader(set, batch_size=batch_size,
                                                shuffle=True, num_workers=0))
            poi_now_test_loader = torch.utils.data.DataLoader(test_data[s], batch_size=batch_size,
                                                shuffle=True, num_workers=0)

            # Update representation via BackProp
            if s < clientNum : extract = False
            else: extract = True
            model.update(s, train_set, class_map, train_data[0:clientNum], pre_poi_data, True)
            model.eval()

            model.n_known = model.n_classes
    
            total = 0.0
            correct = 0.0
            for images, labels in train_loader:
                images = Variable(images).cuda()
                preds = model.classify(images)
                preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
                total += labels.size(0)
                correct += (preds == labels.numpy()).sum()

            # Train Accuracy
            print ('Train Accuracy : %.2f ,' % (100.0 * correct / total))
            print ('Train Accuracy : %.2f ,' % (100.0 * correct / total), file=file)

            total = 0.0
            correct = 0.0
            test_i = 0
            for images, labels in clean_test_loader:
                images = Variable(images).cuda()
                preds = model.classify(images)
                preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
                total += labels.size(0)
                correct += (preds == labels.numpy()).sum()
                #test_i += 1
                #if test_i == testDataNum: break

            # Clean Test Accuracy
            #print ('%.2f' % (100.0 * correct / total), file=file)
            print ('Clean Test Accuracy : %.2f' % (100.0 * correct / total))
            print ('Clean Test Accuracy : %.2f' % (100.0 * correct / total), file = file)
            
            total = 0.0
            correct = 0.0
            test_i = 0
            for images, labels in poi_now_test_loader:
                images = Variable(images).cuda()
                preds = model.classify(images)
                preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
                total += labels.size(0)
                correct += (preds == labels.numpy()).sum()
                #test_i += 1
                #if test_i == testDataNum: break

            # Poi now Test Accuracy
            #print ('%.2f' % (100.0 * correct / total), file=file)
            print ('Data Now Training Test Accuracy : %.2f' % (100.0 * correct / total))
            print ('Data Now Training Test Accuracy : %.2f' % (100.0 * correct / total), file = file)
    
            total = 0.0
            correct = 0.0
            if(s > clientNum):
                for loader in poi_test_loader:
                    test_i = 0
                    for images, labels in loader:
                        images = Variable(images).cuda()
                        preds = model.classify(images)
                        preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
                        total += labels.size(0)
                        correct += (preds == labels.numpy()).sum()
                        #test_i += 1
                        #if test_i == testDataNum: break
                # Poison Test Accuracy
                print ('Poison Test Accuracy : %.2f' % (100.0 * correct / total))
                print ('Poison Test Accuracy : %.2f' % (100.0 * correct / total), file=file)
            
            
            model.train()
            t2 = time.time()
            print (f'Time : {t2 - t1}', file=file)
            print ('\n', file=file)
            if(s % model_batch == 0): torch.save(model, f".\\Models\\Incremental\\new_{s//model_batch}.pth")
    return model
            

def re_joint_train(outfile, class_map, map_reverse, num_iters, train_data, test_data):
    with open(outfile, 'w') as file:
        print("----------Start Retraining!\n--------", file=file)
        t1 = time.time()
        model = Model(1, class_map, class_num, 100, joint_init_lr, joint_epoch, batch_size)
        model.cuda()
        acc_matr = np.zeros((int(total_classes/num_classes), num_iters))
    
        # Set up the parameters
        model.num_epochs = joint_epoch
        model.init_lr = joint_init_lr

        # Load Datasets and initialize test loader
        train_loader = []
        clean_test_loader = []
        train_set_joint = []
        for set in train_data:
            train_set_joint += set
        train_loader += torch.utils.data.DataLoader(train_set_joint, batch_size=batch_size,
                                            shuffle=True, num_workers=0)
        for set in test_data[0 : clientNum]:
            clean_test_loader.append(torch.utils.data.DataLoader(set, batch_size=batch_size,
                                                shuffle=True, num_workers=0))
        poi_test_set = test_data[clientNum : ]
        poi_test_loader = []
        for set in poi_test_set:
            poi_test_loader.append(torch.utils.data.DataLoader(set, batch_size=batch_size,
                                                shuffle=True, num_workers=0))
            
        # Update representation via BackProp
        model.update(0, train_set_joint, class_map, [], [], False)
        model.eval()
        model.n_known = model.n_classes
    
        total = 0.0
        correct = 0.0
        for images, labels in train_loader:
            images = Variable(images).cuda()
            preds = model.classify(images)
            preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
            total += labels.size(0)
            correct += (preds == labels.numpy()).sum()

        # Train Accuracy
        print ('Train Accuracy : %.2f ,' % (100.0 * correct / total))
        print ('Train Accuracy : %.2f ,' % (100.0 * correct / total), file=file)
        total = 0.0
        correct = 0.0
        for loader in clean_test_loader:
            for images, labels in loader:
                images = Variable(images).cuda()
                preds = model.classify(images)
                preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
                total += labels.size(0)
                correct += (preds == labels.numpy()).sum()

        # Clean Test Accuracy
        #print ('%.2f' % (100.0 * correct / total), file=file)
        print ('Clean Test Accuracy : %.2f' % (100.0 * correct / total))
        print ('Clean Test Accuracy : %.2f' % (100.0 * correct / total), file = file)

        total = 0.0
        correct = 0.0


        for loader in poi_test_loader:
            for images, labels in loader:
                images = Variable(images).cuda()
                preds = model.classify(images)
                preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
                total += labels.size(0)
                correct += (preds == labels.numpy()).sum()

        # Poison Test Accuracy
        print ('Poison Test Accuracy : %.2f' % (100.0 * correct / total))
        print ('Poison Test Accuracy : %.2f' % (100.0 * correct / total), file=file)
        
        
        model.train()
        t2 = time.time()
        print (f'Time : {t2 - t1}', file=file)
        torch.save(model, f".\\Models\\Joint\\Joint.pth")
    return model

            






