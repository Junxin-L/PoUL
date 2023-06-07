from torch.autograd import Variable
import torch.nn as nn
import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
import torchvision.models as models
import copy
from tqdm import tqdm
import torch.optim as optim
import numpy as np
import random
from Param import *

def MultiClassCrossEntropy(logits, labels, T):
	labels = Variable(labels.data, requires_grad=False).cuda()
	outputs = torch.log_softmax(logits/T, dim=1)   # compute the log of softmax values
	labels = torch.softmax(labels/T, dim=1)
	outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
	outputs = -torch.mean(outputs, dim=0, keepdim=False)
	return Variable(outputs.data, requires_grad=True).cuda()

def kaiming_normal_init(m):
	if isinstance(m, nn.Conv2d):
		nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
	elif isinstance(m, nn.Linear):
		nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')


# 如果是非增量学习，调用一次即可
class Model(nn.Module):
	def __init__(self, classes, classes_map, class_num, num_samples_poi, init_lr, epoch, batch_size):
		# Hyper Parameters
		self.init_lr = init_lr
		self.num_epochs = epoch
		self.batch_size = batch_size
		self.lower_rate_epoch = [int(0.7 * self.num_epochs), int(0.9 * self.num_epochs)] #hardcoded decay schedule
		self.lr_dec_factor = 10
		
		self.pretrained = False
		self.momentum = 0.9
		self.weight_decay = 0.0001
		# Constant to provide numerical stability while normalizing
		self.epsilon = 1e-16
		self.class_num = class_num
		self.num_samples_poi = num_samples_poi

		# Network architecture
		super(Model, self).__init__()
		self.model = models.resnet34(pretrained=self.pretrained)
		self.model.conv1 = nn.Conv2d(channel_count, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.model.apply(kaiming_normal_init)

		num_features = self.model.fc.in_features
		self.model.fc = nn.Linear(num_features, classes, bias=False)
		self.fc = self.model.fc
		self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
		self.feature_extractor = nn.DataParallel(self.feature_extractor) 


		# n_classes is incremented before processing new data in an iteration
		# n_known is set to n_classes after all data for an iteration has been processed
		self.n_classes = 0
		self.n_known = 0
		self.classes_map = classes_map

	def forward(self, x):
		x = self.feature_extractor(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x

	def increment_classes(self, new_classes):
		"""Add n classes in the final fc layer"""
		n = len(new_classes)
		print('new classes: ', n)
		in_features = self.fc.in_features
		out_features = self.fc.out_features
		weight = self.fc.weight.data

		if self.n_known == 0:
			new_out_features = n
		else:
			new_out_features = out_features + n
		print('new out features: ', new_out_features)
		self.model.fc = nn.Linear(in_features, new_out_features, bias=False)
		self.fc = self.model.fc
		
		kaiming_normal_init(self.fc.weight)
		self.fc.weight.data[:out_features] = weight
		self.n_classes += n

	def classify(self, images):
		"""Classify images by softmax

		Args:
			x: input image batch
		Returns:
			preds: Tensor of size (batch_size,)
		"""
		_, preds = torch.max(torch.softmax(self.forward(images), dim=1), dim=1, keepdim=False)

		return preds

	def update(self, s, input_dataset, class_map, clean_data, pre_poi_data, extract):
		
		with open("logfile", 'a') as logfile:
			dataset = copy.copy(input_dataset)
			if(extract): #是否抽取
				#从clean data里抽出一点来学习
				for client in range(clientNum):
					# 遍历每个类别
					clean_data_client = clean_data[client]
					for c in range(self.class_num):
						# 找出 clean_data 中所有类别为 c 的样本
						class_indices = [i for i in range(len(clean_data_client)) if clean_data_client[i][1] == c]
						if(len(class_indices) > 0):
							# 随机选择 num_samples 个样本
							selected_indices = np.random.choice(class_indices, size=num_samples(s), replace=False)
							# 将这些样本添加到 poi_data 中
							for index in selected_indices:
								dataset.append(clean_data_client[index])

				for data in pre_poi_data:
					class_indices = [i for i in range(len(data))]
					selected_indices = np.random.choice(class_indices, size=self.num_samples_poi, replace=False)
					for index in selected_indices:
						dataset.append(data[index])

			if(len(dataset) % self.batch_size == 1): # 防止batch size为1的情况出现
				dataset.pop(0)
			loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
												shuffle=True, num_workers=0)
			
			self.compute_means = True

			# Save a copy to compute distillation outputs
			prev_model = copy.deepcopy(self)
			prev_model.cuda()

			#classes = list(set(dataset.classes))
			print('Known: ', self.n_known, logfile)
	
			#这里是为了数据集修改的，默认第一次增加了全部的类别，之后不会再有新的类别。
			if self.n_classes == 0:
				new_classes = [i for i in range(self.class_num)]
			else:
				new_classes = []
		
			if len(new_classes) > 0:
				self.increment_classes(new_classes)
				self.cuda()

			print("Batch Size (for n_classes classes) : ", len(dataset), logfile)
			optimizer = optim.SGD(self.parameters(), lr=self.init_lr, momentum = self.momentum, weight_decay=self.weight_decay)
			with tqdm(total=self.num_epochs) as pbar:
				for epoch in range(self.num_epochs):
					e_loader = enumerate(loader) # 必须在每个epoch时更新一次，因为它是迭代器类型
					for i, (images, labels) in e_loader:
						seen_labels = []
						images = Variable(torch.FloatTensor(images)).cuda()
						seen_labels = torch.LongTensor([class_map[label] for label in labels.numpy()])
						labels = Variable(seen_labels).cuda()
						optimizer.zero_grad()
						logits = self.forward(images)
						cls_loss = nn.CrossEntropyLoss()(logits, labels)
						if len(new_classes)!=0 and self.n_classes//len(new_classes) > 1:
							dist_target = prev_model.forward(images)
							logits_dist = logits[:,:-(self.n_classes-self.n_known)]
							dist_loss = MultiClassCrossEntropy(logits_dist, dist_target, 2)
							loss = dist_loss+cls_loss
						else:
							loss = cls_loss

						loss.backward()
						optimizer.step()

						if (i+1) % 1 == 0:
							tqdm.write('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
								%(epoch+1, self.num_epochs, i+1, np.ceil(len(dataset)/self.batch_size), loss.data), logfile)
						if np.ceil(len(dataset)/self.batch_size) % 50 == 0 and epoch % 5 == 0:	
							print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
								%(epoch+1, self.num_epochs, i+1, np.ceil(len(dataset)/self.batch_size), loss.data), logfile)
					pbar.update(1)
