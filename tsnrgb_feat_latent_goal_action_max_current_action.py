#!/usr/bin/env python
# coding: utf-8

# In[1]:

import time, os, copy, numpy as np

import torch, torchvision
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn import Parameter, init
import torch.nn.functional as F
import sys
import pickle
from queue import PriorityQueue
import heapq as hq 

import lmdb
import operator

from torch.utils.tensorboard import SummaryWriter

from torchsummary import summary

#torch.backends.cudnn.enabled = False
#torch.backends.cudnn.benchmark = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


from collections import defaultdict
# ### Debug features class

import pickle

from collections import defaultdict
import random

import pandas as pd

class TSN_EPICval(Dataset):
    def __init__(self, ann_file, obs_seg, seg_length, lmdb_path, transform=None):
        self.tsn_feat_list = []
        self.verbs_list = []
        self.nouns_list = []
        self.actions_list = []
        count_empty_segs = 0
        
        action_annotation = pd.read_csv(ann_file, header=None)
        # [VID, Obs_start, Obs_end, Obs_noun, Obs_verb, Fut_start, Fut_end, Fut_noun,Fut_verb]
        
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        
        video_ids = list(action_annotation[1].unique())
        
        for video_id in video_ids:
            video_id = video_id.strip('\n')
            starts = list(action_annotation.loc[action_annotation[1] == video_id][2].values)[:-1]
            stops = list(action_annotation.loc[action_annotation[1] == video_id][3].values)[:-1]
            verbs = list(action_annotation.loc[action_annotation[1] == video_id][4].values)[1:]
            nouns = list(action_annotation.loc[action_annotation[1] == video_id][5].values)[1:]
            actions = zip(list(action_annotation.loc[action_annotation[1] == video_id][6].values)[:-1],
                          list(action_annotation.loc[action_annotation[1] == video_id][6].values)[1:])
            #print(stops[-1])
            
            # print(tsn_feat.shape)
            video_id = video_id.split()[0]
            with env.begin() as tsn_feats:    
                verbs_in_video = []
                nouns_in_video = []
                actions_in_video = []
                tsn_feat_in_video = []
                for start, stop, verb, noun, action in zip(starts, stops, verbs, nouns, actions):
                    feat_frames = []
                    tsn_feat_in_seg = []
                    # stop = stop - 30
                    # if stop - start > seg_length:
                    for i in range(int(stop)-seg_length,int(stop)):
                        # 'P24_03_frame_0000000578.jpg'
                        frame_num = video_id+'_frame_'+str(i).zfill(10)+'.jpg'
                        ff = tsn_feats.get(frame_num.encode('utf-8'))
                        if ff is not None:
                            tsn_feat = np.frombuffer(ff, 'float32')
                        else:
                            continue
                        tsn_feat_in_seg.append(tsn_feat.copy())
                    # else:
                        # continue
                    #print(len(tsn_feat_in_seg))    
                    if len(tsn_feat_in_seg) != 0:
                        tsn_feat_in_video.append(tsn_feat_in_seg)
                        verbs_in_video.append(verb)
                        nouns_in_video.append(noun)
                        actions_in_video.append(action)
                    else:
                        count_empty_segs += 1
                if len(tsn_feat_in_video) != 0:
                    self.tsn_feat_list.extend(tsn_feat_in_video)
                    self.verbs_list.extend(verbs_in_video)
                    self.nouns_list.extend(nouns_in_video) 
                    self.actions_list.extend(actions_in_video)
                else:
                    print(video_id)
        print(count_empty_segs)         
        
    def __getitem__(self, index):

        tsn_feat_seq = self.tsn_feat_list[index]
        actions = self.actions_list[index]
        verb = self.verbs_list[index]
        noun = self.nouns_list[index]
        
        return tsn_feat_seq, actions, verb, noun

    def __len__(self):
        return len(self.actions_list)



# class definition 
class BeamSearchNode(object):
    def __init__(self, action_state, feat_state, goal_state, hidden, score, prev_node): 
        self.action_state = action_state
        self.feat_state = feat_state 
        self.goal_state = goal_state
        self.hidden = hidden
        self.prev_node = prev_node
        self.score = score

    def __lt__(self, nxt): 
        return self.score < nxt.score
        
    def __gt__(self, nxt): 
        return self.score > nxt.score
                
    
class Anticipator(nn.Module):
    def __init__(self, feature_dim, goal_smoothness, goal_closeness, hidden_size):
        super(Anticipator, self).__init__()
        
        self.hidden_size = hidden_size
        #self.goalpredictor = GoalPredictor()
        self.feat_embedding = nn.Linear(feature_dim, self.hidden_size)
        self.goal_steps = 5
        self.goalpredictor = nn.LSTM(2*self.hidden_size, self.hidden_size, self.goal_steps)
        self.epsilon = goal_closeness
        self.rnn = nn.LSTM(2*self.hidden_size, self.hidden_size, 1)
        self.predictor = nn.Linear(3*self.hidden_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.delta = goal_smoothness
        self.embedding2action = nn.Linear(self.hidden_size, 2513)
        self.embedding2verb = nn.Linear(self.hidden_size, 125)
        self.embedding2noun = nn.Linear(self.hidden_size, 352)
        self.embedding2feature = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(p=0.7)
        
        self.softmax = nn.Softmax(dim=1)
        
    def featpredictor(self, input1, input2, input3):
        #print(input1.shape)
        pred_feat = torch.cat((input1, input2),1)
        # print(input3.shape)
        # print(pred_feat.shape)
        pred_feat = self.predictor(torch.cat((pred_feat, input3),1))
        pred_feat = self.dropout(pred_feat)
        pred_feat = self.relu(pred_feat)
        
        return pred_feat
    
    def heapsort(iterable):
        h = []
        for value in iterable:
            hq.heappush(h, value)
        return [hq.heappop(h) for i in range(len(h))]
    
    def target_feat_embedding(self, target_feat):
        # print(target_feat.shape)
        feat_state, _ = torch.max(target_feat,1)
        #feat_state = feat_state[1,:,:]
        #feat_state = feat_state.squeeze(0)
        return feat_state
    
    def forward(self, tsn_feat_seq, batch_size=None):
        
        
        if len(tsn_feat_seq.shape) == 1:
            tsn_feat_seq = tsn_feat_seq.unsqueeze(0)
        #print(tsn_feat_seq.shape)
        feat_state, _ = torch.max(tsn_feat_seq,0)
        
        if len(feat_state.shape) == 1:
            feat_state = feat_state.unsqueeze(0)
        #print('feat_state',feat_state.shape)
        feat_state = self.feat_embedding(feat_state)

        cur_action = self.embedding2action(feat_state)
        
        obs_acts = feat_state.shape[0]+1
        # print(torch.isnan(feat_states).any())
        #print(obs_acts)
        
        #print('feat_state',feat_state.shape)
        
        best_feat_state = feat_state
        
        nodes = PriorityQueue()
        
        end_nodes = []
        
        action_state = torch.zeros(1, self.hidden_size).to(device)
        score = 1.0
        
        #print(torch.cat((feat_state, action_state),2).shape)
        # print(best_action_state.shape)
        # print(best_feat_state.shape)
        hidden = torch.zeros(self.goal_steps, 1, self.hidden_size).to(device)
        cell = torch.zeros(self.goal_steps, 1, self.hidden_size).to(device)
        # print(feat_state.shape)
        # print(action_state.shape)
        
        goal_state, (hidden, _) = self.goalpredictor(torch.cat((feat_state.unsqueeze(0), action_state.unsqueeze(0)), 2), (hidden, cell))
        goal_state = goal_state.squeeze(1)
        node = BeamSearchNode(action_state, feat_state, goal_state, hidden, score, None)
        nodes.put(node)
        qsize = 1
        endnodes = []
        while True:
            #print(best_feat_state.shape)
            # print(best_action_state.shape)
            if qsize > obs_acts*6:
                break
            if nodes.qsize() != 0:    
                n = nodes.get()
            else:
                break
            feat_state = n.feat_state
           
            action_state = n.action_state
            goal_state = n.goal_state
            hidden = n.hidden
            #print('feat_state',feat_state)
            #print('dist',torch.square(torch.dist(feat_state, goal_state,2))/self.hidden_size)
            if torch.square(torch.dist(feat_state, goal_state,2))/self.hidden_size > self.epsilon:
                h_act = torch.zeros(1, 1, self.hidden_size).to(device)
                c_act = torch.zeros(1, 1, self.hidden_size).to(device)

                #print(len(end_nodes))
                #print(number_required)
                endnodes.append(n)
                if len(end_nodes) >= obs_acts:
                    break
                
                action_state_list = []
                for j in range(10): 
                    _, (h_act, c_act) = self.rnn(torch.cat((action_state.unsqueeze(0), feat_state.unsqueeze(0)), 2), (h_act, c_act))
                    #print(j)
                    action_state_list.append(h_act.squeeze(0))   
                next_nodes = []
                
                # print(len(action_state_list))
                for action_state in action_state_list:
                    next_feat = self.featpredictor(feat_state, action_state, goal_state)    
                    #next_feat = feat_states[i,:].unsqueeze(0)
                    #next_hidden, next_goal_state = self.goalpredictor(feat_state, action_state, hidden)
                    # next_goal_state, (next_hidden, _) = \
                    #         self.goalpredictor(torch.cat((feat_state.unsqueeze(0), action_state.unsqueeze(0)), 2), (hidden, cell))
                    # next_goal_state = self.relu(next_goal_state.squeeze(1))
                    score = torch.square(torch.dist(best_feat_state, goal_state, 2))/self.hidden_size \
                                - torch.square(torch.dist(next_feat, goal_state, 2))/self.hidden_size 
                                #+ torch.square(torch.dist(next_goal_state, goal_state,2))/self.hidden_size
                    next_node = BeamSearchNode(action_state, next_feat, goal_state, hidden, score, n)
                    next_nodes.append(next_node)
                    if torch.square(torch.dist(next_feat, goal_state,2))/self.hidden_size < torch.square(torch.dist(best_feat_state, goal_state, 2))/self.hidden_size:\
                        #torch.square(torch.dist(next_goal_state, goal_state,2))/self.hidden_size < self.delta:
                        best_feat_state = next_feat
                
                for k in range(len(next_nodes)):
                    node = next_nodes[k]
                    nodes.put(node)
                qsize += len(next_nodes) - 1
        #print('num nodes',nodes.qsize())
        #print(len(end_nodes))
        if len(end_nodes) == 0:
            if nodes.qsize() != 0:
                end_nodes = [nodes.get() for _ in range(obs_acts)]
        #print('num end_nodes', len(end_nodes))
        if nodes.qsize() == 0:
            print(len(end_nodes))
            print('empty')
        pred_action_states = []
        pred_feat_states = []
        pred_goal_states = []
        #print('before sorted')
        # print(obs_acts)
        l = 0
        for n in sorted(end_nodes, key=operator.attrgetter('score')):
            # print(n.score)
            l += 1
            pred_action_state = []
            pred_feat_state = []
            pred_goal_state = []
            pred_action_state.append(n.action_state)
            pred_feat_state.append(n.feat_state)
            pred_goal_state.append(n.goal_state)
            # back trace
            while n.prev_node != None:
                #print(l)
                n = n.prev_node
                pred_action_state.append(n.action_state)
                pred_feat_state.append(n.feat_state)
                pred_goal_state.append(n.goal_state)
                
            pred_action_state = pred_action_state[::-1]
            pred_feat_state = pred_feat_state[::-1]
            pred_goal_state = pred_goal_state[::-1]
            pred_action_states.append(pred_action_state)
            pred_feat_states.append(pred_feat_state)
            pred_goal_states.append(pred_goal_state)
                
        
        # for pred_feat_state in pred_feat_states:
            # for pred_feat in pred_feat_state:
                # print(pred_feat.shape)
        pred_action_states = [ torch.stack(state).squeeze(1) for state in pred_action_states]
        pred_action_states = [ torch.mean(state, 0, keepdim=True) for state in pred_action_states]
        pred_action_states = torch.stack(pred_action_states).squeeze(1)
        # print(pred_action_states.shape)
        pred_action_states = self.dropout(pred_action_states)
        pred_actions = self.embedding2action(pred_action_states)
        pred_verbs = self.embedding2verb(pred_action_states)
        pred_nouns = self.embedding2noun(pred_action_states)
        #pred_actions = self.relu()
        
        # print(pred_actions.shape)
        pred_feat_states = [ torch.stack(state).squeeze(1) for state in pred_feat_states]
        pred_feat_states = [ torch.mean(state, 0, keepdim=True) for state in pred_feat_states]
        pred_feat_states = torch.stack(pred_feat_states).squeeze(1)
        # print(pred_feat_states.shape)
        pred_goal_states = [ torch.stack(state).squeeze(1) for state in pred_goal_states]
        pred_goal_states = [ torch.mean(state, 0, keepdim=True) for state in pred_goal_states]
        pred_goal_states = torch.stack(pred_goal_states).squeeze(1)
        # print(pred_goal_states.shape)
        return pred_actions, pred_verbs, pred_nouns, pred_feat_states,\
            pred_goal_states, feat_state, cur_action


class TrainTest():
    
    def __init__(self, model, trainset, testset, batch_size, nepoch, ckpt_path, goal_smoothness, goal_closeness, writer):
        self.model = model
        self.optimizer = optim.AdamW(model.parameters())
        #self.optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001)
        self.celoss = nn.CrossEntropyLoss()
        self.mseloss = nn.MSELoss()
        self.mmloss = nn.MarginRankingLoss()
        self.l1_crit = nn.L1Loss()
        self.goal_smoothness = goal_smoothness
        self.goal_closeness = goal_closeness
        self.hidden_size = 1024
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=True, num_workers=0)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                          shuffle=False, num_workers=0)
          
        self.model.to(device)    
        self.chkpath = ckpt_path
        self.batch_size = batch_size
        self.nepoch = nepoch
        self.trainset_size = len(trainset)
        self.writer = writer
        if not os.path.exists('ckpt/'):
            os.mkdir('ckpt/')            
        print(self.chkpath)
        if os.path.exists(self.chkpath) == True:
            print('load from ckpt', end=' ')
            self.state = torch.load(self.chkpath)
            self.model.load_state_dict(self.state['model'])
            best_acc = self.state['acc']
            start_epoch = self.state['epoch']
            print('Epoch {}'.format(start_epoch))
            if start_epoch == self.nepoch:
                print('existing as epoch is max.')
            self.details = self.state['details']    
            self.best_acc = best_acc
            self.start_epoch = start_epoch + 1
            self.model.to(device)                    
        else:
            self.best_acc = -1.
            self.details = []   
            self.start_epoch = 0
    
    def test(self):
        running_loss = 0.0
        correct_action = 0
        correct_verb = 0
        correct_noun = 0
        count = 0
        iterations = 0
        sequence = []
        with torch.no_grad():
            for i, data in enumerate(self.testloader, 0): 
                loss = 0.
                tsn_feat_seq = []
                obs_label_seq = []
                tsn_seq_segs, target_actions, target_verb, target_noun = data 
                for tsn_seq in tsn_seq_segs: 
                    #print(tsn_seq.shape)
                    tsn_feat_seq.append(tsn_seq)
                tsn_feat_seq = torch.stack(tsn_feat_seq)
                
                # tsn_feat_seq = (tsn_feat_seq - mean_values)/(std_values+1e-9)
                tsn_feat_seq = tsn_feat_seq.float().squeeze(0).to(device)
                # print(tsn_feat_seq.shape)
                if tsn_feat_seq.shape[1] == 0:
                    continue
                cur_action = torch.LongTensor([int(target_actions[0])]).to(device)
                next_action = torch.LongTensor([int(target_actions[1])]).to(device)
                target_verb = torch.LongTensor([int(target_verb)]).to(device)
                target_noun = torch.LongTensor([int(target_noun)]).to(device)
                # print(target_actions.shape)

                # print(action_label_tensor.is_cuda)
                pred_actions, pred_verbs, pred_nouns, pred_feats,\
                    goal_states, obs_feat_states, pred_cur_action = self.model(tsn_feat_seq)
                # print(pred_actions)
                # target_feats_new = torch.cat((obs_feat_states[1:,:], target_feat),0)
                # print(target_feats_new.shape)
                # print(pred_feats.shape)
                
                pred_next_action = pred_actions[-1,:].view(1,-1)
                pred_verb = pred_verbs[-1,:].view(1,-1)
                pred_noun = pred_nouns[-1,:].view(1,-1)
                # print(pred_action.shape)
                
                # if target_actions.shape[0] != pred_actions.shape[0]:
                    # continue
                loss = self.celoss(pred_cur_action, cur_action)
                loss += self.celoss(pred_next_action, next_action)
                loss += self.celoss(pred_verb, target_verb)
                loss += self.celoss(pred_noun, target_noun)
                
                ant_action = torch.argmax(pred_next_action,1)
                ant_verb = torch.argmax(pred_verb,1)
                ant_noun = torch.argmax(pred_noun,1)
                #print(ant_actions)
                correct_action = correct_action + torch.sum(ant_action == next_action).item() 
                correct_verb = correct_verb + torch.sum(ant_verb == target_verb).item()
                correct_noun = correct_noun + torch.sum(ant_noun == target_noun).item()
                
                #loss = loss/len(obs_label_seq)
                print("\rIteration: {}/{}, Loss: {}.".format(i+1, len(self.testloader), loss), end="")
                #sys.stdout.flush()
                count += 1
                running_loss = running_loss + loss.item()
                iterations += 1
                #print(count)    
        TEST_LOSS = running_loss/iterations
        return [correct_action/count*100., correct_verb/count*100., correct_noun/count*100.],\
               [correct_action, correct_verb, correct_noun], TEST_LOSS 
        
    def train(self):        
        for epoch in range(self.start_epoch,self.nepoch):  
            start_time = time.time()        
            running_loss = 0.0
            correct_action = 0
            correct_verb = 0
            correct_noun = 0
            count = 0
            total_loss = 0
            self.optimizer.zero_grad()   
            loss = 0.
            iterations = 0
            count_unequal = 0
            ce_per_class = torch.zeros(48).to(device)
            freq = torch.zeros(48).to(device)
            batch_pred_actions = []
            batch_target_actions = []
            batch_y2 = []
            batch_y4 = []
            batch_pred_feats = []
            batch_target_feats = []
            for i, data in enumerate(self.trainloader, 0):        
                #print(len(data))
                tsn_feat_seq = []
                obs_label_seq = []
                tsn_seq_segs, target_actions, target_verb, target_noun = data 
                for tsn_seq in tsn_seq_segs: 
                    #print(tsn_seq.shape)
                    tsn_feat_seq.append(tsn_seq)
                tsn_feat_seq = torch.stack(tsn_feat_seq)
                
                # tsn_feat_seq = (tsn_feat_seq - mean_values)/(std_values+1e-9)
                tsn_feat_seq = tsn_feat_seq.float().squeeze(0).to(device)
                # print(tsn_feat_seq.shape)
                if tsn_feat_seq.shape[1] == 0:
                    continue
                cur_action = torch.LongTensor([int(target_actions[0])]).to(device)
                next_action = torch.LongTensor([int(target_actions[1])]).to(device)
                target_verb = torch.LongTensor([int(target_verb)]).to(device)
                target_noun = torch.LongTensor([int(target_noun)]).to(device)
                # print(target_actions.shape)

                # print(action_label_tensor.is_cuda)
                pred_actions, pred_verbs, pred_nouns, pred_feats,\
                    goal_states, obs_feat_states, pred_cur_action = self.model(tsn_feat_seq)
                # print(pred_actions)
                # target_feats_new = torch.cat((obs_feat_states[1:,:], target_feat),0)
                # print(target_feats_new.shape)
                # print(pred_feats.shape)
                
                pred_next_action = pred_actions[-1,:].view(1,-1)
                pred_verb = pred_verbs[-1,:].view(1,-1)
                pred_noun = pred_nouns[-1,:].view(1,-1)
                # print(pred_action.shape)
                
                # if target_actions.shape[0] != pred_actions.shape[0]:
                    # continue
               
                # print(pred_actions.shape)
                #print(target_action.shape)
                # if target_actions.shape[0] != pred_actions.shape[0]:
                    # count_unequal += 1
                    # continue
                # ce_per_class[target_action.item()] += self.celoss(pred_action, target_action)
                # freq[target_action.item()] += 1
                # batch_pred_actions.append(pred_action)
                # batch_target_actions.append(target_action)
                loss = self.celoss(pred_cur_action, cur_action)
                loss += self.celoss(pred_next_action, next_action)
                loss += self.celoss(pred_verb, target_verb)
                loss += self.celoss(pred_noun, target_noun)
                # reg_loss = 0.
                # for param in self.model.parameters():
                    # reg_loss += torch.sum(abs(param))

                # loss += 0.0001*reg_loss
                #print(loss)
                
                # batch_pred_feats.append(pred_feats)
                # target_feats_new = torch.cat((obs_feat_states[1:,:], target_feat),0)
                # batch_target_feats.append(target_feats_new)
                # print(target_feats_new.shape)
                # print(pred_feats.shape)
                #loss += self.mseloss(pred_feats, target_feats_new)
                
                # loss.backward(retain_graph=True)
                
                # Max-Margin Loss
                t = torch.ones(pred_feats.shape[0]-1).to(device)
                y1 = torch.FloatTensor([self.goal_closeness]).repeat(goal_states.shape[0]-1).to(device)
                y2 = torch.ones(pred_feats.shape[0]-1).to(device)
                
                
                for k in range(pred_feats.shape[0]-1):
                    y2[k] = torch.square(torch.dist(pred_feats[k,:], goal_states[k,:], 2))/self.hidden_size \
                    - torch.square(torch.dist(pred_feats[k+1,:], goal_states[k,:], 2))/self.hidden_size
                loss += self.mmloss(y1, y2, t)
                # batch_y2.append(y2)
                #print(y1)
                #print(y2)
                # print(loss)
                # loss.backward(retain_graph=True)
                t = torch.ones(goal_states.shape[0]-1).to(device)
                y3 = torch.FloatTensor([self.goal_smoothness]).repeat(goal_states.shape[0]-1).to(device)
                y4 = torch.ones(goal_states.shape[0]-1).to(device)

                for j in range(goal_states.shape[0]-1):
                    y4[j] = torch.square(torch.dist(goal_states[j,:], goal_states[j+1,:], 2))/self.hidden_size
                # print(y4)
                batch_y4.append(y4)
                loss += self.mmloss(y3, y4, t)
                
                #loss.backward(retain_graph=True)
                #print(loss)   
                # print(pred_seq.shape)
                loss.backward()
                ant_action = torch.argmax(pred_next_action,1)
                ant_verb = torch.argmax(pred_verb,1)
                ant_noun = torch.argmax(pred_noun,1)
                #print(ant_actions)
                with torch.no_grad():
                    correct_action = correct_action + torch.sum(ant_action == next_action).item() 
                    correct_verb = correct_verb + torch.sum(ant_verb == target_verb).item()
                    correct_noun = correct_noun + torch.sum(ant_noun == target_noun).item()                  
                  
                if i % (batch_size-1) == 0 and i>1:
                    #loss = loss/(batch_size*len(obs_label_seq))
                    #norm_inv_freq = 1/(freq/batch_size)
                    #ce_per_class = ce_per_class/norm_inv_freq
                    #print(ce_per_class)
                    #loss = torch.sum(ce_per_class)/48
                    '''
                    preds = torch.stack(batch_pred_actions).squeeze(1).view(-1,48)
                    #print(preds.shape)
                    targets = torch.stack(batch_target_actions).squeeze(1).view(-1,)
                    #print(targets.shape)
                    loss = self.celoss(preds, targets)
                    
                    # batch_pred_feats = torch.stack(batch_pred_feats)
                    # batch_target_feats = torch.stack(batch_target_feats)
                    
                    # loss += self.mseloss(batch_pred_feats, batch_target_feats)
                    #print(loss)
                    ### goal closness loss
                    batch_y2 = torch.stack(batch_y2,0).view(-1,)
                    y1 = torch.FloatTensor([self.goal_closeness]).repeat(batch_y2.shape[0]).to(device)
                    t1 = torch.ones(batch_y2.shape[0]).to(device)
                    loss += self.mmloss(y1, batch_y2, t1)
                    
                    ## goal smoothness loss
                    batch_y4 = torch.stack(batch_y4,0).view(-1,)
                    #print(batch_y4.shape)
                    y3 = torch.FloatTensor([self.goal_smoothness]).repeat(batch_y4.shape[0]).to(device)
                    t2 = torch.ones(batch_y4.shape[0]).to(device)
                    loss += self.mmloss(y3, batch_y4, t2)
                    
                    # reg_loss = 0
                    # for param in self.model.parameters():
                      # reg_loss += self.l1_crit(param,target=torch.zeros_like(param))
                      
                    # factor = 0.01
                    # loss += factor * reg_loss
                    
                    loss.backward()
                    
                    with torch.no_grad():                
                        self.writer.add_scalar('celoss/train', self.celoss(preds, targets).item(), (epoch+1)*iterations)
                        # self.writer.add_scalar('mseloss/train', self.mseloss(batch_pred_feats, batch_target_feats).item(), (epoch+1)*iterations)
                        self.writer.add_scalar('goal_closeness/train', self.mmloss(y1, batch_y2, t1).item(), (epoch+1)*iterations)
                        self.writer.add_scalar('goal_smoothness/train', self.mmloss(y3, batch_y4, t2).item(), (epoch+1)*iterations)
                    '''
                    self.optimizer.step()
                    self.optimizer.zero_grad()                    
                    running_loss = running_loss + loss.item()
                    print("\rIteration: {}/{}, Loss: {}.".format(i+1, len(self.trainloader), loss), end="")
                    loss = 0.
                    count += self.batch_size
                    iterations += 1
                    batch_pred_actions = []
                    batch_target_actions = []
                    batch_y2 = []
                    batch_y4 = []
                    batch_pred_feats = []
                    batch_target_feats = []
                    
            # print('count_unequal',count_unequal)
            TRAIN_LOSS = running_loss/iterations
            TRAIN_ACC = [correct_action/count*100, correct_verb/count*100, correct_noun/count*100]
            TEST_ACC, TEST_COUNT, TEST_LOSS = self.test()
            self.details.append((TRAIN_LOSS,TRAIN_ACC,0.,TEST_ACC))
            

            if TEST_ACC[0] > self.best_acc:                
                self.state = {
                    'model': self.model.state_dict(),
                    'acc': TEST_ACC[0],
                    'epoch': epoch,
                    'details':self.details,            
                }        
                torch.save(self.state, self.chkpath)
                self.best_acc = TEST_ACC[0]
            else:
                self.state['epoch'] = epoch
                torch.save(self.state, self.chkpath)
            elapsed_time = time.time() - start_time
            print('[{}] [{:.1f}] [Loss {:.3f}] [A {:.2f}] [V {:.2f}] [N {:.2f}]'.format(epoch, elapsed_time,
                    TRAIN_LOSS, TRAIN_ACC[0], TRAIN_ACC[1], TRAIN_ACC[2] ),end=" ")
            print('[A {:.2f}] [V {:.2f}] [N {:.2f}]'.format(TEST_ACC[0], TEST_ACC[1], TEST_ACC[2]))
# In[17]:


# ### define hyperparameters

goal_closeness = 1e-5
goal_smoothness = 1e-5
#instantiate the model
feature_dim = 1024
hidden_size = 1024
obs_segs = [1, 2, 3, 4]
seg_length_secs = [0.5, 1, 2, 3, 5, 10, 15, 20 ,25]

obs_seg = obs_segs[0]
seg_length_sec = seg_length_secs[3]
frame_rate = 30 # tsn features calculated @ 10fps
    
writer = SummaryWriter('runs/epic_action_max_{:1.1f}x{:d}'.format(seg_length_sec, obs_seg))
nepochs = 20

anticipation_model = VerbAnticipator(feature_dim, goal_smoothness, goal_closeness, hidden_size)
ckpt_path = 'ckpt/tsnrgb_action_latent_goal_{:1.1f}sx{:d}_obs_max_current_action.pt'.format(seg_length_sec, obs_seg)
# ckpt_path = 'ckpt/tsnrgb_verb_latent_goal_{:1.1f}sx{:d}_obs_max_hidden{:d}.pt'.format(seg_length_sec, obs_seg, hidden_size)

goal_closeness = 1e-5
goal_smoothness = 1e-5
criterion = nn.CrossEntropyLoss()

batch_size = 256

lmdb_path = 'data/ek55/rgb'

train_ann_file = 'training.csv'
test_ann_file = 'validation.csv'
test_set = TSN_EPICval(test_ann_file, obs_seg, int(seg_length_sec*frame_rate), lmdb_path)
print('{} test instances.'.format(len(test_set)))
training_set = TSN_EPICval(train_ann_file, obs_seg, int(seg_length_sec*frame_rate), lmdb_path)
print('{} train instances.'.format(len(training_set)))

EXEC = TrainTest(anticipation_model, training_set, test_set, batch_size, nepochs, ckpt_path, goal_smoothness, goal_closeness, writer)
EXEC.train()
