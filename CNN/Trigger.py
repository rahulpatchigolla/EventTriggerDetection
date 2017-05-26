from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import numpy as np
import random
import re
import collections
from nltk.tokenize import WordPunctTokenizer
from gensim.models import Word2Vec
tokenizer = WordPunctTokenizer()
import pickle
import sys

sys.path.append("source")
# Load pretrained word embeddings
def loadWordEmbeddings(wordVocab,embSize):
    print "Loading Word Embeddings..."
    print "Total Words:", len(wordVocab)
    model = Word2Vec.load_word2vec_format("/home/rahul/PycharmProjects/PubMed-w2v.bin", binary=True)
    wordemb = []
    count = 0
    for word in wordVocab:
        if model.__contains__(word):
            wordemb.append(model[word])
        else:
            count += 1
            #print word
            wordemb.append(np.random.rand(embSize))
            #	print (wordemb)
    # wordemb = np.asarray(map(float, wordemb))
    wordemb[wordVocab["unk"]] = np.zeros(embSize)
    wordemb = np.asarray(wordemb,dtype='float32')
    print ("number of unknown word in word embedding", count)
    return wordemb

# reading the data into an appropriate format
def dataRead(path):
    fp=open(path,"r")
    labels=[]
    words=[]
    count=0
    for line in fp:
        count+=1
        line=line.strip("\n").split("\t")
        word=line[0].strip(" ").split(" ")
        label=line[1]
        #print word,entity,label
        '''if count<10:
            print line
            print word
            print label'''
        words.append(word)
        labels.append(label)
    return words,labels

#finding sentence lengths
def findSentLengths(tr_te_list):
    lis = []
    for lists in tr_te_list:
        lis.append([len(l) for l in lists])
    return lis

# creating vocabulary
def makeWordList(sent_lista, sent_listb):
    sent_list = sent_lista + sent_listb
    wf = {}
    for sent in sent_list:
        for w in sent:
            if w in wf:
                wf[w] += 1
            else:
                wf[w] = 0

    wl = collections.OrderedDict()  # orederd dictionary
    i = 1
    wl['unkown'] = 0
    for w, f in wf.items():
        wl[w] = i
        i += 1
    return wl
    
#map words to the vocabulary ids
def mapWordToId(sent_contents, word_dict):
    T = []
    for sent in sent_contents:
        t = []
        for w in sent:
            t.append(word_dict[w])
        T.append(t)
    return T

# map class labels to label ids
def mapLabelToId(sent_labels, label_dict):
    #	print"sent_lables", sent_lables
    #	print"label_dict", label_dict
    # return [label_dict[label] for label in sent_lables]
    rval = []
    for label in sent_labels:
        if label_dict.has_key(label):
            rval.append(label_dict[label])
        else:
            label1 = label[1:]
            label2 = label[:len(label) - 1]
            if label_dict.has_key(label1):
                rval.append(label_dict[label1])
            elif label_dict.has_key(label2):
                rval.append(label_dict[label2])
            else:
                print "Not found label"
                exit(0)
    return rval

# padding the input data
def paddData(listL, maxl):  # W_batch, d1_tatch, d2_batch, t_batch)
    rlist = []
    for mat in listL:
        mat_n = []
        for row in mat:
            lenth = len(row)
            t = []
            for i in range(lenth):
                t.append(row[i])
            for i in range(lenth, maxl):
                t.append(0)
            mat_n.append(t)
        rlist.append(np.array(mat_n))
    return rlist

# Test the result using the trained model at every epoch
def test_step(W_te,Y_te):
    n = len(W_te)
    num = int(n/batch_size) + 1
    sample = []
    for batch_num in range(num):
        start_index = batch_num*batch_size
        end_index = min((batch_num + 1) * batch_size, n)
        sample.append(range(start_index, end_index))
    acc = []
    pred = []
    for i in sample:
        a,p = cnn.test_step(W_te[i], Y_te[i])
        pred.extend(p)
    return pred
# from combine_cnn_rnn import *
# from att_rnn_sum_max import *
# from rnn_train import *
from CNNModel import *

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

embSize = 200
type_emb_size = 50
numfilter = 1500
out_file = 'cnn.txt'
num_epochs = 30
N = 1
batch_size = 128
ftrain = "./train_new.txt"
ftest = "./test_new.txt"


Tr_sent_contents, Tr_sent_lables = dataRead(ftrain)
print ('Tr_sent_contents', len(Tr_sent_contents))
Te_sent_contents, Te_sent_lables = dataRead(ftest)
print ("Te_sent_contents", len(Te_sent_contents))
print ("train_size", len(Tr_sent_contents))
print ("test_size", len(Te_sent_contents))
train_sent_lengths, test_sent_lengths = findSentLengths([Tr_sent_contents, Te_sent_contents])
sentMax = max(train_sent_lengths + test_sent_lengths)
print ("max sent length", sentMax)

train_sent_lengths = np.array(train_sent_lengths, dtype='int32')
test_sent_lengths = np.array(test_sent_lengths, dtype='int32')
# Wordlist
label_dict = {'Development':0,'Growth':1,'Breakdown':2,
                'Death':3,'Cell_proliferation':4,'Blood_vessel_development':5,
                'Localization':6,'Binding':7,'Gene_expression':8,'Regulation':9,'Positive_regulation':10,
                'Negative_regulation':11,'Planned_process':12,'Other':13
                }
rev_label_dict = {label_dict[k]: k for k in label_dict}

word_dict = makeWordList(Tr_sent_contents, Te_sent_contents)


rev_word_dict = {word_dict[k]: k for k in word_dict}

print ("word dictonary length", len(word_dict))

# Word Embedding
wv = loadWordEmbeddings(word_dict,embSize)

# Mapping Train
W_train = mapWordToId(Tr_sent_contents, word_dict)
Y_t = mapLabelToId(Tr_sent_lables, label_dict)
Y_train = np.zeros((len(Y_t), len(label_dict)))
for i in range(len(Y_t)):
    Y_train[i][Y_t[i]] = 1.0

# Mapping Test
W_test = mapWordToId(Te_sent_contents, word_dict)
Y_t = mapLabelToId(Te_sent_lables, label_dict)
Y_test = np.zeros((len(Y_t), len(label_dict)))
for i in range(len(Y_t)):
    Y_test[i][Y_t[i]] = 1.0
# padding
W_train, W_test = paddData([W_train, W_test], sentMax)
print ("train", len(W_train))
print ("test", len(W_test))
with open('train_test_data.pickle', 'wb') as handle:
    pickle.dump(W_train, handle)
    pickle.dump(Y_train, handle)
    pickle.dump(train_sent_lengths, handle)

    pickle.dump(W_test, handle)
    pickle.dump(Y_test, handle)
    pickle.dump(test_sent_lengths, handle)

    pickle.dump(wv, handle)
    pickle.dump(word_dict, handle)
    pickle.dump(label_dict, handle)
    pickle.dump(sentMax, handle)
with open('train_test_data.pickle', 'rb') as handle:
	W_train = pickle.load(handle)
	Y_train = pickle.load(handle)
	train_sent_lengths = pickle.load(handle)

	W_test = pickle.load(handle)
	Y_test = pickle.load(handle)
	test_sent_lengths = pickle.load(handle)

	wv = pickle.load(handle)
	word_dict= pickle.load(handle)
	label_dict = pickle.load(handle)
	sentMax = pickle.load(handle)

# vocabulary size
word_dict_size = len(word_dict)
label_dict_size = len(label_dict)
print ("train_size", len(W_train))
print ("test_size", len(W_test))
assert len(W_train)==len(Y_train)
assert len(W_test)==len(Y_test)

fp = open(out_file, 'w')

cnn = CNN_Relation(label_dict_size, sentMax, word_dict_size, wv,batch_size=batch_size,w_emb_size=embSize,type_emb_size=type_emb_size,filter_sizes=[7,8,9],num_filters=numfilter)

num_train = len(W_train)
y_true_list = []
y_pred_list = []
num_batches_per_epoch = int(num_train / batch_size) + 1
for j in range(num_epochs):
    # Shuffling
    shuffle_indices = np.random.permutation(np.arange(num_train))
    W_train = W_train[shuffle_indices]
    Y_train = Y_train[shuffle_indices]
    sam = []
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, num_train)
        sam.append(range(start_index, end_index))

    for rang in sam:
        cnn.train_step(W_train[rang], Y_train[rang])

    if (j % N) == 0:
        pred = test_step(W_test, Y_test)
        print "test data size ", len(pred)
        y_true = np.argmax(Y_test, 1)
        y_pred = pred
        y_true_list.append(y_true)
        y_pred_list.append(y_pred)
#getting the F1-Score 
for y_true, y_pred in zip(y_true_list, y_pred_list):
    fp.write(str(precision_score(y_true, y_pred, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], average='micro')))
    fp.write('\t')
    fp.write(str(recall_score(y_true, y_pred, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], average='micro')))
    fp.write('\t')
    fp.write(str(f1_score(y_true, y_pred, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], average='micro')))
    fp.write('\t')
    fp.write('\n')

fp.write('\n')
fp.write('\n')






