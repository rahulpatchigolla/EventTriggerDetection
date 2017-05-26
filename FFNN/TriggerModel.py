from __future__ import print_function
__docformat__ = 'restructedtext en'
import os
import sys
import timeit
import numpy
import gzip
import theano
import six.moves.cPickle as pickle
import theano.tensor as T
from Utils import *
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
srng = RandomStreams()
class Dropout(object):
    def __init__(self,X, p=0.):
        self.retain_prob=1 - p
        X *= srng.binomial(X.shape, p=self.retain_prob, dtype=theano.config.floatX)
        X /= self.retain_prob
        self.output=X
class HiddenLayer(object):
    def __init__(self,rng,in_dim,op_dim,input):
        W_ar=numpy.asarray(
            rng.uniform(
            low=-numpy.sqrt(6. / (in_dim+op_dim)),
            high=numpy.sqrt(6. / (in_dim+op_dim)),
            size=(in_dim,op_dim)
            ),
            dtype=theano.config.floatX
        )
        W = theano.shared(value=W_ar, name='W', borrow=True)
        b_ar = numpy.zeros((op_dim,), dtype=theano.config.floatX)
        b = theano.shared(value=b_ar, name='b', borrow=True)
        self.W = W
        self.b = b
        self.params = [self.W, self.b]
        self.input = input
        self.output=T.tanh(T.dot(input,self.W)+self.b)
class EmbedLayer(object):
    def __init__(self,VocabSize,embSize,wordids,embPresent,wordVocab={}):
        self.VocabSize=VocabSize
        self.embSize=embSize
        if embPresent:
            self.E=theano.shared(loadWordEmbeddings(wordVocab,embSize))
        else:
            self.E=theano.shared(value=np.asarray(np.random.uniform(low = -1,
							high = 1,
							size=(self.VocabSize,self.embSize)), dtype=theano.config.floatX), borrow=True)
        self.params=self.E
        self.output=self.E[wordids]
class FFLayer(object):
    def __init__(self, input, in_dim, winDim, op_dim):

        self.W_h = theano.shared(randomMatrix(in_dim*(2*winDim+1),op_dim))
        self.b_h = theano.shared(randomArray(op_dim,))

        self.w_sos = theano.shared(randomMatrix(1, in_dim))
        self.w_eos = theano.shared(randomMatrix(1, in_dim))

        self.input = T.concatenate(winDim * [self.w_sos] + [input] + winDim * [self.w_eos])

        def get_n_gram_h(*arg):
            return T.tanh(T.dot( (T.concatenate(list(arg))).T, self.W_h) + self.b_h)

        h, _ = theano.scan( fn=get_n_gram_h, \
            sequences = dict(input= self.input, taps = range(-winDim,winDim+1)) )
        self.output = h
        self.params = [self.W_h,self.b_h,self.w_sos,self.w_eos]
class InputLayer(object):
    def __init__(self,wordVocabSize,wordVocab,entityVocabSize,entityVocab,embSizeWord,embSizeEntity,input1,input2):
        self.WordEmdLayer=EmbedLayer(VocabSize=wordVocabSize,embSize=embSizeWord,wordids=input1,wordVocab=wordVocab,embPresent=True)
        self.EntityEmdLayer=EmbedLayer(VocabSize=entityVocabSize,embSize=embSizeEntity,wordids=input2,embPresent=False)
        self.output=T.concatenate([self.WordEmdLayer.output,self.EntityEmdLayer.output],axis=1)
        self.params=[self.WordEmdLayer.params,self.EntityEmdLayer.params]


class OutputLayer(object):
    def __init__(self,rng,in_dim,op_dim,input):
        W_ar = numpy.asarray(rng.uniform(low=-numpy.sqrt(6 / (in_dim + op_dim)),
                                         high=numpy.sqrt(6 / (in_dim + op_dim)),
                                         size=(in_dim , op_dim))
                             , dtype=theano.config.floatX
                             )
        W = theano.shared(value=W_ar, name='W', borrow=True)
        b_ar = numpy.zeros((op_dim,), dtype=theano.config.floatX)
        b = theano.shared(value=b_ar, name='b', borrow=True)
        self.W = W
        self.b = b
        self.params=[self.W,self.b]
        self.input = input
        self.prob = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.prob, axis=1)
    def predict(self):
        return self.y_pred
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.prob)[T.arange(y.shape[0]), y])
    def errors(self,y):
        return T.mean(T.neq(self.y_pred,y))



class MyTriggerModel(object):
    def __init__(self,rng,wordVocab,entityVocab,eventVocab,embSizeWord,embSizeEntity,FFhiddLayerDim,input1,input2,windowsize,dropout):
        self.wordVocab=wordVocab
        self.entityVocab=entityVocab
        self.eventVocab=eventVocab
        self.wordVocabSize=len(wordVocab)
        self.entityVocabSize=len(entityVocab)
        self.eventVocabSize=len(eventVocab)
        self.embSizeWord=embSizeWord
        self.embSizeEntity=embSizeEntity
        self.inpDim=self.embSizeWord+self.embSizeEntity
        self.FFhidDim=FFhiddLayerDim
        self.inputLayer=InputLayer(wordVocabSize=self.wordVocabSize,wordVocab=self.wordVocab,entityVocabSize=self.entityVocabSize,entityVocab=self.entityVocab,embSizeWord=self.embSizeWord,embSizeEntity=self.embSizeEntity,input1=input1,input2=input2)
        self.WindowLayer=FFLayer(input=self.inputLayer.output,in_dim=self.inpDim,op_dim=500,winDim=windowsize)
        self.hiddenLayer1=HiddenLayer(rng=rng,in_dim=500,op_dim=self.FFhidDim,input=self.WindowLayer.output)
        self.hiddenLayer1WithDropout=Dropout(self.hiddenLayer1.output,dropout)
        self.hiddenLayer2= HiddenLayer(rng=rng, in_dim=self.FFhidDim, op_dim=100,input=self.hiddenLayer1WithDropout.output)
        self.hiddenLayer2WithDropout = Dropout(self.hiddenLayer2.output, dropout)
        self.outputLayer=OutputLayer(rng=rng,in_dim=100,op_dim=self.eventVocabSize,input=self.hiddenLayer2WithDropout.output)
        self.negative_log_likelihood = self.outputLayer.negative_log_likelihood
        self.errors = self.outputLayer.errors
        self.predict=self.outputLayer.predict
        self.params = self.inputLayer.params + self.hiddenLayer1.params + self.hiddenLayer2.params + self.outputLayer.params+ self.WindowLayer.params
def compute(sentence,entities,model,events):
    input1=wordsToIndexes(sentence,model.wordVocab,True,False)
    #input1=np.asarray(input1,dtype='int32')
    input2 =wordsToIndexes(entities,model.entityVocab,False,False)
    #input2=np.asarray(input2,dtype='int32')
    events=wordsToIndexes(events,model.eventVocab,False,True)
    #events=np.asarray(events,dtype='int32')
    return input1,input2,events
