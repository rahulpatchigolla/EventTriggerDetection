import numpy as np
import theano
import theano.tensor as T
import os
from os import listdir
from nltk.tokenize import PunktSentenceTokenizer
import cPickle
import string
import numpy
import sys
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
distanceDict={}
NoDict={}
#Creates random numpy datastructures
def randomMatrix(r, c,):
    W_bound = numpy.sqrt(6. / (r + c))
    #W_bound = 1.
    return np.random.uniform(low=-W_bound, high=W_bound,\
                   size=(r, c)).astype(theano.config.floatX)
def randomArray(size):
    return np.random.uniform(
        low=-np.sqrt(6. / np.sum(size)),
        high=np.sqrt(6. / np.sum(size)),
        size=size).astype(theano.config.floatX)
def zeroTensor(shape):
    return np.zeros(shape).astype(theano.config.floatX)
#Converts the words into appropriate indexes based on the vocabulary
def wordsToIndexes(words,vocab,wordflag,eventflag):
    punctuations=string.punctuation
    wordIndexes=[]
    for word in words:
        if wordflag == True:
            while len(word) > 0 and word[-1] in punctuations:
                word = word[:-1]
            while len(word) > 0 and word[0] in punctuations:
                word = word[1:]
            word = word.lower()
        if vocab.has_key(word):
            index=vocab[word]
            wordIndexes.append(index)
        elif wordflag==False and eventflag==False:
            index=vocab['None']
            wordIndexes.append(index)
        else:
            if not eventflag==True:
                print word
                print "Look up error!!!"
                exit()
            else:
                print "Event error!!!"
                exit()
    return wordIndexes

#loads the word embeddings from the binary file mentioned in my paper
def loadWordEmbeddings(wordVocab,embSize):
    print "Loading Word Embeddings..."
    print "Total Words:", len(wordVocab)
    model = Word2Vec.load_word2vec_format('/home/rahul/PycharmProjects/PubMed-w2v.bin', binary=True)
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
    wordemb[wordVocab["UNK"]] = np.zeros(embSize)
    wordemb = np.asarray(wordemb, dtype=theano.config.floatX)
    print ("number of unknown word in word embedding", count)
    return wordemb
#method for reading files from the directory
def readDir(path):
    dirList=listdir(path)
    dirDict = {}
    for file in dirList:
        id = file.split('.')
        if (not dirDict.has_key(id[0])):
            dirDict[id[0]] = 0
    return dirDict
#method for getting F1-score from using connll script
def getF1Score(otherWord,filename):
    import commands
    output = commands.getoutput("perl connlleval.pl -l -r -o "+otherWord+"<"+filename)
    output = output.split('\n')
    output = output[-1]
    output = output.split(' ')
    return  float(output[-2])
#method to update the current result with the best result
def updateGlobalFile(file,globalfile):
    f=open(file,"r")
    fw=open(globalfile,"w")
    for line in f.readlines():
        fw.write(line)
    f.close()
    fw.close()
#method for merging train and test vocabularies
def mergeVocabs(trainVocab,testVocab,removeRareEvents=False):
    rareEvents = {"Phosphorylation": "", "Synthesis": "", "Transcription": "", "Catabolism": "",
                  "Dephosphorylation": "","Remodeling":""}
    for key in testVocab.keys():
        if not trainVocab.has_key(key):
            trainVocab[key]=0
    i=0
    if removeRareEvents==True:
        for key in trainVocab.keys():
            if rareEvents.has_key(key):
                del trainVocab[key]
    for key in trainVocab.keys():
        trainVocab[key]=i
        i+=1
    return trainVocab
#method to get context window for the given word
def getContextWindow(input1Sent,windowSize):
    win = (windowSize - 1) / 2
    extra = ["UNK"] * win
    input1Sent = extra + input1Sent
    input1Sent = input1Sent + extra
    input1=[]
    for i in range(0, len(input1Sent) - 2 * win):
        input1.append(input1Sent[i:i + windowSize])
    return input1
#suroutine for postprocess results
def generateMergingList(events):
    newindex = 0
    indexList = []
    for index in range(0, len(events)):
        word = events[index]
        if index == 0:
            if word == "Other":
                indexList.append(newindex)
            else:
                indexList.append(newindex)
        else:
            if word == "Other":
                newindex += 1
                indexList.append(newindex)
            else:
                prevword = events[index - 1]
                if prevword == word :
                    indexList.append(newindex)
                else:
                    newindex += 1
                    indexList.append(newindex)
    return indexList
#subroutine for postprocess results
def mergeMultiWordTriggers(words,actual,predicted):
    indexList=generateMergingList(actual)
    assert(len(words)==len(actual)==len(predicted)==len(indexList))
    newwords=[]
    newactual=[]
    newpredicted=[]
    newindex = 0
    newword = ""
    newactuallabel=""
    newpredictedlabel=""
    for word,actuallabel,predictedlabel,index in zip(words,actual,predicted,indexList):
        if newindex == index:
            if newword=="":
                newword=word
                newactuallabel=actuallabel
                newpredictedlabel=predictedlabel
            else:
                newword+="_"
                newword+=word
                if not newactuallabel==actuallabel:
                    print "Something is wrong!!"
                    exit(0)
                if newactuallabel==predictedlabel:
                    newpredictedlabel=newactuallabel
        else:
            newwords.append(newword)
            newactual.append(newactuallabel)
            newpredicted.append(newpredictedlabel)
            newword=word
            newactuallabel=actuallabel
            newpredictedlabel=predictedlabel
            newindex+=1
    newwords.append(newword)
    newactual.append(newactuallabel)
    newpredicted.append(newpredictedlabel)
    assert len(newwords)==len(newactual)==len(newpredicted)
    return newwords,newactual,newpredicted
#method for merging multiword triggers to a single trigger
def postProcessResults(filename):
    fp=open(filename,"r")
    finalLinesWords=[]
    finalLinesActual = []
    finalLinesPredicted = []
    finalLineWords=[]
    finalLineActual = []
    finalLinePredicted = []
    for line in fp:
        if line=="\n":
            finalLineWords,finalLineActual,finalLinePredicted=mergeMultiWordTriggers(finalLineWords,finalLineActual,finalLinePredicted)
            #print finalLineWords
            #print finalLineActual
            #print finalLinePredicted
            #print "\n"
            finalLinesWords.append(finalLineWords)
            finalLinesActual.append(finalLineActual)
            finalLinesPredicted.append(finalLinePredicted)
            finalLineWords = []
            finalLineActual = []
            finalLinePredicted = []
        else:
            line=line.strip('\n').split(' ')
            finalLineWords.append(line[0])
            finalLineActual.append(line[1])
            finalLinePredicted.append(line[2])

    fp.close()
    fpwrite=open(filename,"w")
    for line,actual,predicted in zip(finalLinesWords,finalLinesActual,finalLinesPredicted):
        for word,actuallabel,predictedlabel in zip(line,actual,predicted):
            fpwrite.write(word+" "+actuallabel+" "+predictedlabel+"\n")
        fpwrite.write("\n")
    fpwrite.close()
