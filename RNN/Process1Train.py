import cPickle
import string
from Utils import readDir
from nltk.tokenize import PunktSentenceTokenizer
from SentenceExtractor import *
trainpath="./Preprocessed_Corpus_train/"
#This file generates the pickle files(train data) which store the representational information regarding the data for sentence extractor to work
def ExtractEntities(key):
    entityIndexDict={}
    entityIdDict={}
    entityIdIndexMap={}
    fileName=key
    fileName+=".a1"
    fp=open(fileName,"r")
    line=fp.readline()
    line=line.replace('\n','')
    while line:
        line=line.split('\t')
        #print line
        if(entityIdDict.has_key(line[0])):
            print "Duplicate key error!!! in entityIdDict"
            exit(0)
        else:
            entityIdDict[line[0]]=line[1]
        indexLine=line[1].split(' ')
        #print indexLine
        indexLine[2]+=" "
        indexLine[2]+=indexLine[0]
        if(entityIndexDict.has_key(int(indexLine[1]))):
            tempList=entityIndexDict[int(indexLine[1])]
            tempList.append(indexLine[2])
            entityIndexDict[int(indexLine[1])]=tempList
        else:
            tempList=[]
            tempList.append(indexLine[2])
            entityIndexDict[int(indexLine[1])]=tempList
        if(entityIdIndexMap.has_key(int(indexLine[1]))):
            entityIdIndexMap[int(indexLine[1])].append(line[0])
        else:
            entityIdIndexMap[int(indexLine[1])]=[line[0]]
        line=fp.readline()
        line=line.replace('\n','')
    fp.close()
    return entityIdDict,entityIndexDict,entityIdIndexMap
def ExtractEvents(key):
    eventIndexDict={}
    eventIdDict={}
    eventTriggerDict={}
    eventIdIndexMap={}
    fileName=key
    fileName += ".a2"
    fp = open(fileName, "r")
    line = fp.readline()
    line = line.replace('\n', '')
    while line:
        if line[0] == 'T':
            line = line.split('\t')
            # print line
            if (eventIdDict.has_key(line[0])):
                print "Duplicate key error!!! in entityIdDict"
                exit(0)
            else:
                eventIdDict[line[0]] = line[1]
            indexLine = line[1].split(' ')
            # print indexLine
            indexLine[2] += " "
            indexLine[2] += indexLine[0]
            if (eventIndexDict.has_key(int(indexLine[1]))):
                tempList = eventIndexDict[int(indexLine[1])]
                tempList.append(indexLine[2])
                eventIndexDict[int(indexLine[1])] = tempList
            else:
                tempList = []
                tempList.append(indexLine[2])
                eventIndexDict[int(indexLine[1])] = tempList
            if (eventIdIndexMap.has_key(int(indexLine[1]))):
                eventIdIndexMap[int(indexLine[1])].append(line[0])
            else:
                eventIdIndexMap[int(indexLine[1])] = [line[0]]
        line = fp.readline()
        line = line.replace('\n', '')
    fp.close()
    return eventIdDict,eventIndexDict,eventIdIndexMap
def getSentIndex(sentences):
    sentIndex=[]
    for sentence in sentences:
        sentIndex.append(len(sentence))
    for index in range(0,len(sentIndex)):
        if index==0:
            continue
        elif index==1:
            sentIndex[index]=sentIndex[index-1]+sentIndex[index]+2
        else:
            sentIndex[index]=sentIndex[index-1]+sentIndex[index]+1
    return sentIndex
def buildWordVocab(dirDict,path):
    vocab={}
    dummy={}
    for key in dirDict.keys():
        print key
        fileName=key+".txt"
        fp=open(path+fileName,"r")
        content=fp.read()
        extractor = sentenceExtractor()
        sentIndex, entityIdDict, entityIndexDict, entityIdIndexMap, eventIdDict, eventIndexDict, eventIdIndexMap = cPickle.load(
            open(trainpath + key + ".pkl", 'rb'))
        sentences, _, _=extractor.entitiesAndEvents(entityIdDict,entityIndexDict,eventIdDict,eventIndexDict,content,sentIndex,eventVocab,dummy,entityIdIndexMap,eventIdIndexMap,removeRareTriggers=True)
        punctuations=string.punctuation
        for sentence in sentences:
            for word in sentence:
                while len(word)>0 and word[-1] in punctuations:
                    word = word[:-1]
                while len(word)>0 and word[0] in punctuations:
                    word = word[1:]
                word=word.lower()
                if not vocab.has_key(word):
                    #print word
                    vocab[word]=0
    print "Word Vocabulary size",len(vocab)
    i=0
    for w in vocab.keys():
        vocab[w]=i
        i+=1
    vocab['UNK']=i
    return vocab

def buildEntityVocab(entityIdDict,entityVocab):
    for key,value in entityIdDict.items():
        value=value.split(' ')
        if key[0]=='T':
            if not entityVocab.has_key(value[0]):
                entityVocab[value[0]]=0
    return entityVocab

def buildEntityVocabFinal(entityVocab,eventVocab):
    i=0
    rareEventVocab={'Dissociation': 0,'Protein_processing':0,'Pathway': 0,'Cell_division': 0,'Metabolism': 0,'Translation': 0,'Ubiquitination': 0,'Acetylation': 0,'Reproduction': 0,'DNA_methylation': 0,'Protein_domain_or_region':0,'DNA_domain_or_region': 0}
    keys = eventVocab.keys() + rareEventVocab.keys()
    for key in keys:
        if (eventVocab.has_key(key) or rareEventVocab.has_key(key)) and not key=='Other':
            if entityVocab.has_key(key):
                #print "removing",key
                del entityVocab[key]
    for w in entityVocab.keys():
        entityVocab[w]=i
        i+=1
    entityVocab["None"]=i
    return entityVocab
def buildEventVocab():
    eventVocab={'Development':0,'Growth':1,'Remodeling':2,'Breakdown':3,
                'Death':4,'Cell_proliferation':5,'Blood_vessel_development':6,
                'Localization':7,'Binding':8,'Catabolism':9,'Synthesis':10,
                'Gene_expression':11,'Phosphorylation':12,'Dephosphorylation':13,
                'Transcription':14,'Regulation':15,'Positive_regulation':16,
                'Negative_regulation':17,'Planned_process':18,'Other':19
                }
    return eventVocab
def dumpEntitiesAndEvents(fileName,entityIdDict,entityIndexDict,entityIdIndexMap,eventIdDict,eventIndexDict,eventIdIndexMap,sentIndex):
    cPickle.dump([sentIndex, entityIdDict, entityIndexDict, entityIdIndexMap,eventIdDict, eventIndexDict,eventIdIndexMap],
                 open("./Preprocessed_Corpus_train/" + fileName + ".pkl", "wb"))
if __name__ == "__main__":
    path="./Corpus_filtered/train/"
    dirDict=readDir(path)
    for key,value in dirDict.items():
        print key
    entityVocab={}
    #print wordVocab
    for key in dirDict.keys():
        #key="PMID-15975645"
        print "*******************",key,"*******************\n\n\n"
        fileName=key
        entityIndexDict={}
        entityIdDict={}
        eventIdDict={}
        eventIndexDict={}
        entityIdDict,entityIndexDict,entityIdIndexMap=ExtractEntities(path+fileName)
        eventIdDict, eventIndexDict,eventIdIndexMap=ExtractEvents(path+fileName)
        assert len(entityIdIndexMap)==len(entityIndexDict)
        assert len(eventIdIndexMap)==len(eventIndexDict)
        '''print "entityidDict"
        for key,value in entityIdDict.items():
            print key,":",value
        print "\nentityindexDict\n"
        for key,value in entityIndexDict.items():
            print key,":",value
        print "\nentityidindexDict\n"
        for key,value in entityIdIndexMap.items():
            print key,":",value
        print "\neventidDict\n"
        for key,value in eventIdDict.items():
            print key,":",value
        print "\neventindexDict\n"
        for key,value in eventIndexDict.items():
            print key,":",value
        print "\neventidindexDict\n"
        for key,value in eventIdIndexMap.items():
            print key,":",value'''
        entityVocab=buildEntityVocab(entityIdDict,entityVocab)
        fp=open(path+fileName+".txt","r")
        content=fp.read()
        #print content
        tokenizer = PunktSentenceTokenizer(content)
        sentences=tokenizer.tokenize(content)
        index=0
        #print sentences
        sentIndex=getSentIndex(sentences)
        '''print sentIndex,len(sentIndex)
        print len(content)'''
        dumpEntitiesAndEvents(fileName,entityIdDict,entityIndexDict,entityIdIndexMap,eventIdDict,eventIndexDict,eventIdIndexMap,sentIndex)
        fp.close()
    eventVocab = buildEventVocab()
    entityVocab=buildEntityVocabFinal(entityVocab,eventVocab)
    wordVocab = buildWordVocab(dirDict, path)
    cPickle.dump(entityVocab, open("./Preprocessed_Corpus_train/entityVocab.pkl", "wb"))
    cPickle.dump(eventVocab, open("./Preprocessed_Corpus_train/eventVocab.pkl", "wb"))
    cPickle.dump(wordVocab, open("./Preprocessed_Corpus_train/wordVocab.pkl", "wb"))
    print "EntityVocabulary size:",len(entityVocab)
    print "EventVocabulary size",len(eventVocab)
    print "WordVocabulary size", len(wordVocab)
    print eventVocab
    print entityVocab
    print wordVocab