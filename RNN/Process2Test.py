from Utils import *
from SentenceExtractor import *
processedpathtrain="./Preprocessed_Corpus_train/"
corpuspathtrain="./Corpus_filtered/train/"
processedpathtest="./Preprocessed_Corpus_test/"
corpuspathtest="./Corpus_filtered/test/"
#This file generates test data for the window based CNN model mentioned in the paper
if __name__ == "__main__":
    dirDicttrain = readDir(corpuspathtrain)
    dirDicttest=readDir(corpuspathtest)
    wordVocabtrain = cPickle.load(open(processedpathtrain + "wordVocab.pkl", 'rb'))
    entityVocabtrain = cPickle.load(open(processedpathtrain + "entityVocab.pkl", 'rb'))
    eventVocabtrain = cPickle.load(open(processedpathtrain + "eventVocab.pkl", 'rb'))
    wordVocabtest = cPickle.load(open(processedpathtest + "wordVocab.pkl", 'rb'))
    entityVocabtest = cPickle.load(open(processedpathtest + "entityVocab.pkl", 'rb'))
    eventVocabtest = cPickle.load(open(processedpathtest + "eventVocab.pkl", 'rb'))
    wordVocab=mergeVocabs(wordVocabtrain,wordVocabtest)
    eventVocab=mergeVocabs(eventVocabtrain,eventVocabtest,removeRareEvents=True)
    #print eventVocab
    entityVocab = mergeVocabs(entityVocabtrain, entityVocabtest)
    train_set = dirDicttrain.keys()
    test_set = dirDicttest.keys()
    # print train_set
    # print test_set
    for key in test_set:
        if dirDicttrain.has_key(key):
            print "something is wrong",key
            exit(0)
    fpwrite=open("test.txt","w")
    for key in test_set:
        sentIndex, entityIdDict, entityIndexDict, entityIdIndexMap, eventIdDict, eventIndexDict, eventIdIndexMap = cPickle.load(
            open(processedpathtest + key + ".pkl", 'rb'))
        fp = open(corpuspathtest + key + ".txt", 'r')
        content = fp.read()
        extractor = sentenceExtractor()
        sentenceList, entityList, eventList = extractor.entitiesAndEvents(entityIdDict, entityIndexDict, eventIdDict,
                                                                          eventIndexDict, content, sentIndex,
                                                                          eventVocab, wordVocab, entityIdIndexMap,
                                                                          eventIdIndexMap, removeRareTriggers=True)
        for index in range(0, len(sentenceList)):
            sentence=sentenceList[index]
            entities=entityList[index]
            events=eventList[index]
            winsent,winents=getContextWindow(sentence,9),getContextWindow(entities,9)
            #print winsent,winents
            assert len(winsent)==len(winents)==len(events)
            for winword,winent,event in zip(winsent,winents,events):
                dumpline=""
                for word in winword:
                    punctuations = string.punctuation
                    while len(word) > 0 and word[-1] in punctuations:
                        word = word[:-1]
                    while len(word) > 0 and word[0] in punctuations:
                        word = word[1:]
                    word = word.lower()
                    dumpline+=word
                    dumpline+=" "
                dumpline+="\t"
                for entity in winent:
                    dumpline+=entity
                    dumpline+=" "
                dumpline+="\t"
                dumpline+=event
                fpwrite.write(dumpline)
                fpwrite.write("\n")


