from Utils import *
class sentenceExtractor:
    def __init__(self):
        self.rareEvents = {"Phosphorylation": "", "Synthesis": "", "Transcription": "", "Catabolism": "",
                  "Dephosphorylation": "","Remodeling":""}
    def entitiesAndEvents(self,entityIdDict,entityIndexDict,eventIdDict,eventIndexDict,content,sentIndex,eventVocab,wordVocab,entityIdIndexMap,eventIdIndexMap,removeRareTriggers=False):
        sentenceList = []
        entityList = []
        eventList = []
        '''print entityIdDict
        print "\n"
        print entityIndexDict
        print "\n"
        print sentIndex'''
        for index in range(0,len(sentIndex)):
            if index==0:
                sentence=content[0:sentIndex[index]]
                #print sentence
                indexList, wordList=self.assignIndextoWord(sentence,0,0,entityIndexDict,eventIndexDict)
                '''print indexList
                print wordList'''
                entities=["None"]*len(indexList)
                events=["Other"]*len(indexList)
                '''i = 0
                for index in indexList:
                    print content[int(index)], wordList[i]
                    i += 1'''
                for wordIndex in range(0,len(indexList)):
                    if entityIndexDict.has_key(indexList[wordIndex]):
                        entity = entityIndexDict[indexList[wordIndex]]
                        entity=entity[0]
                        entity=entity.split(' ')
                        startindex=indexList[wordIndex]
                        endindex=int(entity[0])
                        words=content[startindex:endindex]
                        words=words.split(' ')
                        j=wordIndex
                        for word in words:
                            entities[j]=entity[1]
                            j+=1
                for wordIndex in range(0,len(indexList)):
                    if eventIndexDict.has_key(indexList[wordIndex]):
                        event = eventIndexDict[indexList[wordIndex]]
                        event=event[0]
                        event=event.split(' ')
                        startindex=indexList[wordIndex]
                        endindex=int(event[0])
                        words=content[startindex:endindex]
                        words=words.split(' ')
                        j=wordIndex
                        for word in words:
                            events[j]=event[1]
                            j+=1

            else:
                sentence = content[sentIndex[index-1]:sentIndex[index]]
                #print sentence
                if index==1:
                    indexList, wordList = self.assignIndextoWord(sentence, sentIndex[index-1], 2,entityIndexDict,eventIndexDict)
                else:
                    indexList, wordList = self.assignIndextoWord(sentence, sentIndex[index-1], 1,entityIndexDict,eventIndexDict)
                '''print indexList
                print wordList'''
                entities = ["None"] * len(indexList)
                events = ["Other"] * len(indexList)
                '''i = 0
                for index in indexList:
                    print content[int(index)], wordList[i]
                    i += 1'''
                for wordIndex in range(0, len(indexList)):
                    if entityIndexDict.has_key(indexList[wordIndex]):
                        entity = entityIndexDict[indexList[wordIndex]]
                        entity = entity[0]
                        entity = entity.split(' ')
                        startindex = indexList[wordIndex]
                        endindex = int(entity[0])
                        words = content[startindex:endindex]
                        words = words.split(' ')
                        j = wordIndex
                        for word in words:
                            entities[j] = entity[1]
                            j += 1
                for wordIndex in range(0,len(indexList)):
                    if eventIndexDict.has_key(indexList[wordIndex]):
                        event = eventIndexDict[indexList[wordIndex]]
                        event = event[0]
                        event = event.split(' ')
                        startindex = indexList[wordIndex]
                        endindex = int(event[0])
                        words = content[startindex:endindex]
                        words = words.split(' ')
                        j = wordIndex
                        for word in words:
                            events[j] = event[1]
                            j += 1
                '''print wordList
                print entities
                print events
                print indexList'''
            events = self.removeRareTriggerWords(events, eventVocab, removeRareTriggers)
            sentenceList.append(wordList)
            entityList.append(entities)
            eventList.append(events)
        self.correctList(entityList,eventList,eventVocab)#should take care of the correction
        #self.printList(sentenceList,entityList,eventList)
        return sentenceList,entityList,eventList
    def assignIndextoWord(self,sentence,index,flag,entityIndexDict,eventIndexDict):
        indexList=[]
        wordList=sentence.strip().split(' ')
        i=flag
        for word in wordList:
            indexList.append(flag)
            flag+=len(word)+1
        for i in range(0,len(indexList)):
            indexList[i]+=index
        newwordList=[]
        newindexList=[]
        for word,index in zip(wordList,indexList):
            high = index + len(word) - 1
            low=index
            curr=low
            tempindexlist=[]
            tempwordlist=[]
            while curr<=high:
                if entityIndexDict.has_key(curr) or eventIndexDict.has_key(curr) or curr==low:
                    tempindexlist.append(curr)
                curr+=1
            wordindex=0
            for iterator in range(0,len(tempindexlist)):
                if iterator==len(tempindexlist)-1:
                    tempwordlist.append(word[wordindex:])
                else:
                    low=tempindexlist[iterator]
                    high=tempindexlist[iterator+1]-1
                    tempwordlist.append(word[wordindex:wordindex+high-low+1])
                    wordindex+=high-low+1
            assert(len(tempwordlist)==len(tempindexlist))
            newwordList+=tempwordlist
            newindexList+=tempindexlist
        indexList=newindexList
        wordList=newwordList
        newindexList=[]
        newwordList=[]
        for word,index in zip(wordList,indexList):
            high = index + len(word) - 1
            low = index
            curr = low
            tempindexlist = []
            tempwordlist = []
            if entityIndexDict.has_key(curr) or eventIndexDict.has_key(curr):
                length =0
                if entityIndexDict.has_key(curr):
                    length=self.getLength(curr,entityIndexDict)
                else:
                    length=self.getLength(curr,eventIndexDict)
                tempindexlist=[curr]
                tempwordlist=[word[0:length]]
                newindexList+=tempindexlist
                newwordList+=tempwordlist
            else:
                newindexList.append(index)
                newwordList.append(word)

        return newindexList,newwordList
    def correctList(self,entityList,eventList,eventVocab):
        for gindex in range(0,len(entityList)):
            entities=entityList[gindex]
            events=eventList[gindex]
            for index in range(0,len(entities)):
                if eventVocab.has_key(entities[index]) or self.rareEvents.has_key(entities[index]):
                    entities[index]="None"

    def printList(self, sentenceList, entityList, eventList):
        for index in range(0, len(sentenceList)):
            print "\n******************sentence*********************\n"
            '''print "sentence:", sentenceList[index]
            print "entities:", entityList[index]
            print "events", eventList[index]'''
            for word, entity, trigger in zip(sentenceList[index], entityList[index], eventList[index]):
                print word,entity,trigger
    def removeRareTriggerWords(self,events,eventVocab,removeRareTriggers):
        for index in range(0,len(events)):
            if removeRareTriggers==True:
                if self.rareEvents.has_key(events[index]) or not eventVocab.has_key(events[index]):
                    events[index]="Other"
            else:
                if not eventVocab.has_key(events[index]):
                    events[index]="Other"
        return events
    def getBestWordIndex(self,indexList,currentindex,indexdict,wordlength):
        low=indexList[currentindex]
        oldlow=low
        high=indexList[currentindex]+wordlength-1
        if high-low<0:
            #print "Empty string"
            return oldlow
        while low<=high:
            if indexdict.has_key(low):
                return low
            low+=1
        return oldlow
    def getLength(self,index,indexDict):
        low=index
        val=indexDict[index][0]
        val=val.split(" ")
        high=int(val[0])
        return high-low
    def loadParseTrees(self,key,sentenceList,entitylist,eventlist,prefix):
        newSentenceList=[]
        newEntityList=[]
        newEventList=[]
        sentenceIdList=[]
        sentenceRootList=[]
        sentenceChunkList=[]
        sentencePosList=[]
        sentenceParentList=[]
        sentenceParseList=[]
        id, root, chunk, pos, parentid, parse=cPickle.load( open("./Parsed_output_"+prefix + key + ".pkl", "rb"))
        for sentence,sentids,sentroots,sentchunks,sentpos,sentparentids,sentparses,entities,events in zip(sentenceList,id,root,chunk,pos,parentid,parse,entitylist,eventlist):
            newsentence=[]
            newentities=[]
            newevents=[]
            for word,entity,event in zip(sentence,entities,events):
                if not word=="":
                    newsentence.append(word)
                    newentities.append(entity)
                    newevents.append(event)
                else:
                    print key
                    exit(0)
            '''print newsentence
            print newentities
            print sentids
            print sentroots
            print sentchunks
            print sentpos
            print sentparentids
            print sentparses
            print "\n\n"
            print len(newsentence)
            print len(newentities)
            print len(sentids)
            print len(sentroots)
            print len(sentchunks)
            print len(sentpos)
            print len(sentparentids)
            print len(sentparses)'''
            assert(len(newsentence)==len(sentids)==len(sentroots)==len(sentchunks)==len(sentpos)==len(sentparentids)==len(sentparses)==len(newentities)==len(newevents))
            newSentenceList.append(newsentence)
            newEntityList.append(newentities)
            newEventList.append(newevents)
            sentenceIdList.append(sentids)
            sentenceRootList.append(sentroots)
            sentenceChunkList.append(sentchunks)
            sentencePosList.append(sentpos)
            sentenceParentList.append(sentparentids)
            sentenceParseList.append(sentparses)
        return newSentenceList,newEntityList,newEventList,sentenceIdList,sentenceRootList,sentenceChunkList,sentencePosList,sentenceParentList,sentenceParseList