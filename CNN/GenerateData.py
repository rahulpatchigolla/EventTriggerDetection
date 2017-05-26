def checkPresenceandReturnEntities(entities):
    final={}
    for entity in entities:
        if not entity=="None" and not entity=="UNK":
            if not final.has_key(entity):
                final[entity]=0
    finalList=[]
    if len(final.keys())==0:
        finalList.append("No")
    else:
        finalList.append("Yes")
        finalList.extend(final.keys())
    return finalList
def convertToSentence(sentenceList,delimiter):
    sentence=""
    for word in sentenceList:
        sentence+=word
        sentence+=delimiter
    return sentence
trainfile="train"
testfile="test"
fp=open(trainfile+".txt","r")
fpwrite=open(trainfile+"_new"+".txt","w")
count=0
for line in fp:
    line =line.split('\t')
    words=line[0]
    entities=line[1]
    tag=line[2]
    count+=1
    words=words.strip(' ').split(' ')
    entities=entities.strip(' ').split(' ')
    '''if count<2:
        print line
        print entities
        print words'''
    final=checkPresenceandReturnEntities(entities)
    words.extend(final)
    fpwrite.write(convertToSentence(words," ")+"\t"+tag)
    '''if count<2:
        print words
        print convertToSentence(words," ")+"\t"+tag'''
fpwrite.close()
fp.close()
fp=open(testfile+".txt","r")
fpwrite=open(testfile+"_new"+".txt","w")
for line in fp:
    line =line.split('\t')
    words=line[0]
    entities=line[1]
    tag=line[2]
    words=words.strip(' ').split(' ')
    entities=entities.strip(' ').split(' ')
    final=checkPresenceandReturnEntities(entities)
    words.extend(final)
    fpwrite.write(convertToSentence(words," ")+"\t"+tag)
fpwrite.close()
fp.close()
