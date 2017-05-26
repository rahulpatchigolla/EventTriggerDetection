from Process1Test import readDir
import os
from shutil import copyfile
eventCount={}
multiWordDict={}
multiwordcount=0
#This file is to change the test data according to my preprocessing requirements
def removeEventAnnotations(dirDict,destpath,path):
    for fileName in dirDict.keys():
        print fileName
        fp = open(path+fileName+".a2", "r")
        fpwrite=open(destpath+fileName+".a2","w")
        for line in fp:
            if line[0]=="T":
                fpwrite.write(line)
        fp.close()
        fpwrite.close()
        copyfile(path + fileName + ".txt", destpath + fileName + ".txt")
        copyfile(path + fileName + ".a1", destpath + fileName + ".a1")
if __name__ == "__main__":
    path = "./Corpus/standoff/test/test/"
    destpath="./Corpus_filtered/test/"
    dirDict = readDir(path)
    for file in dirDict.keys():
        #print file
        file+=".txt"
        fp = open(path+file, "r")
        content=fp.read()
        fp.close()
        if content[0]=='[':
            content='@'+content[1:]
            index=1
            while True:
                if content[index]==']':
                    content=content[:index]+'.'+content[index+1:]
                    break
                index+=1
        content=content.replace("\n\n","  ")
        fp = open(path+file, "w")
        fp.write(content)
        fp.close()
    removeEventAnnotations(dirDict,destpath,path)
