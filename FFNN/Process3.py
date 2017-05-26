from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
#This file is to generate the final F1-score of the data by considering the rareevents as false negatives and also generates analysis results also
fp=open("bestTestPredictionsTrigger.txt","r")
actual = []
predicted = []
for line in fp:
    if not line=="\n":
        line =line.strip('\n').split(' ')
        actual.append(line[1])
        predicted.append(line[2])
fp.close()
rareEvents = {"Phosphorylation": 3, "Synthesis": 4, "Transcription": 7, "Catabolism": 4,"Dephosphorylation":1, "Remodeling": 10}
for key in rareEvents.keys():
    #print key
    actualval=rareEvents[key]
    takenval=0
    diff=actualval-takenval
    while diff>0:
        diff-=1
        actual.append(key)
        predicted.append('Other')
x=precision_score(actual,predicted,labels=['Development','Growth','Breakdown','Death','Cell_proliferation','Blood_vessel_development','Remodeling'],average='micro')
y=recall_score(actual,predicted,labels=['Development','Growth','Breakdown','Death','Cell_proliferation','Blood_vessel_development','Remodeling'],average='micro')
z=f1_score(actual,predicted,labels=['Development','Growth','Breakdown','Death','Cell_proliferation','Blood_vessel_development','Remodeling'],average='micro')
print "Anatomical",x,y,z
x=precision_score(actual,predicted,labels=['Gene_expression','Synthesis','Transcription','Catabolism','Phosphorylation','Dephosphorylation'],average='micro')
y=recall_score(actual,predicted,labels=['Gene_expression','Synthesis','Transcription','Catabolism','Phosphorylation','Dephosphorylation'],average='micro')
z=f1_score(actual,predicted,labels=['Gene_expression','Synthesis','Transcription','Catabolism','Phosphorylation','Dephosphorylation'],average='micro')
print "Molecular",x,y,z

x=precision_score(actual,predicted,labels=['Localization','Binding','Regulation','Positive_regulation','Negative_regulation'],average='micro')
y=recall_score(actual,predicted,labels=['Localization','Binding','Regulation','Positive_regulation','Negative_regulation'],average='micro')
z=f1_score(actual,predicted,labels=['Localization','Binding','Regulation','Positive_regulation','Negative_regulation'],average='micro')
print "General",x,y,z

x=precision_score(actual,predicted,labels=['Planned_process'],average='micro')
y=recall_score(actual,predicted,labels=['Planned_process'],average='micro')
z=f1_score(actual,predicted,labels=['Planned_process'],average='micro')
print "Planned",x,y,z

x=precision_score(actual,predicted,labels=['Development','Growth','Breakdown','Death','Cell_proliferation','Blood_vessel_development',
                                                         'Localization','Binding','Gene_expression','Regulation','Positive_regulation',
                                                         'Negative_regulation','Planned_process','Phosphorylation', 'Synthesis', 'Transcription', 'Catabolism','Dephosphorylation','Remodeling'],average='micro')
y=recall_score(actual,predicted,labels=['Development','Growth','Breakdown','Death','Cell_proliferation','Blood_vessel_development',
                                                         'Localization','Binding','Gene_expression','Regulation','Positive_regulation',
                                                         'Negative_regulation','Planned_process','Phosphorylation', 'Synthesis', 'Transcription', 'Catabolism','Dephosphorylation','Remodeling'],average='micro')

z=f1_score(actual,predicted,labels=['Development','Growth','Breakdown','Death','Cell_proliferation','Blood_vessel_development',
                                                         'Localization','Binding','Gene_expression','Regulation','Positive_regulation',
                                                         'Negative_regulation','Planned_process','Phosphorylation', 'Synthesis', 'Transcription', 'Catabolism','Dephosphorylation','Remodeling'],average='micro')

print "Overall",x,y,z
matrix=confusion_matrix(actual,predicted,labels=['Development','Growth','Breakdown','Death','Cell_proliferation','Blood_vessel_development',
                                                         'Localization','Binding','Gene_expression','Regulation','Positive_regulation',
                                                         'Negative_regulation','Planned_process','Phosphorylation', 'Synthesis', 'Transcription', 'Catabolism','Dephosphorylation','Remodeling','Other'])

labels=['DEV','GRO','BRK','DTH','CELLP','BVD','LOC','BIND','GENXP','REG','PREG','NREG','PLP','PHO', 'SYN', 'TRANS', 'CATA','DEPHO','REMDL','OTH']
print matrix
df_cm = pd.DataFrame(matrix,index = [i for i in labels],columns=[i for i in labels])
y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
plt.figure(figsize = (21,15))
sn.set(font_scale=1.1)
sn.heatmap(df_cm, annot=True,fmt=".0f",annot_kws={"size":20},linewidths=1,linecolor="Black",robust=True)
plt.savefig('confusion_matrix.png', format='png')
