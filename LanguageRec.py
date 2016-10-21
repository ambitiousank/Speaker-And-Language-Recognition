from python_speech_features import mfcc
from os import listdir
from os.path import isfile, join,isdir
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
import os
import scipy.io.wavfile as wav
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 


def getMFCCFeature(path):
	rate,sig=wav.read(path)
	mfcc_feat = mfcc(sig,rate,winlen=0.025,winstep=0.01)
	print mfcc_feat.shape
	return mfcc_feat


def createClassLabel(length,label):
	k=np.zeros((length,1), dtype=int)
	k=k+label
	return k
	

trainPath="lang_rec_data_with_division/train/"
testPath="lang_rec_data_with_division/test/"
vec_length=13

dict_label={}
mfcc_feature=np.empty((0,vec_length))
label_array=np.empty((0,1))
mean_arr=np.empty((0,vec_length))
mean_temp=np.empty((0,vec_length))


languageFiles = [ f for f in listdir(trainPath) if isdir(join(trainPath,f)) ]
for i in range (0,len(languageFiles)):
	dict_label[languageFiles[i].upper()]=i
	audioPath=join(trainPath,languageFiles[i])
	audioFiles= [ l for l in listdir(audioPath) if isfile(join(audioPath,l)) ]
	print "--------------------------",languageFiles[i]
	mfcc_temp=np.empty((0,vec_length))
	for j in range (0,len(audioFiles)):
		print "---------",audioFiles[j]
		filePath=join(audioPath,audioFiles[j])
		temp_feat=getMFCCFeature(filePath)
		print "temp_feat",temp_feat.shape
		mfcc_temp=np.append(mfcc_temp,temp_feat,axis=0)
	length=mfcc_temp.shape[0]
	avg=np.empty((1,vec_length))
	sm=np.empty((1,vec_length))
	print "length", length
	avg=np.mean(mfcc_temp,axis=0)
	print avg.shape
	feature=avg.reshape(1,vec_length)
	print avg
	arr=np.array(feature,dtype=np.float32)
	print "arr shape", arr.shape
	mean_arr=np.append(mean_arr,arr,axis=0)
	print "mfcc_temp",mfcc_temp.shape
	temp_label_array=createClassLabel(length,i)
	print "temp_label_array",temp_label_array.shape
	label_array=np.append(label_array,temp_label_array,axis=0)
	print "label_array",label_array.shape
	mfcc_feature=np.append(mfcc_feature,mfcc_temp,axis=0)
	print "mfcc_feature",mfcc_feature.shape

print dict_label
print mean_arr.shape


'''estimators= dict((cov_type, GaussianMixture(n_components=n_classes,covariance_type=cov_type,max_iter=20,random_state=0)) for cov_type in ['spherical', 'diag', 'tied', 'full'])

n_estimators=len(estimators)   '''

print "learning GMM"

print mfcc_feature.shape
print mfcc_feature.astype

#mfcc_feature=np.asmatrix(mfcc_feature)
print label_array.shape
print "for 1",sum(mfcc_feature[:][1])
print "for 2",sum(mfcc_feature[:][0])



'''
for i in range(0,mfcc_feature.shape[0]):
	if(label_array[i]==0):
	for l in range(0,mfcc_feature[i].shape[1]):
	avg[0][l]=sum(mfcc_feature[:][l])
	elif(label_array[i]==1):
'''



gmm=GaussianMixture(n_components=4,covariance_type='tied',max_iter=20,random_state=0)
#xd=np.array([mfcc_feature[label_array == i].mean(axis=0)  for i in range(4)]).reshape(4,13)
gmm.means_init=mean_arr
print mean_arr
gmm.fit(mfcc_feature)
model=LogisticRegression()
model.fit(mfcc_feature,label_array)
print "Inference"
languageTestFiles = [ f for f in listdir(testPath) if isdir(join(testPath,f)) ]
for i in range(0,len(languageTestFiles)):
	audioPath=join(testPath,languageTestFiles[i])
	audioFiles= [ l for l in listdir(audioPath) if isfile(join(audioPath,l)) ]
	print "--------------------------",languageTestFiles[i]
	label_current=dict_label[languageTestFiles[i].upper()]
	print "current label",label_current
	for j in range (0,len(audioFiles)):
		print "---------",audioFiles[j]
		filePath=join(audioPath,audioFiles[j])
		temp_feat=getMFCCFeature(filePath)
		#print "temp_feat.shape",temp_feat.shape
		y=gmm.predict(temp_feat)
		y_lr=model.predict(temp_feat)
		y_test=createClassLabel(temp_feat.shape[0],label_current)
		test_accuracy = np.mean(y.ravel() == y_test.ravel()) 
		print "accuracy for ensemble ",metrics.accuracy_score(y_lr,y_test)
		print test_accuracy
		#print y
		#print y.shape
		print "mode", stats.mode(y)[0]
		
				


