from numpy import *
import operator
from os import listdir

def  createDataSet():
	group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	lables=['A','A','B','B']
	return group ,lables

def classify0(inX,dataSet,labels,k):
	dataSetSize=dataSet.shape[0]
	diffMat=tile(inX,(dataSetSize,1))-dataSet
	sqDiffMat=diffMat**2
	sqDistances=sqDiffMat.sum(axis=1)
	distances=sqDistances**0.5
	sortedDistIndicies=distances.argsort()
	classCount={}
	for i in range(k):
		voteIlabel=labels[sortedDistIndicies[i]]
		classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
	sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
	return sortedClassCount[0][0]

def file2matrix(filename):
	fr=open(filename)
	arrayOLines=fr.readlines()
	numberoflines=len(arrayOLines)
	returnMat=zeros((numberoflines,3))
	classLabelVector=[]
	index=0
	for line in arrayOLines:
		line=line.strip()
		listFromLine=line.split('\t')
		returnMat[index,:]=listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index+=1
	return returnMat , classLabelVector

def autoNorm(dataSet):
	minVals=dataSet.min(0)#it is a colume vector,not just a number
	maxVals=dataSet.max(0)
	ranges=maxVals-minVals
	normDataSet=zeros(shape(dataSet))#make a mat the same as dataSet with 0 filled
	m=dataSet.shape[0]#to get the colume of dataSet
	normDataSet=dataSet - tile(minVals,(m,1))#
	normDataSet=normDataSet/tile(ranges,(m,1))
	return normDataSet,ranges,minVals

def datingClassTest():
	hoRatio=0.1#the ratio of Test data
	datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
	normMat,ranges,minVals=autoNorm(datingDataMat)
	numMatLines=int(normMat.shape[0])
	numTestData=int(numMatLines*hoRatio)
	falseCount=0
	for i in range(numTestData):
		testResult=classify0(normMat[i,:],normMat[numTestData:numMatLines,:],datingLabels[numTestData:numMatLines],3)
		if testResult!=datingLabels[i]:
			print "the classifier came back with :%d,the real answer is :%d" %(testResult,datingLabels[i])
			falseCount+=1
	print "the total error rate is :%f "%(float(falseCount)/numTestData)

def mat2Vec(filename):
	returnVec=zeros((1*1024))
	fr=open(filename)
	print "Opened :",filename
	for i in range(32):
		line=fr.readline()
		for j in range(32):
			returnVec[i+j*32]=int(line[j])
	return returnVec
def handWritingClassTest():
	hwlabels=[]
	fileLists=listdir('trainingDigits')
	linesOMatLines=len(fileLists)
	DataSetMat=zeros((linesOMatLines,32*32))
	for i in range(linesOMatLines):
		DataSetMat[i]=mat2Vec('trainingDigits/%s'%fileLists[i])
		splitFileName=fileLists[i].split('_')
		hwlabels.append(int(splitFileName[0]))
	#test the datatesting
	errorCount=0
	testFileLists=listdir('testDigits')
	for file in testFileLists:
		testVec=mat2Vec('testDigits/%s'%file)
		fileName=file.split('_')
		rightLabel=int(fileName[0])
		resultLabel=classify0(testVec,DataSetMat,hwlabels,3)
		if resultLabel!=rightLabel:
			errorCount+=1
	print "The error count is:%d, error rate is :%f"%(errorCount,errorCount/len(testFileLists))
			

	