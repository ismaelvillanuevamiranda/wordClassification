import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def parsefile(filename):  #---------------------------------

  inputfp = open(filename, 'r')
  array = []
  for line in inputfp.readlines():
    array.append(line.split())
  return array 

def get_features_and_labels(data):
    data = np.array(data)
    features = data[:,1:5]
    labels = data[:,5:6]
    return features, labels

def to_onehot_encoded_labels(labels):
    labels = np.ravel(labels) #Changes size of labels from (nSamples, 1) to (nSamples,)
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    labels_onehot_encoded = label_encoder.fit_transform(np.array(labels))
    return labels_onehot_encoded

def to_word_embedded_features(train_features, gloveModel):

    word_embedded_features = []
    for word_list in train_features:
        words_features = []
        for word in word_list:
            try:
                embedding_vector = gloveModel[word]
            except KeyError:
                embedding_vector = np.ndarray(shape=(50,), dtype=float)
            words_features.append(embedding_vector)
                
        word_embedded_features.append(np.array(words_features).flatten())
    return np.array(word_embedded_features, dtype = np.float32)
    
def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    with open(gloveFile,'r', encoding="utf8") as f:
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        print("Done.",len(model)," words loaded!")
    return model

trainData = parsefile('training') 
train_features, train_labels = get_features_and_labels(trainData)
print('Loaded train data')
train_onehot_labels = to_onehot_encoded_labels(train_labels)


devsetData = parsefile('devset') 
devset_features, devset_labels = get_features_and_labels(devsetData)
print('Loaded devset data')

gloveModel = loadGloveModel('glove.6B.50d.txt')
print('Loaded Glove Word embeddings')

train_word_embedded_feats = to_word_embedded_features(train_features, gloveModel)
gnb = GaussianNB()
model = gnb.fit(train_word_embedded_feats, train_onehot_labels)



#Making predictions
testData = parsefile('test')
features, labels = get_features_and_labels(testData)
word_embedded_feats = to_word_embedded_features(features, gloveModel)
preds = gnb.predict(word_embedded_feats)

#Printin results
train_onehot_labels = to_onehot_encoded_labels(labels)
print("Acurracy: ")
print(accuracy_score(train_onehot_labels, preds))

