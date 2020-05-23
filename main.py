import nltk 
from nltk.stem.lancaster import LancasterStemmer 
stemmer = LancasterStemmer() 

import numpy as np
import tflearn
import tensorflow 
import random
import json 
import pickle

#Data preprocessing 
with open("intents.json") as file:
    data = json.load(file)

#If the data already exists and we already saved it as "data.pickle" we won't bother doing it again. 
try:
    #rb is read bytes. 
    with open("data.pickle", "rb") as f:
        #We're going to save these four variables: words, labels, training, and output into our pickle file. 
        #We're going to load in these lists. 
        words, labels, training, output = pickle.load(f)

except: 
    words = []
    labels = []
    docs_x = [] #list of all the different pattern in patterns 
    docs_y =[] #corresponding tag for that pattern 

    #For each intent in data["intents"] --> returns each dictionary in the intent list.
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            #Stemming: It takes each word in our "patterns" and brings it down to the root word.
            #Tokenize: Get all the words in our pattern. 
            wrds = nltk.word_tokenize(pattern)
            #Put all the tokenized words into our words list. 
            #wrds is already a list, so instead of looping through it and appending each one
            # we can just extend the list. The extend method takes a single argument (a list) and adds it to the end. 
            words.extend(wrds) 
            #Each entry in docs_x corresponds to an entry in docs_y. 
            docs_x.append(wrds) #We want to append the tokenized words. 
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            #the append() method adds an item to the end of the list
            labels.append(intent["tag"])

    #We're going to stem all the words that we have in the words list, and remove any duplicates.
    #We want to figure out the vocab size of the model is (how many words it has seen already)
    #For each word in words, turn it into lowercase, and use stemmer to stem the words list. 
    words = [stemmer.stem(w.lower()) for w in words if w != "?"] 
    words = sorted(list(set(words))) #set removes duplicates, and you can only sort a list. 
    labels = sorted(labels)

    #Create a bag of words. 
    training = [] #A bunch of bags of words. List of lists of 0s and 1s. 
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = [] #if the word exists, then have a 1 to represent it. If not, a 0 to represent it. 

        wrds = [stemmer.stem(w) for w in doc]

        #Looping through all the words we have 
        for w in words:
            #If the word exists in the current pattern we're looping through 
            if w in wrds: 
                bag.append(1)
            else: 
                bag.append(0)
        
        output_row = out_empty[:]
        #Look through the labels list, see where the tag is in that list, and set that value to 1 in our output_row. 
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    #Turn into np arrays because we need to work with arrays with tflearn 
    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        #Write all these variables into a pickle file so we can save it. 
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()
#The model should expect a length of our training data (length of words)
net = tflearn.input_data(shape=[None, len(training[0])])
#We're going to add this fully_connected layer to our neural network that starts at this input data 
# and we're going to have 8 neurons for this hidden layer. 
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
#allow us to get probabiltiies for each output. 
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

#Our model is predicting which tag that we should take a response from to give to the user. 
#We have six tags, we get prediction values for all of them. 
#We take in as input a bag of words, and as output, we get a random response from the label we predicted. 
#We are classifying sentences of words to some kind of output/tag. 

#to train our model
#DNN is a type of neural network. 
model = tflearn.DNN(net)

#If a model exists, then we won't retrain the model. 
try:
    model.load("model.tflearn")
except: 
#n_epoch is the number of times it's going to see the data. In this case, it's going to see it 1000 times. 
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


#The first step in classifying sentences. 
#Turn a sentence input from the user into a bag of words. 
# s is a sentence
# words is a list of words 
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))] #store all the words
    s_words = nltk.word_tokenize(s) #tokenize the sentence 
    s_words = [stemmer.stem(word.lower()) for word in s_words] #stem the words. 

    #generate bag list properly 
    for se in s_words: 
        for i, w in enumerate(words):
            #the current word that we're looking at in this words list is equal to the word in our sentence (se)
            if w == se: 
                bag[i] = 1
            
    return np.array(bag)

def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        #inp stands for input
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = np.argmax(results) #this gives us the index of the greatest value in the list 
        #We can use that index to figure out which response we want to display
        tag = labels[results_index] #It'll give us the label it thinks our message is 

        if results[results_index] > 0.7: 
            #Open up the json file. Pick a random repsonse with that tag and return to the user. 
            #Loop through all the dictionaries in "intents"
            for tg in data["intents"]:
                #If that dictionary's tag is equal to the tag predicted 
                if tg['tag'] == tag:
                    #Save all the possible responses into "responses"
                    responses = tg['responses']
                    
            print (random.choice(responses))
        else:  
            print("I didn't quite get that, try again.")


            
chat()




