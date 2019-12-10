import random
import numpy
import sys



def initWeights():
    global weights
    weights = 2 * numpy.random.random((390,1)) - 1      #initialises random weights for every arc
    
    
def chooseLanguage():
    global languageName1, languageName2
    languages = [languageName1, languageName2]          
    languageChoice = random.choice(languages)           #randomly choosing between the two chosen languages
    if languageChoice == languageName1:
        outputWordValue = [0]                           #setting the desired value of the language to either 0/1
    elif languageChoice == languageName2:
        outputWordValue = [1]

    return languageChoice

def createWordMatrix(languageChoice):
    global languageName1, languageName2
    word = ''
    if languageChoice == languageName1:
        lines = open(languageName1 + 'Words.txt').read().splitlines()       #getting a random word from the training words text file
        outputWordValue = [0]
        word = random.choice(lines)
    elif languageChoice == languageName2:
        lines = open(languageName2 + 'Words.txt').read().splitlines()
        outputWordValue = [1]
        word = random.choice(lines)
    

    word = word[0:15]
    if len(word) < 15:                                                      #ensuring the chosen word length is less than 15 characters
        for x in range(15-len(word)):
            word += " "                                                     #adds spaces to the end of the word to make it 15 length
    alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    base = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]            #26 long list corresponding to each letter in the alphabet
    inputWordMatrix = []

    for letter in word:
        letter_descriptor = base
        for pos, char in enumerate(alphabet):
            if letter == char:
                letter_descriptor[pos] = 1                                  #a 1 is set in the base list where there is that letter in the alphabet
        inputWordMatrix.extend(letter_descriptor)

    return inputWordMatrix


def sigmoid(x):
    return 1 /(1 + numpy.exp(-x))                                           #used in the dot product of the word matrix and weighted arcs

def sigmoidDerivative(x):
    return x * (1-x)                                                        

def train(inputWordMatrix, outputWordValue, numberOfIterations):
    global weights
    for iteration in range(numberOfIterations):                             #performs the algorithm for the user inputted number of times
        inputWordMatrix = numpy.reshape(createWordMatrix(chooseLanguage), (1, 390))
        calculatedOutput = calculate(inputWordMatrix)                       
        error = outputWordValue - calculatedOutput                          #calculate error by finding difference between the desired value and calculated value
        
        adjustment = numpy.dot(numpy.transpose(inputWordMatrix), error * sigmoidDerivative(calculatedOutput))   #calculating the adjustment value to be applied to the weights
        weights = adjustment + weights
        
        if iteration == numberOfIterations / 4:                             #updates the user on the progress of the training
            print('25% complete')
        elif iteration == numberOfIterations / 2:
            print('50% complete')
        elif iteration == (numberOfIterations / 4) * 3:
            print('75% complete')
    print("Training complete")
    
def calculate(inputWordMatrix):
    return sigmoid(numpy.dot(inputWordMatrix, weights))                     #calculating dot product of the word and the weights and applying sigmoid function to put it between 0 and 1

def userInputWordMatrix(word):                                              #same process as createWordMatrix but using user input word
    word = word[0:15]
    if len(word) < 15:
        for x in range(15-len(word)):
            word += " "                                                     
    alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    base = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    inputWordMatrix = []

    for letter in word:
        letter_descriptor = base
        for pos, char in enumerate(alphabet):
            if letter == char:
                letter_descriptor[pos] = 1
        inputWordMatrix.extend(letter_descriptor)

    return inputWordMatrix

#Main Program
outputWordValue = [0]
weights = []
initWeights()
while True:
    print("LANGUAGE IDENTIFICATION NEURAL NETWORK")                 #visual main menu                
    print("1. Train the neural network")
    print("2. Predict language of own word")
    print("3. Exit")
    userSelection = input("Enter menu choice: ")

    if userSelection == "1" :
        print("The program is capable of predicting between 2 languages. Here are the available languages to choose from:")
        print("1. English \n2. Spanish \n3. Dutch \n4. French \n5. Polish \n6. Chinese \n7. Japanese \n8. German")
        while True:
            try:
                userLanguageChoice1 = int(input("Enter the number of the first language selection: "))          #user input for language selection error checking to ensure input is an integer and is in range
            except:
                print("Please enter a number corresponding to your choice of language.")
            else:
                if 1 <= userLanguageChoice1 <= 8:
                    print("Language number ", userLanguageChoice1 , " selected")
                    break
                else:
                    print("Number entered does not correspond to a language")
        while True:
            try:
                userLanguageChoice2 = int(input("Enter the number of the second language selection: "))
            except:
                print("Please enter a number corresponding to your choice of language.")
            else:
                if 1 <= userLanguageChoice2 <= 8 and userLanguageChoice1 != userLanguageChoice2 :           #user cannot select the same language as before
                    print("Language number ", userLanguageChoice2 , " selected")
                    break
                else:
                    print("Number entered does not correspond to a language or is the same as the first language choice")
                    
        while True:
            try:
                iterationValue = int(input("How many iterations of training would you like to perform: "))
            except:                                    
                print("Please enter an integer number value.")
            else:
                print(iterationValue, " iterations will be performed.")
                break
                         


        languageList = ['English', 'Spanish', 'Dutch', 'French', 'Polish', 'Chinese', 'Japanese', 'German'] #assigning the user's selection to the language name
        languageName1 = languageList[userLanguageChoice1 - 1]       #language name is used in the opening of the word file in the createWordMatrix function
        languageName2 = languageList[userLanguageChoice2 - 1]


            
        print("Training. Please wait...")    
        train(createWordMatrix(chooseLanguage), outputWordValue, iterationValue) #using the user's word to train

    elif userSelection == "2":
        again = True                                                                    #loops unless the user chooses to not continue
        while again == True:
            userWord = str(input("Enter a word "))
            userPrediction = (calculate(userInputWordMatrix(userWord)))
            userPrediction = int(round(userPrediction[0]))                              #checks if the word is closer to 0 / 1 by rounding
            if userPrediction == 0:                                                     #0/1 corresponds to either the first or second language selected
                print("I predict", languageName1, ". How did I do?")
            elif userPrediction == 1:
                print("I predict", languageName2, ". How did I do?")

            userAgain = input("Again Y/N ")
            if userAgain == "N" or userAgain == "n":
                again = False
                
        

    elif userSelection == "3":
        sys.exit(0)

