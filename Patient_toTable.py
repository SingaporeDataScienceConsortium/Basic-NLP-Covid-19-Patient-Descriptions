# # The code is shared on SDSC Github
'''
'Rule-based Information Extraction from Covid-19 Patient Descriptions'

This code aims to extract the following information (if available)
from descriptions of COVID-19 patients
(1) Case number
(2) Age
(3) Nationality
(4) Hospital
(5) Address
'''

import nltk # Natural Language Toolkit
from nltk.stem import PorterStemmer # An algorithm for suffix stripping
import string # String processing
import pandas as pd # create and generate table
import numpy as np # mathematical calculation
import os # operations related to operation system


# create a new folder for results
if not os.path.exists('result'): # check whether the result folder already exists
    os.makedirs('result') # create a new folder

country_list = pd.read_excel('country_list.xlsx') # get country list
country_list = country_list['Country'].tolist() # get country list
country_list = [x.lower() for x in country_list] # convert all country names to lowercase

hospital_list = pd.read_excel('hospital_list.xlsx') # get hospital list
hospital_list = hospital_list['Hospital'].tolist() # get hospital list
hospital_list = [x.lower() for x in hospital_list] # convert all hospital names to lower-case

f=open('patients.txt','r') # specify the file containing patient descriptions
Lines = f.readlines() # read the file line by line and save into a list
Lines = [x for x in Lines if x!='\n'] # for each line: remove 'newline' character
Lines = [x.replace('.',' .') for x in Lines] # add a space before '.'

# names of entities to be extracted
col_labels = ['Case Number',
              'Age',
              'Nationality',
              'Hospital',
              'Address']

# create an empty table using dataframe
patients = pd.DataFrame(columns=col_labels )

counter = 0 # create a counter to count the number of patients registered
for i in range(0, len(Lines), 2): # read every other line from the descriptions
    print('processing', i, '/', len(Lines), '.') #indicate progress

    case_number = np.nan
    age = np.nan
    nationality = ''
    hospital = ''
    address = ''

    # get case number
    _, case_number = Lines[i].split() # split 'Case XX' into 'Case' and 'XX'
    case_number = int(case_number) # from string to integer

    sentences = nltk.sent_tokenize(Lines[i+1].lower()) # split a description into multiple sentences and change them to lower-case
    sentences = [x.replace(',','') for x in sentences] # replace symbols that are not important in the processing
    sentences_token = [nltk.word_tokenize(sent) for sent in sentences] # tokenize each sentence into multiple words

    # this for loop aims to extract age and nationality
    for s in sentences_token: # go through each sentence
        for i in range(len(s)): # go through the words of each sentence

            # get age
            if s[i]=='year-old': # format 1: year-old
                age = int(s[i-1]) # the previous word is the age

            if i!=(len(s)-1) and s[i]=='years' and s[i+1]=='old': # format 2: years old
                age = int(s[i-1]) # the previous word is the age

            # get nationality
            if s[i]=='citizen' and (s[i-1] in country_list): # format 1: 'Country_Name Citizen'
                nationality = s[i-1].upper() # the previous word is the country name
            if s[i]=='permanent' and s[i+1]=='resident' and (s[i-1] in country_list): # format 1: 'Country_Name Permanent Resident'
                nationality = s[i-1].upper() + ' PR' # the previous word is the country name

    # this for loop aims to extract hospital
    for s in sentences_token: # go through each sentence
        for i in range(len(s)):
            if PorterStemmer().stem(s[i])=='ward': # if 'ward' appears
                temp_hosp = s[i+1:] # the remaining string after 'ward' may contain a hospital name
                temp_hosp = ' '.join([x for x in temp_hosp]) # convert the remaining tokenized words into a string separated by a space
                for h in hospital_list: # go through all hospital names
                    if h in temp_hosp: # hospital detected if the name of a hopital exists in the remaining string
                        hospital = string.capwords(h)


    # extract address below
    sentences_tag = [nltk.pos_tag(sent) for sent in sentences_token] # add tags (properties) to all tokenized words in all sentences
    for s_tag in sentences_tag: # go through each tagged sentence

        # split words and their tags into 2 lists
        words = [x[0] for x in s_tag]
        tags = [x[1] for x in s_tag]

        for c, w in enumerate(words[:-1]): # go through each word

            if PorterStemmer().stem(w)=='stay' and tags[c+1]=='IN' and (words[c+1]=='at' or words[c+1]=='in'):
            # if 'stay' appears and its next word is preposition or subordinating conjunction ('at' or 'in')

                temp_str = [] # create an empty string to store a possible address
                while c+2 < len(words): # this loop ends when the sentence ends

                    if tags[c+2]!='IN' and tags[c+2]!='CC': # if the word is not IN or CC
                    # IN: preposition or subordinating conjunction
                    # CC coordinating conjunction

                        temp_str.append(words[c+2]) # append words one by one
                        c = c + 1
                    else:

                        if tags[c+2]=='CC': # CC coordinating conjunction
                            if 'home' in temp_str: # if 'home' exists in the text before 'CC', address may not be available
                                temp_str = []
                                break
                            if 'home' not in temp_str:
                                if tags[c+3]=='VBN' or tags[c+3]=='VBD': # maybe no more information about address if a verb follows
                                    break
                                else: # may stay in multiple places
                                    temp_str.append(words[c+2])
                                    c = c + 1

                        if tags[c+2]=='IN': # IN: preposition or subordinating conjunction
                            if 'home' in temp_str:
                                if words[c+2]=='at' or words[c+2]=='in':
                                    temp_str = [] # clear 'temp_str' because the address may be after this
                                    c = c + 1
                                else: # other prepositions may lead to non-address information
                                    temp_str = []
                                    break
                            if 'home' not in temp_str:
                                if words[c+2]=='at' or words[c+2]=='in': # may further describe the address
                                    temp_str.append(words[c+2])
                                    c = c + 1
                                else: # other prepositions may lead to non-address information
                                    break

                address = temp_str # the stored information may be the address
                address = [x for x in address if x!='.' and x!=','] # remove symbols
                address = ' '.join([x for x in address]) # covert tokenized words into a string
                address = string.capwords(address)
    # extract address above

    patients.loc[counter] = [case_number] + [age] + [nationality] + [hospital] + [address] # update the table
    counter = counter + 1

patients.to_excel('result/Patients_Table.xlsx') # export the table into an excel file




'''
POS tag list:

CC coordinating conjunction
CD cardinal digit
DT determiner
EX existential there (like: "there is" ... think of it like "there exists")
FW foreign word
IN preposition/subordinating conjunction
JJ adjective 'big'
JJR adjective, comparative 'bigger'
JJS adjective, superlative 'biggest'
LS list marker 1)
MD modal could, will
NN noun, singular 'desk'
NNS noun plural 'desks'
NNP proper noun, singular 'Harrison'
NNPS proper noun, plural 'Americans'
PDT predeterminer 'all the kids'
POS possessive ending parent's
PRP personal pronoun I, he, she
PRP$ possessive pronoun my, his, hers
RB adverb very, silently,
RBR adverb, comparative better
RBS adverb, superlative best
RP particle give up
TO to go 'to' the store.
UH interjection errrrrrrrm
VB verb, base form take
VBD verb, past tense took
VBG verb, gerund/present participle taking
VBN verb, past participle taken
VBP verb, sing. present, non-3d take
VBZ verb, 3rd person sing. present takes
WDT wh-determiner which
WP wh-pronoun who, what
WP$ possessive wh-pronoun whose
WRB wh-abverb where, when
IN Preposition or subordinating conjunction, in
'''
