import PyPDF2
import nltk
from nltk import pos_tag,word_tokenize
import enchant
from tabula import read_pdf
from nameparser.parser import HumanName
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

pdfFileObj = open("/home/user/Downloads/deep resume.pdf", "rb")
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
pageObj = pdfReader.getPage(0)
pdf_data = pageObj.extractText()

print(pdf_data.split())
tab_pdf = read_pdf("/home/user/Downloads/deep resume.pdf",output_format='json')

print(type(pdf_data))

# print(tab_pdf)
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
text = ("Diwakar Shukla is good man. and Pratik Kalathiya is good developer. Deep Kothadiya is good at AI")
matcher = Matcher(nlp.vocab)
print(type(text))
def extract_name(resume_text):
    try:
        nlp_text = nlp(resume_text)
    except:
        print("Error")
    print(matcher, "matcher")
    matcher.add("HelloWorld", None, [{'POS': 'PROPN'}, {'POS': 'PROPN'}])
    matches = matcher(nlp_text)
    for match_id, start, end in matches:
        span = nlp_text[start:end]
        print(span.text)

extract_name(pdf_data)

def remove_wild(tag):
    for wild in ['@', ':', ';', ',', '.', '?', '/','-','(',')','+']:
        # print(wild,"wild")
        if wild in tag:
            # print("tag", tag,wild)
            check_i = tag.strip(wild)
            return check_i
    else:
        return tag

def get_human_names(text):
    tokens = nltk.tokenize.word_tokenize(text)
    pos_trim = []
    for i in tokens:
        # print(i)
        trim = remove_wild(i)
        # print(trim)
        if trim:
            pos_trim.append(trim)
    # print(pos_trim)
    pos = pos_tag(pos_trim)

    sentt = nltk.ne_chunk(pos, binary = False)
    person_list = []
    person = []
    name = ""
    # print(pos)
    for k in range(len(pos)):
        porce = [pos[k]]
        # print(porce)
        for i,j in porce:
            if j == 'NNP':
                d = enchant.Dict("en_US")
                # print(d.check("%s" % i))
                check_i = i

                if check_i:
                    checker = d.check("%s" % check_i)
                    if not checker and i is not ['@',':',';',',','.','?','/','-']:
                        check_porce = pos[k-2]
                        check_up_porce = pos[k-1]
                        # print(check_porce)
                        # print(check_up_porce)
                        # print("----break----")
    return (person_list)
#
names = get_human_names(pdf_data)
# print("LAST, FIRST")
for name in names:
    last_first = HumanName(name).last + ', ' + HumanName(name).first + ', ' + HumanName(name).middle
    # print("here",last_first)

import re
import spacy
from nltk.corpus import stopwords

# load pre-trained model
nlp = spacy.load('en_core_web_sm')

# Grad all general stop words
STOPWORDS = set(stopwords.words('english'))

# Education Degrees
EDUCATION = [
            'BE','B.E.', 'B.E', 'BS', 'B.S',
            'ME', 'M.E', 'M.E.', 'MS', 'M.S',
            'BTECH', 'B.TECH', 'M.TECH', 'MTECH',
            'SSC', 'HSC', 'CBSE', 'ICSE', 'X', 'XII'
        ]

def extract_education(resume_text):
    nlp_text = nlp(resume_text)

    # Sentence Tokenizer
    nlp_text = [sent.string.strip() for sent in nlp_text.sents]

    edu = {}
    # Extract education degree
    for index, text in enumerate(nlp_text):
        for tex in text.split():
            # Replace all special symbols
            tex = re.sub(r'[?|$|.|!|,]', r'', tex)
            if tex.upper() in EDUCATION and tex not in STOPWORDS:
                edu[tex] = text + nlp_text[index + 1]

    # Extract year
    education = []
    for key in edu.keys():
        year = re.search(re.compile(r'(((20|19)(\d{2})))'), edu[key])
        if year:
            education.append((key, ''.join(year[0])))
        else:
            education.append(key)
    return education

result = extract_education(pdf_data)
print(result)
d = enchant.Dict("en_US")
print(d.check("jack"))

# from nltk.tag.stanford import StanfordNERTagger
# st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz',
#                        'stanford-ner.zip')
# text = pdf_data
# print(nltk.sent_tokenize(text))
# for sent in nltk.sent_tokenize(text):
#     tokens = nltk.tokenize.word_tokenize(sent)
#     print(tokens)
#     tags = st.tag(tokens)
#     for tag in tags:
#         if tag[1]=='PERSON': print(tag)

# for subtree in sentt.subtrees():
#     for leaf in subtree.leaves():
#         person.append(leaf[0])
#     if len(person) > 1:  # avoid grabbing lone surnames
#         print("hello", person)
#         for part in person:
#             print("hey", part)
#             name += part + ' '
#         print("hi", name[:-1], person_list)
#         if name[:-1] not in person_list:
#             person_list.append(name[:-1])
#         name = ''
#     person = []

# print(pos)
# print(sentt)
# for i,j in sentt:
#     print("here:",i,j)
# print(sentt.subtrees(filter= sentt.node=='NN'))
# filter=lambda t: t.node == 'NNP'\

# chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
# chunkParser = nltk.RegexpParser(chunkGram)
# chunked = chunkParser.parse(pos)
# print(chunkParser)
