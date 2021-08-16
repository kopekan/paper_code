import platform
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
from opencc import OpenCC
import warnings
def Space2Tab(data):
    for i in range(len(data)):
        if data[i]!='\n':
            data[i] = data[i].replace(' ','\t')
    return data
def transferWeibo(data):
    for i in range(len(data)):
        if data[i]!='\n':
            if data[i][1] in '0123456789':
                data[i] = data[i].replace(data[i][1], '')
            if len(data[i])>8:
                data[i]=data[i][:-4]
    return data

def read_data(fh):
    #  in windows the new line is '\r\n\r\n' the space is '\r\n' . so if you use windows system,
    #  you have to use recorsponding instructions
    if platform.system() == 'Windows':
        split_text = '\r\n'
    else:
        split_text = '\n'
        
    # 以字符串读取，全文都在一个字符串上“'从 O\n效益 O\n上 O\n来 O\n看 O\n...”
    string = fh.read().decode('utf-8')
    rows_data = [row.replace("#","") for row in string.strip().split(split_text)]
    rows_data = Space2Tab(rows_data)
    # 分割句子
    sentence_data = []
    sentence_tmp= []
    for row in rows_data:
        if row.strip():
            if row.endswith('\r'):
                sentence_tmp.append(row[:-1])
            else:
                sentence_tmp.append(row)
        else:            
            sentence_data.append(transferWeibo(sentence_tmp))
#             sentence_data.append(sentence_tmp)
            sentence_tmp = []
    if len(sentence_tmp)>0:
        sentence_data.append(transferWeibo(sentence_tmp))

    fh.close()
    return sentence_data

def _process_data(data, tags, maxlen=128, use_bert=False, char_emb=None, mylabel2idx=None, toTaiwan=True, embed='CKIP'):
    tag_split_text = '\t'
    cc = OpenCC('s2tw')
    special_words = ['<PAD>', '<UNK>'] # 特殊词表示
    if embed=='CKIP':
        char_vocabs = list(np.load('CKIP/token_list.npy'))
    else:
        with open('zh-nlp-demo/data/char_vocabs.txt', "r", encoding="utf8") as fo:
            char_vocabs = [line.strip() for line in fo]
    char_vocabs = special_words + char_vocabs
    char_vocabs.sort()
    idx2word = {idx: char for idx, char in enumerate(char_vocabs)}
    word2idx = {char: idx for idx, char in idx2word.items()}
    # set to <unk> (index end) if not in vocab
    if use_bert:
        warnings.simplefilter('ignore')
        temp_x, temp_y = trans2BERT_data(BERT_pre_add_notation(data))
        y = [np.array([tags.index(c) for c in s]) for s in temp_y]
        x=temp_x
    else:
        if toTaiwan:
            x = [np.array([word2idx.get(cc.convert(row.split(tag_split_text)[0]), word2idx['<UNK>']) for row in s])for s in data]
        else:
            x = [np.array([word2idx.get(row.split(tag_split_text)[0], word2idx['<UNK>']) for row in s])for s in data]
        x = pad_sequences(x, maxlen=maxlen, value=word2idx['<PAD>'], padding='post', truncating="post")
        y = [np.array([tags.index(row.split(tag_split_text)[1]) for row in s]) for s in data]

    origin_y = y
    y = pad_sequences(y, maxlen=maxlen, value = mylabel2idx['O'], padding='post', truncating="post")
    return x, list(y), origin_y

def load_data(train_dest, test_dest, use_bert=False, maxlen=128, char_emb = None, mylabel2idx = None, toTaiwan=True, embed='CKIP'):
    tag_split_text = '\t'
    train = read_data(open(train_dest, 'rb'))
    test = read_data(open(test_dest, 'rb'))
    if mylabel2idx==None:
        mylabel2idx, _ = get_idxlabel(train, test)
    for data in range(len(train)):
        for row in range(len(train[data])):
            if train[data][row].startswith('\t\t'): #'\t\t'
                train[data][row]=' '+train[data][row][1:]                
            elif train[data][row].split(tag_split_text)[1]=='':
                train[data][row]+='O'
            if train[data][row].endswith('\r'):
                train[data][row]=train[data][row][:-1]
    for data in range(len(test)):
        for row in range(len(test[data])):
            if test[data][row].startswith('\t\t'): #'\t\t'
                test[data][row]=' '+test[data][row][1:]                
            elif test[data][row].split(tag_split_text)[1]=='':
                test[data][row]+='O'
            if test[data][row].endswith('\r'):
                test[data][row]=test[data][row][:-1]

    word_counts = Counter(word.split(tag_split_text)[0].lower() for sentence in train for word in sentence)
    
    tags = list(mylabel2idx.keys())

    train = _process_data(train, tags, use_bert=use_bert, maxlen=maxlen, char_emb=char_emb, mylabel2idx=mylabel2idx, toTaiwan=toTaiwan, embed=embed)
    if train_dest==test_dest:
        test = train
    else:
        test = _process_data(test, tags, use_bert=use_bert, maxlen=maxlen, char_emb=char_emb, mylabel2idx=mylabel2idx, toTaiwan=toTaiwan, embed=embed)

    return train,test, tags

import tensorflow_hub as hub
import model.tokenization as tokenization

def BERT_pre_add_notation(NER_data):    
    notation_dictionary = {'PB':'≦', 'PN':'▼', 'OW':'↗', 'AT': '‥', 'PER':'—', 'ORG':'÷', 'LOC':'﹍', 'Si':'↘', 'GPE':'╳'}
    notation_list = [value for key, value in notation_dictionary.items()]
    outputdata = []
    for s in NER_data:
        temp_s = []
        for row in s:
            c = row.split('\t')[0]
            t = row.split('\t')[1]
            if c not in notation_list:
                if t.startswith('B'):
                    temp_s.append(' '+notation_dictionary[t.split('-')[1]]+' '+'\tO')
                    temp_s.append(row)
                elif t.startswith('E'):
                    temp_s.append(row)
                    temp_s.append(' '+notation_dictionary[t.split('-')[1]]+' '+'\tO')
                elif t.startswith('S'):
                    temp_s.append(' '+notation_dictionary[t.split('-')[1]]+' '+'\tO')
                    temp_s.append(row)
                    temp_s.append(' '+notation_dictionary[t.split('-')[1]]+' '+'\tO')
                else:
                    temp_s.append(row)
        outputdata.append(temp_s)
    return outputdata

def trans2BERT_data(BERT_pre_data): #一堆字串
    #tokenize
    tag_split_text='\t'
    cc = OpenCC('s2tw')
    BERT_LAYER = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/3', trainable=False)
    VOCAB_FILE = BERT_LAYER.resolved_object.vocab_file.asset_path.numpy()
    #DO_LOWER_CASE = BERT_LAYER.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(VOCAB_FILE, True) 
    outputx={'input_word_ids':[], 'input_mask':[], 'input_type_ids':[]}
    outputy = []
    sentencedata = [cc.convert(''.join([row.split(tag_split_text)[0] for row in s])) for s in BERT_pre_data]
    for sentence in sentencedata:
        temp_s = tokenizer.tokenize(sentence)
        temp_s, temp_y = return_bert_data(temp_s)
        temp_word, temp_mask, temp_type = return_word_mask_type(temp_s, tokenizer)
        outputx['input_word_ids'].append(temp_word)
        outputx['input_mask'].append(temp_mask)
        outputx['input_type_ids'].append(temp_type)
        outputy.append(temp_y)
    return outputx, outputy

def return_word_mask_type(inputx, tokenizer):
    tempword=[]
    tempmask=[]
    temptype=[]
    for x in inputx:
        try:
            token = tokenizer.convert_tokens_to_ids([x.lower()])
        except:
            token = tokenizer.convert_tokens_to_ids(['[UNK]'])
        tempword.append(token[0])
        tempmask.append(1)
        temptype.append(0)
    while len(tempword)<128:
        tempword.append(0)
        tempmask.append(0)
        temptype.append(0)
    if len(tempword)>128:
        tempword=tempword[:128]
        tempmask=tempmask[:128]
        temptype=temptype[:128]
    return tempword, tempmask, temptype

def return_bert_data(sentence_tokenized):
    notation_dictionary = {'PB':'≦', 'PN':'▼', 'OW':'↗', 'AT': '‥', 'PER':'—', 'ORG':'÷', 'LOC':'﹍', 'Si':'↘', 'GPE':'╳'}
    outputy = []
    outputx = []
    lastflag ='O'
    nowflag = 'O'
    ne = ''
    lastne=''
    begin_flag = False
    for c in sentence_tokenized:
        flag = True
        for key, value in notation_dictionary.items():
            if value in c:                    
                flag=False
                begin_flag = True
                if nowflag=='O':
                    ne=key
                    nowflag='B-'
                else:
                    nowflag='O'
                break
        if flag:
            outputx.append(c)
            if nowflag!='O' and lastflag!='O' and ne==lastne:
                if not begin_flag:
                    nowflag='I-'
            if nowflag == 'O':
                ne = ''
            begin_flag=False
            outputy.append(nowflag+ne)
            lastne=ne
            lastflag=nowflag
    return outputx, addBIOES(outputy)

def addBIOES(y_list):
    tagging = False
    for y in range(len(y_list)):
        if y_list[y]!='O':
            #S
            if y_list[y][0]=='B':
                if (y==len(y_list)-1) or (y_list[y+1][0]!='I'):            
                    y_list[y]='S'+y_list[y][1:]
            #E
            elif y_list[y][0]=='I':
                if (y==len(y_list)-1) or (y_list[y+1][0]!='I'):
                    y_list[y]='E'+y_list[y][1:]
                    
#             if tagging==False: #B / S
#                 if y==len(y_list)-1 or y_list[y+1]!=y_list[y]:
#                     y_list[y]='S-'+y_list[y]
#                 else:
#                     y_list[y]='B-'+y_list[y]
#                     tagging=True
#             else: # I / E
#                 if y==len(y_list)-1 or y_list[y+1]!=y_list[y]:
#                     y_list[y]='E-'+y_list[y]
#                     tagging=False
#                 else:
#                     y_list[y]='I-'+y_list[y]
#         else:
#             tagging=False
    return y_list

#read label
def get_idxlabel(train, test, read=True):
    if read==False: #輸入的是位置，不是read後的資料
        train = read_data(open(train, 'rb'))
        test = read_data(open(test, 'rb'))
        
    tag_split_text = "\t"
    for sentence in range(len(train)):
        for c in range(len(train[sentence])):
            if train[sentence][c].split(tag_split_text)[1]=='':
                train[sentence][c]=' '+train[sentence][c][1:]
    for sentence in range(len(test)):
        for c in range(len(test[sentence])):
            if test[sentence][c].split(tag_split_text)[1]=='':
                test[sentence][c]=' '+test[sentence][c][1:]
    label_list_all = [row.split(tag_split_text)[1] for s in train for row in s ]+[row.split(tag_split_text)[1] for s in test for row in s ]
    label_list = list(set(label_list_all))
    for i in range(len(label_list)):
        if len(label_list[i])>5:
            label_list[i]=label_list[i][:-4]
        if label_list[i].endswith('\r'):
            label_list[i]=label_list[i][:-1]
        if len(label_list[i])>1:
            label_list[i]=label_list[i][2:]
    label_list_temp = list(set(label_list))
    label_list = []
    for label in label_list_temp:
        if len(label)>1:
            label_list.append('B-'+label)
            label_list.append('I-'+label)
            label_list.append('E-'+label)
            label_list.append('S-'+label)
    label_list.append('O')
    label_list.sort()
    try:
        label_list.remove('')
    except:
        pass

    label2idx = {char: idx for idx, char in enumerate(label_list)}
    idx2label = {idx: label for label, idx in label2idx.items()}
    return label2idx, idx2label

def get_pretrain_char_emb(mode='CKIP'):
    if mode=='CKIP':
        ckip_token = list(np.load('CKIP/token_list.npy'))
        char_vocabs = ckip_token.copy()
        ckip_vector = list(np.load('CKIP/vector_list.npy'))
        char_embeddings = {}
        if len(ckip_token)!=len(ckip_vector):
            print('CKIP token size not equal to CKIP vector size!')
            return None
        for emb in range(len(ckip_token)):
            char_embeddings[ckip_token[emb]]=ckip_vector[emb]
    elif mode=='Lattice':
        with open('zh-nlp-demo/data/char_vocabs.txt', "r", encoding="utf8") as fo:
            char_vocabs = [line.strip() for line in fo]
        with open('./NER_data/embedding/gigaword_chn.all.a2b.uni.ite50.vec') as emb:
                Emb = emb.readlines()
        char_embeddings = {}
        for emb in Emb:    
            tempemb = emb.split(' ')
            for i in range(50):
                tempemb[i+1]=float(tempemb[i+1])
            char_embeddings[tempemb[0]]=tempemb[1:-1]
    special_words = ['<PAD>', '<UNK>'] # 特殊词表示
    char_vocabs = special_words + char_vocabs
    char_vocabs.sort()
    idx2word = {idx: char for idx, char in enumerate(char_vocabs)}
    word2idx = {char: idx for idx, char in idx2word.items()}
    return idx2word, word2idx, char_embeddings