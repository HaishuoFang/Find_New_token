'''
Created on 2018年7月10日

@author: fanghaishuo
'''
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import re
class FindNewToken(object):
    def __init__(self,txt_path,write_path = r'../data/all_token.txt',min_count=25,token_length=4,min_proba={2:5,3:25,4:125}):
        self.txt_path = txt_path
        self.min_count = min_count
        self.token_length = token_length
        self.min_proba = min_proba
        self.write_path = write_path
        self.read_text()
        self.statistic_ngrams()
        self.filter_ngrams()
        self.sentences_cut()
        self.judge_exist()
        self.write()

    def read_text(self):
        print("reading text！")
        with open(txt_path,encoding='utf-8') as f:
            texts = f.readlines()
        texts = list(map(lambda x:x.strip(),texts))
        self.texts = list(map(lambda x:re.sub('[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）0-9a-zA-Z]+',"",x),texts))
        print(self.texts[0:2])

    def statistic_ngrams(self): #粗略统计1，2..ngrams
        print('Starting statistic ngrams!')
        ngrams = defaultdict(int)
        for txt in self.texts:
            for char_id in range(len(txt)):
                for step in range(1,self.token_length+1): 
                    if char_id+step <=len(txt):
                        ngrams[txt[char_id:char_id+step]] += 1
        self.ngrams = {k:v for k,v in ngrams.items() if v>=self.min_count}
        # print("ngrams:",self.ngrams)
        
    def calculate_prob(self,token): #计算2grams及以上的凝固度 
        self.total = sum([v for k,v in self.ngrams.items() if len(k)==1])
        if len(token)>=2:
           score = min([self.total*self.ngrams[token]/(self.ngrams[token[:i+1]]*self.ngrams[token[i+1:]])  for i in range(len(token)-1)]) 
           if score > self.min_proba[len(token)]:
               return True
        else:
            return False
    
    def filter_ngrams(self):#过滤凝固度小于设定阈值的词
        self.ngrams_ = set(token for token in self.ngrams if self.calculate_prob(token))
        # print("根据凝固度过滤的:",self.ngrams_)

    def cut_sentence(self,txt):
        mask =np.zeros(len(txt)-1)#从第二个字开始标注
        for char_id in range(len(txt)-1):
            for step in range(2,self.token_length+1):
                if txt[char_id:char_id+step] in self.ngrams_:
                    mask[char_id:char_id+step-1] += 1
        sent_token = [txt[0]]
        for index in range(1,len(txt)):
            if mask[index-1]>0:
                sent_token[-1] += txt[index]
            else:
                sent_token.append(txt[index])

        return (txt,sent_token)
    
    def sentences_cut(self):
        self.sentences_tokens = []
        all_tokens = defaultdict(int)
        for txt in self.texts:
            if len(txt)>2:
                for token in self.cut_sentence(txt)[1]:
                    all_tokens[token] +=1
                self.sentences_tokens.append(self.cut_sentence(txt))
        self.all_tokens = {k:v for k,v in all_tokens.items() if v >=self.min_count}
    
    def is_real(self,token):
        if len(token)>=3:
            for i in range(3,self.token_length+1):
                for j in range(len(token)-i+1):
                    if token[j:j+i] not in self.ngrams_:
                        return False
            return True
        else:
            return True
    
    def judge_exist(self):
        self.pairs = []##按照句子-token  进行显示
        for sent,token in self.sentences_tokens:
            real_token = []
            for tok in token:
                if self.is_real(tok) and len(tok)!=1:
                    real_token.append(tok)
            self.pairs.append((sent,real_token))
        
        self.new_word = {k:v for k,v in self.all_tokens.items() if self.is_real(k)}
    
    def statistic_token(self):#统计发现的新词的个数
        count = defaultdict(int)
        length = list(map(lambda x:len(x),self.new_word.keys()))
        for i in length:
            count[i] +=1
        print("每个词的字符串长度的个数统计：",count)
        
                
    def write(self):
        with open(self.write_path,'w',encoding='utf-8') as f:
            # for key in tqdm(self.new_word):
            #     if len(key)!= 1:  
            #         f.write(key+'\n')
            for sent,token in self.pairs:
                f.write(sent+','+','.join(token)+'\n')
    



if __name__ =='__main__':
    txt_path = r'../data/corpus.txt'
    findtoken = FindNewToken(txt_path)
    findtoken.statistic_token()

        
        
