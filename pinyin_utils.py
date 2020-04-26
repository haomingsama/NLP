from pypinyin import pinyin, lazy_pinyin, Style
import numpy as np
import re


def serperate_pinyin(w):
    '''
    将输入的单个汉字声母，韵母，声调提取出来
    
    input:
    w -- 输入的汉字，必须是单个，如果是多个汉字只会匹配第一个汉字
    
    ouput:
    shengmu -- 该汉字的声母，如果该汉字没有声母，则输出空字符串
    yunmu --  该汉字的韵母，没有声调标注
    tone -- 该汉字的声调，如果是亲生则声调为0
    '''
    #检测输入的是否是中文
    is_chinese = re.compile('[\u4e00-\u9fa5]')
    if not is_chinese.findall(w):
        return '', '',0
    
    #提取声母
    shengmu_list = pinyin(w,style=Style.INITIALS,strict = False) #不然 y， w不会算在里面
    shengmu = shengmu_list[0][0]
    
    
    #提取韵母
    yunmu_list= pinyin(w,style=Style.FINALS,strict = True)   #strict =  True 主要是为了区分 un和vn的区别
    yunmu = yunmu_list[0][0]
    
    #提取声调
    pre_tone = pinyin(w, style=Style.FINALS_TONE3)            #先把韵母和声调一起提取出来 比如 中 => ong1
    pattern=re.compile('[1-4]')                              #用正则表达式匹配其中的数字
    tone_list = pattern.findall(pre_tone[0][0]) 
    if tone_list:
        tone = int(tone_list[0])                                 #如果是轻声则检测不到声调
    else:
        tone = 0              
    return shengmu,yunmu,tone


def get_index(s,y,t,shengmu_to_index,yunmu_to_index):
    '''
    获取声母s，韵母y，音调t在向量中的下标位
    input:
        s -- 表示输入的声母
        y -- 表示输入的韵母
        t -- 表示输入的音调

    output:
        s_index -- 声母在向量中的下标
        y_index -- 韵母在向量中的下标
        t_index -- 音调在向量中的下标
    '''
    if not s: # 如果没有声母的情况
        s_index =None
    else:
        s_index = shengmu_to_index[s]
    
    if not y : #如果没有韵母的情况
        y_index = None
    else:
        y_index = yunmu_to_index[y]
    
    
    if t ==0:  #没有音调的情况
        t_index  =  None
    else:
        t_index = t+len(shengmu_to_index)+len(yunmu_to_index)-1    
    
    return s_index,y_index,t_index


def create_word_phonetics_vector(word, shengmu_to_index,yunmu_to_index,vocab_size):

    '''
    对于单个汉字，生成属于他的representation vector
    input：
    word  -- 单个汉字
    shengmu_to_index -- 声母对下标的哈希表
    yunmu_to_index -- 韵母对下标的哈希表
    vocab_size -- 一共有多少汉字
    
    
    output：
    vector -- 代表这个汉字的向量 其dimension 为 （vocab_size+extra_length,1 ）
    '''
    extra_length = len(shengmu_to_index)+len(yunmu_to_index)+4  # 除了one-hot部分的长度

    vector = np.zeros([vocab_size+extra_length,1])
    
    if word == None: return vector    #如果输入的是none，直接输出全是0的vector
    
    s,y,t = serperate_pinyin(word) #声母, 韵母， 音调提取
    s_index,y_index,t_index = get_index(s,y,t,shengmu_to_index,yunmu_to_index)
    w = char_to_index[word]

    if s_index:    #代表有声母的情况
        vector[s_index]=1
    if y_index:    #代表有韵母的情况
        vector[y_index]=1
    if t_index:    #代表有声调的情况
        vector[t_index]=1

    vector[w] = 1  # 代表one-hot 的部分
   
    return vector


def extract_names():
    '''
    从名字列表里正则匹配出中文名字,剔除重复的名字
    output:
    names -- 含有所有名字的列表
    '''
    pattern = re.compile('[\u4e00-\u9fa5]+')   
    names = []
    with open('names.txt','r') as f:
        for line in f.readlines(): 
            name = pattern.findall(line)
            if name: names.append(name[0])
    return list(set(names))

def extract_word():
    '''
    将所有组成名字的字都给列出来
    output:
    words -- 所有包含字组成的列表
    
    '''
    data = open('names.txt','r').read()
    data= list(set(data))
    pattern = re.compile('[\u4e00-\u9fa5]')
    words = [pattern.findall(line)[0] for line in data if pattern.findall(line)]
    return words




