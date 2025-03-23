import csv
import json
import random
from random import shuffle

import numpy
csv.field_size_limit(2147483647)
from bert_serving.client import BertClient
IP = 'localhost' # 在本机调用服务
bc = BertClient(ip = IP, check_version = False, check_length = False)


CWEList = ["CWE-787","CWE-79","CWE-89","CWE-20","CWE-125","CWE-78","CWE-416","CWE-22","CWE-352","CWE-476"]
# CWEList = ["CWE-352"]

if __name__ == "__main__":
    writelist = []
    cvelist={}
    for CWE in CWEList:
        file1 = '../' + CWE + '.csv'
        with open(file1, "r", newline="", encoding="utf-8") as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                if row[5] == 'CVE ID':
                    continue
                try:
                    if row[5] not in cvelist:
                        cvelist[row[5]] = {
                            'score': row[13],
                            'desc': row[14],
                            'mess': row[20]
                        }
                except Exception as e:
                    continue
    print(len(cvelist))
    dict_key_ls = list(cvelist.keys())
    random.shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = cvelist.get(key)

    cnt = 0
    tranlist = open('./trainlist.txt', 'a+')
    testlist = open('./testlist.txt', 'a+')
    print(new_dic)
    for cve in new_dic:
        if new_dic[cve]['desc']=='':
            desc_feature = numpy.zeros((1,768),dtype = float)
        else:
            desc_feature = bc.encode([new_dic[cve]['desc']])
        if new_dic[cve]['mess']=='':
            mess_feature = numpy.zeros((1,768),dtype = float)
        else:
            mess_feature = bc.encode([new_dic[cve]['mess']])
        res = {}
        res["desc_feature"] = desc_feature.tolist()
        res["mess_feature"] = mess_feature.tolist()
        score = int(float(new_dic[cve]['score']))
        if score < 10:
            score = score+1
        targetfilePath = './data/'+ cve +'.json'
        targetfile= open(targetfilePath, 'w+')
        print(res)
        targetfile.write(str(json.dumps(res)))
        targetfile.close()
        cnt+=1
        if cnt <=770:
            tranlist.write('../BERT/data/'+ cve +'.json' +' '+str(score) + '\n')
        else:
            testlist.write('../BERT/data/'+ cve +'.json' +' '+str(score) + '\n')

    tranlist.close()
    testlist.close()
