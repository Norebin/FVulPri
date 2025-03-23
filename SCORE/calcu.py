import csv
import math
import re
import json
import random
from random import shuffle
import numpy

import numpy
csv.field_size_limit(2147483647)

CWEList = {"CWE-787":[],"CWE-79":[],"CWE-89":[],"CWE-20":[],"CWE-125":[],"CWE-78":[],"CWE-416":[],"CWE-22":[],"CWE-476":[]}
# CWEList = ["CWE-352"]

if __name__ == "__main__":
    # all_score= open('score_all3.csv', mode='a', encoding="utf=8", newline='')
    fieldnames = ["CWEid", "CVEid", "filename", "func", "funname", "Complexity", "Attack Origin","Authentication Required", "description", "commit_message", "CVSS", "Availability", "Confidentiality", "Integrity", "Vulnerability Classification","LeaderRank","ACVSS","LoC","num_parm","call","fan-in","fan-out","final",'nofun']  # 列名
    # csv_writer = csv.DictWriter(all_score, fieldnames=fieldnames)
    # csv_writer.writeheader()
    file1 = 'score_all2 [3].csv'
    stat_LR=0
    jiegou = 0
    diaoyong = 0
    cnt1 =0
    cnt2= 0
    our = [0,0,0,0,0,0,0,0,0,0,0]
    cvss = [0,0,0,0,0,0,0,0,0,0,0]
    ourlist = []
    cvsslist = []

    with open(file1, "r", newline="", encoding="ISO-8859-1") as csvfile:
        csvreader = csv.reader(csvfile)
        Complexity = {
            'High': 1,
            'Medium': 2,
            'Low': 3
        }
        Attack_Origin = {
            'Remote': 3,
            'Local Network': 2,
            'Local': 1
        }
        Authentication_Required = {
            'Single system': 2,
            'Not required':3
        }
        Availability = {
            'None': 1,
            'Partial': 2,
            'Complete': 3
        }
        Confidentiality = {
            'None': 1,
            'Partial': 2,
            'Complete': 3
        }
        Integrity = {
            'None': 1,
            'Partial': 2,
            'Complete': 3
        }
        Vulnerability_Classification = {
                'Dir. Trav.':3.5,
                'DoS':0.5,
                'Sql':5,
                'Exec Code':2.1,
                'Overflow':0.9,
                'Mem. Corr.':0.3,
                'Priv':0.8,
                'CSRF':1.6,
                'XSS':1.4,
                'Info':0.3,
                'Bypass':0.4,
                'Http R.Spl.':0.4,
                'File Inclusion':3.5,
                '#NAME?': 0.8
                }
        try:
            for row in csvreader:
                if row[1] == 'CVEid':
                    continue
                try:
                    row.append('')
                    row.append('')
                    tmpdict = {fieldnames[idx]: row[idx] for idx in range(len(fieldnames))}
                    type = tmpdict["Vulnerability Classification"]
                    comp = tmpdict["Complexity"]
                    origin = tmpdict["Attack Origin"]
                    auth = tmpdict["Authentication Required"]
                    LR = tmpdict["LeaderRank"]
                    ACVSS = tmpdict["ACVSS"]
                    LoC = tmpdict["LoC"]
                    num_parm = tmpdict["num_parm"]
                    call = tmpdict["call"]
                    fanin = tmpdict["fan-in"]
                    fanout = tmpdict["fan-out"]
                    avail = tmpdict["Availability"]
                    conf = tmpdict["Confidentiality"]
                    inte = tmpdict["Integrity"]

                    type_score = 0
                    for cla in Vulnerability_Classification:
                        if cla in type:
                            type_score = max(type_score, Vulnerability_Classification[cla])

                    comp_score = 0
                    if comp in Complexity:
                        comp_score = Complexity[comp]

                    origin_score = 0
                    if origin in Attack_Origin:
                        origin_score = Attack_Origin[origin]


                    auth_score = 0
                    if auth in Authentication_Required:
                        auth_score = Authentication_Required[auth]

                    LR_score = float(LR)
                    # stat_LR = max(stat_LR, LR_score)
                    if LR_score>=2:
                        LR_score = 2
                    

                    ACVSS_score = float(ACVSS)

                    LoC_score = float(LoC)
                    num_parm_score = float(num_parm)
                    tmp1 = 100/(10*num_parm_score+LoC_score)*(1/9.09)
                    # jiegou = max(jiegou,tmp1)

                    call_score = float(call)
                    fanin_score = float(fanin)
                    fanout_score = float(fanout)
                    tmp2 = call_score + fanout_score - fanin_score
                    if tmp2>40:
                        tmp2 = 0.1
                    elif tmp2>25:
                        tmp2 = 0.2
                    elif tmp2>17:
                        tmp2 = 0.3
                    elif tmp2>12:
                        tmp2 = 0.4
                    elif tmp2>7:
                        tmp2 = 0.5
                    elif tmp2>5:
                        tmp2 = 0.6
                    elif tmp2>4:
                        tmp2 = 0.7
                    elif tmp2>3:
                        tmp2 = 0.8
                    elif tmp2>2:
                        tmp2 = 0.9
                    else:
                        tmp2 = 1

                    # if tmp2>=0:
                    #     cnt+=1
                    # diaoyong = max(diaoyong, tmp2)

                    avail_score = 0
                    if avail in Availability:
                        comp_score = Availability[avail]

                    conf_score = 0
                    if conf in Availability:
                        conf_score = Availability[conf]

                    inte_score = 0
                    if inte in Integrity:
                        inte_score = Integrity[inte]

                    final = (type_score+(comp_score+origin_score+auth_score)/3+LR_score+ACVSS_score)/4+tmp1+tmp2+(avail_score+conf_score+inte_score)/3
                    nofun = (type_score+(comp_score+origin_score+auth_score)/3+ACVSS_score)/4+(avail_score+conf_score+inte_score)/3
                    # print(final)
                    tmpdict["final"] = str(final)
                    tmpdict["nofun"] = str(nofun*4/3)
                    # csv_writer.writerow(tmpdict)

                    CWEID = row[0]
                    CWEList[CWEID].append(float(tmpdict["CVSS"]))
                    CVSS = float(tmpdict["CVSS"])
                    our[math.floor(final)]+=1
                    cvss[math.floor(nofun*4/3)]+=1
                    ourlist.append(final)
                    cvsslist.append(nofun*4/3)
                    # CWEList[CWEID].append(final)


                except Exception as e:
                    continue
        except Exception as e:
            print(row)
            print(e)

    print(our,cvss)
    print(numpy.mean(ourlist),numpy.mean(cvsslist))
    # print(CWEList)
    # for CWE in CWEList:
    #     print(CWE, numpy.mean(CWEList[CWE]))

