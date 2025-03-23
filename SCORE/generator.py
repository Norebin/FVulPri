import csv
import math
import re
import json
import random
from random import shuffle
import numpy

import numpy
csv.field_size_limit(2147483647)

CWEList = ["CWE-787","CWE-79","CWE-89","CWE-20","CWE-125","CWE-78","CWE-416","CWE-22","CWE-352","CWE-476"]
# CWEList = ["CWE-352"]

if __name__ == "__main__":
    LR = numpy.load('../LR/LR.npy',allow_pickle=True).item()
    writelist = []
    all_score= open('score_all2.csv', mode='a', encoding="utf=8", newline='')
    fieldnames = ["CWEid", "CVEid", "filename", "func", "funname", "Complexity", "Attack Origin","Authentication Required", "description", "commit_message", "CVSS", "Availability", "Confidentiality", "Integrity", "Vulnerability Classification","LeaderRank","ACVSS","LoC","num_parm","call","fan-in","fan-out"]  # 列名
    csv_writer = csv.DictWriter(all_score, fieldnames=fieldnames)
    csv_writer.writeheader()
    file1 = 'score_all1.csv'
    filelist = {}


    with open(file1, "r", newline="", encoding="utf-8") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if row[1] == 'CVEid':
                continue
            try:
                fun = row[3]
                tmpdict = {fieldnames[idx]: row[idx] for idx in range(len(fieldnames))}
                funname = re.findall("\s[^\s]+\(", fun)[0].lstrip()
                funname = funname[:-1]
                # if funname == "int f2fs_quota_off":
                # print(funname)
                count = len(re.findall("\n",fun))
                # print(count)
                filename = row[2]
                if filename in filelist:
                    arr = filelist[filename]
                    arr.append(funname)
                    filelist[filename] = arr
                else:
                    filelist[filename] = [funname]
                # tmpdict["funname"] = funname
                # tmpdict["LoC"] = str(count)
                # pram = re.findall("\([^\)]+\)", fun)[0].lstrip()
                # pram = pram.split(',')
                # # print(len(pram))
                # tmpdict["num_parm"] = str(len(pram))
                # try:
                #     tmpdict["LeaderRank"] = LR[funname]
                # except:
                #     tmpdict["LeaderRank"] = str(1)

                # csv_writer.writerow(tmpdict)
            except Exception as e:
                continue
    csvfile.close()
    with open(file1, "r", newline="", encoding="utf-8") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if row[1] == 'CVEid':
                continue
            if len(re.findall('CWE',row[0])) == 0:
                print(row[0])
                continue
            tmpdict = {fieldnames[idx]: row[idx] for idx in range(len(fieldnames))}
            fun = row[3]
            filename = row[2]
            calls = re.findall("[^\s\(]+\(", fun)[1:]
            fanin = 0
            for call in calls:
                call = call[:-1]
                if call in filelist[filename]:
                    fanin += 1
            acvss = math.ceil(float(tmpdict["CVSS"]))
            tmpdict["call"] = str(len(calls))
            tmpdict["fan-in"] = str(fanin)
            tmpdict["fan-out"] = str(len(calls) - fanin)
            tmpdict["ACVSS"] = acvss

            # csv_writer.writerow(tmpdict)
