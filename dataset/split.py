import csv
import time

import numpy as np

csv.field_size_limit(2147483647)
CWE = ["CWE-787","CWE-79","CWE-89","CWE-20","CWE-125","CWE-78","CWE-416","CWE-22","CWE-352","CWE-434","CWE-476",]
def chunk_csv(file1, chunk_size=500):
    """
    输入csv文件、一片包含的数量（可自定义）

    """
    chunk = []  # 存放500个数据，每个是一行，dict格式
    id_list = []
    time_start = time.time()
    small787 = open(CWE[0]+'.csv', mode='a', encoding="utf=8", newline='')
    small79 = open(CWE[1]+'.csv', mode='a', encoding="utf=8", newline='')
    small89 = open(CWE[2]+'.csv', mode='a', encoding="utf=8", newline='')
    small20 = open(CWE[3]+'.csv', mode='a', encoding="utf=8", newline='')
    small125 = open(CWE[4]+'.csv', mode='a', encoding="utf=8", newline='')
    small78 = open(CWE[5]+'.csv', mode='a', encoding="utf=8", newline='')
    small416 = open(CWE[6]+'.csv', mode='a', encoding="utf=8", newline='')
    small22 = open(CWE[7]+'.csv', mode='a', encoding="utf=8", newline='')
    small352 = open(CWE[8]+'.csv', mode='a', encoding="utf=8", newline='')
    small434 = open(CWE[9]+'.csv', mode='a', encoding="utf=8", newline='')
    small476 = open(CWE[10]+'.csv', mode='a', encoding="utf=8", newline='')
    cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    with open(file1, "r", newline="", encoding="utf-8") as csvfile:
        csvreader = csv.reader(csvfile)
        fieldnames = ["no","Access Gained","Attack Origin","Authentication Required","Availability","CVE ID","CVE Page","CWE ID","Complexity","Confidentiality","Integrity","Known Exploits","Publish Date","Score","Summary","Update Date","Vulnerability Classification","add_lines","codeLink","commit_id","commit_message","del_lines","file_name","files_changed","func_after","func_before","lang","lines_after","lines_before","parentID","patch","project","project_after","project_before","vul","vul_func_with_fix"]  # 列名
        csv_writer787 = csv.DictWriter(small787, fieldnames=fieldnames)
        csv_writer79 = csv.DictWriter(small79, fieldnames=fieldnames)
        csv_writer89 = csv.DictWriter(small89, fieldnames=fieldnames)
        csv_writer20 = csv.DictWriter(small20, fieldnames=fieldnames)
        csv_writer125 = csv.DictWriter(small125, fieldnames=fieldnames)
        csv_writer78 = csv.DictWriter(small78, fieldnames=fieldnames)
        csv_writer416 = csv.DictWriter(small416, fieldnames=fieldnames)
        csv_writer22 = csv.DictWriter(small22, fieldnames=fieldnames)
        csv_writer352 = csv.DictWriter(small352, fieldnames=fieldnames)
        csv_writer434 = csv.DictWriter(small434, fieldnames=fieldnames)
        csv_writer476 = csv.DictWriter(small476, fieldnames=fieldnames)

        csv_writer787.writeheader()
        csv_writer79.writeheader()
        csv_writer89.writeheader()
        csv_writer20.writeheader()
        csv_writer125.writeheader()
        csv_writer78.writeheader()
        csv_writer416.writeheader()
        csv_writer22.writeheader()
        csv_writer352.writeheader()
        csv_writer434.writeheader()
        csv_writer476.writeheader()

        csv_writer = [csv_writer787,csv_writer79,csv_writer89,csv_writer20,csv_writer125,csv_writer78,csv_writer416,csv_writer22,csv_writer352,csv_writer434,csv_writer476]

        for row in csvreader:
            if row[7] not in CWE:
                continue
            try:
                index = CWE.index(row[7])
                tmpdict = {fieldnames[idx]: row[idx] for idx in range(len(fieldnames))}
                id_ = row[0]
                csv_writer[index].writerow(tmpdict)
                cnt[index] += 1
                # print(tmpdict)
            except Exception as e:
                continue
        end_time = time.time()
        print('{}文件处理完成共耗时：{}'.format(file1, end_time - time_start))
        print(cnt)

chunk_csv("./MSR_data_cleaned.csv")