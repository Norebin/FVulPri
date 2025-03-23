import csv
import time

import numpy as np

csv.field_size_limit(2147483647)
CWE = ["CWE-787","CWE-79","CWE-89","CWE-20","CWE-125","CWE-78","CWE-416","CWE-22","CWE-352","CWE-476"]
def chunk_csv(file1=CWE[9]+'.csv', chunk_size=500):
    """
    输入csv文件、一片包含的数量（可自定义）

    """
    chunk = []  # 存放500个数据，每个是一行，dict格式
    id_list = []
    time_start = time.time()

    cnt = {}
    filecnt = 0

    with open(file1, "r", newline="", encoding="utf-8") as csvfile:
        csvreader = csv.reader(csvfile)

        for row in csvreader:
            if row[5] == 'CVE ID':
                continue
            try:
                index = row[5]
                fn = row[22]
                if index in cnt:
                    if fn in cnt[index]:
                        cnt[index][fn] += 1
                    else:
                        filecnt += 1
                        cnt[index][fn] = 1
                else:
                    cnt[index] = {}
                    cnt[index][fn] = 1
                    filecnt += 1
                # print(tmpdict)
            except Exception as e:
                continue
        end_time = time.time()
        print('{}文件处理完成共耗时：{}'.format(file1, end_time - time_start))
        print(cnt)
        print(len(cnt), filecnt)
        # sum = 0
        # for cve in cnt:
        #     sum+=len(cnt[cve])
        # print(sum)
        fncnt = {}
        for cve in cnt:
            for file in cnt[cve]:
                if cnt[cve][file] in fncnt:
                    fncnt[cnt[cve][file]] += 1
                else:
                    fncnt[cnt[cve][file]] = 1
        print(fncnt)
        small = open('file-fn.csv', mode='a', encoding="utf=8", newline='')
        fieldnames = ["file-fn-nums", "cnt"]
        csvwriter = csv.DictWriter(small, fieldnames = fieldnames)
        csvwriter.writeheader()
        for cnt in fncnt:
            tmp = {"file-fn-nums": cnt, "cnt": fncnt[cnt]}
            csvwriter.writerow(tmp)

chunk_csv()