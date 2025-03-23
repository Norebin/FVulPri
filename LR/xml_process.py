import xml.etree.ElementTree as ET
import os
import re
import random

with open("/home/hc/DeepWukong/data/sensiAPI.txt", "r", encoding="utf-8") as f:
    sensis = set([api.strip() for api in f.read().split(",")])

if __name__ == "__main__":
    tree = ET.ElementTree(file='LR/edges/_c_w_e-476_8c.xml')
    root = tree.getroot()
    targetfile = writer = open('LR/edgelist.txt', 'a+')
    # nodes = root.findall('compounddef')
    # nodes = nodes[0].findall('sectiondef')
    # nodes = nodes[0].findall('memberdef')
    # print(nodes)
    for node in root.iter('memberdef'):
        refers = node.findall('references')
        if len(refers) == 0:
            continue
        name = node.find('name').text
        # name = name[0].text
        for refer in refers:
            rname = refer.text
            targetfile.write(name+ ' '+ rname + '\n')
            print(name,rname)

    targetfile.close()
