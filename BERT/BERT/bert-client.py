from bert_serving.client import BertClient
IP = 'localhost' # 在本机调用服务
bc = BertClient(ip = IP, check_version = False, check_length = False)
vector = bc.encode(['In the uncurl_ws_accept function in uncurl.c in uncurl before 0.07, as used in Parsec before 140-3, insufficient Origin header validation'])
print(vector)