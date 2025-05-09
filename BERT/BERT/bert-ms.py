from bert_serving.server import BertServer
from bert_serving.server.helper import get_run_args
import tensorflow as tf


if __name__ == '__main__':
    args = get_run_args()
    server = BertServer(args)
    server.start()
    server.join()
