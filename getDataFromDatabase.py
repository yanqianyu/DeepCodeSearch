import redis
import pymysql
import json
import re
import random
import configs
from operator import itemgetter
import collections


def parseInput(sent):
    return [z for z in sent.split(' ')]


def toNum(data, vocab_to_int):
    # 转为编号表示
    res = []
    for z in parseInput(data):
        res.append(vocab_to_int[z])
    return res


def getVocabForOther(datas, vocab_size):
    # 为其他特征生成词表
    vocab = set()
    counts = {}

    vocab_to_int = {}
    int_to_vocab = {}

    for data in datas:
        words = parseInput(data)
        for word in words:
            counts[word] = counts.get(word, 0) + 1
        vocab.update(words)

    _sorted = sorted(vocab, reverse=True, key=lambda x: counts[x])
    for i, word in enumerate(["<PAD>", "<UNK>", "<START>", "<STOP>"] + _sorted):
        if vocab_size is not None and i > vocab_size:
            break

        vocab_to_int[word] = i
        int_to_vocab[i] = word

    return vocab_to_int, int_to_vocab


def getVocabForAST(asts, vocab_size):
    # 为ast的type和value生成词表 获得所有的type和value
    vocab = set()
    counts = {}

    vocab_to_int = {}
    int_to_vocab = {}

    for ast in asts:
        for node in ast:
            if "type" in node.keys():
                counts[node["type"]] = counts.get(node["type"], 0) + 1
                vocab.update([node["type"]])
            # code2seq中path不包括value
            # if "value" in node.keys():
            #     counts[node["value"]] = counts.get(counts[node["value"]], 0) + 1
            #     vocab.update(node["value"])

    _sorted = sorted(vocab, reverse=True, key=lambda x: counts[x])
    for i, word in enumerate(["<PAD>", "<UNK>", "<START>", "<STOP>"] + _sorted):
        if vocab_size is not None and i > vocab_size:
            break

        vocab_to_int[word] = i
        int_to_vocab[i] = word

    return vocab_to_int, int_to_vocab


def dfs(ast, root, path, totalpath):
    # 深度遍历 得到多条路径
    if "children" in ast[root["index"]].keys():
        path.append(root["type"])
        for child in root["children"]:
            dfs(ast, ast[child], path, totalpath)
            path.pop()
    else:
        # path.append(root["value"])
        # code2seq中叶节点内容不包含在path中 而是subtoken
        totalpath.append(' '.join(path))
        return


def getNPath(ast, n):
    # 随机得到n条路径
    path = []
    totalpath = []
    dfs(ast, ast[0], path, totalpath)
    nPath = []
    for i in range(n):
        a = random.randint(0, len(totalpath)-1)
        b = random.randint(0, len(totalpath)-1)
        sent = ' '.join(reversed(totalpath[a].split(' ')[1:])) + ' ' + totalpath[b]
        nPath.append(sent)
    return nPath


def getSBT(ast, root):
    # 得到李戈的sbt树 （效果已经在多篇文章里证明不行了）
    cur_root = ast[root["index"]]
    tmp_list = []

    tmp_list.append("(")
    if "value" in cur_root.keys() and len(cur_root["value"]) > 0:
        str = cur_root["type"] + "_" + cur_root["value"] # 没有孩子
    else:
        str = cur_root["type"]
    tmp_list.append(str)
    if "children" in cur_root.keys():
        chs = cur_root["children"]
        for ch in chs:
            tmpl = getSBT(ast, ast[ch])
            tmp_list.extend(tmpl)

    tmp_list.append(")")
    return tmp_list


def getIndex(node):
    return node["index"]


def str2list(ast):
    nodes = []
    while len(ast) > 0:
        idx = ast.find("},")
        if idx == -1:
            idx = ast.find("}")
        node = ast[:idx + 1]

        idx1 = node.find("type")
        if idx1 != -1:
            idx3 = node.find(",", idx1)
            if idx3 == -1:
                idx3 = node.find("}", idx1)
            type = node[idx1 + 6: idx3]
            new_type = '"' + type + '"'
            node = node[0: idx1 + 6] + new_type + node[idx3:]
            # node = node.replace(type, new_type)

        idx2 = node.find("value")
        if idx2 != -1:
            idx4 = node.find(",", idx2)
            if idx4 == -1:
                idx4 = len(node) - 1
                # idx4 = node.find("}", idx2)
            value = node[idx2 + 7: idx4]
            new_value = '"' + value + '"'
            node = node[0: idx2 + 7] + new_value + node[idx4:]
            # node = node.replace(value, new_value)
        nodes.append(json.loads(node))
        # print(node)

        if idx + 2 > len(ast):
            break
        ast = ast[idx + 3:]
    return sorted(nodes, key=getIndex)


def getVocab():
    # 获得几种特征的词表
    # ast是json格式 n是需要抽取的路径数
    connect = pymysql.Connect(
        host="0.0.0.0",
        port=3306,
        user="root",
        passwd="Taylorswift-1997",
        db="githubreposfile",
        charset='utf8'
    )
    cursor = connect.cursor()
    sql = "SELECT * FROM reposfile;"
    cursor.execute(sql)

    data = cursor.fetchall()

    cursor.close()
    connect.close()

    asts = []
    methNames = []
    tokens = []
    descs = []
    rawcodes = []
    apiseqs = []

    datas = []

    for i in range(len(data)):
        id = data[i][0]

        methName = str(data[i][1], encoding="utf-8")
        methNames.append(methName)

        token = str(data[i][2], encoding="utf-8")
        tokens.append(token)

        desc = str(data[i][3], encoding="utf-8")
        descs.append(desc)

        rawcode = str(data[i][4], encoding="utf-8")
        rawcodes.append(rawcode)

        apiseq = str(data[i][5], encoding="utf-8")
        apiseqs.append(apiseq)

        # with open("example.txt", 'r') as f:
        #     ast = f.readlines()
        #     ast = " ".join(ast)

        ast = str(data[i][-1], encoding="utf-8")[1:-1].replace("=", ":").replace("\n", " ")
        # ast = ast[1:-1].replace("=", ":").replace("\n", " ")
        # 这一步替换注
        ast = ast.replace("children:", "\"children\":").replace("index:", "\"index\":").replace("value:", "\"value\":").replace("type:", "\"type\":")
        ast = str2list(ast)
        asts.append(ast)

        info = {}
        info["id"] = id
        info["methName"] = methName
        info["token"] = token
        info["desc"] = desc
        info["apiseq"] = apiseq
        info["ast"] = ast

        datas.append(info)

    cf = configs.conf()

    methName_vocab_to_int, methName_int_to_vocab = getVocabForOther(methNames, cf.n_words)
    token_vocab_to_int, token_int_to_vocab = getVocabForOther(tokens, cf.n_words)
    desc_vocab_to_int, desc_int_to_vocab = getVocabForOther(descs, cf.n_words)
    apiseq_vocab_to_int, apiseq_int_to_vocab = getVocabForOther(apiseqs, cf.n_words)

    # 以上这些特征可以转为编号后重新写入数据库
    connect = pymysql.Connect(
        host="0.0.0.0",
        port=3306,
        user="root",
        passwd="Taylorswift-1997",
        db="githubreposfile",
        charset='utf8'
    )
    cursor = connect.cursor()
    # todo
    sql = """CREATE TABLE numrepresent (
         id  INT(20) NOT NULL,
         methName  BLOB,
         token BLOB,  
         desc BLOB,
         apiseq BLOB )"""

    cursor.execute(sql)

    for info in datas:
        methNameNum = " ".join(toNum(info["methName"], methName_vocab_to_int))
        tokenNum = " ".join(toNum(info["token"], token_vocab_to_int))
        descNum = " ".join(toNum(info["desc"], desc_vocab_to_int))
        apiseqNum = " ".join(toNum(info["apiseq"], apiseq_vocab_to_int))

        insert = """"INSERT INTO NUMREPRESENT(id, methName, token, desc, apiseq)
                     VALUES(%d,%s,%s,%s,%s,%s)"""
        data = (info["id"], methNameNum, tokenNum, descNum, apiseqNum)
        cursor.execute(insert, data)
        connect.commit()

    cursor.close()
    connect.close()

    # methNamesNum = []
    # for methName in methNames:
    #     methNamesNum.append(toNum(methName, methName_vocab_to_int))
    #
    # tokensNum = []
    # for token in tokens:
    #     tokensNum.append(toNum(token, token_vocab_to_int))
    #
    # descsNum = []
    # for desc in descs:
    #     descsNum.append(toNum(desc, desc_vocab_to_int))
    #
    # apiseqsNum = []
    # for apiseq in apiseqs:
    #     apiseqsNum.append(toNum(apiseq, apiseq_vocab_to_int))

    ast_vocab_to_int, ast_int_to_vocab = getVocabForAST(asts, cf.n_words)

    # ast的词表保存在本地
    save_vocab("vocab_ast.json", ast_vocab_to_int)
    save_vocab("vocab_methName.json", methName_vocab_to_int)
    save_vocab("vocab_token.json", token_vocab_to_int)
    save_vocab("vocab_desc.json", desc_vocab_to_int)
    save_vocab("vocab_apiseq_json", apiseq_vocab_to_int)


def save_vocab(path, params):
    with open(path, 'w') as f:
        json.dump(params, f)


def load_vocab(path):
    with open(path,'r') as f:
        return json.load(f)


def getPath(asts, pathNum, ast_vocab_to_int):
    # 每次训练路径都是随机抽取的
    astPathNum = [] # 所有ast的所有path的编号表示 三维数组
    for ast in asts:
        nPath = getNPath(ast, pathNum)  # 针对每个ast的n条路径
        nPathNum = []
        for path in nPath:  #每条path的编号表示
            nPathNum.append(toNum(path, ast_vocab_to_int))
        astPathNum.append(nPathNum)
        # sbt = ' '.join(getSBT(ast, ast[0]))  # 得到李戈的sbt树
    return astPathNum


getVocab()
ast_vocab_to_int = load_vocab("vocab_ast.json")

