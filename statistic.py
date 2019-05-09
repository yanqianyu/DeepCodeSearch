import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import json


def apiseqlength():
    apiseqlength = {}
    with open("deepcs_dataset/test.apiseq.txt", "r") as f:
        lines = f.readlines()
        count = 0
        for line in lines:
            split_line = line.strip().replace("[", " ").replace("]", " ").split(" ")
            array = [int(i, base=10) for i in split_line if i != ' ' and i != '']

            length = len(array)
            if length in apiseqlength.keys():
                apiseqlength[length] += 1
            else:
                apiseqlength[length] = 1
            count += 1
            print("apiseq deal with " + str(count))

    with open('statistics/test_apiseq_stat.json', 'w+') as w:
        w.write(json.dumps(apiseqlength))

    labels = apiseqlength.keys()
    values = apiseqlength.values()

    plt.bar(labels, values, color='g', align='center')

    plt.xlabel('length')
    plt.ylabel('numbers')

    plt.title('apiseq length')
    plt.savefig('statistics/test_apiseq_length.jpg')
    plt.show()


def desclength():
    desclength = {}
    with open("deepcs_dataset/test.desc.txt", "r") as f:
        lines = f.readlines()
        count = 0
        for line in lines:
            split_line = line.strip().replace("[", " ").replace("]", " ").split(" ")
            array = [int(i, base=10) for i in split_line if i != ' ' and i != '']

            length = len(array)
            if length in desclength.keys():
                desclength[length] += 1
            else:
                desclength[length] = 1
            count += 1
            print("desc deal with " + str(count))

    with open('statistics/test_desc_stat.json', 'w+') as w:
        w.write(json.dumps(desclength))

    labels = desclength.keys()
    values = desclength.values()

    plt.bar(labels, values, color='g', align='center')

    plt.xlabel('length')
    plt.ylabel('numbers')

    plt.title('desc length')
    plt.savefig('statistics/test_desc_length.jpg')
    plt.show()


def tokenslength():
    tokenslength = {}
    with open("deepcs_dataset/test.tokens.txt", "r") as f:
        lines = f.readlines()
        count = 0
        for line in lines:
            split_line = line.strip().replace("[", " ").replace("]", " ").split(" ")
            array = [int(i, base=10) for i in split_line if i != ' ' and i != '']

            length = len(array)
            if length in tokenslength.keys():
                tokenslength[length] += 1
            else:
                tokenslength[length] = 1
            count += 1
            print("tokens deal with " + str(count))

    with open('statistics/test_tokensc_stat.json', 'w+') as w:
        w.write(json.dumps(tokenslength))

    labels = tokenslength.keys()
    values = tokenslength.values()

    plt.bar(labels, values, color='g', align='center')

    plt.xlabel('length')
    plt.ylabel('numbers')

    plt.title('tokens length')
    plt.savefig('statistics/test_tokens_length.jpg')
    plt.show()


def methnamelength():
    methnamelength = {}
    with open("deepcs_dataset/test.methname.txt", "r") as f:
        lines = f.readlines()
        count = 0
        for line in lines:
            split_line = line.strip().replace("[", " ").replace("]", " ").split(" ")
            array = [int(i, base=10) for i in split_line if i != ' ' and i != '']

            length = len(array)
            if length in methnamelength.keys():
                methnamelength[length] += 1
            else:
                methnamelength[length] = 1
            count += 1
            print("methname deal with " + str(count))

    with open('statistics/test_methname_stat.json', 'w+') as w:
        w.write(json.dumps(methnamelength))

    labels = methnamelength.keys()
    values = methnamelength.values()

    plt.bar(labels, values, color='g', align='center')

    plt.xlabel('length')
    plt.ylabel('numbers')

    plt.title('methname length')
    plt.savefig('statistics/test_methname_length.jpg')
    plt.show()


desclength()
tokenslength()
methnamelength()
apiseqlength()