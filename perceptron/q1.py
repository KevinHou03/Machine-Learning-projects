import argparse
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def model_assessment(filenames):
    """
    Given the entire data, split it into training and test set
    so you can assess your different models
    to compare perceptron, logistic regression,
    and naive bayes.
    """
    filename = pd.read_csv(filenames, header = None)
    dataframe = filename

    sample_num = filename.shape[0]
    indices = np.arange(sample_num)
    np.random.shuffle(indices)
    filename = filename.iloc[indices].reset_index()
    filename.rename(columns={0: 'text'}, inplace=True)
    filename = filename.drop(columns = ['index'])
    filename['text'] = filename['text'].apply(lambda email : email.split())

    xTrain, xTest = train_test_split(filename, test_size=0.3,
                                     random_state=42)  # 0.3 is the commonly used ratio that is believed to generate good result

    yTrain_list = []
    for rows in xTrain.iloc[:, 0]:  # rows: 3917  [1, from, mr, joe, mark, email, fax, number, n...
        yTrain_list.append(int(rows[0]))

    yTest_list = []
    for rows in xTest.iloc[:, 0]:
        yTest_list.append(int(rows[0]))

    xTrain['label'] = yTrain_list
    xTest['label'] = yTest_list

    xTrain['text'] = xTrain['text'].apply(lambda x: x[1:])
    xTest['text'] = xTest['text'].apply(lambda x: x[1:])


    # filename = pd.read_csv(filenames,header = None)
    #
    # # 打乱数据
    # np.random.seed(0)  # 设置随机种子以确保可重复性
    # filename = filename.sample(frac=1).reset_index(drop=True)
    #
    # # 重命名列
    # filename.rename(columns={0: 'text'}, inplace=True)
    #
    # # 拆分文本列
    # filename['text'] = filename['text'].apply(lambda email: email.split())
    #
    # xTrain, xTest = train_test_split(filename, test_size=0.3, random_state=42)
    #
    # yTrain_list = []
    # for rows in xTrain.iloc[:, 0]:  # rows: 3917  [1, from, mr, joe, mark, email, fax, number, n...
    #     yTrain_list.append(int(rows[0]))
    #
    # yTest_list = []
    # for rows in xTest.iloc[:, 0]:
    #     yTest_list.append(int(rows[0]))
    #
    # xTrain['label'] = yTrain_list
    # xTest['label'] = yTest_list
    #
    # xTrain['text'] = xTrain['text'].apply(lambda x: x[1:])
    # xTest['text'] = xTest['text'].apply(lambda x: x[1:])
    #
    #

    return xTrain, xTest


def build_vocab_map(traindf):
    """
    Construct the vocabulary map such that it returns
    (1) the vocabulary dictionary contains words as keys and
    the number of emails the word appears in as values, and
    (2) a list of words that appear in at least 30 emails.
    {'hello', 4} all the words, and their occurance,
    ---input:
    dataset: pandas dataframe containing the 'text' column
             and 'y' label column

    ---output:
    dict: key-value is word-count pair
    list: list of words that appear in at least 30 emails
    """
    traindf = traindf.iloc[:, 0]  # important wtf？
    traindf = pd.DataFrame(traindf)
    words_over_30 = []
    total_list = []
    for rows in traindf.iloc[:, 0]:
        total_list.extend(rows)

    words_dict = dict(Counter(total_list))

    for key, value in words_dict.items():
        if value >= 30:
            words_over_30.append(key)


    return words_dict, words_over_30[1:]


def construct_binary(dataset, freq_words):
    """
    Construct email datasets based on
    the binary representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is 1 if the ith word in the
    vocabulary occurs in the email,
    or 0 otherwise

    ---input:
    dataset: pandas dataframe containing the 'text' column

    freq_word: the vocabulary map built in build_vocab_map() i assume it is a list

    ---output:
    numpy array
    """
    freq_size = len(freq_words)
    all_vectors = []
    for emails_texts in dataset.iloc[:, 0]:
        feat_vector = [1] * freq_size
        for index, words in enumerate(freq_words):
            if words not in emails_texts:
                feat_vector[index] = 0
            if words in emails_texts:
                feat_vector[index] = 1
        all_vectors.append(feat_vector)

    all_vectors = np.array(all_vectors)
    binary_data = all_vectors

    return binary_data


def construct_count(dataset, freq_words):
    """
    Construct email datasets based on
    the count representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is the number of times the ith word in the
    vocabulary occurs in the email, feat_vector [i] = dict[words]
    or 0 otherwise

    ---input:
    dataset: pandas dataframe containing the 'text' column

    freq_word: the vocabulary map built in build_vocab_map()

    ---output:
    numpy array
    """

    all_vectors = []
    for emails_texts in dataset.iloc[:, 0]:  # for each email
        feat_vector = [1] * len(freq_words)
        counts = Counter(emails_texts)
        counts_dict = dict(counts)

        for index, words in enumerate(freq_words):
            if words in counts_dict.keys():
                feat_vector[index] = counts_dict[words]
            if words not in counts_dict.keys():
                feat_vector[index] = 0
        all_vectors.append(feat_vector)
    all_vectors = np.array(all_vectors)
    count_data = all_vectors
    print(count_data.shape)

    return count_data


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        default="spamAssassin.data",
                        help="filename of the input data")
    data = pd.read_csv('/Users/apple/Desktop/2023FALL/CS-334/perceptron-template2/spamAssassin.data', header=None)
    filename = '/Users/apple/Desktop/2023FALL/CS-334/perceptron-template2/spamAssassin.data'
    print(data)
    train_data, test_data = model_assessment(filename)
    yTrain_list = []
    # for rows in train_data.iloc[:, 0]:  # rows: 3917  [1, from, mr, joe, mark, email, fax, number, n...
    #     yTrain_list.append(int(rows[0]))

    yTest_list = []
    # for rows in test_data.iloc[:, 0]:
    #     yTest_list.append(int(rows[0]))
    for label in train_data.iloc[:, -1]:
        yTrain_list.append(label)

    for label in test_data.iloc[:, -1]:
        yTest_list.append(label)

    train_data['text'] = train_data['text'].apply(lambda x: x[1:])
    train_data['label'] = yTrain_list

    test_data['text'] = test_data['text'].apply(lambda x: x[1:])
    test_data['label'] = yTest_list

    print('train_data:\n', train_data)

    print('test_data:\n', test_data)

    words_dict, words_over_30 = build_vocab_map(train_data)

    print("length:", len(words_over_30))
    print('words_over_30:\n', words_over_30)

    # print(len(words_dict))
    # print(words_dict)

    binary_train = construct_binary(train_data, words_over_30)
    print("binary\n", binary_train)
    binary_test = construct_binary(test_data, words_over_30)

    count_train = construct_count(train_data, words_over_30)
    print(count_train)
    count_test = construct_count(test_data, words_over_30)

    binary_train = pd.DataFrame(binary_train)
    binary_train.to_csv('/Users/apple/Desktop/2023FALL/CS-334/binary_train.csv')

    binary_test = pd.DataFrame(binary_test)
    binary_test.to_csv('/Users/apple/Desktop/2023FALL/CS-334/binary_test.csv')

    count_train = pd.DataFrame(count_train)
    count_train.to_csv('/Users/apple/Desktop/2023FALL/CS-334/count_train.csv')

    count_test = pd.DataFrame(count_test)
    count_test.to_csv('/Users/apple/Desktop/2023FALL/CS-334/count_test.csv')

    yTrain_list = pd.DataFrame(yTrain_list)
    yTrain_list.to_csv('/Users/apple/Desktop/2023FALL/CS-334/yTrain.csv')

    yTest_list = pd.DataFrame(yTest_list)
    yTest_list.to_csv('/Users/apple/Desktop/2023FALL/CS-334/yTest.csv')


if __name__ == "__main__":
    main()

'''
cd /Users/apple/Desktop/2023FALL/CS-334/perceptron-template2/
python q1.py --data spamAssassin.data
'''

'''
words_over_30
['wrote', 'hi', 'i', 've', 'got', 'an', 'normal', 'number', 'cd', 'rw', 'id', 'drive', 'that', 'd', 'like', 'to', 'be', 'abl', 'us', 'with', 'a', 'dell', 'laptop', 'doe', 'anyon', 'know', 'ani', 'wai', 'enabl', 'thi', 'for', 'exampl', 'through', 'the', 'of', 'special', 'cabl', 'bai', 'where', 'rom', 'or', 'floppi', 'is', 'there', 'absolut', 'no', 'convent', 'directli', 'howev', 'you', 'can', 'get', 'extern', 'usb', 'and', 'attach', 'it', 'will', 'work', 'in', 'case', 'httpaddr', 'regard', 'irish', 'linux', 'user', 'group', 'emailaddr', 'un', 'subscript', 'inform', 'list', 'maintain', 'john', 'hall', 'wasn', 't', 'wa', 'on', 'file', 'cours', 'palestinian', 'mistak', 'which', 'kei', 'part', 'm', 'not', 'against', 'american', 'school', 'honor', 'killer', 'if', 'we', 'do', 'sure', 'plan', 'renam', 'who', 'kill', 'last', 'few', 'year', 'owen', 'fri', 'aug', 'at', 'numberam', 'look', 'trojan', 'latest', 'openssh', 'amaz', 'bit', 'caught', 'hour', 'thank', 'freebsd', 'port', 'system', 'automat', 'check', 'kevin', 'believ', 'happier', 'than', 'skeptic', 'more', 'fork', 'ed', 'point', 'fact', 'man', 'place', 'home', 'happi', 'cheap', 'danger', 'qualiti', 'g', 'b', '1', 'multi', 'messag', 'mime', 'format', 'nextpart', 'content', 'type', 'multipart', 'altern', 'boundari', 'text', 'html', 'charset', 'transfer', 'encod', 'basenumb', 'lnumber', 'my', 'name', 'jeremi', 'invest', 'properti', 'issu', 'been', 'have', 'everi', 'everyon', 'deal', 'hous', 'ar', 'almost', 'sell', 'mayb', 'thei', 're', 'realli', 'ugli', 'need', 'lot', 'bad', 'but', 'what', 'ever', 'reason', 'just', 'don', 'those', 'most', 'agent', 'were', 'hesit', 'becaus', 'isn', 'much', 'return', 'time', 'effort', 'privat', 'investor', 'fund', 'so', 'both', 'avoid', 'long', 'out', 'loan', 'approv', 'pai', 'cash', 'close', 'week', 'am', 'pre', 'qualifi', 'up', 'dollarnumb', 'purchas', 'monei', 'mortgag', 'base', 'non', 'owner', 'your', 'full', 'commiss', 'licens', 'real', 'estat', 'nor', 'affili', 'firm', 'mean', 'about', 'want', 'pleas', 'under', 'impress', 'market', 'valu', 'interest', 'sometim', 'even', 'retail', 'price', 'onli', 'guarante', 'll', 'make', 'wast', 'tell', 'over', 'phone', 'whether', 'project', 'from', 'our', 'would', 'form', 'allianc', 'contact', 'me', 'email', 'best', 'wish', 'www', 'com', 'unsubscrib', 'send', 'mail', 'bodi', 'dear', 'homeown', 'rate', 'their', 'lowest', 'help', 'find', 'situat', 'by', 'match', 'hundr', 'lender', 'improv', 'refin', 'second', 'equiti', 'elig', 'less', 'perfect', 'credit', 'servic', 'free', 'new', 'buyer', 'without', 'oblig', 'other', 'sai', 'ye', 'take', 'minut', 'complet', 'follow', 'all', 'kept', 'strictli', 'confidenti', 'must', 'least', 'ag', 'avail', 'within', 'unit', 'state', 'fast', 'opt', 'wed', 'robert', 'hmm', 'now', 'faster', 'algorithm', 'better', 'done', 'patent', 'prevent', 'go', 'near', 'first', 'adam', 'l', 'duncan', 'beberg', 'plain', 'iso', 'disposit', 'inlin', 'numberbit', 'view', 'newslett', 'color', 'visit', 'media', 'unspun', 'press', 'report', 'why', 'august', 'brother', 'spare', 'rememb', 'global', 'cross', 'stori', 'serv', 'busi', 'analysi', 'dai', 'annual', 'cost', 'dollar', 'four', 'trial', 'come', 'end', 'soon', 'sign', 'via', 'card', 'advertis', 'offer', 'high', 'net', 'worth', 'access', 'same', 'research', 'he', 'guid', 'client', 'creditor', 'sharehold', 'u', 's', 'airwai', 'found', 'themselv', 'lock', 'posit', 'sundai', 'as', 'nation', 'largest', 'airlin', 'bankruptci', 'protect', 'va', 'carrier', 'cover', 'code', 'chapter', 'troubl', 'could', 'trace', 'sept', 'when', 'airport', 'except', 'washington', 'too', 'white', 'site', 'how', 'befor', 'flight', 'lift', 'guess', 'major', 'hint', 'billion', 'past', 'terror', 'attack', 'anim', 'pain', 'usual', 'industri', 'add', 'these', 'competit', 'discount', 'small', 'area', 'concentr', 'east', 'coast', 'battl', 'highest', 'labor', 'frustrat', 'tri', 'throw', 'two', 'ago', 'remind', 'post', 'bid', 'bought', 'feder', 'regul', 'surviv', 'own', 'fed', 'disagre', 'block', 'ha', 'among', 'whose', 'employe', 'mani', 'whom', 'expect', 'lose', 'job', 'though', 'union', 'million', 'especi', 'plane', 'leas', 'compani', 'collect', 'judg', 'pick', 'account', 'yet', 'hear', 'word', 'folk', 'line', 'wall', 'street', 'journal', 'observ', 'profit', 'decad', 'mind', 'texa', 'pacif', 'down', 'also', 'plai', 'friend', 'pocket', 'custom', 'chief', 'assur', 'york', 'continu', 'track', 'frequent', 'mile', 'still', 'book', 'partner', 'hei', 'accord', 'analyst', 'quot', 'chanc', 'won', 'next', 'announc', 'gone', 'shift', 'caus', 'some', 'hit', 'hard', 'paid', 'requir', 'heard', 'ident', 'manag', 'opportun', 'model', 'emerg', 'result', 'download', 'execut', 'summari', 'coverag', 'releas', 'learn', 'expand', 'applic', 'never', 'strong', 'suit', 'onc', 'page', 'summer', 'corpor', 'back', 'januari', 'try', 'recal', 'chad', 'financi', 'sale', 'fridai', 'await', 'transact', 'seem', 'hold', 'breath', 'shock', 'alon', 'attent', 'worldcom', 'behind', 'sorri', 'ow', 'nearli', 'third', 'stake', 'sound', 'low', 'gotten', 'three', 'veri', 'team', 'asian', 'technolog', 'discov', 'here', 'left', 'pend', 'regulatori', 'six', 'month', 'bank', 'share', 'miss', 'dice', 'noth', 'feel', 'lucki', 'realiti', 'consider', 'wors', 'edward', 'lawyer', 'offici', 'statement', 'extend', 'telecom', 'capabl', 'particularli', 'broadband', 'simpli', 'bui', 'higher', 'turn', 'approxim', 'amount', 'founder', 'gari', 'winnick', 'person', 'rich', 'stock', 'note', 'paper', 'floor', 'accept', 'ap', 'insid', 'sold', 'appar', 'fraud', 'law', 'confus', 'europ', 'top', 'race', 'aol', 'warner', 'seek', 'data', 'interpret', 'lead', 'ban', 'rang', 'microsoft', 'chang', 'tune', 'toward', 'spam', 'world', 'reach', 'audienc', 'detail', 'todai', 'staff', 'written', 'jim', 'editor', 'publish', 'produc', 'inc', 'copyright', 'subscrib', 'alreadi', 'redistribut', 'permit', 'link', 'includ', 'power', 'remov', 'receiv', 'futur', 'e', 'n', 'p', 'pass', 'mondai', 'spin', 'sponsor', 'enter', 'address', 'box', 'below', 'nbsp', 'provid', 'sent', 'click', 'instantli', 'prefer', 'didn', 'max', 'might', 'paypal', 'them', 'secur', 'eas', 'pm', 'pt', 'encrypt', 'start', 'attend', 'univers', 'late', 'colleg', 'co', 'onlin', 'payment', 'sinc', 'attract', 'ten', 'gain', 'reput', 'premier', 'internet', 'processor', 'auction', 'ebai', 'acquir', 'king', 'old', 'kid', 'programm', 'famili', 'move', 'chicago', 'then', 'pursu', 'mission', 'creat', 'startup', 'right', 'build', 'password', 'palm', 'pilot', 'met', 'peter', 'target', 'ceo', 'vision', 'begin', 'matter', 'remain', 'focus', 'good', 'fortun', 'explain', 'spend', 'design', 'doesn', 'step', 'conveni', 'around', 'trade', 'off', 'fundament', 'mountain', 'allow', 'consum', 'countri', 'particip', 'led', 'web', 'although', 'rival', 'relationship', 'between', 'juli', 'acquisit', 'agreement', 'subject', 'review', 'stai', 'merg', 'collabor', 'explor', 'domin', 'featur', 'abil', 'carri', 'open', 'commun', 'fit', 'grow', 'peopl', 'anoth', 'gener', 'ii', 'confer', 'natur', 'save', 'exploit', 'vice', 'presid', 'combat', 'crimin', 'establish', 'program', 'monitor', 'warn', 'import', 'shop', 'safe', 'arriv', 'beauti', 'websit', 'refer', 'give', 'plenti', 'java', 'button', 'feedback', 'construct', 'tast', 'specif', 'templat', 'host', 'mo', 'control', 'panel', 'front', 'graphic', 'statist', 'call', 'center', 'such', 'yahoo', 'hotmail', 'etc', 'instead', 'respons', 'method', 'w', 'ac', 'current', 'comment', 'neither', 'parti', 'circumst', 'elimin', 'upon', 'matthia', 'thu', 'jul', 'saou', 'see', 'valhalla', 'freshrpm', 'should', 'bug', 'problem', 'copi', 'deep', 'directori', 'into', 'fix', 'well', 'him', 'brown', 'bag', 'column', 'header', 'put', 'version', 'hope', 'edificio', 'nort', 'planta', 'network', 'engin', 'barcelona', 'spain', 'electron', 'interact', 'rpm', 'printabl', 'introduc', 'seri', 'filter', 'dc', 'electr', 'anumb', 'dnumber', 'test', 'x', 'capac', 'temperatur', 'inquiri', 'sampl', 'request', 'kind', 'forward', 'film', 'ltd', 'f', 'po', 'st', 'tel', 'fax', 'apolog', 'inconveni', 'repli', 'stream', 'j', 'cap', 'jpg', 'filenam', 'numberj', 'numberp', 'jnumber', 'vs', 'qnumber', 'numberb', 'z', 'v', 'pa', 'td', 'k', 'bu', 'nf', 'c', 'numberanumb', 'bnumber', 'pnumber', 'ad', 'numbera', 'rh', 'q', 'bf', 'al', 'numberi', 'tnumber', 'au', 'numberx', 'ee', 'cb', 'r', 'tremend', 'toner', 'inkjet', 'secret', 'weapon', 'lower', 'printer', 'suppli', 'ourselv', 'rapid', 'outstand', 'compat', 'replac', 'epson', 'canon', 'hewlett', 'product', 'meet', 'often', 'exce', 'origin', 'manufactur', 'stylu', 'cartridg', 'hp', 'laserjet', 'similar', 'differ', 'consol', 'hook', 'thing', 'think', 'desk', 'talk', 'robot', 'becom', 'human', 'mai', 'trick', 'seattl', 'answer', 'machin', 'intellig', 'friendli', 'therefor', 'empti', 'practic', 'technic', 'support', 'while', 'process', 'fail', 'dramat', 'brain', 'ill', 'ey', 'surfac', 'databas', 'queri', 'later', 'red', 'ask', 'quickli', 'short', 'keyword', 'quick', 'adapt', 'interfac', 'averag', 'comput', 'slow', 'wake', 'weather', 'icon', 'sit', 'desktop', 'fun', 'convers', 'decid', 'game', 'lawrenc', 'murphi', 'teledynam', 'blog', 'biz', 'useless', 'picasso', 'url', 'date', 'giant', 'scienc', 'fiction', 'di', 'cancer', 'discuss', 'michael', 'cnet', 'cool', 'toshiba', 'pc', 'enumb', 'handheld', 'tx', 'powershot', 'gnumber', 'camera', 'pioneer', 'creativ', 'lab', 'portabl', 'dvd', 'player', 'associ', 'reader', 'everybodi', 'launch', 'advic', 'read', 'wireless', 'devic', 'numbermhz', 'style', 'soni', 'tool', 'captur', 'digit', 'video', 'perform', 'excel', 'light', 'demand', 'imag', 'everyth', 'deliv', 'pictur', 'twice', 'resolut', 'expens', 'gave', 'mark', 'handi', 'perhap', 'os', 'pda', 'great', 'screen', 'mpnumber', 'appeal', 'compar', 'wave', 'wait', 'wing', 'tough', 'progress', 'scan', 'sub', 'tag', 'rest', 'budget', 'img', 'src', 'width', 'height', 'built', 'audio', 'listen', 'christma', 'movi', 'record', 'divx', 'true', 'fals', 'question', 'plug', 'onto', 'watch', 'none', 'propos', 'entertain', 'roll', 'far', 'live', 'tech', 'submit', 'cio', 'love', 'zdnet', 'director', 'had', 'enough', 'choic', 'award', 'faq', 'comparison', 'reserv', 'exhibit', 'hotel', 'show', 'resort', 'middl', 'itself', 'region', 'hospit', 'event', 'local', 'intern', 'uk', 'daili', 'worldwid', 'index', 'big', 'held', 'relat', 'depart', 'commerc', 'organis', 'rank', 'showcas', 'varieti', 'bed', 'profession', 'star', 'apart', 'repres', 'housekeep', 'financ', 'develop', 'architect', 'consult', 'space', 'contract', 'run', 'immedi', 'sincer', 'pro', 'legal', 'novel', 'sourc', 'forget', 'awai', 'join', 'membership', 'smaller', 'cell', 'memori', 'plu', 'window', 'connect', 'ear', 'enjoi', 'choos', 'select', 'eventu', 'stephen', 'beach', 'jame', 'y', 'expir', 'error', 'activ', 'cancel', 'void', 'prohibit', 'titl', 'info', 'updat', 'mobil', 'th', 'o', 'anywher', 'deathtospamdeathtospamdeathtospam', 'sf', 'jabber', 'fastest', 'platform', 'im', 'spamassassin', 'sight', 'numberpm', 'actual', 'gather', 'rare', 'option', 'did', 'attempt', 'order', 'dsl', 'larg', 'obtain', 'vari', 'either', 'honest', 'god', 'numbercnumb', 'numberdnumb', 'imagin', 'rout', 'exit', 'obvious', 'yourself', 'warranti', 'car', 'truck', 'direct', 'exist', 'fill', 'easi', 'numbertnumb', 'weekend', 'googl', 'damn', 'blue', 'angl', 'again', 'wow', 'suck', 'couldn', 'haven', 'anyth', 'somewher', 'thinkgeek', 'welcom', 'geek', 'heaven', 'razor', 'junk', 'weekli', 'digest', 'calendar', 'confirm', 'someth', 'hot', 'org', 'techniqu', 'ibm', 'war', 'canada', 'alter', 'cat', 'app', 'channel', 'forb', 'california', 'sue', 'declar', 'scam', 'amazon', 'privaci', 'polici', 'int', 'aim', 'broadcast', 'studi', 'doubl', 'san', 'francisco', 'fight', 'virus', 'dead', 'regist', 'anti', 'wire', 'arm', 'boston', 'globe', 'rais', 'optim', 'cultur', 'theft', 'independ', 'sm', 'delet', 'lo', 'angel', 'tire', 'vnumber', 'lawsuit', 'stop', 'eu', 'ireland', 'face', 'spammer', 'porn', 'cite', 'rule', 'victim', 'children', 'alert', 'numberm', 'member', 'interview', 'cnn', 'push', 'upgrad', 'focu', 'search', 'instant', 'touch', 'someon', 'western', 'submiss', 'glad', 'individu', 'student', 'settl', 'charg', 'oh', 'doctor', 'administr', 'dave', 'peer', 'board', 'canadian', 'polit', 'voic', 'rise', 'store', 'fbi', 'flaw', 'after', 'viru', 'wild', 'diet', 'outlook', 'exchang', 'fcc', 'mac', 'mike', 'child', 'softwar', 'pr', 'broken', 'morn', 'exec', 'head', 'appl', 'habea', 'author', 'public', 'trademark', 'compil', 'archiv', 'intend', 'enhanc', 'claim', 'made', 'accuraci', 'authent', 'materi', 'assum', 'transmiss', 'storag', 'section', 'fee', 'curiou', 'capit', 'indic', 'percentag', 'charact', 'letter', 'raw', 'rohit', 'datapow', 'xml', 'acceler', 'unlik', 'compet', 'solut', 'hardwar', 'achiev', 'greater', 'mass', 'proprietari', 'core', 'pars', 'steve', 'kelli', 'adopt', 'anticip', 'awar', 'infrastructur', 'said', 'convert', 'increas', 'size', 'tax', 'server', 'addit', 'firewal', 'inspect', 'http', 'traffic', 'packet', 'straight', 'blind', 'approach', 'initi', 'speed', 'path', 'knowledg', 'mostli', 'switch', 'ssl', 'socket', 'layer', 'load', 'balanc', 'byte', 'languag', 'deploi', 'proxi', 'mode', 'identifi', 'deploy', 'earlier', 'intel', 'field', 'enterpris', 'earli', 'eugen', 'effect', 'protocol', 'assembl', 'class', 'uniqu', 'leverag', 'experi', 'bar', 'purpos', 'bring', 'draw', 'leader', 'sever', 'success', 'venrock', 'mobiu', 'ventur', 'seed', 'round', 'bill', 'invent', 'transform', 'prepar', 'cto', 'critic', 'singl', 'extraordinari', 'incompat', 'economi', 'decis', 'laser', 'encount', 'rapidli', 'surround', 'quit', 'compel', 'seen', 'nice', 'seriou', 'effici', 'reli', 'tradit', 'entrepreneur', 'span', 'seven', 'stage', 'promis', 'experienc', 'goal', 'term', 'assist', 'consist', 'life', 'proven', 'growth', 'organ', 'offic', 'citi', 'ma', 'park', 'ca', 'respond', 'former', 'senior', 'primarili', 'compon', 'combin', 'broad', 'asset', 'portfolio', 'forc', 'locat', 'provinc', 'ensur', 'taylor', 'chairman', 'dure', 'twenti', 'involv', 'prior', 'resid', 'function', 'subsequ', 'revenu', 'equip', 'clean', 'room', 'numer', 'integr', 'mit', 'advanc', 'accomplish', 'router', 'ip', 'forum', 'magazin', 'institut', 'task', 'vast', 'understand', 'extens', 'lan', 'wan', 'optic', 'previous', 'cach', 'nasdaq', 'contribut', 'wide', 'hill', 'print', 'advisor', 'hoover', 'bell', 'role', 'edg', 'georg', 'mr', 'level', 'known', 'regular', 'speak', 'engag', 'expert', 'marshal', 'rose', 'dozen', 'standard', 'implement', 'pop', 'smtp', 'msg', 'probabl', 'unwant', 'auto', 'contain', 'prioriti', 'set', 'mailer', 'relai', 'whitelist', 'adjust', 'background', 'border', 'bottom', 'numberpx', 'solid', 'ffffff', 'font', 'numberpt', 'weight', 'bold', 'hello', 'notic', 'definit', 'maximum', 'amp', 'newest', 'fulli', 'bound', 'congratul', 'winner', 'florida', 'vacat', 'grand', 'prize', 'night', 'magic', 'trip', 'ship', 'win', 'contest', 'ticket', 'exclud', 'norton', 'packag', 'ton', 'edit', 'pack', 'util', 'valuabl', 'backup', 'easili', 'superior', 'lt', 'gt', 'fall', 'destruct', 'hacker', 'justin', 'mason', 'nonspam', 'log', 'output', 'ie', 'along', 'default', 'score', 'entir', 'input', 'ga', 'alwai', 'exact', 'let', 'main', 'verdana', 'arial', 'helvetica', 'serif', 'decor', 'hover', 'yellow', 'debt', 'consolid', 'transmit', 'specialist', 'advantag', 'correspond', 'struggl', 'ahead', 'outsid', 'mention', 'consid', 'reduc', 'entri', 'prompt', 'present', 'until', 'cannot', 'advis', 'proper', 'mon', 'rick', 'gpl', 'went', 'correct', 'suse', 'unless', 'separ', 'permiss', 'grant', 'thirti', 'leav', 'french', 'agre', 'defend', 'death', 'rat', 'ass', 'franc', 'care', 'tuesdai', 'septemb', 'tue', 'sep', 'numberst', 'amend', 'shout', 'recent', 'opinion', 'thought', 'behavior', 'freedom', 'speech', 'buri', 'besid', 'basic', 'appropri', 'bush', 'writer', 'begun', 'scientif', 'advisori', 'committe', 'health', 'conclus', 'odd', 'hate', 'hadn', 'valid', 'discoveri', 'els', 'brian', 'conserv', 'voyag', 'icq', 'bullet', 'behalf', 'thursdai', 'cc', 'write', 'sl', 'okai', 'hire', 'resum', 'defin', 'tom', 'mh', 'emac', 'idea', 'evolv', 'evil', 'exmh', 'perl', 'headlin', 'translat', 'km', 'beta', 'pudg', 'chat', 'messeng', 'spamd', 'spamc', 'dcc', 'setup', 'instal', 'properli', 'document', 'otherwis', 'brief', 'appreci', 'llc', 'osdn', 'php', 'wednesdai', 'each', 'govern', 'given', 'seeker', 'moment', 'exactli', 'imposs', 'across', 'foundat', 'everydai', 'appli', 'possibl', 'deni', 'agenc', 'oper', 'keep', 'deposit', 'citizen', 'entitl', 'scholarship', 'train', 'rush', 'incred', 'america', 'wouldn', 'half', 'econom', 'boss', 'huge', 'equal', 'somehow', 'tape', 'walk', 'benefit', 'incom', 'repair', 'rent', 'fuel', 'cloth', 'camp', 'music', 'lesson', 'art', 'medic', 'suffer', 'fire', 'educ', 'primari', 'men', 'women', 'further', 'foreign', 'congression', 'mandat', 'togeth', 'comprehens', 'thousand', 'instruct', 'procedur', 'guidelin', 'resourc', 'land', 'employ', 'black', 'christian', 'faith', 'instanc', 'loss', 'dream', 'confid', 'refund', 'insist', 'risk', 'taken', 'aren', 'satisfi', 'bonus', 'gift', 'bonu', 'teach', 'tip', 'littl', 'rental', 'insur', 'heat', 'telephon', 'food', 'prescript', 'drug', 'manual', 'succe', 'simpl', 'strategi', 'concept', 'determin', 'blank', 'predict', 'happen', 'shirt', 'dog', 'explan', 'prospect', 'door', 'readi', 'easiest', 'ignor', 'stand', 'nobodi', 'told', 'common', 'partnership', 'cut', 'mine', 'reduct', 'sens', 'succeed', 'reveal', 'solv', 'five', 'seminar', 'career', 'whole', 'delai', 'longer', 'regardless', 'handl', 'describ', 'abov', 'map', 'navig', 'smith', 'gambl', 'gmt', 'she', 'rss', 'flow', 'joseph', 'iii', 'fair', 'suppos', 'fuck', 'attitud', 'hand', 'exclus', 'club', 'stuff', 'dont', 'anymor', 'pull', 'freez', 'cant', 'skill', 'apt', 'dist', 'fine', 'catch', 'kernel', 'meant', 'depend', 'categori', 'whatev', 'older', 'hat', 'recommend', 'final', 'root', 'usr', 'hassl', 'broker', 'liter', 'act', 'attornei', 'action', 'netscap', 'browser', 'ok', 'debug', 'presum', 'merchant', 'visa', 'mastercard', 'express', 'trust', 'substanti', 'premium', 'deserv', 'threat', 'equival', 'solicit', 'blood', 'random', 'perlnumb', 'basi', 'australia', 'shouldn', 'occur', 'northern', 'legisl', 'label', 'remark', 'british', 'england', 'south', 'split', 'assign', 'opposit', 'side', 'india', 'loos', 'english', 'ah', 'coupl', 'sexual', 'curv', 'la', 'en', 'period', 'extrem', 'promiscu', 'sex', 'distribut', 'gai', 'inde', 'evid', 'band', 'argu', 'argument', 'essenti', 'explicitli', 'spread', 'aid', 'accur', 'particular', 'survei', 'conduct', 'proof', 'stimul', 'pleasur', 'pill', 'famou', 'worri', 'paragraph', 'perfectli', 'lockergnom', 'chri', 'gui', 'kate', 'fairli', 'overal', 'larri', 'configur', 'finish', 'shoot', 'wife', 'potenti', 'insert', 'tim', 'shape', 'buck', 'himself', 'spring', 'knew', 'dump', 'figur', 'dig', 'notebook', 'easier', 'tini', 'larger', 'sheet', 'satellit', 'bird', 'planet', 'signific', 'float', 'minimum', 'quarter', 'tcp', 'affect', 'significantli', 'ground', 'extra', 'pgp', 'vulner', 'expos', 'rather', 'difficult', 'numberxnumb', 'violat', 'xp', 'ebook', 'intro', 'correctli', 'bunch', 'tabl', 'lazi', 'complain', 'displai', 'stupid', 'mainten', 'ran', 'null', 'spent', 'anywai', 'spot', 'mous', 'dot', 'chip', 'deliveri', 'penguin', 'shell', 'suggest', 'feed', 'catalog', 'pretti', 'visual', 'collector', 'zoom', 'domain', 'dn', 'central', 'btw', 'corpu', 'devel', 'flash', 'numberg', 'wealthi', 'wealth', 'earn', 'hawaii', 'jersei', 'mexico', 'north', 'island', 'virginia', 'west', 'zip', 'desir', 'monthli', 'pgi', 'craig', 'hugh', 'vipul', 'ved', 'imho', 'sort', 'bulk', 'recogn', 'minim', 'decent', 'reliabl', 'certainli', 'due', 'te', 'doubt', 'neg', 'took', 'patch', 'binari', 'tar', 'cv', 'weird', 'egg', 'abus', 'fl', 'de', 'db', 'promot', 'ref', 'notif', 'numberth', 'serial', 'consequ', 'sum', 'usdollarnumb', 'total', 'asia', 'mix', 'complic', 'smoker', 'male', 'cigarett', 'femal', 'social', 'smoke', 'flat', 'tobacco', 'render', 'ms', 'smart', 'wonder', 'harvest', 'paul', 'master', 'silent', 'prove', 'presenc', 'raid', 'dublin', 'martin', 'truli', 'heck', 'tv', 'millionair', 'popular', 'investig', 'respect', 'attain', 'testimoni', 'comfort', 'shown', 'appear', 'envelop', 'corner', 'resel', 'sequenc', 'rel', 'reward', 'legitim', 'treat', 'cycl', 'count', 'modifi', 'disk', 'classifi', 'goe', 'worst', 'calcul', 'inexpens', 'strongli', 'illeg', 'wrong', 'clear', 'steal', 'currenc', 'wrap', 'postal', 'numberc', 'germani', 'mlm', 'relax', 'batch', 'limit', 'excit', 'popul', 'jump', 'feet', 'her', 'lai', 'laugh', 'surpris', 'dan', 'came', 'carefulli', 'numbernd', 'road', 'complianc', 'sprai', 'gun', 'break', 'mess', 'water', 'plant', 'flexibl', 'wind', 'pound', 'accessori', 'foot', 'serious', 'occup', 'yr', 'forev', 'h', 'oct', 'specifi', 'environ', 'variabl', 'tweak', 'spec', 'necessarili', 'mechan', 'rewrit', 'tree', 'alsa', 'driver', 'greatli', 'simplifi', 'andrew', 'oop', 'librari', 'strip', 'cur', 'dollarfold', 'dollarmsgid', 'invok', 'flist', 'folder', 'dollarexmh', 'script', 'remot', 'scott', 'sun', 'june', 'syntax', 'ftoc', 'inbox', 'command', 'unseen', 'drop', 'tcl', 'garrigu', 'vircio', 'congress', 'austin', 'doer', 'signatur', 'gnupg', 'gnu', 'worker', 'histori', 'sa', 'die', 'settlement', 'multipl', 'annuiti', 'teen', 'guido', 'incorpor', 'procmail', 'spambay', 'pickl', 'usag', 'statu', 'skip', 'tk', 'ba', 'edt', 'symbol', 'lib', 'inumb', 'razornumb', 'relev', 'modul', 'david', 'comp', 'thoma', 'admin', 'ffnumber', 'numberfnumb', 'vehicl', 'dealer', 'ct', 'toll', 'interrupt', 'brought', 'britain', 'abandon', 'nuclear', 'missil', 'phase', 'scientist', 'numberk', 'thread', 'fresh', 'makefil', 'export', 'unsolicit', 'href', 'viagra', 'bandwidth', 'mirror', 'luck', 'babi', 'retir', 'mother', 'father', 'nov', 'greatest', 'centuri', 'mailbox', 'afford', 'extract', 'verifi', 'numbermb', 'evalu', 'per', 'threaten', 'successfulli', 'detect', 'certifi', 'bgcolor', 'numberd', 'numberctr', 'numberctd', 'align', 'numberftd', 'numberftr', 'numberft', 'numbercbr', 'numberff', 'numbercfont', 'numberf', 'numberffont', 'numbercp', 'numbercb', 'numberdari', 'numbereeasi', 'searcher', 'ffnumberff', 'numberfb', 'numberfp', 'numbercli', 'nt', 'disconnect', 'sender', 'demo', 'numberca', 'numberhttp', 'numberfwww', 'numberewldinfo', 'numberecom', 'numberfdownload', 'numberfnewe', 'numberezip', 'numberedownload', 'numberfa', 'numberenumb', 'numbercstrong', 'keyboard', 'numberfstrong', 'numbercdiv', 'numberfdiv', 'numberverdana', 'numberrd', 'overlook', 'commerci', 'herbal', 'formul', 'nb', 'sp', 'super', 'cooper', 'banner', 'gif', 'fe', 'fool', 'suffici', 'geeg', 'schuman', 'da', 'senat', 'roi', 'joe', 'innov', 'possess', 'sustain', 'commit', 'council', 'judgment', 'scale', 'variou', 'ethic', 'strike', 'facil', 'transit', 'implic', 'measur', 'gnomenumb', 'alt', 'gnome', 'air', 'william', 'sir', 'european', 'articl', 'colleagu', 'lookup', 'cafe', 'april', 'sector', 'percent', 'heavi', 'recov', 'boost', 'competitor', 'depress', 'rebuild', 'dev', 'strateg', 'mount', 'corp', 'poor', 'strength', 'pressur', 'appoint', 'maker', 'seat', 'audit', 'kit', 'photo', 'fnumber', 'snumber', 'shot', 'concern', 'maxim', 'casino', 'proud', 'bet', 'sick', 'si', 'buffer', 'slightli', 'failur', 'beat', 'pci', 'volum', 'physic', 'unabl', 'lack', 'necessari', 'factor', 'energi', 'convinc', 'ultim', 'blame', 'craft', 'invit', 'corrupt', 'ipod', 'bigger', 'imac', 'pentium', 'msn', 'guru', 'menu', 'gatewai', 'dedic', 'enforc', 'ideal', 'kick', 'merger', 'trend', 'clearli', 'quicktim', 'wi', 'fi', 'perman', 'highlight', 'crash', 'burn', 'vendor', 'poll', 'crime', 'preserv', 'session', 'impact', 'topic', 'rsa', 'assert', 'item', 'mere', 'encourag', 'fault', 'previou', 'vote', 'profil', 'weak', 'alloc', 'billi', 'young', 'husband', 'son', 'favorit', 'forth', 'usa', 'gold', 'ab', 'ea', 'fc', 'cnumber', 'bc', 'bd', 'ec', 'aa', 'ae', 'bb', 'plugin', 'mozilla', 'et', 'crap', 'beyond', 'modem', 'televis', 'redhat', 'deliber', 'sat', 'alan', 'circl', 'greg', 'funni', 'radio', 'mathemat', 'matthew', 'court', 'china', 'bright', 'roger', 'ian', 'logic', 'latter', 'distanc', 'mutual', 'chenei', 'lost', 'fetch', 'var', 'februari', 'afraid', 'inch', 'pipe', 'rock', 'precis', 'oil', 'bear', 'sweet', 'weblog', 'realiz', 'violenc', 'yeah', 'song', 'logo', 'artist', 'heart', 'cheaper', 'branch', 'repositori', 'multimedia', 'javamail', 'commentari', 'wholesal', 'visitor', 'scheme', 'green', 'somebodi', 'piec', 'examin', 'massiv', 'brand', 'fraction', 'wisdom', 'desert', 'bomb', 'prosper', 'constant', 'condit', 'hide', 'op', 'isp', 'virtual', 'replica', 'pair', 'cheer', 'warm', 'dinosaur', 'earth', 'despit', 'belief', 'degre', 'environment', 'fear', 'drink', 'unusu', 'cold', 'ic', 'climat', 'conclud', 'scenario', 'highli', 'yield', 'richard', 'alfr', 'professor', 'complex', 'el', 'assumpt', 'march', 'empir', 'demonstr', 'season', 'societi', 'hettinga', 'bearer', 'underwrit', 'farquhar', 'antiqu', 'agreeabl', 'gibbon', 'declin', 'roman', 'cvsroot', 'tmp', 'length', 'rc', 'retriev', 'revis', 'diff', 'rnumber', 'numbernumb', 'ham', 'constantli', 'config', 'imap', 'van', 'ratio', 'zero', 'hang', 'python', 'eastern', 'tribe', 'moor', 'fashion', 'hurt', 'rob', 'flag', 'sourceforg', 'formula', 'self', 'rebat', 'ram', 'compaq', 'hnumber', 'elsewher', 'builder', 'healthi', 'protest', 'speaker', 'iraq', 'russia', 'sake', 'arrest', 'jack', 'polic', 'peac', 'moral', 'hell', 'eff', 'harm', 'init', 'xmm', 'object', 'xine', 'nvidia', 'ti', 'brent', 'welch', 'panasa', 'scalabl', 'dmca', 'aggress', 'lifetim', 'disabl', 'town', 'sea', 'disrupt', 'lame', 'neighbor', 'saturdai', 'octob', 'woman', 'wear', 'pic', 'peni', 'candid', 'millennium', 'skin', 'cent', 'estim', 'newspap', 'hunt', 'divis', 'escap', 'ilug', 'moder', 'rip', 'biggest', 'studio', 'forest', 'ben', 'frnumber', 'gcc', 'stack', 'frame', 'pointer', 'hack', 'element', 'dir', 'decemb', 'republ', 'su', 'ch', 'tr', 'distributor', 'began', 'chines', 'se', 'reflect', 'disclaim', 'retain', 'alli', 'brows', 'saw', 'trivial', 'spamtrap', 'sh', 'ftp', 'ascii', 'knumber', 'manner', 'bitbitch', 'er', 'fantasi', 'gordon', 'hash', 'poverti', 'truth', 'margin', 'marri', 'bottl', 'becam', 'compens', 'ge', 'daniel', 'principl', 'campaign', 'territori', 'le', 'proce', 'sed', 'fly', 'est', 'tend', 'adult', 'capitalist', 'tm', 'drag', 'acknowledg', 'damag', 'russel', 'commonli', 'emploi', 'shopper', 'supplier', 'penni', 'soft', 'obviou', 'niall', 'excess', 'myself', 'portion', 'holidai', 'genet', 'signal', 'modern', 'activist', 'spirit', 'intent', 'phrase', 'deriv', 'tone', 'exhaust', 'apach', 'grab', 'notifi', 'movement', 'igtt', 'peak', 'evolut', 'limbo', 'suspect', 'herba', 'supplement', 'kathmandu', 'templ', 'substanc', 'ragga', 'dagga', 'botan', 'viripot', 'mood', 'sophist', 'stress', 'inspir', 'sleep', 'alcohol', 'invas', 'virtu', 'herb', 'potent', 'amalgam', 'compris', 'southern', 'jigget', 'structur', 'tea', 'aspect', 'calm', 'flower', 'capillari', 'muscl', 'diseas', 'ladi', 'boi', 'descript', 'bless', 'kingdom', 'oz', 'vjestika', 'aphrodisia', 'liquid', 'seventh', 'prosaka', 'tablet', 'gentl', 'feroc', 'appetit', 'remedi', 'dragon', 'sensit', 'dynam', 'therebi', 'agenda', 'bone', 'joint', 'tin', 'jar', 'tab', 'reg', 'pure', 'domest', 'clue', 'girl', 'toler', 'il', 'rm', 'dd', 'ex', 'euro', 'catalogu', 'benchmark', 'crazi', 'swap', 'travel', 'compress', 'armi', 'pablo', 'ou', 'ce', 'fa', 'cf', 'af', 'fb', 'eb', 'df', 'introduct', 'li', 'whatsoev', 'disappear', 'timeout', 'crack', 'hidden', 'tower', 'receipt', 'urgent', 'henc', 'shall', 'sur', 'static', 'sql', 'batteri', 'zone', 'bob', 'bio', 'album', 'bundl', 'privileg', 'restrict', 'gate', 'schedul', 'invoic', 'bond', 'num', 'sole', 'boot', 'filesystem', 'bin', 'char', 'invalid', 'donat', 'felt', 'mysql', 'randomli', 'typic', 'recipi', 'unix', 'anonym', 'pub', 'interoper', 'certain', 'txt', 'africa', 'plc', 'nigeria', 'unfortun', 'arrang', 'conf', 'afternoon', 'dark', 'stick', 'finger', 'dumb', 'bother', 'annoi', 'shut', 'nomin', 'arrai', 'ui', 'attribut', 'clock', 'scope', 'repeat', 'depth', 'endors', 'joke', 'mod', 'anytim', 'satisfact', 'liber', 'reject', 'hopefulli', 'pnumberp', 'resist', 'station', 'fd', 'partit', 'lilo', 'mdnumber', 'xxx', 'toni', 'underlin', 'cbyi', 'histor', 'safeti', 'ir', 'litig', 'reform', 'entiti', 'collaps', 'motiv', 'termin', 'md', 'parent', 'orbit', 'admit', 'terribl', 'elect', 'difficulti', 'riaa', 'favor', 'theori', 'revers', 'anybodi', 'scratch', 'coloni', 'belong', 'eat', 'recognit', 'london', 'routin', 'tomorrow', 'vou', 'greet', 'resolv', 'accid', 'democrat', 'dr', 'till', 'mbr', 'rid', 'destroi', 'restor', 'silli', 'bridg', 'sport', 'boom', 'dial', 'clinton', 'stabl', 'farm', 'ps', 'autom', 'gzip', 'regularli', 'russian', 'echo', 'embed', 'sedit', 'solari', 'sendmail', 'milter', 'occasion', 'whenev', 'goto', 'steven', 'loop', 'recurs', 'enemi', 'registr', 'minor', 'debian', 'chart', 'refus', 'ing', 'distro', 'militari', 'democraci', 'civil', 'increasingli', 'perspect', 'throughout', 'distinct', 'analog', 'anthoni', 'doc', 'reboot', 'rep', 'defens', 'string', 'stuck', 'bind', 'xnumber', 'rsync', 'regim', 'formal', 'novemb', 'nnumber', 'tour', 'architectur', 'hole', 'villag', 'reciev', 'emot', 'marriag', 'alb', 'scene', 'hasn', 'hair', 'transport', 'numbergb', 'shadow', 'gpg', 'ng', 'trader', 'everywher', 'disput', 'hors', 'prison', 'indian', 'yesterdai', 'mm', 'somewhat', 'bugtraq', 'addr', 'stabil', 'overnight', 'oppos', 'coalit', 'fat', 'theme', 'dude', 'challeng', 'oracl', 'mid', 'chain', 'wise', 'recruit', 'ny', 'min', 'squar', 'negoti', 'wit', 'enorm', 'stamp', 'ext', 'classic', 'mplayer', 'ne', 'av', 'newer', 'eval', 'born', 'dad', 'numberw', 'sentenc', 'unlimit', 'chosen', 'familiar', 'unknown', 'strang', 'religion', 'religi', 'nativ', 'influenc', 'conflict', 'debat', 'muslim', 'usd', 'mini', 'conumb', 'du', 'disclosur', 'preced', 'cloudmark', 'duplic', 'sync', 'context', 'em', 'disc', 'mnumber', 'prix', 'neverfail', 'liberti', 'revok', 'endeavor', 'synchron', 'addict', 'mental', 'exercis', 'medicin', 'wrinkl', 'snapshot', 'ssh', 'numberv', 'numberz', 'numbern', 'py', 'numbert', 'dq', 'vb', 'es', 'frontier', 'rhnumber', 'modif', 'token', 'fp', 'penalti', 'openssl', 'gtk', 'gs', 'sdram', 'pref', 'numbercent', 'hormon', 'lean', 'erect', 'tmda', 'farmer', 'illustr', 'inclus', 'affair', 'handbook', 'numbermm', 'numberin', 'motor', 'api', 'dw', 'patrick', 'monkei', 'wine', 'coffe', 'fan', 'mahonei', 'integ', 'overflow', 'vital', 'murder', 'japan', 'und', 'spell', 'der', 'pattern', 'forg', 'leadership', 'terrorist', 'secretari', 'certif', 'bounc', 'realist', 'dollarac', 'cpp', 'numberl', 'foo', 'lcd', 'prefix', 'framework', 'screw', 'afghanistan', 'telecommun', 'cisco', 'constitut', 'con', 'trigger', 'sie', 'numberpnumberp', 'ez', 'wd', 'dollarb', 'chemic', 'justic', 'newsgroup', 'incent', 'defeat', 'numberbnumb', 'fake', 'celebr', 'iiu', 'queue', 'renew', 'crisi', 'corn', 'stonei', 'aqueou', 'enumberrnumb', 'credibl', 'ownership', 'draft', 'arraylist', 'linkedlist', 'zimbabw', 'numbercnumberenumb', 'powel', 'beneficiari', 'provis', 'shore', 'secreci', 'ministri', 'passphras', 'crossov', 'bowl', 'ryanair', 'cna', 'marriott', 'african', 'strengthen', 'glm', 'para', 'rmi', 'diesel', 'unison', 'tt', 'propag', 'jurisdict', 'gh', 'apg', 'nri', 'rogu', 'treati', 'mso', 'numbercnumberabafnumberefnumbercnumb', 'eacut', 'minist', 'imgsnumb', 'asp', 'dilbert', 'numberdocu', 'afnumb', 'realnetwork', 'cnumbera', 'monarch', 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'idnumb', 'enenkio', 'taongi', 'atol', 'fraudul', 'marshalles', 'hermio', 'majesti', 'remio', 'sovereign', 'eneen', 'kio', 'iroijlaplap', 'murjel', 'ratak', 'referenc', 'nato', 'oraclenumberi', 'jnumbere', 'maxaman', 'wap', 'sarun', 'wnumberp', 'numberad', 'pjnumber', 'aaaa', 'dnumberdnumberdnumberdnumberdnumberdnumberdnumberdnumberdnumberdnumberdnumberdnumberdnumberdnumberdnumberdnumberdnumberdnumberdnumberdnumberdnumberdnumberdnumberdnumberdnumberdnumberdnumberdnumberdnumberdnumberdnumberdnumberdnumberdnumberdnumberdnumberdnumberdnumb', 'iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii', 'piiiiiiii', 'iiiiiiihnumberjnumberhnumberjnumberhnumb', 'jnumberiiiiiiihepihepihf', 'iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii', 'numberpd', 'iiiiiiiiiiiiiiiiiiiiiiip', 'iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii', 'iiiiiiiiip', 'iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiij', 'isonumb', 'amz', 'numberazp', 'wdm']
(4199, 3395)
'''
