import re

def sentence_split(text):
    regex = re.compile("[.。]|[!?！？]+")
    return [''.join(item) for item in zip(re.split(regex,text),re.findall(regex, text))]