import requests

class smoothNlpRequest(object):
    def __init__(self):
        url = "http://data.service.nlp.smoothnlp.com/"
        self.url = url

    def set_url(self,url):
        self.url = url

    def __call__(self,text):
        content = {"text":text}
        r = requests.get(self.url, params=content)
        self.result = r.json()['payload']['response']

    def dependencyrelationships(self,text):
        self.__call__(text)
        return self.result['dependencyRelationships']

    def ner(self,text):
        self.__call__(text)
        return self.result['entities']

    def number_recognize(self,text):
        entities = self.ner(text)
        if entities is None :
            return
        numbers = []
        for entity in entities:
            if entity['nerTag'].lower() == "number":
                numbers.append(entity)
        return numbers

    def company_recognize(self, text):
        entities = self.ner(text)
        if entities is None:
            return
        financial_agency = []
        for entity in entities:
            if entity['nerTag'].lower() == "company_name":
                financial_agency.append(entity)
        return financial_agency

    def segment(self,text):
        self.__call__(text)
        return [v['token'] for v in self.result['tokens']]

    def postag(self,text):
        self.__call__(text)
        return self.result['tokens']

    def analyze(self, text):
        self.__call__(text)
        return self.result

class smoothnlpDateRange(object):
    def __init__(self,url:str="http://api.smoothnlp.com/querydate"):
        self.url = url

    def __call__(self, pubdate:str="", givendate:str=""):
        content = {"givendate":givendate}
        r = requests.get(self.url, params=content)
        self.result = r.json()['payload']['response']

    def getDateRange(self,pubdate,givendate):
        self.__call__(pubdate,givendate)
        return self.result
