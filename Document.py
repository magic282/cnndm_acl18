class Document(object):
    def __init__(self, doc_sents, summary_sents):
        self.doc_sents = doc_sents
        self.summary_sents = summary_sents
        self.doc_len = len(self.doc_sents)
        self.summary_len = len(self.summary_sents)
        self.concat_summary = " ".join(self.summary_sents)
