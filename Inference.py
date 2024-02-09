import sys
import os


# Path to the directory containing DPR and subsequently DPR_module
base_path = os.path.join(os.path.dirname(__file__), 'DPR')
print(os.getcwd())

# Add both DPR and DPR_module directories to the PYTHONPATH
sys.path.append(base_path)
sys.path.append(os.path.join(base_path, 'DPR_module'))

import sys
print(sys.path)
import os
import json
import os.path as path
from DPR.DPR_Retriever import DPR_Retriever
from argparse import ArgumentParser
from sklearn.preprocessing import normalize
from whoosh.index import open_dir

import csv
csv.field_size_limit(sys.maxsize)
from sklearn.preprocessing import normalize

class Inference:
    def __init__(self, tsv_file_path):
        self.dpr = DPR_Retriever()
        self.paragraphs = self.load_paragraphs(tsv_file_path)

    def load_paragraphs(self, file_path):
        """Load paragraphs from a TSV file."""
        paragraphs = {}
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                paragraphs[row['id']] = {'text': row['text'], 'title': row['title']}
        return paragraphs

    def __normalize(self, results: dict):
        scores = [list(results.values())]
        scores = normalize(scores)
        for i, id in enumerate(results.keys()):
            results[id] = scores[0][i]

    def get_docs(self, question):
        # Retrieve documents using DPR
        dpr_result = self.dpr.get_top_docs_dpr(question,100)

        if len(dpr_result) > 0:
            self.__normalize(dpr_result)

        # Match IDs and extract context
        final_result = {}
        for id, score in dpr_result.items():
            paragraph_data = self.paragraphs.get(id, {})
            final_result[id] = {
                'paragraph_id': id,
                'context': paragraph_data.get('text', 'Context not found.'),
                'title': paragraph_data.get('title', 'Title not found.'),
                'score': score
            }

        # Sort the results based on scores
        final_result = {k: v for k, v in sorted(final_result.items(), key=lambda item: item[1]['score'], reverse=True)}

        return final_result


def main():
    """
    parser = ArgumentParser()
    parser.add_argument('--question', type=str, required=True)
    args = parser.parse_args()

    question = args.question
    """
    tsv_file_path = './DPR/wiki/wikiAr.tsv'

    inference = Inference(tsv_file_path)
    #while True:
    question = "محمد حسني مبارك" #input('Enter a question:')
    final_result = inference.get_docs(question)
    with open('result.json', mode='w', encoding='utf-8') as f:
        json.dump(final_result, f , indent=4, ensure_ascii=False )

if __name__ == '__main__':
    main()
