# ArabicaQA: Comprehensive Dataset for Arabic Question Answering

ArabicaQA is a robust dataset designed to support and advance the development of Arabic Question Answering (QA) systems. This dataset encompasses a wide range of question types, including both Machine Reading Comprehension (MRC) and Open-Domain questions, catering to various aspects of QA research and application. The dataset is structured to facilitate training, validation, and testing of Arabic QA models.

## Dataset Overview

ArabicaQA is divided into several segments to address different QA challenges:

- **Machine Reading Comprehension (MRC)**: Contains questions with provided context paragraphs and specified answers. It includes both answerable and unanswerable questions to mimic real-world scenarios where some questions may not have straightforward answers.
- **Open-Domain QA**: Designed for scenarios where questions are asked in an open context, encouraging models to retrieve relevant information from a broad dataset.
- **Retriever Training Data**: Offers structured data to train retriever models, which are crucial for identifying relevant context or documents from a large corpus.

### Dataset Statistics

| Category             | Training | Validation | Test  |
|----------------------|----------|------------|-------|
| MRC (with answers)   | 62,186   | 13,483     | 13,426|
| MRC (unanswerable)   | 2,596    | 561        | 544   |
| Open-Domain          | 62,057   | 13,475     | 13,414|
| Open-Domain (Human)  | 58,676   | 12,715     | 12,592|

## Download Links

### MRC Dataset

Structured as JSON files, the MRC dataset includes `train.json`, `val.json`, and `test.json` for training, validation, and testing phases, respectively, along with a metadata CSV file.

- **Data Structure**:
```json
{
  "data": [
    {
      "title": "",
      "paragraphs": [
        {
          "context": "",
          "qas": [
            {
              "question": "",
              "id": "",
              "answers": [
                {
                  "answer_start": 0,
                  "text": ""
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

- **Training Set**: [Download](https://huggingface.co/datasets/abdoelsayed/ArabicaQA/resolve/main/MRC/train.json?download=true)
- **Validation Set**: [Download](https://huggingface.co/datasets/abdoelsayed/ArabicaQA/resolve/main/MRC/val.json?download=true)
- **Test Set**: [Download](https://huggingface.co/datasets/abdoelsayed/ArabicaQA/resolve/main/MRC/test.json?download=true)
- **Metadata**: [Download](https://huggingface.co/datasets/abdoelsayed/ArabicaQA/resolve/main/MRC/all_data_meta.csv?download=true)
### Open-Domain QA Dataset

Available in both JSON and JSONL formats, this part of the dataset is annotated by humans for realistic QA scenarios.

- **Data Structure**:
```
[
    {
        "question_id": "",
        "answer_id": "",
        "question": "",
        "answer": ""
    }
]
```

- **JSON Format**: [Train](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/human-annotated/train-open.json?download=true) | [Validation](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/human-annotated/val-open.json?download=true) | [Test](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/human-annotated/test-open.json?download=true)
- **JSONL Format**: [Train](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/human-annotated/train-open.jsonl?download=true) | [Validation](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/human-annotated/val-open.jsonl?download=true) | [Test](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/human-annotated/test-open.jsonl?download=true)

### Retriever Training Data

This section provides datasets for training retrieval models, crucial for efficient information extraction and context identification.

- **Data Structure**:
```
[
    {
        "question": "...",
        "answers": ["...", "...", "..."],
        "positive_ctxs": [{
            "title": "...",
            "text": "...."
            }],
        "negative_ctxs": ["..."],
        "hard_negative_ctxs": ["..."]
    }
]
```


- **Haystack Annotated**: [Train](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/retreiver/haystack/arabica-train.json?download=true) | [Validation](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/retreiver/haystack/arabica-dev.json?download=true) | [Test](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/retreiver/haystack/arabica-test.json?download=true)
- **Human Annotated**: [Train](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/retreiver/human-annotated/arabica-train.json?download=true) | [Validation](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/retreiver/human-annotated/arabica-dev.json?download=true) | [Test](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/retreiver/human-annotated/arabica-test.json?download=true)
- **CSV Format**: [Train](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/retreiver/csv/arabica-train.csv?download=true) | [Validation](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/retreiver/csv/arabica-dev.csv?download=true) | [Test](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/blob/main/retreiver/csv/arabica-test.csv)

### Retriever Data Output

Outputs from the retrieval models, showcasing the effectiveness of different retrieval strategies (DPR, BM25) in context selection.

- **Data Structure**:
```
[
    {
    "question": "...",
    "answers": ["...", "..."],
    "ctxs": [
        {
            "id": "...",
            "title": "",
            "text": "....",
            "score": "...",
            "has_answer": true|false
        }
     ]
    }
]
```

- **DPR Output**: [Train](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/retreiver_output/DPR/arabica-train.json?download=true) | [Validation](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/retreiver_output/DPR/arabica-dev.json?download=true) | [Test](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/retreiver_output/DPR/arabica-test.json?download=true)
- **BM25 Output**: [Train](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/retreiver_output/BM25/arabica-train.json?download=true) | [Validation](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/blob/main/retreiver_output/BM25/arabica-dev.json) | [Test](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/retreiver_output/BM25/arabica-test.json?download=true)