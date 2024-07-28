# [ArabicaQA: Comprehensive Dataset for Arabic Question Answering](https://arxiv.org/abs/2403.17848)

ArabicaQA is a robust dataset designed to support and advance the development of Arabic Question Answering (QA) systems. This dataset encompasses a wide range of question types, including both Machine Reading Comprehension (MRC) and Open-Domain questions, catering to various aspects of QA research and application. The dataset is structured to facilitate training, validation, and testing of Arabic QA models.
## Demo
Try Our Demo [here](http://3.145.70.14:8080/)
## Requirements
```bash
# for inference
pip install torch==1.5.1
pip install faiss-cpu==1.7.3
pip install transformers==3.0.0
```
## Using AraDPR
To use our AraDPR for question answering, follow the steps below:
### Step 1: Clone AraDPR Repository
First, download the AraDPR model by cloning the repository:
```
git clone https://huggingface.co/abdoelsayed/AraDPR
```
After cloning, move the AraDPR model directory to `DPR/Model` within your project structure:
### Step 2: Clone AraDPR Index
Next, download the DPR index required for running AraDPR:
```
git clone https://huggingface.co/abdoelsayed/AraDPR_index

```
Once downloaded, move the AraDPR index directory to `DPR/DPR_index` within your project structure:
### Step 3: Wikipeda Data
Next, download the TSV:

[TSV](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/wikipedia_split/wikiAr.tsv?download=true)

Once downloaded, move the wikiAr.tsv to `wiki` within your project structure:
### Step 4: Running Inference
With the AraDPR model and index in place, you can run inference to answer questions. Edit the `inference.py` script to include your questions or use the example provided in the script.

To run the inference, execute:
```
python inference.py

```
### Step 5: Review Results
The results of your inference will be saved in `result.json`. Open this file to review the answers provided by the AraDPR model to your questions.


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
<details>
  <summary>Click to maximize</summary>
<pre><code>
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
</code></pre>
</details>


- **Training Set**: [Download](https://huggingface.co/datasets/abdoelsayed/ArabicaQA/resolve/main/MRC/train.json?download=true)
- **Validation Set**: [Download](https://huggingface.co/datasets/abdoelsayed/ArabicaQA/resolve/main/MRC/val.json?download=true)
- **Test Set**: [Download](https://huggingface.co/datasets/abdoelsayed/ArabicaQA/resolve/main/MRC/test.json?download=true)
- **Metadata**: [Download](https://huggingface.co/datasets/abdoelsayed/ArabicaQA/resolve/main/MRC/all_data_meta.csv?download=true)
### Open-Domain QA Dataset

Available in both JSON and JSONL formats, this part of the dataset is annotated by humans for realistic QA scenarios.

- **Data Structure**:
<details>
  <summary>Click to maximize</summary>
<pre><code>
[
    {
        "question_id": "",
        "answer_id": "",
        "question": "",
        "answer": ""
    }
]
</code></pre>
</details>

- **JSON Format**: [Train](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/human-annotated/train-open.json?download=true) | [Validation](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/human-annotated/val-open.json?download=true) | [Test](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/human-annotated/test-open.json?download=true)
- **JSONL Format**: [Train](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/human-annotated/train-open.jsonl?download=true) | [Validation](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/human-annotated/val-open.jsonl?download=true) | [Test](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/human-annotated/test-open.jsonl?download=true)

### Retriever Training Data

This section provides datasets for training retrieval models, crucial for efficient information extraction and context identification.

- **Data Structure**:
<details>
  <summary>Click to maximize</summary>
<pre><code>
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
</code></pre>
</details>


- **Haystack Annotated**: [Train](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/retreiver/haystack/arabica-train.json?download=true) | [Validation](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/retreiver/haystack/arabica-dev.json?download=true) | [Test](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/retreiver/haystack/arabica-test.json?download=true)
- **Human Annotated**: [Train](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/retreiver/human-annotated/arabica-train.json?download=true) | [Validation](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/retreiver/human-annotated/arabica-dev.json?download=true) | [Test](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/retreiver/human-annotated/arabica-test.json?download=true)
- **CSV Format**: [Train](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/retreiver/csv/arabica-train.csv?download=true) | [Validation](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/retreiver/csv/arabica-dev.csv?download=true) | [Test](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/blob/main/retreiver/csv/arabica-test.csv)

### Retriever Data Output

Outputs from the retrieval models, showcasing the effectiveness of different retrieval strategies (DPR, BM25) in context selection.

- **Data Structure**:
<details>
  <summary>Click to maximize</summary>
<pre><code>
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
</code></pre>
</details>

- **DPR Output**: [Train](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/retreiver_output/DPR/arabica-train.json?download=true) | [Validation](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/retreiver_output/DPR/arabica-dev.json?download=true) | [Test](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/retreiver_output/DPR/arabica-test.json?download=true)
- **BM25 Output**: [Train](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/retreiver_output/BM25/arabica-train.json?download=true) | [Validation](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/blob/main/retreiver_output/BM25/arabica-dev.json) | [Test](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/retreiver_output/BM25/arabica-test.json?download=true)

### Wikipedia data
- **Data Structure**:
```
id	text	title
```
- **Wikipedia**: [TSV](https://huggingface.co/datasets/abdoelsayed/Open-ArabicaQA/resolve/main/wikipedia_split/wikiAr.tsv?download=true)


## Trainin AraDPR

Will be avaiable soon

## Citation

If you find these codes or data useful, please consider citing our paper as:

```
@inproceedings{10.1145/3626772.3657889,
author = {Abdallah, Abdelrahman and Kasem, Mahmoud and Abdalla, Mahmoud and Mahmoud, Mohamed and Elkasaby, Mohamed and Elbendary, Yasser and Jatowt, Adam},
title = {ArabicaQA: A Comprehensive Dataset for Arabic Question Answering},
year = {2024},
isbn = {9798400704314},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3626772.3657889},
doi = {10.1145/3626772.3657889},
abstract = {In this paper, we address the significant gap in Arabic natural language processing (NLP) resources by introducing ArabicaQA, the first large-scale dataset for machine reading comprehension and open-domain question answering in Arabic. This comprehensive dataset, consisting of 89,095 answerable and 3,701 unanswerable questions created by crowdworkers to look similar to answerable ones, along with additional labels of open-domain questions marks a crucial advancement in Arabic NLP resources. We also present AraDPR, the first dense passage retrieval model trained on the Arabic Wikipedia corpus, specifically designed to tackle the unique challenges of Arabic text retrieval. Furthermore, our study includes extensive benchmarking of large language models (LLMs) for Arabic question answering, critically evaluating their performance in the Arabic language context. In conclusion, ArabicaQA, AraDPR, and the benchmarking of LLMs in Arabic question answering offer significant advancements in the field of Arabic NLP. The dataset and code are publicly accessible for further research https://github.com/DataScienceUIBK/ArabicaQA.},
booktitle = {Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {2049–2059},
numpages = {11},
keywords = {arabic question answering, information retrieval, llm, question generation},
location = {Washington DC, USA},
series = {SIGIR '24}
}
```
