# DIRECT-Feedback (data/feedback_data_partial.csv)

### Citation
Liermann, W., Huang X., Lee, Y., Lee, K. (2024, November). More Insightful Feedback for Tutoring: Enhancing Generation Mechanisms and Automatic Evaluation. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing.

#### Data Descriptions

This dataset holds feedback presented in the context of interactive reading comprehension exercises. It is an extension of DIRECT based on RACE. For DIRECT, incorrect answers were constructed by selecting faulty answer options from a pre-defined multiple choice form. Those are often totally unrelated to the actual reading passage. We decided to construct additional data with more natural answers, including mistakes that students are likely to make in an environment where only the reading passage and no answer options are provided. For each question in the DIRECT dataset, one annotator in the student role constructs such an answer, then another annotator in the tutor role constructs the corresponding feedback. Both annotators are presented with the reading passage, the question, its correct answer, and the corresponding key sentences. Five annotators with some level of English proficiency worked on the student role, while two native English-speaking annotators worked on the tutor role. The latter were also asked to periodically review randomly selected portions of the constructed data, including both incorrect answers and tutor feedback (constructed by the other worker). They ensured that the percentage of erroneous data items remained below 5%.

#### Data Statistics

We maintain the split of the RACE data set which assigns each reading passage with all its questions to exactly one split.

| | Train | Validation | Test | | Total |
| :---: | :---: | :---: | :---: | :-: | :---: |
| **DIRECT**  |   |   |   | |   |
| Dialogues | 2707 (5066) | 301 | 301 || 5668  |
| Feedback Turns | 5026 (9431) | 475 | 525 || 10431  |
| **DIRECT-Feedback** |   |   |   | |   |
| Dialogues | 2722 (5095) | 302 | 307 || 5704 |
| Feedback Turns | **11440** (21463) | **1239** | **1280** || 23982 |
| **Total** |   |   |   | |   |
| Dialogues | 2722 (5095)  | 302  | 307  | | 5704  |
| Feedback Turns | **16466** (30894)  | **1714**  | **1805**  | | 34413  |

We publish the whole validation and test set. The training set is published in parts (around 50% of the original data).


# Use Baseline Model
We provide our whole model specifically finetuned for the task of feedback generation at ... This model was trained on the entire train set as given in brackets above.

## Usage

#### Test the model
```
python test.py data/config/default.yaml --load PATH_TO_DOWNLOADED_MODEL
```

## Set Up

#### Step 0:
Create and activate a new clean conda environment:
```
>>> conda create -n myenv python=3.9
>>> conda activate myenv
```

#### Step 1:
In order to install the appropriate pytorch version, first find your CUDA version:
In Windows Powershell or Linux standard terminal:
```
>>> nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.65.01    Driver Version: 516.94       CUDA Version: **11.7**     |
|-------------------------------+----------------------+----------------------+
...
```

#### Step 2:
Find your matching pytorch version and copy the comand from:
https://pytorch.org/get-started/previous-versions/
```
>>> conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

#### Step 3
Install all standard dependencies:
```
>>> pip install -r requirements.txt
```


#### Step 4:
Prepare the data. You will need two files `article-id_mapping.json` and `feedback_data_partial.csv` placed in `data/` folder.  

`article-id_mapping.json` holds a mapping from article ids to the article text:
```
{
    "1": "This is an example article.",
    ...
}
```
`feedback_data_partial.csv` is a tab-separated file with the following columns:
```
set\tfile_id\tquestion_id\tquestion\tkey_sentence\tcorrect_answer\twrong_answer\tfeedback
```
