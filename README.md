# DIRECT-F Dataset (data/feedback_data_partial.csv)

#### Paper
[Wencke Liermann, Jin-Xia Huang, Yohan Lee, and Kong Joo Lee. 2024. More Insightful Feedback for Tutoring: Enhancing Generation Mechanisms and Automatic Evaluation](https://aclanthology.org/2024.emnlp-main.605/). In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP 2024), pages 10838-10851, Miami, Florida, USA. Association for Computational Linguistics.

#### Data Descriptions

This dataset holds feedback presented in the context of interactive reading comprehension tutoring. It is an extension of DIRECT (Dialogue-based Reading Comprehension Tutoring) Dataset ([Huang et al., 2022](https://ieeexplore.ieee.org/document/10003215). DIRECT and DIRECT-F datasets incorporates elements derived from RACE dataset(Lai et al., 2017). 
For DIRECT, incorrect answers were constructed by selecting faulty answer options from a pre-defined multiple choice form. Those are often totally unrelated to the actual reading passage. We decided to construct additional data with more natural answers, including mistakes that students are likely to make in an environment where only the reading passage and no answer options are provided. For each question in the DIRECT dataset, one annotator in the student role constructs such an answer, then another annotator in the tutor role constructs the corresponding feedback. Both annotators are presented with the reading passage, the question, its correct answer, and the corresponding key sentences. Five annotators with some level of English proficiency worked on the student role, while two native English-speaking annotators worked on the tutor role. The latter were also asked to periodically review randomly selected portions of the constructed data, including both incorrect answers and tutor feedback (constructed by the other worker). They ensured that the percentage of erroneous data items remained below 5%. We call this new dataset DIRECT-F.

#### Format
Data is provided as a single CSV file that holds one data item per line:
```
set\tfile_id\tquestion_id\tquestion\tkey_sentence\tcorrect_answer\twrong_answer\tfeedback
```
The first column indicates whether the feedback is part of the original DIRECT dataset or our augmentation DIRECT-F.  
The second column provides a unique identifer that can be used to retrieve a reading passage using the provided JSON file `article-id_mapping.json`.  
The third and fourth column hold information on the question the student was asked to answer.  
The fifth column holds a subset of the reading passage's sentences that are relevant to the question.  
The sixth column holds the expected/correct answer to the question.  
The seventh column holds the incorrect answer provided by the student.  
The eigth column holds feedback in the form of hints, explanations or corrections as it was provided by the tutor role.  

#### Data Statistics

We maintain the split of the RACE data set which assigns each reading passage with all its questions to exactly one split.

| | Train | Validation | Test | | Total |
| :---: | :---: | :---: | :---: | :-: | :---: |
| **DIRECT**  |   |   |   | |   |
| Dialogues | 2707 (5066) | 301 | 301 || 5668  |
| Feedback Turns | 5026 (9431) | 475 | 525 || 10431  |
| **DIRECT-F** |   |   |   | |   |
| Dialogues | 2722 (5095) | 302 | 307 || 5704 |
| Feedback Turns | 11440 (21463) | 1239 | 1280 || 23982 |
| **Total** |   |   |   | |   |
| Dialogues | 2722 (5095)  | 302  | 307  | | 5704  |
| Feedback Turns | 16466 (30894)  | 1714  | 1805  | | 34413  |

We publish the whole validation and test set. The training set is published in parts (around 50% of the original data).

## Rectify Model
We provide our whole model specifically finetuned for the task of feedback generation at [HuggingFace](https://huggingface.co/etri-lirs/t5-base-rc-feedback) This model was trained on the entire train set as given in brackets above.

#### Test the model
```
python test.py data/config/default.yaml --load t5-base-rc-feedback
```

#### Set Up
1. Create and activate a new clean conda environment
```
>>> conda create -n myenv python=3.9
>>> conda activate myenv
```

2. CUDA and Pytorch setup
In order to install the appropriate pytorch version, first find your CUDA version:
In Windows Powershell or Linux standard terminal:
```
>>> nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.65.01    Driver Version: 516.94       CUDA Version: **11.7**     |
|-------------------------------+----------------------+----------------------+
...
```

Find your matching pytorch version and copy the comand from:
https://pytorch.org/get-started/previous-versions/
```
>>> conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

3. Install all standard dependencies:
```
>>> pip install -r requirements.txt
```

4. Prepare the data:
You will need two files `article-id_mapping.json` and `feedback_data_partial.csv` placed in `data/` folder.  
'article-id_mapping.json' contains a mapping of article ids to the corresponding articles from the RACE dataset. This file is provided solely for non-commercial research purposes and adheres to the terms of use of the RACE dataset.
`feedback_data_partial.csv` is a tab-separated file with the following columns:
```
set\tfile_id\tquestion_id\tquestion\tkey_sentence\tcorrect_answer\twrong_answer\tfeedback
```

## Attribution and Terms of Use:
The DIRECT-F dataset and Rectify model are released under CC BY-NC-SA 4.0 License. By using the dataset and model, you agree to the following:  

1. Non-Commercial Use Only:   
The dataset is strictly for non-commercial research purposes. Commercial use of any kind is prohibited.  
  
2. Attribution:   
Proper credit must be given to the DIRECT-F dataset (Liermann et al., 2024), the DIRECT dataset (Huang et al., 2022) and the original RACE-M dataset (Lai et al., 2017) from which parts of it are derived.  
  
3. Disclaimer:   
The dataset is provided "as-is" without warranty of any kind. The authors are not liable for any issues or outcomes arising from its use.  

4. Compliance with RACE Terms
Elements derived from the RACE dataset are subject to the terms outlined by [RACE](https://www.cs.cmu.edu/~glai1/data/race/#:~:text=notes). Users must ensure compliance with those terms.
The elements derived from the RACE dataset including:
- "file_id": Match the original "file_id" in RACE.
- "key_sentences": Annotated or extracted from the RACE "article."
- "question": Adapted to dialogue format based on the RACE "question."
- "correct_answer": Reformatted to dialogue format from the RACE "answer."
- "wrong_answer": Converted to dialogue format from the RACE "options", or newly constructed based on the "article" to simulate natural mistakes students might make without "options."

## Reference
[Jin-Xia Huang, Yohan Lee, Oh-Woog Kwon. 2022. DIRECT: Toward Dialogue-Based Reading Comprehension Tutoring](https://ieeexplore.ieee.org/document/10003215). in IEEE Access, vol. 11, pp. 8978-8987, 2023, doi: 10.1109/ACCESS.2022.3233224.  
[Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. 2017. RACE: Large-scale ReAding Comprehension Dataset From Examinations](https://aclanthology.org/D17-1082/). In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 785?794, Copenhagen, Denmark. Association for Computational Linguistics.  
