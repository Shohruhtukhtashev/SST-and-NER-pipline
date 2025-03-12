# Project Docs

# STT(Speech to Text)
## About model
Whisper Small is an AI model that can recognize and translate speech with impressive accuracy. Trained on 680,000 hours of audio data, it can handle a wide range of languages and accents, and even translate speech in real-time.

#### I used the [Common Voice](http://huggingface.co/mozilla-foundation/common_voice_11_0) dataset for fine-tuning the Whisper model.
#### Dataset size is 0.82 GB

## STT for Uzbek language

### Fine tuning results
Model is trained on P100 GPU with 16 GB of VRAM on Kaggle for ~8.5 hours

### Whisper fine tuning codes
### 📗 [kaggle notebook](https://www.kaggle.com/code/shohruhtuxtashev/whisper-small-uz-github/edit)

### It achieves the following results on the evaluation set:
- Loss: 0.2628
- Wer: 23.1694

## Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 16
- eval_batch_size: 8
- seed: 42
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- training_steps: 4000
- mixed_precision_training: Native AMP

## Training results

| Training Loss | Epoch  | Step | Validation Loss | Wer     |
|:-------------:|:------:|:----:|:---------------:|:-------:|
| 0.4646        | 0.6720 | 1000 | 0.3688          | 32.9186 |
| 0.268         | 1.3441 | 2000 | 0.2925          | 26.4408 |
| 0.1436        | 2.0161 | 3000 | 0.2646          | 23.4813 |
| 0.1436        | 2.6882 | 4000 | 0.2628          | 23.1694 |

### Use the model from the Hugging Face platform, you can use the following code:
**Whisper fine tuning model link**
  - ✨[Fine tuning model](https://huggingface.co/tukhtashevshohruh/whisper-small-uz)

## Loading and Using the Model
```python
from transformers import pipeline

# Load the model
pipe = pipeline("automatic-speech-recognition", model="tukhtashevshohruh/whisper-small-uz")

# Convert the audio file to text
audio_file = "my_audio.wav"  # Replace with the name of your own file
text = pipe(audio_file)

# Print the result
print("Text:", text['text'])
```
## Examples

```python
from transformers import pipeline

# Load the model
pipe = pipeline("automatic-speech-recognition", model="tukhtashevshohruh/whisper-small-uz")
audio_file = "test_1.wav"
text = pipe(audio_file)

# print result
print("Matn:", text['text'])
```

Results:
```ptyhon
Matn: propaganda uch turt besh yoxud miyaning chirishi bu kontentdan olgan xulasalarim go‘yo ichimdagi ovozga o‘xshaydi ustozlar jamoasiga hurmat ham ilm fan xantibi va diniy tomondan hozirgi kun vabosini yoritishibdi baraka topkorlar
```
----

----

# Named Entity Recognition (NER)

## About model
#### Fine tuning the xlm-roberta-base (270M) multilingual transformer model on [uzbek_ner](http://huggingface.co/risqaliyevds/uzbek_ner) dataset
####  XLM-RoBERTa-base is a powerful multilingual transformer model, pre-trained on 2.5TB of filtered CommonCrawl data across 100 languages
#### Dataset size is 24.7 MB

## XLM-RoBERTa-base model Fine tuning for Uzbek language
### Preparing dataset codes 👉 [Notebook](https://github.com/Shohruhtukhtashev/STT-and-NER-pipeline/blob/master/ner-preparing-data.ipynb)
I have explained in the notebook how I cleaned and processed the dataset.

### xlm-roberta-base fine-tuning codes
### 📗[Notebook](https://github.com/Shohruhtukhtashev/STT-and-NER-pipeline/blob/master/fine%20tuning%20codes/xlm-roberta-lowercase-github.ipynb)

## Hyperparameter tuning
### Find best hyperparameter codes
### 📗[Notebook](https://github.com/Shohruhtukhtashev/STT-and-NER-pipeline/blob/master/fine%20tuning%20codes/find-best-hyperparameter.ipynb)
* ### Learning rate
| Learning Rate | Training Loss | Validation Loss | Precision | Recall  | F1       |
|--------------|--------------|----------------|-----------|---------|---------|
| 1e-05       | 0.144500     | 0.138469       | 0.500501  | 0.482579 | 0.491376 |
| 2e-05       | 0.131000     | 0.126477       | 0.549881  | 0.624554 | 0.584843 |
| 3e-05       | 0.130000     | 0.126789       | 0.565236  | 0.631792 | 0.596664 |
| 5e-05       | 0.128600     | 0.127545       | 0.577286  | 0.626484 | 0.600879 |

From these results, it is clear that when **lr = 5e-05**, the model achieved the best **F1-score**. This learning rate value is the default setting.  
F1-score measures the balance between **Precision** and **Recall**, making it the most important metric for **NER**.

- ### Scheduler and Weight decays
![f1_scores_visualization_small](https://github.com/user-attachments/assets/9d9131ab-251d-491d-81d8-638aa65088b3)

The graph shows that the best values are **Scheduler = linear** and **Weight Decay = 0.01**  
The best **F1 Score** is **0.550**.
> [!NOTE]
> The values for hyperparameter tuning were calculated with **epoch = 1**.

## Fine tuning results
#### Model is trained on P100 GPU with 16 GB of VRAM on Kaggle for 38 minutes
### Categories
The model can identify the following NER categories:
- **DATE(Date expressions)**
- **LAW(Laws or regulations)**
- **LOC(Location names)**
- **ORG(Organization names)**
- **PERSON(Person names)**
### It achieves the following results on the evaluation set:
* Loss: 0.124728
* Precision: 0.588804
* Recall: 0.611965
* F1: 0.600161

### Training hyperparameters

#### The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 2
- eval_batch_size: 4
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 8
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- num_epochs: 3
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Precision | Recall | F1     |
|:-------------:|:-----:|:----:|:---------------:|:---------:|:------:|:------:|
| 0.1407        | 1.0   | 2206 | 0.1291          | 0.5569    | 0.5721 | 0.5644 |
| 0.1169        | 2.0   | 4412 | 0.1228          | 0.5769    | 0.6093 | 0.5926 |
| 0.0959        | 3.0   | 6618 | 0.1247          | 0.5888    | 0.6120 | 0.6002 |


### XLM-RoBERTa-base fine tuning model link

- **✨[Fine tuning model](https://huggingface.co/tukhtashevshohruh/xlm-roberta-base-lowercase-high-accuracy)**

## Loading and Using the Model
```python
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

model_name = "tukhtashevshohruh/xlm-roberta-base-lowercase-high-accuracy"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)

nm = pipeline("ner", model=model, tokenizer=tokenizer)
```
## Examples
```python
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

model_name_or_path = "tukhtashevshohruh/xlm-roberta-base-lowercase-high-accuracy"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)

text = "Samarqanddanmikan yoki Toshkentdanmikan anig'ini bilmadim' dedi Shohruh umirov"
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

ner = nlp(text)

for entity in ner:
    print(entity)
```
Results:
```python
{'entity': 'B-LOC', 'score': 0.74778676, 'index': 2, 'word': 'Sam', 'start': 1, 'end': 4}
{'entity': 'I-LOC', 'score': 0.76890993, 'index': 3, 'word': 'ar', 'start': 4, 'end': 6}
{'entity': 'I-LOC', 'score': 0.77585006, 'index': 4, 'word': 'qan', 'start': 6, 'end': 9}
{'entity': 'I-LOC', 'score': 0.7489506, 'index': 5, 'word': 'd', 'start': 9, 'end': 10}
{'entity': 'B-LOC', 'score': 0.76504, 'index': 10, 'word': '▁Toshkent', 'start': 24, 'end': 32}
{'entity': 'B-PERSON', 'score': 0.9199773, 'index': 23, 'word': '▁Sho', 'start': 65, 'end': 68}
{'entity': 'I-PERSON', 'score': 0.925115, 'index': 24, 'word': 'h', 'start': 68, 'end': 69}
{'entity': 'I-PERSON', 'score': 0.926187, 'index': 25, 'word': 'ruh', 'start': 69, 'end': 72}
{'entity': 'I-PERSON', 'score': 0.93724865, 'index': 26, 'word': '▁um', 'start': 73, 'end': 75}
{'entity': 'I-PERSON', 'score': 0.9402853, 'index': 27, 'word': 'i', 'start': 75, 'end': 76}
{'entity': 'I-PERSON', 'score': 0.9250853, 'index': 28, 'word': 'rov', 'start': 76, 'end': 79}
```

----

----

# STT and NER pipeline
## About
#### At this stage, you integrate STT and NER:  
* Audio file is input → The STT model converts it into text → The NER model identifies key entities in the text

### pipeline codes
### 📗 [Notebook](https://www.kaggle.com/code/shohruhtuxtashev/notebooka9776e3a4f/edit)
