# Data Augmentatons Techniques in NLP:

## 1. Random Deletion :
In Random Deletion, we randomly delete a word if a uniformly generated number between 0 and 1 is smaller than a pre-defined threshold. This allows for a random deletion of some words of the sentence.
```py
def random_deletion(words, p):

    words = words.split()
    
    #obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    #randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    #if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    sentence = ' '.join(new_words)
    
    return sentence
```

```py
print(random_deletion(trial_sent,0.2))
# 2am feedings for the baby fun are when he is all smiles and coos

print(random_deletion(trial_sent,0.3))
# feedings for the baby he all smiles and coos

print(random_deletion(trial_sent,0.4))
# 2am for the baby are fun when all and
```

## 2. Random Swap :
In Random Deletion, we randomly delete a word if a uniformly generated number between 0 and 1 is smaller than a pre-defined threshold. This allows for a random deletion of some words of the sentence.
```py
def swap_word(new_words):
    
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        
        if counter > 3:
            return new_words
    
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
    return new_words
```

```py
def random_swap(words, n):
    
    words = words.split()
    new_words = words.copy()
    # n is the number of words to be swapped
    for _ in range(n):
        new_words = swap_word(new_words)
        
    sentence = ' '.join(new_words)
    
    return sentence
```

```py
print(random_swap(trial_sent,2))
# 2am feedings for the baby and fun when coos is all smiles are he
```

## 3. Back Translation for Text Augmentation
### 3.1 Using Back Translation in Google Sheets
- Step 1: Load your data

<img src="../images/Screenshot from 2021-12-08 14-09-52.png">

- Step 2: Add a column to hold augmented data

Add a new column and use the GOOGLETRANSLATE() function to translate from English to French and back to English. The Command is:

`=GOOGLETRANSLATE(GOOGLETRANSLATE(A2, "en", "fr"), "fr", "en")`

<img src="../images/Screenshot from 2021-12-08 14-11-47.png">

- Step 3: Filter out duplicated data

For texts where the original text and what get back are the same, we can filter them out programaticaly by comparing the original text column and augumented columns. Then, only keep responses that have True value in the Changed column

<img src="../images/Screenshot from 2021-12-08 14-14-17.png">

#

## 4. Word Embeddings for Data Augmentation & Over Sampling

### Step 1: Load Word Embedding

For Vietnamese Word Embedding, we can use [PhoW2V](https://github.com/datquocnguyen/PhoW2V)

```py
def get_coefficient(word,*arr): 
    return word, np.asarray(arr, dtype='float32')

def load_embedding(file):
    embedding_index = embeddings_index = dict(get_coefficient(*index.split(" ")) for index in open(file, encoding='utf8'))
    return embedding_index

phoW2V = load_embedding('../input/embeddings/word2vec_vi_syllables_100dims.txt')
```

### Step 2: Loading Dataset

|qid|question_text|target|
|---|-------------|------|
|1|How did Quebec nationalists see their province..|0
|2|Do you have an adopted dog, how would you enco..|0
|3|Why does velocity affect time? Does velocity a..|1



### Step 3: Tokenizing

I am using Keras' Tokenizer to apply some text processing and to limit the size of the vocabulary

```py
def make_tokenizer(texts, len_vocab=100000):
    from keras.preprocessing.text import Tokenizer
    t = Tokenizer(num_words=len_vocab)
    t.fit_on_texts(texts)
    return t

tokenizer = make_tokenizer(df['question_text'], len_vocab)
```
Apply padding, mostly to store X as an array.

```py
from keras.preprocessing.sequence import pad_sequences

X = tokenizer.texts_to_sequences(df['question_text'])
X = pad_sequences(X, 70)

y = df['target'].values
```
For visualization, I'm gonna need to see which index corresponds to which word

```py
index_word = {0: ''}
for word in tokenizer.word_index.keys():
    index_word[tokenizer.word_index[word]] = word
```
**Embedding Matrix**
```py
def make_embedding_matrix(embedding, tokenizer, len_voc):
    all_embs = np.stack(embedding.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]
    word_index = tokenizer.word_index
    embedding_matrix = np.random.normal(emb_mean, emb_std, (len_voc, embed_size))
    
    for word, i in word_index.items():
        if i >= len_voc:
            continue
        embedding_vector = embedding.get(word)
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix

embed_matrix = make_embedding_matrix(phoW2V, tokenizer, len_vocab)
```

### Step 4 : Making a Synonym Dictionary

Word vectors are made in a way that similar words have similar representation. Therefore we can use the  k -nearest neighbours to get  k  synonyms.

As the process takes a bit of time, I chose to compute 5 synonyms for the 20000 most frequent words

```py
from sklearn.neighbors import NearestNeighbors

synonyms_number = 5
word_number = 20000

knn = NearestNeighbors(n_neighbors=synonyms_number + 1).fit(embed_matrix)

neighbours_matrix = nn.kneighbors(embed_matrix[1:word_number])[1]

synonyms = {x[0]: x[1:] for x in neighbours_matrix}

for x in np.random.randint(1, word_number, 10):
    print(f"{index_word[x]} : {[index_word[synonyms[x][i]] for i in range(synonyms_number-1)]}")
```

```
kj : ['sj', 'jj', 'kev', 'jn']
trainee : ['trainees', 'apprentice', 'intern', 'supervisor']
polar : ['arctic', 'bears', 'bear', 'tundra']
sino : ['iba', 'aun', 'ito', 'eso']
substitution : ['substitutions', 'substituting', 'substitutes', 'substitute']
elastic : ['stretchy', 'waistband', 'stretchable', 'straps']
task : ['tasks', 'accomplish', 'difficult', 'effort']
creatures : ['creature', 'beasts', 'monsters', 'beings']
insufficient : ['inadequate', 'sufficient', 'adequate', 'lack']
sucking : ['cock', 'dick', 'licking', 'suck']
```

```py
# Visuzlizing Wordcloud for synonyms
index = np.random.randint(1, word_number, 9)
plt.figure(figsize=(20,10))

for k in range(len(index)):
    plt.subplot(3, 3, k+1)
    
    x = index[k]
    text = ' '.join([index_word[x]] + [index_word[synonyms[x][i]] for i in range(synonyms_number-1)]) 
    wordcloud = WordCloud(stopwords=[]).generate((text))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
```

### Step 5 - Data Augmentation / Oversampling

    We work on 1 labelled texts. We apply the following algorithm to modify a sentence :
    For each word in the sentence :
    - Keep it with probability  p  (or if we don't have synonyms for it)
    - Randomly swap it with one of its synonyms with probability  1 − p

```py
def modify_sentence(sentence, synonyms, p=0.5):
    for i in range(len(sentence)):
        if np.random.random() > p:
            try:
                syns = synonyms[sentence[i]]
                sentence[i] = np.random.choice(syns)
            except KeyError:
                pass
    return sentence
```
Let's preview our function

```py
X_positive = X[y==1]

indexes = np.random.randint(0, X_positive.shape[0], 5)

for x in X_pos[indexes]:
    sample =  np.trim_zeros(x)
    sentence = ' '.join([index_word[x] for x in sample])
    print(sentence)

    modified = modify_sentence(sample, synonyms)
    sentence_m = ' '.join([index_word[x] for x in modified])
    print(sentence_m)
    
    print(' ')
```

```
i caught my son having sex with my little sister what should i do
i caught got son same sex and me much sister what should i would
 
why are so many jews so frightened that they join the craziest groups like marching on saturday is nothing really going to teach them
why are so those jew it fearful that would invite the craziest other think march both sunday is something think going up teach they
 
why do you bash ron weasley he is better than you can ever be
what n't can bash ron weasley himself is better even 'll can ever be
 
can a woman really reach orgasm while having sex with dog
can a woman really reach orgasms even one teen with puppy
```
Looks pretty good, we now generate some texts
```py
n_texts = 30000

indexes = np.random.randint(0, X_positive.shape[0], n_texts)

X_gen = np.array([modify_sentence(x, synonyms) for x in X_positive[indexes]])
y_gen = np.ones(n_texts)
```

## 5 NLP Data Augmentation using 🤗 Transformers

### 5.1 Back Translation using Transformers
This is the technique I find most interesting, here we first convert the sentence to a different language using a model and then convert it back to the target language. As we use a ML model to this, it results in sentence equivalent to the original sentence but with different words. There are various pre-trained models available on Huggingface model hub like Google T5, Facebook NMT (Neural Machine Translation) etc. In the below code I am using T5-base for English to German translation and then using Bert2Bert model for German to English translation. We can also use Fairseq models which are available for both English to German and German to English.
```py
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#English to German using the Pipeline and T5
translator_en_to_de = pipeline("translation_en_to_de", model='t5-base')

#Germal to English using Bert2Bert model
tokenizer = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_de_en", pad_token="<pad>", eos_token="</s>", bos_token="<s>")
model_de_to_en = AutoModelForSeq2SeqLM.from_pretrained("google/bert2bert_L-24_wmt_de_en")

input_text = "I went to see a movie in the theater"
en_to_de_output = translator_en_to_de(input_text)
translated_text = en_to_de_output[0]['translation_text']
print("Translated text->",translated_text)
#Ich ging ins Kino, um einen Film zu sehen.

input_ids = tokenizer(translated_text, return_tensors="pt", add_special_tokens=False).input_ids
output_ids = model_de_to_en.generate(input_ids)[0]
augmented_text = tokenizer.decode(output_ids, skip_special_tokens=True)
print("Augmented Text->",augmented_text)
```
We can see that for input text “I went to see a movie in the theater” we get the output text as “I went to the cinema to see a film” which conveys the same meaning but uses different words and in different order! We can also use different languages like English-to-French etc. to create even more variations.
#

### 5.2 Random Insertion using Transformers
In this technique we randomly insert a word in a give sentence. One way is to just randomly insert any word, however we can also use pre-trained models like BERT to insert a word based on context. Here we can use the “fill-mask” task in transformer pipeline, to insert a word.
```py

from transformers import pipeline
import random

unmasker = pipeline('fill-mask', model='bert-base-cased')

input_text = "I went to see a movie in the theater"

orig_text_list = input_text.split()
len_input = len(orig_text_list)
#Random index where we want to insert the word except at the start or end
rand_idx = random.randint(1,len_input-2)

new_text_list = orig_text_list[:rand_idx] + ['[MASK]'] + orig_text_list[rand_idx:]
new_mask_sent = ' '.join(new_text_list)
print("Masked sentence->",new_mask_sent)
#I went to see a [Mask] movie in the theater

augmented_text_list = unmasker(new_mask_sent)
augmented_text = augmented_text_list[0]['sequence']
print("Augmented text->",augmented_text)
#I went to see a new movie in the theater
```
We can see for a input text “I went to see a movie in the theater” the BERT model inserts a word “new” at a random place to create a new sentence “I went to see a new movie in the theater”, it actually gives 5 different options like “I went to see a little movie in the theater”. As we choose the index at random the word is insert at different places every time. After this we can use a similarity metric using Universal Sentence Encoder to choose the most similar sentence.
#
### 5.3 Random Replacement using Transformers
In this technique we replace a random word with a new word, we could use pre-build dictionaries to replace with synonyms or we can use pre-trained models like BERT. Here we again use the “fill-mask” pipeline.
```py
from transformers import pipeline
import random

unmasker = pipeline('fill-mask', model='bert-base-cased')

input_text = "I went to see a movie in the theater"

orig_text_list = input_text.split()
len_input = len(orig_text_list)
#Random index where we want to replace the word 
rand_idx = random.randint(1,len_input-1)

new_text_list = orig_text_list.copy()
new_text_list[rand_idx] = '[MASK]'
new_mask_sent = ' '.join(new_text_list)
print("Masked sentence->",new_mask_sent)
#I went to [MASK] a movie in the theater

augmented_text_list = unmasker(new_mask_sent)
#To ensure new word and old word are not name
for res in augmented_text_list:
  if res['token_str'] != orig_word:
    augmented_text = res['sequence']
    break
print("Augmented text->",augmented_text)
#I went to watch a movie in the theater
```
In above code in one example we random select the word “see” and using BERT replace it with the word “watch”, to produce a sentence “I went to watch a movie in the theater” with the same meaning but different words. We can also replace more than one word using the same technique. For both random insert and replacement we can also use other models which support the “fill-mask” task like Distilbert (small and fast), Roberta and even multilingual models!
#
### 5.4 Text Generation
In this technique we use generative models like GPT2, distilgpt2 etc. to make the sentences longer. We feed the original text as the start and then the model generates additional words based on the input text, this way we add random noise to our sentence. If we add only few words and make sure the sentence is similar to the original sentence using similarity score then we can generate additional sentence without changing the meaning!
```py
from transformers import pipeline
generator = pipeline('text-generation', model='gpt2')

input_text = "I went to see a movie in the theater"
input_length = len(input_text.split())
num_new_words = 5
output_length = input_length + num_new_words
gpt_output = generator(input_text, max_length=output_length, num_return_sequences=5)
augmented_text = gpt_output[0]['generated_text']
print("Augmented text->",augmented_text)
#I went to see a movie in the theater, and the director was
```
Here we use the “text-generation” pipeline and the GPT-2 model to add 5 new words to our original sentence, to get a new sentence like “I went to see a movie in the theater, and the director was”, if we decide to add 10 new words we can get a sentence like “I went to see a movie in the theater, and as I looked the other way and thought”. So we can see that there are lot of different length sentence we can generate depending on our use case.