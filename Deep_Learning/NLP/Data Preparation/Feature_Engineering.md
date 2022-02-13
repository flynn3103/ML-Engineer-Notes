# Feature Engineering for NLP

## 1. Text preprocessing techniques:

- Converting words into lowercase: 
`my_string.lower()`
- Removing leading and trailing whitespaces
```py
s1 = '  abc  '

print(f'String =\'{s1}\'')

print(f'After Removing Leading Whitespaces String =\'{s1.lstrip()}\'')

print(f'After Removing Trailing Whitespaces String =\'{s1.rstrip()}\'')

print(f'After Trimming Whitespaces String =\'{s1.strip()}\'')
```

- Removing punctuation
```py
>>> import string
>>> string.punctuation
'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
>>> '!Hello.'.strip(string.punctuation)
'Hello'
```

- Removing stopwords (Words that occur extremely commonly)
```py
file = open('vietnamese-stopwords.txt')
stop_words = file.readlines()
file.close()

stop_words = set([word.strip('\n') for word in stop_words])

word_list = [word for word in word_list if not in stop_words]
```
- Removing special characters(numbers,emojis,etc.)
```py
  # remove urls and hashtags
    s = re.sub(r'http\S+', '', s)
    s = re.sub(r'#\S+', '', s)
 # remove email address
    s = re.sub(r'\S*@\S*\s?', '', s)
```
- Removing HTML/XMLtags
```py
import lxml.html as LH

content='''
<!--This is the XML comment -->
<p>This is a Test Paragraph</p></br>
<b>Sample Bold</b>
<Table>Sampe Text</Table>
'''

doc=LH.fromstring(content)
print(doc.text_content())
```

- Word Segmentation & Tokenization
```py
from underthesea import word_tokenize
tokens = word_tokenize(string)
# remove punctuation and number
words = [word for word in tokens if re.sub(r"\s+", "", word).isalpha()]
```
- Lemmatization
```py
lemmatization_dict={
    "ko":"không",
    "hok":"không",
    "hông":"không",
    "mn":"bạn",
    "ae":"bạn",
    "bợn":"bạn",
    "vs":"với",
    "sr":"xin lỗi",
    "help":"giúp",
    "thank":"cảm ơn",
    "thanks":"cảm ơn",
    "cám ơn":"cảm ơn",
    "tks":"cảm ơn",
    "kq":"kết quả",
    "kqua":"kết quả",
    "nhe":"nhé",
    "nhá":"nhé",
    "nha":"nhé",
    "h":"giờ",
    "s":"sao",
    "pls":"",
    "plz":"",
    "me":"mình",
    "mị":"mình",
    "đc":"được",
    "dc":"được",
    "tgian": "thời gian"
}

def replace_lemmatization(text):
    if text in lemmatization_dict:
        return lemmatization_dict[text]
    else:
        return text
```

## 2. Word Embeddings

Bag Of Word

Continous Bag Of Word

Tf-Idf

n-gram

skip-gram

Word2Vec

GloVe

Doc2Vec

