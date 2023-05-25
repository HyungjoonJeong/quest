## Code Peer Review Template
---
* 코더 : 정형준
* 리뷰어 : 김창완


## PRT(PeerReviewTemplate)
---
- 코드가 정상적으로 동작하고 주어진 문제를 해결했나요? X
	- 아쉽게도 끝까지는 못하셨습니다
	- 시각화와 85%의 acc에 도달하지 못하여 아쉽게도 X 드렸습니다

-  주석을 보고 작성자의 코드가 이해되었나요? O  
	- 대부분의 코드에 주석이 있어 의도는 파악이 되었습니다
```python
def load_data(train_data, test_data, num_words=10000):
    # 데이터의 중복 제거
    train_data = list(set(train_data))
    test_data = list(set(test_data))
    
    # NaN 결측치 제거
    train_data = [data for data in train_data if str(data) != 'nan']
    test_data = [data for data in test_data if str(data) != 'nan']
    
    # 토큰화 및 불용어 제거
    X_train = [remove_stopwords(tokenizer.morphs(sentence)) for sentence in train_data]
    X_test = [remove_stopwords(tokenizer.morphs(sentence)) for sentence in test_data]
    
    # 단어 집합 생성
    words = np.concatenate(X_train).tolist()
    counter = Counter(words)
    if num_words is None:
        counter = counter.most_common(num_words)
    else:
        counter = counter.items()
    word_to_index = {word: index + 2 for index, (word, _) in enumerate(counter)}
    word_to_index['<PAD>'] = 0  # 패딩을 위한 토큰
    word_to_index['<UNK>'] = 1  # OOV(out-of-vocabulary)를 위한 토큰
```

-  코드가 에러를 유발할 가능성이 있나요? O
	- numpy array가 아니라 리스트가 들어간거같습니다
```python
/tmp/ipykernel_122/203380902.py in <listcomp>(.0)
      3 
      4 def get_encoded_sentences(sentences, word_to_index):
----> 5     return [word_to_index['<PAD>']] + [word_to_index[word] if word in word_to_index else word_to_index['<UNK>'] for word in sentences]
      6 
      7 # def get_encoded_sentences(sentences, word_to_index):

TypeError: unhashable type: 'list'
```
	
- [x] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)  
	- 아쉽게도 경력이 길지는 않으셔서 그런지 코드를 이해하지는 못하셨습니다. 다만 GPT를 활용해 어떻게든 정답을 찾으시려는 노력은 있었습니다.

- [x] 코드가 간결한가요?  
	- 코드 자체는 간결했습니다
