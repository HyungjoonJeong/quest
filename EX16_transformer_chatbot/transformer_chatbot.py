{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "09416e09",
   "metadata": {},
   "outputs": [],
   "source": [
    "import re\n",
    "import tensorflow_datasets as tfds\n",
    "import tensorflow as tf\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a0ee9443",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "51854215",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "7fa95b22",
   "metadata": {},
   "source": [
    "# 질문과 답변의 쌍인 데이터셋을 구성하기 위한 데이터 로드 함수\n",
    "def load_conversations():\n",
    "    id2line = {}\n",
    "    with open(path_to_movie_lines, errors='ignore') as file:\n",
    "        lines = file.readlines()\n",
    "    for line in lines:\n",
    "        parts = line.replace('\\n', '').split(' +++$+++ ')\n",
    "        id2line[parts[0]] = parts[4]\n",
    "\n",
    "    inputs, outputs = [], []\n",
    "    with open(path_to_movie_conversations, 'r') as file:\n",
    "        lines = file.readlines()\n",
    "\n",
    "    for line in lines:\n",
    "        parts = line.replace('\\n', '').split(' +++$+++ ')\n",
    "        conversation = [line[1:-1] for line in parts[3][1:-1].split(', ')]\n",
    "\n",
    "        for i in range(len(conversation) - 1):\n",
    "            inputs.append(preprocess_sentence(id2line[conversation[i]]))\n",
    "            outputs.append(preprocess_sentence(id2line[conversation[i + 1]]))\n",
    "\n",
    "            if len(inputs) >= MAX_SAMPLES:\n",
    "                return inputs, outputs\n",
    "    return inputs, outputs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "4eed5fa7",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'path_to_movie_lines' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m/tmp/ipykernel_164/849143256.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;31m# 데이터를 로드하고 전처리하여 질문을 questions, 답변을 answers에 저장\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0mquestions\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0manswers\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mload_conversations\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;32m/tmp/ipykernel_164/1676036879.py\u001b[0m in \u001b[0;36mload_conversations\u001b[0;34m()\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0;32mdef\u001b[0m \u001b[0mload_conversations\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m     \u001b[0mid2line\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m{\u001b[0m\u001b[0;34m}\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 4\u001b[0;31m     \u001b[0;32mwith\u001b[0m \u001b[0mopen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpath_to_movie_lines\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0merrors\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'ignore'\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mfile\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      5\u001b[0m         \u001b[0mlines\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mfile\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mreadlines\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      6\u001b[0m     \u001b[0;32mfor\u001b[0m \u001b[0mline\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mlines\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mNameError\u001b[0m: name 'path_to_movie_lines' is not defined"
     ]
    }
   ],
   "source": [
    "# 데이터를 로드하고 전처리하여 질문을 questions, 답변을 answers에 저장\n",
    "questions, answers = load_conversations()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f3a1d6e0",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 질문과 답변 데이터셋에 대해서 Vocabulary 생성\n",
    "\n",
    "tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(questions + answers, target_vocab_size=2**13)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2f8654d0",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 시작 토큰과 종료 토큰에 고유한 정수 부여\n",
    "\n",
    "START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]\n",
    "\n",
    "print('START_TOKEN의 번호 :' ,[tokenizer.vocab_size])\n",
    "print('END_TOKEN의 번호 :' ,[tokenizer.vocab_size + 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "53d76b56",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 시작 토큰과 종료 토큰을 고려하여 +2를 하여 단어장의 크기 산정\n",
    "\n",
    "VOCAB_SIZE = tokenizer.vocab_size + 2\n",
    "\n",
    "print(VOCAB_SIZE)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3b128c6a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 샘플의 최대 허용 길이 또는 패딩 후의 최종 길이\n",
    "\n",
    "MAX_LENGTH = 40\n",
    "\n",
    "print(MAX_LENGTH)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e55e7c9e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 정수 인코딩, 최대 길이를 초과하는 샘플 제거 및 패딩\n",
    "\n",
    "def tokenize_and_filter(inputs, outputs):\n",
    "  tokenized_inputs, tokenized_outputs = [], []\n",
    "  \n",
    "  for (sentence1, sentence2) in zip(inputs, outputs):\n",
    "    # 정수 인코딩 과정에서 시작 토큰과 종료 토큰 추가\n",
    "    sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN\n",
    "    sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN\n",
    "\n",
    "    # 최대 길이 MAX_LENGTH 이하인 경우에만 데이터셋으로 허용\n",
    "    if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:\n",
    "        tokenized_inputs.append(sentence1)\n",
    "        tokenized_outputs.append(sentence2)\n",
    "  \n",
    "    # 최대 길이 MAX_LENGTH로 모든 데이터셋을 패딩\n",
    "    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(\n",
    "        tokenized_inputs, maxlen=MAX_LENGTH, padding='post')\n",
    "    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(\n",
    "        tokenized_outputs, maxlen=MAX_LENGTH, padding='post')\n",
    "  \n",
    "    return tokenized_inputs, tokenized_outputs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f9393c47",
   "metadata": {},
   "outputs": [],
   "source": [
    "questions, answers = tokenize_and_filter(questions, answers)\n",
    "print('단어장의 크기 :',(VOCAB_SIZE))\n",
    "print('필터링 후의 질문 샘플 개수: {}'.format(len(questions)))\n",
    "print('필터링 후의 답변 샘플 개수: {}'.format(len(answers)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "64789ae0",
   "metadata": {},
   "outputs": [],
   "source": [
    "BATCH_SIZE = 64\n",
    "BUFFER_SIZE = 20000\n",
    "\n",
    "# 디코더는 이전의 target을 다음의 input으로 사용하므로 outputs에서 START_TOKEN 제거\n",
    "dataset = tf.data.Dataset.from_tensor_slices((\n",
    "    {\n",
    "        'inputs': questions,\n",
    "        'dec_inputs': answers[:, :-1]\n",
    "    },\n",
    "    {\n",
    "        'outputs': answers[:, 1:]\n",
    "    },\n",
    "))\n",
    "\n",
    "dataset = dataset.cache()\n",
    "dataset = dataset.shuffle(BUFFER_SIZE)\n",
    "dataset = dataset.batch(BATCH_SIZE)\n",
    "dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d3993f89",
   "metadata": {},
   "outputs": [],
   "source": [
    "def transformer(vocab_size,\n",
    "                num_layers,\n",
    "                units,\n",
    "                d_model,\n",
    "                num_heads,\n",
    "                dropout,\n",
    "                name=\"transformer\"):\n",
    "    inputs = tf.keras.Input(shape=(None,), name=\"inputs\")\n",
    "    dec_inputs = tf.keras.Input(shape=(None,), name=\"dec_inputs\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3f6f45f2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 인코더에서 패딩을 위한 마스크\n",
    "\n",
    "enc_padding_mask = tf.keras.layers.Lambda(\n",
    "    create_padding_mask, output_shape=(1, 1, None),\n",
    "    name='enc_padding_mask')(inputs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "851fe13c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 디코더에서 미래의 토큰을 마스크 하기 위해 사용\n",
    "# 내부적으로 패딩 마스크도 포함됨\n",
    "\n",
    "look_ahead_mask = tf.keras.layers.Lambda(\n",
    "    create_look_ahead_mask,\n",
    "    output_shape=(1, None, None),\n",
    "    name='look_ahead_mask')(dec_inputs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "48585e2b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 두 번째 어텐션 블록에서 인코더의 벡터들을 마스킹\n",
    "# 디코더에서 패딩을 위한 마스크\n",
    "\n",
    "dec_padding_mask = tf.keras.layers.Lambda(\n",
    "    create_padding_mask, output_shape=(1, 1, None),\n",
    "    name='dec_padding_mask')(inputs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3dc46910",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 인코더\n",
    "\n",
    "enc_outputs = encoder(\n",
    "    vocab_size=vocab_size,\n",
    "    num_layers=num_layers,\n",
    "    units=units,\n",
    "    d_model=d_model,\n",
    "    num_heads=num_heads,\n",
    "    dropout=dropout,\n",
    ")(inputs=[inputs, enc_padding_mask])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7e070c60",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 디코더\n",
    "\n",
    "dec_outputs = decoder(\n",
    "    vocab_size=vocab_size,\n",
    "    num_layers=num_layers,\n",
    "    units=units,\n",
    "    d_model=d_model,\n",
    "    num_heads=num_heads,\n",
    "    dropout=dropout,\n",
    ")(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "13522028",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 완전연결층\n",
    "\n",
    "outputs = tf.keras.layers.Dense(units=vocab_size, name=\"outputs\")(dec_outputs)\n",
    "\n",
    "    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "53cb0179",
   "metadata": {},
   "outputs": [],
   "source": [
    "tf.keras.backend.clear_session()\n",
    "\n",
    "# 하이퍼파라미터\n",
    "NUM_LAYERS = 2 # 인코더와 디코더의 층의 개수\n",
    "D_MODEL = 256 # 인코더와 디코더 내부의 입, 출력의 고정 차원\n",
    "NUM_HEADS = 8 # 멀티 헤드 어텐션에서의 헤드 수 \n",
    "UNITS = 512 # 피드 포워드 신경망의 은닉층의 크기\n",
    "DROPOUT = 0.1 # 드롭아웃의 비율\n",
    "\n",
    "model = transformer(\n",
    "    vocab_size=VOCAB_SIZE,\n",
    "    num_layers=NUM_LAYERS,\n",
    "    units=UNITS,\n",
    "    d_model=D_MODEL,\n",
    "    num_heads=NUM_HEADS,\n",
    "    dropout=DROPOUT)\n",
    "\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2fab45a3",
   "metadata": {},
   "outputs": [],
   "source": [
    "def loss_function(y_true, y_pred):\n",
    "  y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))\n",
    "  \n",
    "  loss = tf.keras.losses.SparseCategoricalCrossentropy(\n",
    "      from_logits=True, reduction='none')(y_true, y_pred)\n",
    "\n",
    "  mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)\n",
    "  loss = tf.multiply(loss, mask)\n",
    "\n",
    "  return tf.reduce_mean(loss)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1b6b1829",
   "metadata": {},
   "outputs": [],
   "source": [
    "class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):\n",
    "\n",
    "  def __init__(self, d_model, warmup_steps=4000):\n",
    "    super(CustomSchedule, self).__init__()\n",
    "\n",
    "    self.d_model = d_model\n",
    "    self.d_model = tf.cast(self.d_model, tf.float32)\n",
    "\n",
    "    self.warmup_steps = warmup_steps\n",
    "\n",
    "  def __call__(self, step):\n",
    "    arg1 = tf.math.rsqrt(step)\n",
    "    arg2 = step * (self.warmup_steps**-1.5)\n",
    "\n",
    "    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b1155044",
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_learning_rate = CustomSchedule(d_model=128)\n",
    "\n",
    "plt.plot(sample_learning_rate(tf.range(200000, dtype=tf.float32)))\n",
    "plt.ylabel(\"Learning Rate\")\n",
    "plt.xlabel(\"Train Step\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4e9a178b",
   "metadata": {},
   "outputs": [],
   "source": [
    "learning_rate = CustomSchedule(D_MODEL)\n",
    "\n",
    "optimizer = tf.keras.optimizers.Adam(\n",
    "    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)\n",
    "\n",
    "def accuracy(y_true, y_pred):\n",
    "  y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))\n",
    "  return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)\n",
    "\n",
    "model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "12fce8c6",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "EPOCHS = 10\n",
    "model.fit(dataset, epochs=EPOCHS, verbose=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5c1f607b",
   "metadata": {},
   "outputs": [],
   "source": [
    "def decoder_inference(sentence):\n",
    "  sentence = preprocess_sentence(sentence)\n",
    "\n",
    "  # 입력된 문장을 정수 인코딩 후, 시작 토큰과 종료 토큰을 앞뒤로 추가.\n",
    "  # ex) Where have you been? → [[8331   86   30    5 1059    7 8332]]\n",
    "  sentence = tf.expand_dims(\n",
    "      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)\n",
    "\n",
    "  # 디코더의 현재까지의 예측한 출력 시퀀스가 지속적으로 저장되는 변수.\n",
    "  # 처음에는 예측한 내용이 없음으로 시작 토큰만 별도 저장. ex) 8331\n",
    "  output_sequence = tf.expand_dims(START_TOKEN, 0)\n",
    "\n",
    "  # 디코더의 인퍼런스 단계\n",
    "  for i in range(MAX_LENGTH):\n",
    "    # 디코더는 최대 MAX_LENGTH의 길이만큼 다음 단어 예측을 반복합니다.\n",
    "    predictions = model(inputs=[sentence, output_sequence], training=False)\n",
    "    predictions = predictions[:, -1:, :]\n",
    "\n",
    "    # 현재 예측한 단어의 정수\n",
    "    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)\n",
    "\n",
    "    # 만약 현재 예측한 단어가 종료 토큰이라면 for문을 종료\n",
    "    if tf.equal(predicted_id, END_TOKEN[0]):\n",
    "      break\n",
    "\n",
    "    # 예측한 단어들은 지속적으로 output_sequence에 추가됩니다.\n",
    "    # 이 output_sequence는 다시 디코더의 입력이 됩니다.\n",
    "    output_sequence = tf.concat([output_sequence, predicted_id], axis=-1)\n",
    "\n",
    "  return tf.squeeze(output_sequence, axis=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5d1ed741",
   "metadata": {},
   "outputs": [],
   "source": [
    "def sentence_generation(sentence):\n",
    "  # 입력 문장에 대해서 디코더를 동작 시켜 예측된 정수 시퀀스를 리턴받습니다.\n",
    "  prediction = decoder_inference(sentence)\n",
    "\n",
    "  # 정수 시퀀스를 다시 텍스트 시퀀스로 변환합니다.\n",
    "  predicted_sentence = tokenizer.decode(\n",
    "      [i for i in prediction if i < tokenizer.vocab_size])\n",
    "\n",
    "  print('입력 : {}'.format(sentence))\n",
    "  print('출력 : {}'.format(predicted_sentence))\n",
    "\n",
    "  return predicted_sentence"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d162a011",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 사용할 샘플의 최대 개수\n",
    "MAX_SAMPLES = 50000\n",
    "print(MAX_SAMPLES)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "315fedbb",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 사용할 샘플의 최대 개수\n",
    "MAX_SAMPLES = 50000\n",
    "print(MAX_SAMPLES)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "117c09e6",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3b9cc225",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "77489da2",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
