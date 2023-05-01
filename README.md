# SHINRA2022 Task2: LayoutLM / BERT

[Wikipedia構造化プロジェクト（森羅プロジェクト）](http://shinra-project.info)の[森羅2022 Task2: 属性値抽出タスク](https://2022.shinra-project.info)で用いた手法の実装です。
本手法では、属性値抽出タスクを系列ラベリング問題として扱い、ENE 毎の属性値を区別せずに1つのモデルを学習して解いています。
モデルは LayoutLM と BERT を用いており、森羅2022 のリーダーボード上では JRIRD チームとして結果を提出しています。

詳細については、[言語処理学会第29回年次大会(NLP2023)](https://www.anlp.jp/nlp2023/)にて発表した「[日本語情報抽出タスクのためのLayoutLMモデルの評価](https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/Q2-7.pdf)」も併せてご確認ください。

## 動作環境

- Ubuntu 20.04.5 LTS
- Python 3.8.13
- Google Chrome 106.0.5249.119

## 1. データ準備

[森羅2022のサブタスク固有データ配布ページ](http://2022.shinra-project.info/data-download#subtask-unique)より、以下のデータをダウンロードします。

- 属性値抽出
  - 教師データ（Fine-tuningに利用します）
  - リーダーボード入力データ（JSONL+Wikipedia2021_subset(該当ページのみ、HTML+PlainText)+拡張固有表現ver9.0）

この README では、ダウンロードしたデータセットを dataset 配下に下記の通り展開した場合を想定して記載します。
`ene-20221116` のディレクトリが教師データで、`attribute_extraction` のディレクトリがリーダーボード入力データです。

```sh
.
├── dataset
│   ├── attribute_extraction
│   │   ├── shinra2022_AttributeExtraction_leaderboard_20220530.jsonl
│   │   ├── Shinra2022_ENE_Definition_v9.0.0-with-Attributes_20220714.jsonl
│   │   ├── html
│   │   │   ├── 2246.html
│   │   │   ...
│   │   └── plain
│   │       ├── 2246.txt
│   │       ...
│   └── ene-20221116
│       ├── annotation
│       │   ├── Academic_dist_for_view.jsonl
│       │   ...
│       ├── html
│       │   ├── Academic
│       │   │   ├── 1144.html
│       │   │   ...
│       │   ...
│       └── plain
│           ├── Academic
│           │   ├── 1144.txt
│           │   ...
│           ...
```

## 2. Fine-tuning

### 2.1. 前処理

教師データのうち、HTML ファイルに対して前処理を行います。

LayoutLM の学習ではトークンの文書中における位置情報も必要となるため、HTML をブラウザで描画した際のトークンの描画位置 `bbox` を取得します。
また、属性値抽出タスクではオフセット形式でラベル情報を扱うため、トークンが元の HTML 上で何行目の何文字目に該当するのかを紐づける `offsets` を取得します。

`pretokenize.py` を実行すると、コンソール上に Server ready ... と表示されます。

```sh
$ python3 pretokenize.py \
    --html_dir dataset/ene-20221116/html \
    --output_dir dataset/pretokenized/train

Server ready at http://localhost:8099/
...
```

Server ready の状態で、Headless モードの Chrome からアクセスすると前処理が始まります。Chrome を複数起動すると処理速度が向上します。

```sh
$ /opt/google/chrome/chrome --headless --no-sandbox --disable-gpu --window-size=1280,854 --remote-debugging-port=9222 http://localhost:8099/
```

処理の進捗は `pretokenize.py` を実行したシェル上のプログレスバーから確認できます。プログレスバーが停止した場合、Chrome を一度停止し、再度実行することで処理が再開されます。

上記のコマンドライン引数を指定して実行した場合、処理結果は `dataset/pretokenized/train/{ENE}/{page_id}.json` として出力されます。

### 2.2. 教師データのラベル変換・データ分割

前処理から抽出されたトークン列を元に、教師データのアノテーションをオフセット形式から IOB2 形式に変換します。

```sh
$ python3 make_iob2_dataset.py \
    --inputs dataset/pretokenized/train \
    --annotations dataset/ene-20221116/annotation \
    --ene_definition dataset/attribute_extraction/Shinra2022_ENE_Definition_v9.0.0-with-Attributes_20220714.jsonl
```

上記のコマンドライン引数を指定して実行した場合、処理結果は `dataset/model_input/whole.json` として出力されます。

変換後、page_id 単位で train:dev にデータセットを分割します。
分割後にデータポイントが LayoutLM/BERT の入力上限長の 512 トークンを超えている場合、重複範囲を持たせたスライディングウィンドウによりさらにデータを分割します。

```sh
$ python3 train_dev_splitter.py \
    --input dataset/model_input/whole.json \
    --outputs dataset/model_input
```

上記のコマンドライン引数を指定して実行した場合、処理結果は `dataset/model_input/{train|dev}.json` として出力されます。

### 2.3. 学習

実験で用いた LayoutLM の初期重みは、[訓練データ用の Wikipedia2019 データ](http://2022.shinra-project.info/data-download#subtask-common) を使って Masked-Visual Language Model (MVLM) による事前学習を行ったモデルを利用します。

このモデルは[https://github.com/nishiwakikazutaka/layoutlm-wikipedia-ja](https://github.com/nishiwakikazutaka/layoutlm-wikipedia-ja)にて公開しており、本プロジェクト内の `models/layoutlm-wikipedia-ja` から submodule として参照しています。
本プロジェクトをクローン済みで、submodule を追加でクローンするには、以下のコマンドを実行してください。

```sh
$ git submodule update --init --recursive
```

LayoutLM の fine-tuning には、`model_name_or_path` のコマンドライン引数に models/layoutlm-wikipedia-ja を指定し、run_ner_tokenized.py を実行します。

```sh
$ python3 run_ner_tokenized.py \
    --model_name_or_path models/layoutlm-wikipedia-ja \
    --train_file dataset/model_input/train.json \
    --validation_file dataset/model_input/dev.json \
    --output_dir models/layoutlm_finetuned \
    --learning_rate 3e-5 \
    --per_device_train_batch $((8 / $(nvidia-smi -L | wc -l))) \
    --num_train_epochs 20 \
    --eval_accumulation_steps 16 \
    --save_total_limit 3 \
    --do_train \
    --do_eval \
    --fp16 \
    --fp16_opt_level O2
```

BERT を fine-tuning するには、`model_name_or_path` のコマンドライン引数に東北大　乾研究室が公開している [cl-tohoku/bert-base-japanese-v2](https://huggingface.co/cl-tohoku/bert-base-japanese-v2) を指定し、run_ner_tokenized.py を実行します。

```sh
$ python3 run_ner_tokenized.py \
    --model_name_or_path cl-tohoku/bert-base-japanese-v2 \
    --train_file dataset/model_input/train.json \
    --validation_file dataset/model_input/dev.json \
    --output_dir models/bert_finetuned \
    --learning_rate 4e-5 \
    --per_device_train_batch $((8 / $(nvidia-smi -L | wc -l))) \
    --num_train_epochs 15 \
    --eval_accumulation_steps 16 \
    --save_total_limit 3 \
    --do_train \
    --do_eval \
    --fp16 \
    --fp16_opt_level O2
```

## 3. 推論

### 3.1. 前処理

[2.1](#21-前処理) と同様に、リーダーボード入力データの HTML から、トークン列、 `bbox`、および `offsets` を抽出する前処理を行います。

```sh
$ python3 pretokenize.py \
    --html_dir dataset/attribute_extraction/html \
    --output_dir dataset/pretokenized/leaderboard

Server ready at http://localhost:8099/
...
```

```sh
$ /opt/google/chrome/chrome --headless --no-sandbox --disable-gpu --window-size=1280,854 --remote-debugging-port=9222 http://localhost:8099/
```

上記のコマンド例で実行した場合、処理結果は `dataset/pretokenized/leaderboard/{page_id}.json` として出力されます。

### 3.2. スライディングウィンドウによるデータ分割

各記事のトークン長が LayoutLM/BERT の入力上限長の 512 トークンを超えている場合、重複範囲を持たせたスライディングウィンドウによりさらにデータを分割します。

```sh
$ python3 make_leaderboard_dataset.py \
    --inputs dataset/pretokenized/leaderboard \
    --json_path dataset/attribute_extraction/shinra2022_AttributeExtraction_leaderboard_20220530.jsonl \
    --outputs dataset/model_input
```

上記のコマンドライン引数を指定して実行した場合、処理結果は `dataset/model_input/leaderboard.json` として出力されます。

### 3.3. 推論

分割されたスライディングウィンドウ毎に推論を行った結果のマージ処理と、推論結果の IOB2 からオフセット形式への変換などの後処理を行います。
以下は fine-tuning を行った LayoutLM を推論に用いる場合の実行例です。

```sh
$ python3 shinra_inference.py \
    --model models/layoutlm_finetuned \
    --output inference/layoutlm
```

### 4. 引用

```bibtex
@inproceedings{bibtex-id,
  title = {日本語情報抽出タスクのための{L}ayout{LM}モデルの評価},
  author = {西脇一尊 and 大沼俊輔 and 門脇一真},
  booktitle = {言語処理学会第29回年次大会(NLP2023)予稿集},
  year = {2023},
  pages = {522--527}
}
```
