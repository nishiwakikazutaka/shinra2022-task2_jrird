#!/usr/bin/env python3.8
# coding=utf-8
# Copyright 2023 The Japan Research Institute, Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import html
import json
import os
import re
import unicodedata

import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer, default_data_collator
from transformers.models.layoutlm.modeling_layoutlm import LayoutLMForTokenClassification


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BSZ = 256


class DatasetForInference(torch.utils.data.Dataset):
    """Custom Dataset Class"""
    def __init__(self, dataset_path, tokenizer):
        global DEVICE
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = list(map(json.loads, f.read().splitlines()))
        input_ids = list(map(lambda datapoint: tokenizer.convert_tokens_to_ids(datapoint["tokens"]), dataset))
        attention_mask = []
        bboxes = []
        word_ids = []
        offsets = []
        for i in range(len(input_ids)):
            # inputs
            attention_mask.append([])
            word_ids.append([])
            input_ids[i].insert(0, tokenizer.cls_token_id)
            input_ids[i].append(tokenizer.sep_token_id)
            attention_mask[i] = [1] * len(input_ids[i])
            word_ids[i] = [None] + list(range(len(input_ids[i]) - 1))
            word_ids[i][-1] = None  # for [SEP] token
            # bboxes
            bbox = [b for b in dataset[i]["bboxes"]]
            bbox.insert(0, [0, 0, 0, 0])  # for [CLS] token
            bbox.append([1000, 1000, 1000, 1000])  # for [SEP] token
            # offsets
            offsets.append([])
            offsets[i] = [[-100, -100, -100, -100]] + [o for o in dataset[i]["offsets"]]
            offsets[i].append([-100, -100, -100, -100])
            # padding
            assert dataset[i]["max_seq_length"] + 2 == 512
            while len(input_ids[i]) < 512:
                input_ids[i].append(tokenizer.pad_token_id)
                attention_mask[i].append(0)
                word_ids[i].append(None)
                bbox.append([0, 0, 0, 0])  # for [PAD] token
                offsets[i].append([-100, -100, -100, -100])
            assert len(input_ids[i]) \
                   == len(attention_mask[i]) \
                   == len(word_ids[i]) \
                   == len(bbox) \
                   == len(offsets[i]) \
                   == 512
            bboxes.append(bbox)
        token_type_ids = [[0] * 512] * len(dataset)

        self.input_ids = input_ids
        self.bbox = bboxes
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.page_ids = [e["page_id"] for e in dataset]
        self.word_ids = [[idx if idx is not None else -100 for idx in word_ids[i]] for i in range(len(self.input_ids))]
        split_ranges = [sum(e, []) for e in [e["split_ranges"] for e in dataset]]  # flatten
        self.split_ranges = [",".join(map(str, split_range)) for split_range in split_ranges]  # convert into str
        self.max_seq_length = [e["max_seq_length"] for e in dataset]
        self.margin = [e["margin"] for e in dataset]
        offsets = [sum(e, []) for e in offsets]  # flatten
        self.offsets = [",".join(map(str, offset)) for offset in offsets]  # convert into str

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "bbox": self.bbox[idx],
            "token_type_ids": self.token_type_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "page_ids": self.page_ids[idx],
            "word_ids": self.word_ids[idx],
            "split_ranges": self.split_ranges[idx],
            "max_seq_length": self.max_seq_length[idx],
            "margin": self.margin[idx],
            "offsets": self.offsets[idx],
        }


def main():
    global DEVICE
    global BSZ

    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--input", default="dataset/model_input/leaderboard.json")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument(
        "--ene_pred",
        default="dataset/attribute_extraction/shinra2022_AttributeExtraction_leaderboard_20220530.jsonl"
    )
    parser.add_argument(
        "--pred_column_name",
        default="AUTO.AIP.202204"
    )
    parser.add_argument(
        "--ene_definition",
        default="dataset/attribute_extraction/Shinra2022_ENE_Definition_v9.0.0-with-Attributes_20220714.jsonl",
        help="ENE定義辞書"
    )
    parser.add_argument(
        "--html_dir",
        default="dataset/attribute_extraction/html",
        help="HTMLファイルの格納先ディレクトリ"
    )
    parser.add_argument("--output", default="inference")
    args = parser.parse_args()

    # Model, config, and Tokenizer
    config = AutoConfig.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model, config=config)
    model.eval().to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer if args.tokenizer is not None else args.model)

    # Leaderboardデータに対するENE予測結果を読み込む
    with open(args.ene_pred, "r", encoding="utf-8") as f:
        ene_preds = list(map(json.loads, f.read().splitlines()))
    pageid2title = {ene_pred["page_id"]: ene_pred["title"] for ene_pred in ene_preds}
    pageid2eneid = {}
    for ene_pred in ene_preds:
        if ene_pred["ENEs"][args.pred_column_name]:
            pageid2eneid[ene_pred["page_id"]] = sorted(ene_pred["ENEs"][args.pred_column_name], key=lambda v: v["prob"], reverse=True)[0]["ENE"]
        else:
            pageid2eneid[ene_pred["page_id"]] = "9"  # IGNORED

    # extraction_task=Trueかつ属性値をもつENE一覧
    with open(args.ene_definition, "r", encoding="utf-8") as f:
        ene_definition = list(map(json.loads, f.read().splitlines()))
    eneid2attr = {definition["ENE_id"]: [unicodedata.normalize("NFKC", e["name"]).strip()
                  for e in definition["attributes"] if e["extraction_task"]]
                  for definition in ene_definition if "attributes" in definition.keys()
                  }
    whole_attributes = set(sum(eneid2attr.values(), []))

    # Dataset / DataLoader
    dataset = DatasetForInference(args.input, tokenizer)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BSZ,
        shuffle=False,
        collate_fn=default_data_collator,
    )

    # Inference
    inferences = None
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=(len(dataset) // BSZ) + 1, desc="Inference"):
            inputs = {
                "input_ids": batch["input_ids"].to(DEVICE),
                "token_type_ids": batch["token_type_ids"].to(DEVICE),
                "attention_mask": batch["attention_mask"].to(DEVICE),
            }
            if isinstance(model, LayoutLMForTokenClassification):
                inputs["bbox"] = batch["bbox"].to(DEVICE)
            logits = model(**inputs).logits
            topk = torch.topk(logits, k=5, dim=2).indices.to("cpu")
            if inferences is None:
                inferences = topk
            else:
                inferences = torch.cat((inferences, topk), dim=0)

    # Alignment: トークン列と予測結果のアラインメント
    page_ids = set([e["page_ids"].split("_")[0] for e in dataset])
    predictions = []
    for page_id in tqdm(page_ids, desc="alignment"):
        # HTMLファイルの読み込み。`make_jsonl` の引数として渡し、オフセット範囲のテキスト抽出に用いる
        with open(os.path.join(args.html_dir, f"{page_id}.html"), "r", encoding="utf-8") as f:
            splitted_html = f.read().splitlines()

        # 予測結果
        indices = [i for i, e in enumerate(dataset) if e["page_ids"].split("_")[0] == page_id]
        input_ids = [dataset[idx]["input_ids"] for idx in indices]
        labels = [[[config.id2label[tag_id] for tag_id in infer_tags] for infer_tags in inferences[idx].tolist()] for idx in indices]  # page -> token -> tag
        # オフセット情報の展開
        offsets, offsets_ = [], [dataset[idx]["offsets"].split(",") for idx in indices]
        for offset in offsets_:
            assert len(offset) % 512 == 0
            offsets.append([[offset[i], offset[i + 1], offset[i + 2], offset[i + 3]] for i in range(0, len(offset), 4)])

        # 入力列の前方・後方の特殊トークンに対応するidxを列から削除する
        trim_maps = list(map(make_trimmap, [dataset[idx]["word_ids"] for idx in indices]))
        input_ids = list(map(trim_seq, input_ids, trim_maps))
        labels = list(map(trim_seq, labels, trim_maps))
        offsets = list(map(trim_seq, offsets, trim_maps))

        # スライディングウィンドウをマージする
        split_ranges = set([dataset[idx]["split_ranges"] for idx in indices])
        max_seq_length = set([dataset[idx]["max_seq_length"] for idx in indices])
        margin = set([dataset[idx]["margin"] for idx in indices])
        assert len(split_ranges) == len(max_seq_length) == len(margin) == 1
        split_ranges, max_seq_length, margin = split_ranges.pop(), max_seq_length.pop(), margin.pop()
        split_ranges_ = split_ranges.split(",")
        split_ranges = [[int(split_ranges_[i]), int(split_ranges_[i + 1])] for i in range(0, len(split_ranges_), 2)]
        merge_map = make_merge_sliding_window_map(input_ids, max_seq_length, margin)
        m_input_ids, m_labels, m_offsets = [], [], []
        for i, (s, e) in enumerate(merge_map):
            m_input_ids.extend(input_ids[i][s:e])
            m_labels.extend(labels[i][s:e])
            m_offsets.extend(offsets[i][s:e])
        assert len(m_input_ids) == len(m_labels) == len(m_offsets) == max(sum(split_ranges, []))
        tokens = tuple(map(tokenizer.decode, m_input_ids))

        # ENEに含まれていない属性の予測結果を削除し、出力形式に変換する
        ene_id = pageid2eneid[page_id]
        title = pageid2title[page_id]
        if ene_id in eneid2attr.keys():
            # 予測ENEに含まれない属性を削除する
            attributes_of_ene = eneid2attr[ene_id]
            ner_tags = []
            for topk_labels in m_labels:
                appended = False
                for label in topk_labels:
                    attribute = label[2:] if label != "O" else None
                    if attribute is None:
                        # Oタグが最もスコアの高い予測結果の場合
                        assert label == "O"
                        ner_tags.append(label)
                        appended = True
                        break
                    assert attribute in whole_attributes
                    if attribute not in attributes_of_ene:
                        # 予測されたENEに存在しない属性はスキップする
                        continue
                    if attribute in attributes_of_ene:
                        # 予測されたENEに存在する属性は採用する
                        ner_tags.append(label)
                        appended = True
                        break
                    else:
                        print("Unexpected pattern.")
                if appended is False:
                    ner_tags.append("O")
            # IOBタグの修正
            ner_tags = fix_o_tags(ner_tags)
            assert len(ner_tags) == len(tokens) == len(m_offsets)

            # offsets の情報を使ってオフセット形式の出力に変換し、jsonlineとしてdumpする
            is_inside_entity = False
            attribute = None
            for token, ner_tag, (s_l, s_o, e_l, e_o) in zip(tokens, ner_tags, m_offsets):
                if not is_inside_entity and (ner_tag.startswith("B-") or ner_tag.startswith("I-")):
                    # new
                    o = [[s_l, s_o, e_l, e_o]]
                    t = [token[2:] if token.startswith("##") else token]
                    attribute = ner_tag[2:]
                    is_inside_entity = True
                elif is_inside_entity and ner_tag.startswith("I-"):
                    if ner_tag[2:] != attribute:
                        # attribute が一致しない場合、別の属性の開始とみなす
                        predictions.append(make_jsonl(page_id, title, ene_id, attribute, o, t, splitted_html, args))
                        # renew
                        o = [[s_l, s_o, e_l, e_o]]
                        t = [token[2:] if token.startswith("##") else token]
                        attribute = ner_tag[2:]
                    else:
                        # add
                        o.append([s_l, s_o, e_l, e_o])
                        t.append(token[2:] if token.startswith("##") else token)
                elif is_inside_entity and ner_tag.startswith("B-"):
                    predictions.append(make_jsonl(page_id, title, ene_id, attribute, o, t, splitted_html, args))
                    # renew
                    o = [[s_l, s_o, e_l, e_o]]
                    t = [token[2:] if token.startswith("##") else token]
                    attribute = ner_tag[2:]
                elif is_inside_entity and ner_tag == "O":
                    predictions.append(make_jsonl(page_id, title, ene_id, attribute, o, t, splitted_html, args))
                    # reset
                    del o, t, attribute
                    is_inside_entity = False
                elif not is_inside_entity and ner_tag == "O":
                    continue
                else:
                    print("Unexpected pattern")

        else:
            # 予測結果なしで出力
            pass

    # out
    out = os.path.join(args.output, f"{args.model}_inference_output.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(f"{json.dumps(pred, ensure_ascii=False)}\n")
    print("Done")


def make_trimmap(word_id, ref_i=-1):
    # -100 は special token を表す
    assert word_id[0] == -100  # 冒頭は必ずCLSトークン
    if word_id[ref_i] == -100:
        return make_trimmap(word_id, ref_i=ref_i - 1)
    else:
        ref_i += 1
        trimmap = [True] * len(word_id)
        trimmap[0] = False
        trimmap[ref_i:] = [False] * abs(ref_i)
        return trimmap


def trim_seq(seq, trimmap):
    assert len(seq) == len(trimmap)
    return [s for s, t in zip(seq, trimmap) if t is True]


def make_merge_sliding_window_map(input_ids, seq_length=510, margin=128):
    med_stride = margin // 2
    merge_map = []
    merged_input_ids = []
    for idx in range(len(input_ids)):
        token_length = len(input_ids[idx])
        assert token_length <= seq_length
        s = med_stride if idx > 0 else 0
        e = token_length - med_stride if idx < len(input_ids) - 1 else token_length
        merged_input_ids.extend(input_ids[idx][s:e])
        merge_map.append((s, e))
    return merge_map


def fix_o_tags(ner_tags, search_limit=5):
    """予測結果のOタグを修正する

    エンティティ途中の記号や助詞がOタグとなり、I-タグの予測が途切れる場合があるため、
    次点の予測結果が当該のI-タグと同じであれば間のOタグはすべてI-タグに置換する
    そうでなければ別個のエンティティとしてB-タグを付与する"""
    fix_indices = [i for i, tag in enumerate(ner_tags) if i > 0 and ner_tags[i - 1] == "O" and tag.startswith("I-")]
    for idx in fix_indices:
        itag_idx = idx
        itag = ner_tags[itag_idx]
        attribute = itag[2:]
        assert ner_tags[idx - 1] == "O"
        idx -= 2
        while idx > max(0, itag_idx - search_limit):
            if ner_tags[idx] == itag:
                # 同じ属性のIタグにOタグが挟まれている場合、Oタグを該当するIタグに置き換える
                ner_tags[idx:itag_idx] = [itag] * (itag_idx - idx)
                break
            elif ner_tags[idx] != "O":
                # 別のエンティティとして最初のIタグをBタグに置き換える
                ner_tags[itag_idx] = f"B-{attribute}"
                break
            else:
                idx -= 1
        if idx <= 0:
            # タグ列の冒頭までOタグが続いた場合、最初のタグをBタグに置き換える
            ner_tags[itag_idx] = f"B-{attribute}"
    return ner_tags


def make_jsonl(page_id, title, ene_id, attribute, offset, tokens, splitted_html, args):
    """
    - http://2022.shinra-project.info/data-format
    - http://2022.shinra-project.info/AttributeExtraction2.png
    """
    start_line_id = min([int(each[0]) for each in offset])
    start_offset = min([int(each[1]) for each in [s_l for s_l in offset if int(s_l[0]) == start_line_id]])
    end_line_id = max([int(each[2]) for each in offset])
    end_offset = max([int(each[3]) for each in [e_l for e_l in offset if int(e_l[2]) == end_line_id]])

    if start_line_id == end_line_id:
        text = splitted_html[start_line_id][start_offset:end_offset]
    else:
        # start_line_id と end_line_id の差が2以上の場合を考慮
        for line_id in range(start_line_id, end_line_id + 1):
            if line_id == start_line_id:
                text = splitted_html[line_id][start_offset:]
            elif line_id == end_line_id:
                text += splitted_html[line_id][:end_offset]
            else:
                text += splitted_html[line_id]

    if remove_tags(text) != "".join(tokens):
        out = os.path.join(args.output, f"{args.model}_assertion_error.txt")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "a", encoding="utf-8") as f:
            print(f"Got: {text}, Expected: {''.join(tokens)}", file=f)

    return {
        "page_id": page_id,
        "title": title,
        "ENE": ene_id,
        "attribute": attribute,
        "html_offset": {
            "start": {
                "line_id": start_line_id,
                "offset": start_offset,
            },
            "end": {
                "line_id": end_line_id,
                "offset": end_offset,
            },
            "text": text
        }
    }


def remove_tags(text):
    text = re.sub(r"<script>.*?</script>", "", text)
    text = re.sub(r"<style.*?>.*?</style>", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u0020", "") if text[0] == "\u0020" else text  # \u00b4 を normalize すると \u0020\u0301 になる。空白スペースを削除する
    text = re.sub(r"\s", "", text)
    return text


if __name__ == "__main__":
    main()
