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
import collections
import json
import os
import random

from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset/model_input/whole.json", help="分割対象のjsonパス")
    parser.add_argument("--outputs", default=None, help="分割後のjson格納先パス")
    parser.add_argument("--max_seq_length", default=510, type=int, help="スライディングウィンドウの幅")
    parser.add_argument("--margin", default=128, type=int, help="スライディングウィンドウの重複幅")
    parser.add_argument("--test_ratio", default=0.1, type=float, help="テストセットの割合")
    parser.add_argument("--seed", default=42, type=int, help="乱数シード")
    args = parser.parse_args()

    random.seed(args.seed)

    print("loading json")
    with open(args.input, "r", encoding="utf-8") as f:
        jsonlines = tuple(map(json.loads, f.read().splitlines()))

    # ENE毎のデータ集合からargs.test_ratioの割合だけテストセットとしてサンプリング
    print("sampling")
    train_indices, test_indices = [], []
    enes = sorted(set([e["ene"] for e in jsonlines]))  # 再現性のためソートしておく
    for ene in tqdm(enes):
        indices = [i for i, e in enumerate(jsonlines) if e["ene"] == ene]
        # テストセットに未知のラベルクラスが含まれないようにする
        testset_candidate_indices = []
        for target in tqdm(indices, desc=f"{ene}: "):
            ner_tags_in_target = set(jsonlines[target]["ner_tags"])
            ner_tags_in_rest = set(sum([jsonlines[i]["ner_tags"] for i in indices if i != target], []))
            if ner_tags_in_target - ner_tags_in_rest == set():
                testset_candidate_indices.append(target)
        # args.test_ratioに沿ってサンプリング
        n_test = int(len(indices) * args.test_ratio)
        if len(testset_candidate_indices) < n_test:
            test_indices.extend(testset_candidate_indices)
        else:
            random.shuffle(testset_candidate_indices)
            test_indices.extend(testset_candidate_indices[:n_test])
        train_indices.extend([i for i in indices if i not in test_indices])
    assert set(train_indices) | set(test_indices) == set(range(len(jsonlines)))
    assert len(train_indices) + len(test_indices) == len(jsonlines)
    # サンプリング結果の確認
    train_test_ratio = collections.defaultdict(dict)
    for ene in enes:
        train_test_ratio[ene] = {
            "train": len([jsonlines[i] for i in train_indices if jsonlines[i]["ene"] == ene]),
            "test": len([jsonlines[i] for i in test_indices if jsonlines[i]["ene"] == ene]),
        }
    print(train_test_ratio)

    # スライディングウィンドウでデータセットを分割
    print("splitting")
    train, test = [], []
    for train_index in train_indices:
        train.extend(
            divide_by_sliding_window(
                jsonlines[train_index],
                max_seq_length=args.max_seq_length,
                margin=args.margin)
        )
    for test_index in test_indices:
        test.extend(
            divide_by_sliding_window(
                jsonlines[test_index],
                max_seq_length=args.max_seq_length,
                margin=args.margin)
        )

    # 出力
    outputs = os.path.dirname(args.input) if args.outputs is None else args.outputs
    os.makedirs(outputs, exist_ok=True)
    train_o, test_o = os.path.join(outputs, "train.json"), os.path.join(outputs, "dev.json")
    with open(train_o, "w", encoding="utf-8") as f:
        for e in train:
            f.write(f"{json.dumps(e, ensure_ascii=False)}\n")
    with open(test_o, "w", encoding="utf-8") as f:
        for e in test:
            f.write(f"{json.dumps(e, ensure_ascii=False)}\n")

    print("done")


def divide_by_sliding_window(jsonline, max_seq_length=510, margin=128):
    # 分割範囲を求める
    original_seq_length = len(jsonline["tokens"])
    adopt_length = max_seq_length - margin
    split_ranges = [
        (i, min(i + max_seq_length, original_seq_length))
        for i in range(0, original_seq_length - margin, adopt_length)
    ] if original_seq_length > margin else [(0, original_seq_length)]
    # 分割範囲に基づいてトークン列を分割
    ret = []
    for i, (s, e) in enumerate(split_ranges):
        id_ = f"{jsonline['page_id']}_{i}" if len(split_ranges) > 1 else jsonline["page_id"]
        ret.append({
            "tokens": jsonline["tokens"][s:e],
            "ner_tags": jsonline["ner_tags"][s:e],
            "bboxes": jsonline["bboxes"][s:e],
            "page_id": id_,
            "offsets": jsonline["offsets"][s:e],
            "body_bbox": jsonline["body_bbox"],
            "ene": jsonline["ene"],
            "split_ranges": split_ranges,
            "max_seq_length": max_seq_length,
            "margin": margin,
        })
    return ret


def list_candidates_for_valid(data, page_ids=None):
    page_ids = page_ids if page_ids else set(d["page_id"].split("_")[0] for d in data)
    has_not_unique_tags = []
    for page_id in page_ids:
        # page_id 毎に GroupBy してタグ集合の論理演算を行う
        datapoint = [each_data for each_data in data if each_data["page_id"].split("_")[0] == page_id]
        rest_datapoint = [each_data for each_data in data if not each_data["page_id"].split("_")[0] == page_id]
        assert len(datapoint) + len(rest_datapoint) == len(data)
        ner_tags_in_dp = set(tag for dp in datapoint for tag in dp["ner_tags"])
        ner_tags_in_rdp = set(tag for rdp in rest_datapoint for tag in rdp["ner_tags"])
        if ner_tags_in_dp - ner_tags_in_rdp == set():
            has_not_unique_tags.append(page_id)
    return has_not_unique_tags


if __name__ == "__main__":
    main()
