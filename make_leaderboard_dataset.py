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
import json
import os

from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        default="dataset/pretokenized/leaderboard",
        help="予測対象のJSONファイル格納先"
    )
    parser.add_argument(
        "--json_path",
        default="dataset/attribute_extraction/shinra2022_AttributeExtraction_leaderboard_20220530.jsonl",
        help="リーダーボード予測対象のJSONファイルのパス"
    )
    parser.add_argument("--outputs", default="dataset/model_input", help="変換後のjson格納先パス")
    parser.add_argument("--max_seq_length", default=510, type=int, help="スライディングウィンドウの幅")
    parser.add_argument("--margin", default=128, type=int, help="スライディングウィンドウの重複幅")
    args = parser.parse_args()

    # ベースラインモデルの予測結果を読み込み、トークン列に変換する
    with open(args.json_path, "r", encoding="utf-8") as f:
        riken_preds = tuple(map(json.loads, f.read().splitlines()))
    leaderboard_datasets = []
    for riken_pred in tqdm(riken_preds):
        # riken_pred["ENEs"]["AUTO.AIP.202204"].sort(key=lambda item: item["prob"], reverse=True)
        # ENE_id = riken_pred["ENEs"]["AUTO.AIP.202204"][0]["ENE"]
        # 当該のpage_idについて、HTMLから抽出したトークン列他のデータを読み込む
        with open(os.path.join(args.inputs, f"{riken_pred['page_id']}.json"), "r", encoding='utf-8') as f:
            input_json = json.loads(f.read())
        # bbox=[0,0,0,0]のシーケンスを削除する
        rm_indices = [i for i, bbox in enumerate(input_json["bboxes"]) if bbox == [0, 0, 0, 0]]
        rm_indices.sort(reverse=True)
        for i in rm_indices:
            del input_json["tokens"][i]
            del input_json["bboxes"][i]
            del input_json["offsets"][i]
        assert len(input_json["tokens"]) == len(input_json["bboxes"]) == len(input_json["offsets"])
        # bboxの正規化
        n_bboxes = [normalize_bbox(bbox, input_json["body_bbox"]) for bbox in input_json["bboxes"]]
        assert all([all(map(lambda v: 0 <= v <= 1000, e)) for e in n_bboxes])
        input_json["bboxes"] = n_bboxes
        # スライディングウィンドウでデータ分割
        leaderboard_datasets.extend(divide_by_sliding_window(input_json))

    os.makedirs(args.outputs, exist_ok=True)
    with open(os.path.join(args.outputs, "leaderboard.json"), "w", encoding="utf-8") as f:
        for leaderboard_dataset in leaderboard_datasets:
            f.write(f"{json.dumps(leaderboard_dataset, ensure_ascii=False)}\n")


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
            "bboxes": jsonline["bboxes"][s:e],
            "page_id": id_,
            "offsets": jsonline["offsets"][s:e],
            "body_bbox": jsonline["body_bbox"],
            "split_ranges": split_ranges,
            "max_seq_length": max_seq_length,
            "margin": margin,
        })
    return ret


def normalize_bbox(bbox, body_bbox):
    body_x0, body_y0, body_x1, body_y1 = body_bbox
    width = body_x1 - body_x0
    height = body_y1 - body_y0
    return [
        min(1000, max(0, int(1000 * ((bbox[0] - body_x0) / width)))),
        min(1000, max(0, int(1000 * ((bbox[1] - body_y0) / height)))),
        min(1000, max(0, int(1000 * ((bbox[2] - body_x0) / width)))),
        min(1000, max(0, int(1000 * ((bbox[3] - body_y0) / height)))),
    ]


if __name__ == "__main__":
    main()
