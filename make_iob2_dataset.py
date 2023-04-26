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
import glob
import html
import itertools
import json
import os
import pdb
import re
import unicodedata

from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        default="dataset/pretokenized/train",
        help="教師データのpretokenize結果の出力先ディレクトリのパス"
    )
    parser.add_argument(
        "--annotations",
        default="dataset/ene-20221116/annotation",
        help="アノテーションファイルの格納先ディレクトリのパス"
    )
    parser.add_argument(
        "--ene_definition",
        default="dataset/attribute_extraction/Shinra2022_ENE_Definition_v9.0.0-with-Attributes_20220714.jsonl",
        help="ENE定義辞書のパス"
    )
    parser.add_argument("--outputs", default="dataset/model_input", help="オフセット変換後の格納先ディレクトリのパス")
    args = parser.parse_args()

    input_files = glob.glob(os.path.join(args.inputs, "**", "*.json"), recursive=True)
    annotation_files = glob.glob(os.path.join(args.annotations, "*_dist.jsonl"))
    assert len(annotation_files) == 178, print(len(annotation_files))
    annotations = {k: v for iter in map(parse_annotation, annotation_files) for k, v in iter.items()}

    # ENE定義の読み込み
    with open(args.ene_definition, "r", encoding="utf-8") as f:
        ene_definition = tuple(map(json.loads, f.read().splitlines()))
    target_attrs = {d["name"]["en"]: {a["name"]: a["extraction_task"] for a in d["attributes"]} for d in ene_definition if "attributes" in d.keys()}

    input_jsons = list(
        tqdm(
            map(
                calibration,
                input_files,
                itertools.repeat(annotations),
                itertools.repeat(target_attrs),
            ), total=len(input_files)
        )
    )

    os.makedirs(args.outputs, exist_ok=True)
    with open(os.path.join(args.outputs, "whole.json"), "w", encoding="utf-8") as f:
        for e in input_jsons:
            f.write(f"{json.dumps(e, ensure_ascii=False)}\n")


def calibration(input_file, annotations, target_attrs):
    page_id = os.path.basename(input_file).rstrip(".json")
    ene = os.path.basename(os.path.dirname(input_file))

    # アノテーションを持たないpage_idはcontinue
    try:
        annots = annotations[ene][page_id]
    except KeyError:
        print("keyerror*", ene, page_id)
        return {page_id: None}

    with open(input_file, "r", encoding="utf-8") as f:
        input_json = json.loads(f.read())

    iob_tag = ["O"] * len(input_json["tokens"])
    for annot in annots:
        attribute = annot["attribute"]
        if not target_attrs[ene][attribute]:
            print(f"{attribute} is not a extraction target. The attribute has been removed.")
            continue
        attribute = unicodedata.normalize("NFKC", attribute).strip()

        s_l, s_o = annot["html_offset"]["start"]["line_id"], annot["html_offset"]["start"]["offset"]
        e_l, e_o = annot["html_offset"]["end"]["line_id"], annot["html_offset"]["end"]["offset"]
        if (s_l, s_o) == (0, 0) or (e_l, e_o) == (0, 1):
            continue

        start_token_cand = [(i, each) for i, each in enumerate(input_json["offsets"]) if each[0] == s_l]
        start_token = [i for i, each in start_token_cand if each[1] == s_o]
        if start_token:
            start_token = min(start_token)
        else:
            start_token = [i for i, each in start_token_cand if each[1] <= s_o]
            if start_token:
                start_token = max(start_token)
            else:
                start_token = min([i for i, _ in start_token_cand])
        end_token_cand = [(i, each) for i, each in enumerate(input_json["offsets"]) if each[2] == e_l]
        end_token = [i for i, each in end_token_cand if each[3] == e_o]
        if end_token:
            end_token = max(end_token)
        else:
            end_token = [i for i, each in end_token_cand if each[3] >= e_o]
            if end_token:
                end_token = min(end_token)
            else:
                end_token = max([i for i, _ in end_token_cand])
        end_token += 1

        # オフセット範囲が重複するアノテーションは最初に参照したもののみIOBに変換する
        if not all([tag == "O" for tag in iob_tag[start_token:end_token]]):
            continue

        tokens = [token[2:] if token.startswith("##") else token for token in input_json["tokens"][start_token:end_token]]
        # entity = "".join(tokens)
        part_of_entity = "".join(tokens[1:-1])
        normalized_e_text = re.sub(
            r"\s+",
            "",
            unicodedata.normalize(
                "NFKC",
                html.unescape(
                    re.sub(
                        r"<.*?>",
                        "",
                        annot["html_offset"]["text"]
                    )
                )
            )
        )
        # assert normalized_e_text in entity, pdb.set_trace()
        assert normalized_e_text not in part_of_entity

        if start_token == end_token + 1:
            iob_tag[start_token] = f"B-{attribute}"
        else:
            n_tokens = end_token - start_token
            iob_tag[start_token:end_token] = [f"I-{attribute}"] * n_tokens
            iob_tag[start_token] = f"B-{attribute}"

    # bbox=[0,0,0,0]のシーケンスを削除する
    rm_indices = [i for i, bbox in enumerate(input_json["bboxes"]) if bbox == [0, 0, 0, 0]]
    rm_indices.sort(reverse=True)
    for i in rm_indices:
        del input_json["tokens"][i]
        del input_json["bboxes"][i]
        del input_json["offsets"][i]
        del iob_tag[i]
    assert len(input_json["tokens"]) \
           == len(input_json["bboxes"]) \
           == len(input_json["offsets"]) \
           == len(iob_tag)

    # bbox の正規化
    n_bboxes = [normalize_bbox(bbox, input_json["body_bbox"]) for bbox in input_json["bboxes"]]
    assert all([all(map(lambda v: 0 <= v <= 1000, e)) for e in n_bboxes]), pdb.set_trace()
    input_json["bboxes"] = n_bboxes

    input_json["ene"] = ene
    input_json["ner_tags"] = iob_tag
    return input_json


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


def parse_annotation(annotation_file):
    ene = os.path.basename(annotation_file)[:-11]
    with open(annotation_file, "r", encoding="utf-8") as f:
        annotation_tuple = tuple(map(json.loads, f.read().splitlines()))
    # group by page_id
    annotation = collections.defaultdict(list)
    keys = ["title", "attribute", "ENE", "html_offset"]
    for e in annotation_tuple:
        annotation[e["page_id"]].append({key: e[key] for key in keys})
    return {ene: annotation}


if __name__ == "__main__":
    main()
