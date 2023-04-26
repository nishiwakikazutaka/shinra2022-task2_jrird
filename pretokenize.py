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
import glob
import html
import json
import os
import re
import time
import unicodedata

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing import Pool, Manager
from threading import Thread, Lock
from unittest.mock import patch
from urllib.parse import parse_qs
import tqdm

from transformers import BertJapaneseTokenizer


def pretokenize(html_str, prefix="token_"):
    ubody, ls = [], True    # lstrip
    hc, he = max(0, html_str.find("<body ")), (html_str + "</body>").find("</body>")      # bodyタグ内を抽出
    assert html_str[hc:he].count("<script>") == html_str[hc:he].count("</script>")
    while hc < he:
        if html_str[hc] == "<":       # タグを削除
            if html_str[hc:he].startswith("<script>"):    # JavaScriptを削除
                hc += html_str[hc:he].index("</script>") + len("</script>")
            elif html_str[hc:he].startswith("<br />"):    # 改行がエンティティ区切りの場合がある
                if not ls:
                    ubody.append((hc, hc + len("<br />"), " "))
                ls = True
                hc += len("<br />")
            else:
                hc += html_str[hc:he].index(">") + 1

        elif html_str[hc].isspace():        # 改行などのスペースを削除
            if not ls:
                ubody.append((hc, hc + 1, " "))
            ls = True
            hc += 1

        else:
            ch = cu = html_str[hc]
            if html_str[hc] == "&":                   # html entitiesをデコード
                mo = html._charref.match(html_str[hc:he])
                if mo:
                    ch, cu = mo.group(0), html._replace_charref(mo)
                    if mo.group(1)[0] != "#":
                        while ch and cu and ch[-1] == cu[-1]:
                            ch, cu = ch[:-1], cu[:-1]
                    assert len(ch) >= len(cu), ("an entity reference is shorter than its character", mo.group(1), html._replace_charref(mo))

            combining = False
            if cu and ubody and unicodedata.combining(cu[0]):
                combining = True
            elif cu and ubody:
                # normalize("\u1103\u1172") -> "\ub4c0"; see https://bugs.python.org/issue43925
                # normalize("\u3137\u315f") -> "\u1103\u1171" -> "\ub4a4"
                # normalize("\u30b9\u3099") -> "\u30ba"
                # normalize("\u3099\u0304\u3099\u3099") -> "\u3099\u3099\u3099\u0304"; see 231082.html
                s, e, c = ubody[-1]
                if unicodedata.normalize("NFKC", c + cu[0]) != unicodedata.normalize("NFKC", c) + unicodedata.normalize("NFKC", cu[0]):
                    combining = True
            if combining:
                s, e, c = ubody.pop()
                ubody.append((s, hc + len(ch), c + cu))
            else:
                ubody.append((hc, hc + len(ch), cu))
            ls = False
            hc += len(ch)

    while ubody and ubody[-1][2].isspace():     # rstrip
        ubody.pop()

    # 正規化
    abody = []
    for s, e, cu in ubody:
        n = unicodedata.normalize("NFKC", cu)
        for i, c in enumerate(n):
            abody.append((s, e, i, len(n) - i - 1, c))

    abodyn = "".join(n for s, e, o, l, n in abody)

    # verify
    for s, e, o, l, n in abody:
        assert unicodedata.normalize("NFKC", html.unescape(re.sub(r"\s+", " ", re.sub(r"<.*?>", "", re.sub(r"<script>.*?</script>", "", html_str[s:e].replace("<br />", " ")), flags=re.DOTALL))))[o] == n, ("html mismatch", s, e, l, n, html_str[s:e])
    assert len(abodyn) == len(abody), ("length mismatch", len(abodyn), len(abody))

    # tokenize
    ntokens = tokenize(abodyn)

    # html to lines
    html_offsets, col, row = [], 0, 0
    for c in html_str:
        html_offsets.append((col, row))
        row += 1
        if c == "\n":
            col += 1
            row = 0

    # token to lines
    out_dict, out_offsets, bs, le = [], [], 0, ""
    for idx, token in enumerate(ntokens):
        if token:
            while abodyn[bs] != token[0] and abodyn[bs].isspace():  # tokenizer may drop spaces
                if le:
                    le += abodyn[bs]    # but normalize("\ufdfa") contains space
                bs += 1
            nhtoken = token[2:] if token[:2] == "##" else token  # for BERT-like tokenizer
            be = bs + len(nhtoken)
            assert abodyn[bs:be] == nhtoken, ("body and ntokens mismatch", idx, bs, be, token, nhtoken, abodyn[bs:be])

            hs, he = abody[bs][0], abody[be - 1][1]

            s1 = unicodedata.normalize("NFKC", html.unescape(re.sub(r"\s+", " ", re.sub(r"<.*?>", "", re.sub(r"<script>.*?</script>", "", html_str[hs:he].replace("<br />", " ")), flags=re.DOTALL))))
            s2 = le + nhtoken + abodyn[be:be + abody[be - 1][3]]
            assert s1.lstrip() == s2.lstrip(), ("result mismatch", hs, he, le, s1, token, nhtoken, abodyn[be:be + abody[be - 1][3]])  # lstrip for "\u00b4 を normalize すると \u0020\u0301 になる"
            for i, (s, e, o, l, n) in enumerate(abody[bs:be], start=bs):
                le = "" if l == 0 else le + abodyn[i]

        else:
            hs = he = abody[bs][0]

        (sc, sr), (ec, er) = html_offsets[hs], html_offsets[he]
        out_dict.append((f"{prefix}{idx}", {"token": token, "start": {"line_id": sc, "offset": sr}, "end": {"line_id": ec, "offset": er}}))
        out_offsets.append((-hs, 0, -idx, hs, f'<span id="span_{idx}" style="white-space: nowrap;" class="span-for-offset" data-token="{html.escape(token)}" data-start-byte="{hs}" data-start-line="{sc}" data-start-offset="{sr}" data-end-byte="{he}" data-end-line="{ec}" data-end-offset="{er}">'))
        out_offsets.append((-he, 1, -idx, he, "</span>"))
        bs = be

    assert not le, ("result mismatch", le)

    out_html, e = [], len(html_str)
    if prefix is None:
        out_offsets.sort()
        for _, _, _, o, h in out_offsets:
            out_html.append(html_str[o:e])
            out_html.append(h)
            e = o
        out_html.append(html_str[:e])
        out_html = "".join(reversed(out_html))
        return out_html

    return dict(out_dict)


def init():
    global g_tokenizer
    g_tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-v2")


def tokenize(t):
    global g_tokenizer

    with patch.object(g_tokenizer, "do_subword_tokenize", False):
        words = g_tokenizer.tokenize(t)

    if not g_tokenizer.do_subword_tokenize:
        return words

    words = [w for word in words for w in word.strip().split() if w]
    # subword_tokenizer.tokenize("\u1680") = []; subword_tokenizer.tokenize("\u2028\u300c") = ["\u300c"]; see 142921.html and 202817.html

    no_split_token = set(g_tokenizer.unique_no_split_tokens)
    subwords = []
    for token in words:
        if token in no_split_token:
            subwords.append(token)
        else:
            subwords.extend(g_tokenizer.subword_tokenizer.tokenize(token))

    w, s, c = 0, 0, 0
    while s < len(subwords):
        if c >= len(words[w]):
            w += 1
            c = 0
        o = 2 if c else 0
        if c == 0 and subwords[s] == g_tokenizer.unk_token != words[w]:
            subwords[s] = words[w]
            continue
        if c == 0 and subwords[s].startswith("##") and subwords[s] != g_tokenizer.unk_token:   # subword_tokenizer.tokenize("###") = ["###"]; see 14465.html
            subwords.insert(s, "#")
            subwords.insert(s + 1, "###")
            s += 2
        assert words[w][c:].startswith(subwords[s][o:]), ("tokenization failed", w, s, c, words[w], subwords[s])
        c += len(subwords[s]) - o
        s += 1

    # verify
    r1 = g_tokenizer.tokenize(t)
    r2 = [subwords[i] for i in range(len(subwords)) if ["#", "###"] not in (subwords[i:i + 2], subwords[i - 1:i + 1])]
    assert len(r1) == len(r2) and all(w1 == w2 or w1 == g_tokenizer.unk_token for w1, w2 in zip(r1, r2))

    return subwords


class Service():
    tmpl = """
<form method="POST" action="/" id="form-for-offset"><input name="outpath" value=""><input name="csum" value=""><input name="result" value=""></form>
<script>
window.onload = function() {
    var outpath = "[OUTPATH_HERE]";
    var csum = "[CSUM_HERE]";
    var pageid = "[PAGEID_HERE]";
    var tokens = [];
    var bboxes = [];
    var offsets = [];
    for(var node of document.body.querySelectorAll("span.span-for-offset")) {
        var rect = node.getBoundingClientRect();
        tokens.push(node.dataset.token);
        bboxes.push([rect.x, rect.y, rect.x + rect.width, rect.y + rect.height]);
        offsets.push([parseInt(node.dataset.startLine), parseInt(node.dataset.startOffset), parseInt(node.dataset.endLine), parseInt(node.dataset.endOffset)]);
    }
    var rect = document.body.getBoundingClientRect();
    document.forms["form-for-offset"]["outpath"].value = outpath;
    document.forms["form-for-offset"]["csum"].value = csum;
    document.forms["form-for-offset"]["result"].value = JSON.stringify({
        "page_id": pageid,
        "tokens": tokens,
        "bboxes": bboxes,
        "offsets": offsets,
        "body_bbox": [rect.x, rect.y, rect.x + rect.width, rect.y + rect.height],
    });
    document.forms["form-for-offset"].submit();
}
</script>
"""

    def Handler(self, *args, **kwargs):
        parent = self

        class MyHTTPRequestHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_GET(self):
                if self.path != "/":
                    self.send_error(404, "Not found")
                    return

                try:
                    with parent.lock:
                        outpath, csum, pageid, resp = next(parent.itr)
                except StopIteration:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write("Finished!".encode("utf-8"))
                    return

                parent.sent_outpath[outpath] = str(csum)
                resp += parent.tmpl.replace("\"[OUTPATH_HERE]\"", json.dumps(outpath)).replace("[CSUM_HERE]", str(csum)).replace("[PAGEID_HERE]", pageid)

                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(resp.encode("utf-8"))

            def do_POST(self):
                if self.path != "/":
                    self.send_error(404, "Not found")
                    return

                content_length = int(self.headers["content-length"])
                req_data = self.rfile.read(content_length).decode("utf-8")
                try:
                    req_data = parse_qs(req_data)
                    assert all(len(req_data.get(x, [])) == 1 for x in ("outpath", "csum", "result"))
                    outpath = req_data["outpath"][0]
                    csum = req_data["csum"][0]
                    result = json.loads(req_data["result"][0])
                except Exception:
                    self.send_error(400, "Invalid request")
                    return

                e_csum = None
                try:
                    if parent.sent_outpath.get(outpath) == csum:
                        e_csum = parent.sent_outpath.pop(outpath)      # atomic
                except KeyError:
                    pass

                if e_csum and e_csum == csum:
                    if parent.g_pbar is not None:
                        parent.g_pbar.set_postfix({"file": os.path.splitext(os.path.relpath(outpath, parent.output_dir) if parent.output_dir else outpath)[0]})
                        parent.g_pbar.update()

                    os.makedirs(os.path.dirname(outpath), exist_ok=True)
                    with open(outpath, "w", encoding="utf-8") as f:
                        f.write(f"{json.dumps(result, ensure_ascii=False)}\n")

                self.do_GET()

                if not parent.sent_outpath:
                    parent.finished = True

        return MyHTTPRequestHandler(*args, **kwargs)

    def watcher(self, server):
        while not self.finished:
            time.sleep(1)
        server.shutdown()

    @staticmethod
    def generate_single_response(args):
        fn, outpath, pageid, semaphore_1 = args
        try:
            with open(fn, "r", encoding="utf-8-sig") as f:
                html_str = f.read()
            ret = pretokenize(html_str, prefix=None)
            semaphore_1.acquire()
            return outpath, hash(ret), pageid, ret
        except Exception as e:
            raise ValueError(fn) from e

    def generate_responses(self, arglist):
        with Pool(processes=None) as pool, Manager() as manager:
            semaphore_1 = manager.Semaphore(1024)
            for r in pool.imap_unordered(self.generate_single_response, ((*args, semaphore_1) for args in arglist)):
                semaphore_1.release()
                yield r

    def run(self, port=8000, to_be_processed=[], output_dir=None):
        self.sent_outpath, self.finished = {}, False
        self.output_dir = output_dir
        assert to_be_processed

        self.lock = Lock()
        self.itr = self.generate_responses(to_be_processed)

        init()

        with ThreadingHTTPServer(("", port), self.Handler) as server:
            print(f"Server ready at http://localhost:{port}/")
            print(f"e.g.) /opt/google/chrome/chrome --headless --window-size=1280,854 --remote-debugging-port=9222 http://localhost:{port}/")

            with tqdm.tqdm(total=len(to_be_processed)) as self.g_pbar:
                thread = Thread(target=self.watcher, args=[server])
                thread.start()
                try:
                    server.serve_forever()
                finally:
                    self.finished = True
                    thread.join()
            self.g_pbar = None
        print("Finished!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--html_dir",
        type=str,
        default="dataset/attribute_extraction/html",
        help="教師データ内の`html`ディレクトリまでのパス。もしくはリーダーボード入力データ内の`html`ディレクトリまでのパス",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset/pretokenized",
        help="Pretokenize結果の出力先パス",
    )
    args = parser.parse_args()

    filenames = glob.glob(os.path.join(args.html_dir, "**", "*.html"), recursive=True)
    assert filenames
    to_be_processed = []
    for filename in filenames:
        filename_rel = os.path.relpath(filename, args.html_dir)
        assert filename_rel.endswith(".html")
        outpath = os.path.join(args.output_dir, filename_rel[:-len(".html")] + ".json")
        pageid = os.path.splitext(os.path.basename(filename_rel))[0]
        if pageid.isnumeric() and not os.path.exists(outpath):
            to_be_processed.append((filename, outpath, pageid))

    if to_be_processed:
        Service().run(port=8099, to_be_processed=to_be_processed, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
