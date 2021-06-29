# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Created by Hesensen on 2021.5, thanks for part of the github crawler code provided by Cheng Jun.
"""
import glob
import json
import os
import os.path as osp
import re
import time
from collections import defaultdict

import requests
from tqdm import trange, tqdm

global_suf_to_dotnumber = {}
global_api_list_main_key = []
global_api_list_main_value = []

global_api_list_frameworkOp_key = []
global_api_list_frameworkOp_value = []

global_api_list_selfOp_key = []
global_api_list_selfOp_value = []

global_api_list_total = []
global_selfOp_to_frameworkOp = {}


def preprocess_api_list_from_file(file_path: str) -> None:
    """
    从给定的API列表文本中爬取所有的API
    并将其分为变量本身的API(*.Tensor.op)--global_api_list_extra
    和需要直接调用的API(*.op)--global_api_list_main
    :param file_path: API列表文本, txt文件
    :return: 无
    """
    with open(file_path, "r") as f:
        lines = f.read().splitlines()
        for api in lines:  # 利用framework.Tensor.op 找到所有 framework.op
            # fixed bug : torch.Tensor.tolist是tensorOp，但没有framework.tolist
            # if api == 'paddle.Tensor.mul':
            #     print(1)
            if ('.Tensor.' in api) and (not api.endswith('_')) and (api.replace("Tensor.", "") in lines):
                global_api_list_frameworkOp_key.append(api.replace("Tensor.", "") + '(')  # ["framework.op(", ...]
                global_api_list_frameworkOp_value.append(api.replace("Tensor.", ""))  # ["framework.op", ...]
                global_api_list_total.append(api.replace("Tensor.", ""))  # ["framework.op", ...]
                global_selfOp_to_frameworkOp[api] = global_api_list_frameworkOp_value[-1]

        for api in lines:
            # if api == 'paddle.Tensor.mul':
            #     print(1)
            if api in global_api_list_frameworkOp_value:  # framework.op上面已处理, 跳过
                continue
            if '.Tensor.' in api:  # 获取var.op和var.op_
                global_api_list_selfOp_key.append('.' + api.split('.')[-1] + '(')  # [".op(", ...]
                global_api_list_selfOp_value.append(api)  # ["framework.Tensor.op", ...]
            else:  # 获取普通API e.g. framework.nn.functional.mse_loss, framework.nn.ReLU......
                global_api_list_main_key.append(api + '(')  # ["framework.nn.Conv2D(", ...]
                global_api_list_main_value.append(api)  # ["framework.nn.Conv2D", ...]
            global_api_list_total.append(api)


def parse_high_star_repo(key_word: str, star_min_limit=100, num_repo_limit=1000) -> None:
    """
    用github的API爬取符合条件的repo信息，存入json中，再从json中读取repo地址和star数存入txt中
    :param key_word: 字符串关键字字符串
    :param star_min_limit: star的最少数量, 不少于这个数量的repo才会被考虑
    :param num_repo_limit: 最多考虑多少个repo
    :return:
    """
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Authorization': 'token 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',
        'Content-Type': 'application/json',
        'method': 'GET',
        'Accept': 'application/json'
    }
    t = time.time()
    num_repo = 0
    num_json = 0
    for p in trange(0, 20):
        URL = f"https://api.github.com/search/repositories?q={key_word}+stars:>{star_min_limit}&sort=stars&page={p}&per_page=100"
        result = requests.get(URL, headers=headers)
        if result.status_code != 200:
            print(f"Request status_code error - {result.status_code}")
            break
        response_dict = result.json()
        if len(response_dict['items']) == 0:
            print("There is 0 items in current request")
            break
        if not osp.exists('./json_files'):
            os.makedirs('./json_files')
            print("[*] make json_files directory at [./json_files]")
        with open(f"./json_files/{key_word}_high_star_part_{p}.json", "w") as f:
            json.dump(response_dict, f, indent=4)
        num_repo += len(response_dict['items'])
        num_json += 1
        if num_repo >= num_repo_limit:
            break

    print(f"Get [{num_repo}] repos during {time.time() - t}")

    with open(f"./json_files/repo_star_{key_word}.txt", "w") as f:
        for p in range(0, num_json):
            repo_json_path_part = f"./json_files/{key_word}_high_star_part_{p}.json"
            lines = gen_repo_star_list(repo_json_path_part, star_min_limit)
            for i, line in enumerate(lines):
                f.write(line + '\n')
                if i >= num_repo_limit:
                    break


def gen_clone_candidates(repo_star_file_path: str, key_word: str, method='https') -> None:
    """
    根据repo_star_file_path给的repo地址，用os.system来模拟手工克隆，将这些repo克隆到本地
    :param repo_star_file_path: 记录repo_i star_i的文本文件
    :param key_word: 关键字
    :param method: 用https还是git克隆
    :return: 无
    """
    assert method in ['https', 'git']
    with open(repo_star_file_path, 'r') as f:
        lines = f.read().splitlines()
        repo_urls = [line.split(' ')[0] for line in lines]
        # 下面两种方式选一种
        if method == 'git':
            repo_urls = ["git@github.com:" + repo[1:] + ".git" for repo in repo_urls]  # 设置过git账户, 使用密钥克隆速度可能快一点
        else:
            repo_urls = ["https://github.com/" + repo[1:] + ".git" for repo in repo_urls]  # 未设置过git账户，使用https克隆，速度可能慢一点

        if not osp.exists(f"./{key_word}_cover"):
            os.makedirs(f"./{key_word}_cover")
            print(f"[*] make json_files directory at [./{key_word}_cover]")

        # 1. 从文本中克隆到本地
        # with open(f"./{key_word}_cover/clone_urls.txt", 'w') as wf:
        #     for repo_url in repo_urls:
        #         wf.write(repo_url + '\n')
        # os.chdir(f"./{key_word}_cover")
        # print(f"Now in [{os.getcwd()}], preparing to clone those repos......")
        # os.system("cat clone_urls.txt | while read line ; do git clone $line $(echo $line | awk -F 'com/' '{print $2}')  ; done;")

        # 2. 或者直接用命令逐个克隆
        os.chdir(f"./{key_word}_cover")
        print(f"Now in [{os.getcwd()}], preparing to clone those repos with os.system......")
        for i, repo_url in enumerate(tqdm(repo_urls)):
            try:
                target_repo_dir_name = osp.basename(repo_url.split('.git')[0])
                if not osp.exists(target_repo_dir_name):  # 目录不存在，克隆
                    # print(f"git not exitst, git clone [{repo_url}]")
                    os.system("git clone " + repo_url)
                else:
                    os.chdir(f"./{target_repo_dir_name}")  # 目录存在，pull更新
                    os.system("git pull origin master")
                    # print(f"git exitst, use git pull to update")
                    os.chdir(os.path.dirname(os.getcwd()))  # pull完毕跳回analyse目录
            except Exception as e:
                print(e)

        print(f"Clone finished, exit to the father path in [{os.getcwd()}], preparing to analyze......")
        os.chdir(os.path.dirname(os.getcwd()))  # 克隆完毕跳回analyse目录


def parse_valid_context(file_path: str) -> list:
    """
    从给定的代码文件路径中获取每一行的字符串并, 然后过滤掉被注释的行和注释块
    :param file_path: 代码py文件的路径
    :return: 返回过滤后的行字符串list
    """
    assert osp.exists(file_path), f"path {file_path} must be exist"
    # if 'mnist' not in file_path:
    #     return []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
        valid_lines = []
        for line in lines:
            p = line.find("#")
            if p != -1:
                line = line[:p]
            if line[:4] == "try:" or line[:7] == "except ":
                continue
            for i in range(len(line)):
                if not (32 <= ord(line[i]) < 127):
                    line = line.replace(line[i], " ")
            line = line.rstrip()  # 去除末尾空格
            if len(line) > 0:
                valid_lines.append(line)

        valid_lines2 = []
        flag = 0
        note_string = None
        for i in range(len(valid_lines)):
            if len (valid_lines[i]) >= 6 and ((valid_lines[i].lstrip().startswith("'''") and valid_lines[i].rstrip().endswith("'''")) or
                                              (valid_lines[i].lstrip().startswith('"""') and valid_lines[i].rstrip().endswith('"""'))):
                continue
            if '"""' in valid_lines[i] or "'''" in valid_lines[i]:
                if '"""' in valid_lines[i]:
                    note_string = '"""'
                else:
                    note_string = "'''"
                if flag == 0:
                    flag = 1
                else:
                    if flag == 1:
                        if ('"""' in valid_lines[i] and note_string == '"""') or "'''" in valid_lines[i] and note_string == "'''":
                            flag = 0
            else:
                if flag == 0 and len(valid_lines[i]) > 0:
                    valid_lines2.append(valid_lines[i].replace('\t', ' ').replace('\n', ' ').replace('\r', ' '))

    return valid_lines2


def gen_repo_star_list(repo_info_json_path: str, star_min_limit: int) -> list:
    """
    从单个包含结构化repo信息的json文件中获取符合star条件的(reponame, star数)并保存到list返回
    :param repo_info_json_path: 记录repo信息的json文件，json种的信息有很多，但是本函数只提取repo名和star
    :param star_min_limit: star的最少数量, 不少于这个数量的repo才会被考虑
    :return: ["/repo_name_0 starnum_0", "/repo_name_1 starnum_1", "/repo_name_2 starnum_2",  ...]
    """
    with open(repo_info_json_path, "r") as f:
        info = json.load(f)
        repo_items = info["items"]
        ret_str = []
        for repo in repo_items:
            if repo["stargazers_count"] >= star_min_limit:
                ret_str.append(str(f"/{repo['full_name']} {repo['stargazers_count']}"))
    return ret_str


def analyze_import(context: list):
    """
    利用py文件路径和文件内部的文本，解析并建立别名-完整名的映射关系F
    :param context: 文本字符串list
    :return: 分析得到的别名->完整名映射alias_to_full, 所有可能会出现的名字all_name_list(比如import torch.nn as nn, 那么"nn"就是可能会出现的名字之一), 除了import语句剩下的需要被分析的字符串行res_lines
    """
    # e.g. D:\\framework_cover\\Paddle-Image-Models\\ppim\\models\\vit.py
    alias_to_full = {}
    all_name_list = []
    i = 0
    res_lines = []
    while i < len(context):
        line = context[i]
        if "import " not in line:  # 没有import的行存好返回
            res_lines.append(context[i].replace('\\', ''))
            i += 1
            continue
        # 一定含有import
        line = line.lstrip()
        if line[:5] != "from " and line[:7] != "import ":  # 并不是以"import "或 "from "开头
            res_lines.append(context[i].replace('\\', ''))
            i += 1
            continue
        full_line = ""
        if "(" in line:  # 出现了 "import (a,..." 这种情况
            j = i
            while True:
                full_line += context[j]
                if context[j].endswith(")"):
                    break
                j += 1
            i = j
            full_line.replace("(", "")
            full_line.replace(")", "")
        elif line[-1] == "\\":
            """
            出现了from a import \\
            a, b, c
            """
            j = i
            while True:
                full_line += context[j]
                if not context[j].endswith("\\"):  # 不需要再接了
                    break
                j += 1
            i = j
            full_line.replace("\\", "")
        else:
            full_line = line
        full_line = full_line.replace("\\", "")

        if "from" == full_line[:4]:  # 有from的情况
            out = re.findall(re.compile("from (.+) import (.+)"), full_line)
            assert len(out) == 1, f"strange line [{full_line}] and out [{out}] at 1"
            out = out[0]
            assert len(out) >= 2, f"strange line [{full_line}] and out [{out}] at 2"
            # from out[0] import ...
            if out[1].strip() == "*":  # ::TODO
                # print("not suppot import *, continue....")
                pass
            else:
                groups = out[1].split(",")
                for alias_name in groups:
                    if " as " in alias_name:  # a as b
                        tname, talias = alias_name.split(" as ")
                        all_name_list.append(talias.strip())  # C
                        alias_to_full[all_name_list[-1]] = out[0].strip() + "." + tname.strip()
                    else:
                        all_name_list.append(alias_name.strip())  # C
                        alias_to_full[all_name_list[-1]] = out[0].strip() + "." + all_name_list[-1]
        else:  # 直接import a...的情况
            if " as " in full_line:  # import a as b的情况
                out = re.findall(re.compile("import (.+) as (.+)"), full_line)
                assert len(out) == 1, f"strange line [{full_line}] and out [{out}] at 4"
                out = out[0]
                assert len(out) == 2, f"strange line [{full_line}] and out [{out}] at 5"

                all_name_list.append(out[1].strip())
                alias_to_full[out[1].strip()] = out[0].strip()
            else:  # import a, b, c的情况
                out = re.findall(re.compile("import (.+)"), full_line)
                assert len(out) == 1, f"strange line [{full_line}] and out [{out}] at 6"
                out = out[0]

                all_name_list.extend([name.strip() for name in out.split(",")])
        i += 1
    return alias_to_full, all_name_list, res_lines


class Trie:
    """
    Trie
    """

    def __init__(self):
        super(Trie, self).__init__()
        self.nxt = {}
        self.fail = 0
        self.id = None
        self.len_ = 0


class Automaton:
    """
    Aho-Corasick automaton
    """

    def __init__(self):
        super(Automaton, self).__init__()
        self.node = [Trie(), ]
        self.sz = 1

    def insert(self, s: str, _id: str) -> None:
        u = 0
        for c in s:
            v = ord(c)
            assert 32 <= v < 127, f"v is not valid {v}"
            if self.node[u].nxt.get(v) is None:
                self.node.append(Trie())
                self.node[u].nxt[v] = self.sz
                self.sz += 1
            u = self.node[u].nxt.get(v)
        if self.node[u].id is None:
            self.node[u].id = _id
            self.node[u].len_ = len(s)

    def build(self) -> None:
        self.node[0].fail = 0
        queue = []
        for i in range(32, 127):
            v = self.node[0].nxt.get(i)
            if v is not None:
                self.node[v].fail = 0
                queue.append(v)
            else:
                self.node[0].nxt[i] = 0
        while len(queue) > 0:
            u = queue.pop(0)
            uf = self.node[u].fail
            for i in range(32, 127):
                v = self.node[u].nxt.get(i)
                if v is not None:
                    self.node[v].fail = self.node[uf].nxt[i]
                    queue.append(v)
                else:
                    self.node[u].nxt[i] = self.node[uf].nxt[i]

    def query(self, s: str) -> defaultdict:
        u = 0
        cnt = defaultdict(int)
        for ind, c in enumerate(s):
            v = ord(c)
            assert 32 <= v < 127, f"ord(c) is not valid [{v}] and the char is [{c}]"
            u = self.node[u].nxt.get(v)
            tu = u
            while tu is not None and tu > 0:
                if self.node[tu].id is not None:
                    if (ind - self.node[tu].len_ >= 0 and s[ind - self.node[tu].len_ + 1] == '.') or (ind - self.node[tu].len_ < 0 or (
                            (not s[ind - self.node[tu].len_ - 1].isalpha()) and (not s[ind - self.node[tu].len_].isnumeric()) and s[
                        ind - self.node[tu].len_] != '_')):
                        if ".Tensor." in self.node[tu].id:
                            if s[ind + 1] != '"' and s[ind + 1] != "'":
                                cnt[self.node[tu].id] += 1
                        else:
                            cnt[self.node[tu].id] += 1
                tu = self.node[tu].fail
        return cnt

    def prepare_initial_graph(self, key: list, value: list) -> None:
        assert len(key) == len(value), f"Key and value's len must be same, but got {len(key)} {len(value)}"
        for k, v in zip(key, value):
            self.insert(s=k, _id=v)


def analyze_text(context: list, initial_ac):
    """
    统计单个py文件中所有通过import出现的方法、函数名调用次数
    :param context: 准备要被分析import名字以及统计出现次数的文本每一行所组成的字符串list
    :return: 文件完整名API统计情况statistics_refined = {API_name: cnt}, 文件出现的别名映射alias_to_full = {alias_name: full_name}
    """
    alias_to_full, all_name_list, res_context = analyze_import(context)
    all_name_list = [name for name in all_name_list if (alias_to_full.get(name) is None) or (global_suf_to_dotnumber.get(name) is not None)]
    # build ac automaton
    if initial_ac is not None:
        AC = initial_ac
    else:
        AC = Automaton()
        AC.prepare_initial_graph([t + '(' for t in all_name_list], all_name_list)
        AC.build()

    # 拼合所有文本行变为1行
    squeeze_line = "".join(res_context)

    # 在Trie图上搜索squeeze_line为并统计答案
    statistics = AC.query(squeeze_line)

    statistics_refined = defaultdict(int)  # 把别名的统计情况转换成对应完整名的统计情况
    for name, cnt in statistics.items():
        if alias_to_full.get(name) is not None:
            statistics_refined[alias_to_full.get(name)] += cnt
        else:
            statistics_refined[name] += cnt
    if initial_ac is None:
        del AC
    return statistics_refined, alias_to_full


def build_api_suffixes(api_list):
    """
    把每个再apilist中的完整名api构造出它所有的后缀别名并储存
    :param api_list: 准备拿来建立所有其API后缀的API列表
    :return: 按深度（.个数)从小到大排序好的API后缀名数组all_suffix_api_list, 后缀名-完整名映射ref_dict(可能有多个后缀名映射同一个完整名 , 需要后续根据文件内的import关系来解决这个问题)
    """
    all_suffix_api_list = []
    ref_dict = {}
    for api in api_list:
        dot_cnt = api.count('.')
        if dot_cnt == 0:
            all_suffix_api_list.append(api)
            ref_dict[api] = api
            global_suf_to_dotnumber[api] = 0
        else:
            seg_split = api.split('.')
            for i in range(dot_cnt + 1):
                all_suffix_api_list.append('.'.join(seg_split[i:]))
                ref_dict[all_suffix_api_list[-1]] = api
                global_suf_to_dotnumber[all_suffix_api_list[-1]] = dot_cnt - i
    all_suffix_api_list.sort(key=lambda x: global_suf_to_dotnumber[x])  # 从浅到深排序
    return all_suffix_api_list, ref_dict


def analyze_dir(repo_dir, keyword: str, queries_list_txt_path) -> list:
    """
    递归分析目录下所有的.py文件调用给定的API列表中API的情况
    :param repo_dir: 储存所有repo的文件夹路径, 一般是 ./key_word_cover
    :param keyword: 与统计内容有关的关键字
    :param queries_list_txt_path: 想统计的API列表集合, 文本txt
    :return: 统计结果statistics
    """
    preprocess_api_list_from_file(f"./query_files/{keyword}_queries.txt")

    assert keyword in ["paddle", "pytorch"], f"framework must be pytorch or paddle, but got {keyword}"
    t = time.time()
    py_file_paths = glob.glob(repo_dir + os.path.sep + "**" + os.path.sep + "*.py", recursive=True)
    print(f"Total [{len(py_file_paths)}] files will be parsed......")
    statistics_dir = defaultdict(int)

    global_api_list_main_suf_value, outter_suf_to_full = build_api_suffixes(global_api_list_main_value)  # 处理main的API
    global_api_list_main_suf_key = [t + '(' for t in global_api_list_main_suf_value]

    outter_AC = Automaton()
    outter_AC.prepare_initial_graph(global_api_list_main_suf_key, global_api_list_main_suf_value)
    outter_AC.build()

    frameworkOpAC = Automaton()
    frameworkOpAC.prepare_initial_graph(global_api_list_frameworkOp_key, global_api_list_frameworkOp_value)
    frameworkOpAC.build()

    selfOpAC = Automaton()
    selfOpAC.prepare_initial_graph(global_api_list_selfOp_key, global_api_list_selfOp_value)
    selfOpAC.build()
    for i, py_file_path in enumerate(tqdm(py_file_paths)):
        try:
            # ONLY FOR DEBUG
            # py_file_path = './debug.py'
            # ONLY FOR DEBUG

            # 1. 统计非*.Tensor.的API
            context = parse_valid_context(py_file_path)
            if len(context) > 10000:
                print(f"Big file at [{py_file_path}] has [{len(context)}] lines, so it is skipped, continue scanning......")
                continue
            statistics_inner, inner_alias_to_full = analyze_text(context, None)
            statistics_outter, _ = analyze_text(context, outter_AC)

            statistics_outter = {k: v for (k, v) in statistics_outter.items() if global_suf_to_dotnumber.get(k) is not None}
            statistics_merge = defaultdict(int)

            # 2. 统计关于*.op和*.Tensor.op的API
            for suf_name in sorted(statistics_outter.keys(), key=lambda x: global_suf_to_dotnumber[x]):  # 把从API列表中获取的API统计出来
                if global_suf_to_dotnumber[suf_name] == 0:
                    """
                    对于由单个单词构成的API比如paddle.hub.list的后缀list，除非它在inner_alias_to_full的key中且inner_alias_to_full的key对应的value是, 或者像
                    否则视作其为内置函数，不算在内
                    """
                    if inner_alias_to_full.get(suf_name) is not None:
                        if statistics_merge.get(outter_suf_to_full[suf_name]) is None:  # 自底向上更新，如果被浅层的API更新过，就不用再更新了
                            if inner_alias_to_full.get(suf_name) is not None:  # 如果py文件中有extra的后缀的别名存在，那么就用py文件中的统计结果
                                statistics_merge[inner_alias_to_full[suf_name]] = statistics_inner[suf_name]
                            else:  # 如果py文件中没有，那么就用extraAPI列表中的API统计结果
                                statistics_merge[outter_suf_to_full[suf_name]] = statistics_outter[suf_name]
                else:
                    """
                    否则正常检索
                    """
                    if statistics_merge.get(outter_suf_to_full[suf_name]) is None:  # 自底向上更新，如果被浅层的API更新过，就不用再更新了
                        if inner_alias_to_full.get(suf_name) is not None:  # 如果py文件中有outter的后缀的别名存在，那么就用py文件中的统计结果
                            statistics_merge[inner_alias_to_full[suf_name]] = statistics_inner[suf_name]
                        else:  # 如果py文件中没有，那么就用extraAPI列表中的API统计结果
                            statistics_merge[outter_suf_to_full[suf_name]] = statistics_outter[suf_name]

            for name, cnt in statistics_merge.items():
                statistics_dir[name] += cnt

            # 2. 统计*.Tensor.的API, 做法是把 ".op(" (代表framework.Tensor.op) 统计一遍, 再把 "framework.op(" (代表framework.op) 统计一遍, cnt[".op("] - cnt[
            # "framework.op("]就是.Tensor.op的真正统计情况
            statistics_selfOp, dummy = analyze_text(context, selfOpAC)
            statistics_frameworkOp, dummy = analyze_text(context, frameworkOpAC)

            for name in statistics_selfOp.keys():  # self.op的统计结果肯定包含framework.Tensor.op的统计结果
                statistics_dir[name] += statistics_selfOp[name]
                if global_selfOp_to_frameworkOp.get(name) is not None and statistics_frameworkOp.get(global_selfOp_to_frameworkOp[name]) is not None:
                    statistics_dir[name] -= statistics_frameworkOp[global_selfOp_to_frameworkOp[name]]  # 后者准确，前者可能存在问题
            del inner_alias_to_full, statistics_inner, statistics_outter, _, statistics_merge

        except BaseException as e:
            print(e.__traceback__.tb_lineno)
            print(f"error at file_path [{py_file_path}] - {[e]}")
            continue
    api_items = list(statistics_dir.items())
    api_items = sorted(api_items, key=lambda x: -x[1])
    api_items = [(name, cnt) for (name, cnt) in api_items if cnt > 0]
    if not osp.exists("./statistics"):
        os.makedirs("./statistics")
    with open(f"./statistics/{keyword}_statistics.txt", "w") as f:
        for api, freq in api_items:
            if api in global_api_list_total:
                f.write(f"{api}\t{freq}\n")
    print(f"------------ Finished analyze in {time.time() - t} second(s) and statistics saved at [./statistics/{keyword}_statistics.txt]------------")
    return api_items


if __name__ == "__main__":
    # 0. 设定要爬取的仓库相关的关键字， 并把要爬取的关键字对应repo的目标字符串全部放到f"./query_files/{keyword}_queries.txt"文本中
    key_word = 'paddle'

    # 1. 按star数从高到低爬取含有关键字的项目repo并将这些repo的信息分批保存为结构化的json数据, 再将json中提取有效信息合并写入repo记录文本中
    parse_high_star_repo(key_word, star_min_limit=400, num_repo_limit=2000)

    # 2. repo记录文本中生成对应repo的git地址，存放入clone_urls.txt文件，对这些repo地址执行git clone
    gen_clone_candidates(f"./json_files/repo_star_{key_word}.txt", key_word, method='git')
    # 等待所有候选的git repo clone完毕......

    # 3. 所有repo clone下来之后，对这些repo逐文件用线性匹配算法统计，将统计结果按调用频率从大到小排好序, 保存到f"./statistics/{key_word}_statistics.txt"文件中
    dir_statistics = analyze_dir(f"D:\\analyse2\\{key_word}_cover", f"{key_word}", None)

