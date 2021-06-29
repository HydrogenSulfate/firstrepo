# -*- coding: utf-8 -*-

import glob
import json
import os
import os.path as osp
import re
import time
from collections import defaultdict
import requests
from tqdm import trange, tqdm



def parse_all_api_docs(doc_dir: str) -> list:
    return glob.glob(doc_dir + os.path.sep + "**" + os.path.sep + "*.rst", recursive=True)
    # return glob.glob(doc_dir + os.path.sep + "*.rst", recursive=True)


def parse_name_and_context_from_doc_path(doc_path: str) -> (str, str):
    """
    从给定的rst文档路径中parse出对应的API名字和对应内容
    """

    # 文本内容去掉无意义的空行，把制表符替换成4个空格
    with open(doc_path, "r", encoding="utf-8") as f:
        context = f.read().splitlines()
        context = [line for line in context if len(line.replace(" ", "")) > 0]
        context = [line.replace("\t", "    ") for line in context]

    # 找到API名字，返回API名字和文档内容
    if ".. _cn_api_" in context[0] or "_cn_paddle_nn" in context[0]:
        class_pattern = re.compile(f"py:class::[ ]*([A-Za-z._0-9]+)")
        function_pattern = re.compile(f"py:function::[ ]*([A-Za-z._0-9]+)")
        attribute_pattern = re.compile(f"py:attribute::[ ]*([A-Za-z._0-9]+)")
        method_pattern = re.compile(f"py:method::[ ]*([A-Za-z._0-9]+)")
        decorator_pattern = re.compile(f"py:decorator::[ ]*([A-Za-z._0-9]+)")
        for line in context:
            out_class = re.findall(class_pattern, line)
            if len(out_class) > 0:
                return out_class[0].strip(), context

            out_function = re.findall(function_pattern, line)
            if len(out_function) > 0:
                return out_function[0].strip(), context

            out_attribute = re.findall(attribute_pattern, line)
            if len(out_attribute) > 0:
                return out_attribute[0].strip(), context

            out_method = re.findall(method_pattern, line)
            if len(out_method) > 0:
                return out_method[0].strip(), context

            out_decorator = re.findall(decorator_pattern, line)
            if len(out_decorator) > 0:
                return out_decorator[0].strip(), context

    return None, None


def count_left_whitespace(line: str) -> int:
    cnt = 0
    for c in line:
        if c == ' ':
            cnt += 1
        else:
            return cnt
    return cnt


def parse_codeblocks_from_context(context: list) -> list:
    """
    从文本内容中parse出第一块示例代码，返回示例代码内容
    """
    code_blocks = []  # 代码块内容
    code_block_begin = 0  # 代码块开始标记
    min_indent = 1000000  # 整体缩进
    flag_word1 = u"示例代码"  # 标志着示例代码起始的关键字
    flag_word2 = u"代码示例"
    for i in range(len(context)):
        if context[i].find(flag_word1) != -1 or context[i].find(flag_word2) != -1:  # 找到了示例代码的起始标志
            if code_block_begin == 0:  # 只计第一个代码块
                code_block_begin = 1  # 标记代码块要开始了
            else:  # 遇见第二个代码块起始标志，直接break
                break
        elif "code-block:: python" in context[i] and code_block_begin == 1:
            code_block_begin = 2   # 接下来就是代码块的内容了
        else:
            if code_block_begin != 2:  # 非代码块内容跳过
                continue
            indent = count_left_whitespace(context[i])  # 算一下左侧缩进多少
            if min_indent == 1000000:
                min_indent = indent  # 更新一下最小缩进
                code_blocks.append(context[i])  # 放入代码
            else:
                if indent >= min_indent:  # 只有当后续缩进大于等于最小缩进，才算代码块中的有效代码
                    code_blocks.append(context[i])
                else:
                    break
    if 0 < min_indent < 1000000:
        code_blocks = [line[min_indent:] for line in code_blocks]
    return code_blocks


def gen_pyfile_from_docs_dir(api_run_file_output_dir: str, doc_dir: str) -> None:
    """
    从官方的docs除了Overview之外的*.rst文档中提取所有API的示例代码
    写入对应的python文件，文件名与API同名
    存入指定的目录api_run_file_output_dir中
    """

    api_doc_paths = parse_all_api_docs(doc_dir)
    # ID_to_API = {}

    # Leaky_API_list = []
    # with open("./leaky_API_list.txt", "r") as f:
    #      Leaky_API_list = f.read().splitlines()
    gen_files_list = set()
    for i, api_doc_path in enumerate(tqdm(api_doc_paths)):
        # FOR DEBUG #
        # if api_doc_path == ".\\docs\\docs\\api\\paddle\\nn\\functional\\smooth_l1_loss_cn.rst":
        #     print(1)
        #############
        try:
            if "Overview" in api_doc_path:  # 跳过Overview文档
                continue

            #  从rst文件中parse出API名字和文档内容
            api_name, doc_context = parse_name_and_context_from_doc_path(api_doc_path)

            if api_name is None:
                print(f"API name is None, need check {api_doc_path}")
                continue
            # if api_name not in Leaky_API_list:  # 只处理还存在统计问题的API，正常能获取调用OP数据的跳过
                # continue
            
            # 从文档内容中parse得到一块示例代码（多块只取第一块）
            codeblock = parse_codeblocks_from_context(doc_context)

            # 目前paddle版本静态图需要加上paddle.enable_static()这一句话，所以这里需要特别标记
            static_mode = False
            for line in codeblock:
                if 'fluid.' in line:
                    static_mode = True
                    break

            # # 找到最后一个import的位置，准备在其后插入必要的profile语句
            # for i in range(len(codeblock) - 1, -1, -1):
            #     if "import " in codeblock[i]:
            #         last_import_line_number = i
            #         break
            last_import_line_number = 0

            # 插入两句profile跟踪必要的代码
            codeblock.insert(last_import_line_number, "import paddle")
            last_import_line_number += 1
            codeblock.insert(last_import_line_number, "import paddle.fluid.profiler as profiler")
            last_import_line_number += 1

            # 静态图需要再加一句paddle.enable_static()
            if static_mode:
                codeblock.insert(last_import_line_number, "paddle.enable_static()")
                last_import_line_number += 1

            # 插入开始跟踪的语句start_profiler
            codeblock.insert(last_import_line_number + 1, "profiler.start_profiler('GPU')")
            last_import_line_number += 1

            # 插入结束跟踪语句stop_profiler
            codeblock.append(f"profiler.stop_profiler('total', './{api_name}')")
            last_import_line_number += 1

            # 如果存放py文件的目录没创建，先创建一下
            if not osp.exists(api_run_file_output_dir):
                os.makedirs(api_run_file_output_dir)
                print(f"[*] make directory for api_run_file_output_dir at {api_run_file_output_dir}")

            # 把加入profile代码的codeblock写入py文件中，以备运行
            with open(f"./{api_run_file_output_dir}/{api_name}.py", "w") as f:
                f.write(f"#    {api_name}\n")
                for line in codeblock:
                    f.write(line + '\n')
                gen_files_list.append(api_name)
        except Exception as e:
            print(f"Error {e} occr when look at {api_doc_path}")
            break

    print(f"生成了[{len(gen_files_list)}]个py文件")


def check_bracket_dir(output_txt_path: str, doc_dir: str) -> None:
    """
    从官方的docs除了Overview之外的*.rst文档中提取所有API的文档，进行括号匹配，如果有出现公式错误必然会出现在匹配失败的结果内并被记录下来
    """
    api_doc_paths = parse_all_api_docs(doc_dir)
    check_file_pathes = []
    for i, api_doc_path in enumerate(tqdm(api_doc_paths)):
        # FOR DEBUG #
        # if api_doc_path == ".\\docs\\docs\\api\\paddle\\nn\\functional\\smooth_l1_loss_cn.rst":
        #     print(1)
        #############
        try:
            # if "Overview" in api_doc_path:  # 跳过Overview文档
                # continue

            #  从rst文件中parse出API名字和文档内容
            api_name, doc_context = parse_name_and_context_from_doc_path(api_doc_path)

            # if api_name is None:
                # print(f"API name is None, need check {api_doc_path}")
                # continue
            # if api_name not in Leaky_API_list:  # 只处理还存在统计问题的API，正常能获取调用OP数据的跳过
                # continue
            bracket_stack = []
            flag = 1
            tot_context = "".join(doc_context)
            for c in tot_context:
                if c=="(" or c=="{" or c=="[":
                    bracket_stack.append(c)
                else:
                    if c==")":
                        if bracket_stack[-1] == "(":
                            bracket_stack.pop(-1)
                        else:
                            flag = 0
                            break
                    elif c=="]":
                        if bracket_stack[-1] == "[":
                            bracket_stack.pop(-1)
                        else:
                            flag = 0
                            break
                    elif c=="}":
                        if bracket_stack[-1] == "{":
                            bracket_stack.pop(-1)
                        else:
                            flag = 0
                            break
                    else:
                        continue
                
            if flag == 0:
                check_file_pathes.append(api_doc_path)
        except Exception as e:
            print(f"Error {e} occr when look at {api_doc_path}")
            break
    with open(output_txt_path, "w") as f:
        for file_path in check_file_pathes:
            f.write(file_path + "\n")

    print(f"检测到[{len(check_file_pathes)}]个括号匹配有问题的文档")


def gen_cProfile_output(api_run_files_dir: str, log_dir: str, overlap=True, filterjson=None) -> None:
    """
    构建代码运行命令command
    用os.system(command)模拟运行在api_run_files_dir中的所有api代码
    然后得到每个API的profile记录信息，写入同名文本文件，存放到log_dir中
    filterjson是用已经生成好的statistics_API_OP_rela.json文件做过滤，优先分析filterjson中API对应OP为空的文件
    """
    # with open("./leaky_API_list.txt", "r") as f:
    #      Leaky_API_list = f.read().splitlines()
    if filterjson is not None:
        with open(filterjson, "r") as f:
            exist_api_op_rela_dict = json.load(f)

    # 切换到api_run_files_dir目录下准备执行命令
    ori_path = os.getcwd()
    os.chdir(api_run_files_dir)

    # 爬取所有要运行的API文件
    run_py_files = os.listdir(os.getcwd())

    # 过滤一下
    run_py_files = [pyfile for pyfile in run_py_files if pyfile.endswith('.py') and pyfile.startswith('paddle.')]
    print(f"从[{api_run_files_dir}]中扫描到[{len(run_py_files)}]个API运行文件")

    if filterjson is not None:
        run_py_files = [pyfile for pyfile in run_py_files if len(exist_api_op_rela_dict[pyfile.split('.py')[0]]) == 0]
    # 排序
    run_py_files.sort()
    print(f"过滤后将要运行[{len(run_py_files)}]个py文件")
    # 构造将要执行的shell语句
    command_list = [f"python3.7 {file} > {osp.join('../', log_dir, file.replace('.py', '.txt'))}" for file in run_py_files]

    # 执行失败的记录
    fail_command_list = []
    if not osp.exists(osp.join('../', log_dir)):
        os.makedirs(osp.join('../', log_dir))
        print(f"[*] make directory for log_dir at {osp.join('../', log_dir)}")
    # 逐条执行
    try:
        with tqdm(command_list) as tbar:
            for i, command in enumerate(tbar):
                tbar.set_description(f"{command}")

                # 构造输出文本路径
                output_path = osp.join('../', log_dir, run_py_files[i].replace('.py', '.txt'))
                
                # 如果选择了不覆盖且文本已经存在，那么就不执行，跳过
                if not overlap and osp.exists(output_path):
                    print(f"{output_path} has been exist, skip......")
                else:# 否则执行命令
                    retflag = os.system(command)
                    if retflag != 0:  # 返回值不为0，这条命令执行有问题，记录下来
                        fail_command_list.append((retflag, run_py_files[i]))
    except KeyboardInterrupt:# 如果遇到键盘输入终止，那就终止
        print(f"外部输入终止命令，API模拟运行程序终止")
        tbar.close()
        raise
    tbar.close()

    os.chdir(ori_path)  # 执行完毕, 切换回原上级目录
    with open("./failed_log.txt", "w") as f:
        for fail_info in fail_command_list:
            f.write(str(fail_info[-1]) + '\n')

    print(f"成功执行了[{len(command_list) - len(fail_command_list)}]条命令，失败[{len(fail_command_list)}]条命令，执行失败详细信息已保存到[./failed_log.txt]中")


def parse_rela_context_from_profile(key_word: str, file_path: str) -> list:
    """
    从profile中parse出调用OP的关键信息
    """
    with open(file_path, "r") as f:
        context = f.read().splitlines()
        context = [line for line in context if key_word in line]
        context = [line.split(' ')[0].split('::')[-1] for line in context if 'GpuMemcpySync' not in line]
    return context


def analyze_API_OP(log_dir: str, OP_list_path):
    
    Profile_file_paths = glob.glob(f"{log_dir}/paddle.*.txt", recursive=False)
    # Profile_file_paths = [path for path in Profile_file_paths if osp.basename(path).startswith('API_') and ('profile' not in osp.basename(path)) and any(['0'<=c<='9' for c in osp.basename(path)])]

    # ID_TO_API_NAME_MAP = defaultdict(str)
    # assert osp.exists(f"{log_dir}/API_ID_Map.txt"), f"Make sure {log_dir}/API_ID_Map.txt exist"

    # with open(f"{log_dir}/API_ID_Map.txt", "r") as f:
    #     context = f.read().splitlines()
    #     for line in context:
    #         api_name, _id = line.split(' ')
    #         _id = int(_id)
    #         if not isinstance(api_name, str):
    #             api_name = str(api_name)
    #         ID_TO_API_NAME_MAP[_id] = api_name


    # 如果提供了完整的OP list，那就生成一份，以备后续过滤用
    OP_list = None
    if OP_list_path is not None:
        assert osp.exists(OP_list_path), f"OP_list_path [{OP_list_path}] not exists, please check"
        with open(OP_list_path, "r") as f:
            OP_list = f.read().splitlines()

    # 构造API-OP调用关系的dict，后续写入json文件中
    API_OP_dict = defaultdict(list)

    # 枚举每一个profile
    for i, Profile_file_path in enumerate(tqdm(Profile_file_paths)):
        api_name = osp.basename(Profile_file_path).split('.txt')[0] # 得到API名字
        rela_context = parse_rela_context_from_profile("thread0::", Profile_file_path) # 从文本中解析出与OP有关的含key_word的调用记录

        if OP_list_path is not None:
            rela_context = [op for op in rela_context if op in OP_list]

        # 建立API名字->调用记录的映射关系
        API_OP_dict[api_name] = rela_context
            
    with open(f"./statistics_API_OP_rela.json", "w") as f: # API-OP调用关系写入文件中
        json.dump(API_OP_dict, f, indent=4)


def search_empty_api(result_json_file: str):
    assert osp.exists(result_json_file), f"Make sure json file exists [{result_json_file}]"
    with open(result_json_file, "r") as f:
        api_op_rela_dict = json.load(f)
    for api, op_list in api_op_rela_dict.items():
        if len(op_list) == 0:
            print(api)


def rename():
    os.chdir("./api_run_files")
    files = glob.glob("./API_*.py")
    for i in range(len(files)):
        with open(files[i], "r") as f:
            api_name = f.read().splitlines()[0]
            api_name = api_name.split(' ')[-1].strip()
            # print(files[i], api_name+".py")
            os.system(f"mv {files[i]} {'./'+api_name+'.py'}")


if __name__ == "__main__":
    # 1. 分析docs文件，提取中每个doc的API和对应的示例代码
    gen_pyfile_from_docs_dir("./api_run_files", "./docs/docs/api/paddle/")
    # 2. 运行这些示例代码， 得到每个代码profile记录的信息，保存成文本文件，以备后续分析用
    # gen_cProfile_output("./api_run_files", "./profile_log_files", overlap=True, filterjson="./statistics_API_OP_rela.json")
    # 3. 分析profile的跟踪输出，从中提取OP调用信息, 并储存到json中,  最好提供OP_list的路径，以过滤最终的分析结果，否则有一些OP不在统计目标内也会被算进去
    # analyze_API_OP("./profile_log_files", None)

