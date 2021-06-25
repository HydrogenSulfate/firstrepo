import os
import os.path as osp
import sys


def allwhitespace(line):
    for c in line:
        if c != ' ' and c != '\t' and c != '\n' and c != '\r':
            return False
    return True


def parse_text_pure(code_file_path):
    assert osp.exists(
        code_file_path), f'Code file {code_file_path} must be exists'
    with open(code_file_path, "r") as f:
        context = f.read().splitlines()
    return context, (context[0] == "import paddle.fluid.profiler as profiler")


def get_profiler_code_text(code_file_path):
    assert osp.exists(code_file_path), 'Code file must be exists'
    with open(code_file_path, "r") as f:
        context = f.read().splitlines()
    while (context and allwhitespace(context[-1])):
        context.pop(-1)
    min_indent = len(context[-1]) - len(context[-1].lstrip())
    if context[0] != "import paddle.fluid.profiler as profiler":
        context.insert(0, "import paddle.fluid.profiler as profiler")
        context.insert(1, "profiler.start_profiler('GPU')")
        context.append(' ' * min_indent +
                       "profiler.stop_profiler(sorted_key='total')")
    return context


def write_text(context: list, dump_path: str):
    with open(dump_path, "w") as f:
        for line in context:
            f.write(line + '\n')
        print(f'Modified code has saved at {dump_path}, total {len(context)} lines')


if __name__ == '__main__':
    path = [file for file in sys.argv[2:] if file.endswith(".py")]
    modelname_mode = sys.argv[1]
    sys.argv.pop(1)
    assert modelname_mode.split(
        ".")[-1] in ['train', 'infer'], f"Model' mode must 'train' or 'infer', but got {modelname_mode}"
    assert(len(path) ==
           1), f'There must be a excute python file, but got {path}'
    path = path[0]
    original_context, done_before = parse_text_pure(path)
    profiler_context = get_profiler_code_text(path)
    if not done_before:
        write_text(original_context,
                   dump_path=path.replace(".py", "_backup.py"))
        write_text(profiler_context, dump_path=path)
    os.system(' '.join(['python3.7'] + sys.argv[1:]) + ' > OP_stat.txt')
    OP_list = []
    with open('OP_stat.txt', 'r') as f:
        lines = f.read().splitlines()
    valid_OP_lines = [line for line in lines if "thread0::" in line]
    for line in valid_OP_lines:
        split_lines = line.split(" ")
        if split_lines and len(split_lines) > 0:
            op = split_lines[0].split('thread0::')[1]
            if "GpuMemcpy" not in op:
                OP_list.append(op)
    with open(modelname_mode + '.txt', 'w') as f:
        for op in OP_list:
            f.write(op+'\n')
