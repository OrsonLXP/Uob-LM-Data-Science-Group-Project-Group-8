import inspect
import utils

def generate_function_docs(module):
    # 获取模块中所有的函数
    functions = [func for func, obj in inspect.getmembers(module) if inspect.isfunction(obj)]

    docs = []

    for func in functions:
        # 获取函数的文档字符串
        doc = inspect.getdoc(getattr(module, func))
        docs.append(f"Function: {func}\n{doc}\n")

    # 将文档保存到文件
    with open('function_docs.md', 'w', encoding='utf-8') as f:
        for doc in docs:
            f.write(doc + '\n\n')

# 调用生成函数文档并保存
generate_function_docs(utils)
