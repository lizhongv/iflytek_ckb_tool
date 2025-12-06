import pandas as pd
from conf.settings import logger
result_dict_excel = {}


def write_result(index, result_dict):
    result_dict_excel[index + 1] = result_dict


def save_result(path):
    df = pd.DataFrame(
        columns=['用户问题', '参考溯源', '参考答案', '溯源1', '溯源2', '溯源3', '溯源4',
                 '溯源5','溯源6','溯源7','溯源8','溯源9','溯源10', '大模型返回答案', 
                 '溯源正确率', '回复正确率', '溯源错误原因',
                 '回复错误原因', 'RequestId', 'SessionId']  # '全部匹配字段',
    )
    for index in range(len(result_dict_excel)):
        df.loc[index + 1] = result_dict_excel[index + 1]
    try:
        df.to_excel(path, index=False)
    except PermissionError as e:
        logger.error(f"文件被其它程序占用，无法写入测试结果，程序即将退出...")
        exit(0)
    except Exception as e:
        logger.error(f"保存文件失败，程序即将退出...\n{e}")
        exit(0)


