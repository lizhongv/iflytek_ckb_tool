import pandas as pd
from conf.settings import config_manager, logger

def check_source(standard_l1, l2):
    standard = []
    for i in range(len(standard_l1)):
        if str(standard_l1[i]) != str(l2[i]):
            standard.append((l2[i], standard_l1[i]))
    if not standard:
        return True, []
    return False, standard


def check_source_init():
    check_message = []
    df1 = pd.read_excel(r"db/standard.xlsx")

    actual_path = config_manager.output_file
    df2 = pd.read_excel(actual_path)

    for i in range(1, 11):
        standard_l1 = list(df1[f"溯源{i}"])
        actual_l2 = list(df2[f"溯源{i}"])

        result, message = check_source(standard_l1, actual_l2)
        if result:
            logger.info(f"溯源{i}一致")
            continue

        logger.info(f"溯源{i}不一致：{message}")