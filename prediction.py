import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from airclub_class import AeroclubPipeline

# загразим тестовый датасет
check_df = pd.read_excel('./test_to_check.xlsx')
# необходимо переименовать фичу ValueRu, т.к. на обучающих данных у нее было другое название
check_df.rename(columns={'ValueRu':'TravellerGrade'}, inplace=True)

# создадим экземпляр класса для обработки данныех и получения предсказаний
pipeline = AeroclubPipeline()
# загрузим обученную ранее модель
pipeline.load_model('./Model/ctb_clsf_model.cbm')

# сформируем предсказания
pipeline.get_xls_result(check_df, './result.xls')

