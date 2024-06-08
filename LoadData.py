import pandas as pd

import utils.utils
from utils.utils import get_his_data, load_data_single
import pickle
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from utils.my_functions import read_large_csv_parallel

def load_data_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        date_visibility_matrices = pickle.load(f)
    # 假设每个pickle文件中的数据结构是 {token: matrices}
    return date_visibility_matrices


def save_data_to_pickle(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def get_df(data_path):
    df = get_his_data(data_path)
    # dataprocessor = feature_ori.DataProcessorOri(df, [f'future_{i}d_ret' for i in [1,7,20]], N=3)
    # df = dataprocessor.run()
    print('Done with data processing')
    return df

def main(time_step,his_path,save_path,start_date=None,end_date=None):
    df = pd.read_pickle(his_path)
    print('Loaded Data Successfully')
    # with open('config/zz500股票成分.pkl', 'rb') as file:
    #     stock_list = pickle.load(file)
    # df = df[df['token'].isin(stock_list)]
    load_data_single(df=df, save_path=save_path, time_step=time_step, start_date=start_date, end_date=end_date)


if __name__ == '__main__':
    index = 'zz500' #todo 或者改为'hs300'
    his_path = f'config/{index}.pkl'
    # df = get_df(data_path)
    for time_step in [20]:
        start_date = '2016-01-01' #todo 时间自己改一下
        end_date = '2016-12-31' #todo 时间自己改一下
        save_path = f'config/{index}_data_{time_step}'
        print(f'now generating {time_step} data ')
        main(time_step,his_path,save_path,start_date=start_date,end_date=end_date)
