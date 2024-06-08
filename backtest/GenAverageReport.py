import pandas as pd

def gen_average_report(main_path):
    for bin in range(1,6):
        test_res_path = main_path + '/track_{}'
        dfs = []
        for i in range(1,7):
            df = pd.read_csv(f'{test_res_path.format(i)}/bin{bin}.csv')
            df.set_index('index',inplace=True)
            df.rename(columns={
                'return' : f'return_{i}'
            },inplace=True)
            dfs.append(df)
        cur_df = pd.concat(dfs,axis=1)
        cur_df.fillna(0,inplace=True)
        cur_df['average_return'] = cur_df.mean(axis=1)
        cur_df = cur_df[['average_return']]
        cur_df.to_csv(f'{main_path}/bin{bin}.csv',index=False,encoding='utf-8-sig')

if __name__ == '__main__':
    main_path = 'test_res/train_circle_12m'
    gen_average_report(main_path)