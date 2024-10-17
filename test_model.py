import dill
import json
import pandas as pd


def main():
    with open('data/hit_predict.pkl', 'rb') as file:
        model = dill.load(file)

    with open('data/test_json_1.json') as fin:
        form = json.load(fin)

    df = pd.DataFrame.from_dict([form])
    print('test_model df =', df)
    y = model['model'].predict(df)
    print(y)


if __name__ == '__main__':
    main()