import pandas as pd
try:
    df = pd.read_csv('data/food.com-interaction/RAW_recipes.csv', nrows=2)
    print("Ingredients Sample:")
    print(df['ingredients'].iloc[0])
    print(type(df['ingredients'].iloc[0]))
    print("-" * 20)
    print("Nutrition Sample:")
    print(df['nutrition'].iloc[0])
    print(type(df['nutrition'].iloc[0]))
except Exception as e:
    print(e)
