import pandas as pd


recovered_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
recovered_df = pd.read_csv(recovered_url)
print(recovered_df.head())