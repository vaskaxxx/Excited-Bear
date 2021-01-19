from oauth2client.service_account import ServiceAccountCredentials
import gspread
import pandas as pd
import schedule
import time
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import plotly.express as px

path = input('Enter the download path: ')


def excited_bear():
    #####################################
    # Importing from google sheets
    #####################################
    scope = [
        'https://www.googleapis.com/auth/spreadsheets', ]

    GOOGLE_KEY_FILE = '../experimental/Medium_Data_Extraction_Key.json'

    credentials = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_KEY_FILE, scope)
    gc = gspread.authorize(credentials)

    workbook_key = '1epgfe52DjdSbnvf63ne4IhseGGIHsMa6mB6BVazBcDA'
    workbook = gc.open_by_key(workbook_key)
    sheet = workbook.get_worksheet(0)
    values = sheet.get_all_values()
    df = pd.DataFrame(values[1:], columns=values[0])

    file_name = 'Raw_data'
    file_path = path + '\\' + file_name + '.csv'
    df.to_csv(file_path)
    #####################################
    # Data cleaning
    #####################################
    df = pd.read_csv(file_path, header=7, index_col=False)
    df = df.reset_index(drop=True)

    cols = list(df.columns[df.columns.str.contains('Revenue|impressions')])

    for i in cols:
        df.loc[:, i] = df.loc[:, i].str.replace('\€|\$|\,', "")
        df.loc[:, i] = df.loc[:, i].str.replace('O', "0")
        df.loc[:, i] = df[i].astype('float64')

    df_p_all = df.loc[:, ['Date', 'Placement', 'Advertiser', 'Total impressions', 'Revenue (€)']]
    df_p1 = df.loc[:, ['Date.1', 'Advertisers', 'Key-values ID', 'Total impressions.1', 'Revenue (€).1']]
    df_p2 = df.loc[:, ['Date.2', 'Advertisers.1', 'Key-values ID.1', 'Total impressions.2', 'Revenue (€).2']]
    df_p3 = df.loc[:, ['Date.3', 'Advertisers.2', 'Key-values ID.2', 'Total impressions.3', 'Revenue (USD)']]

    p_cols = ['Date', 'Advertiser', 'Key-values ID', 'Total impressions', 'Revenue (€)']
    df_p1.columns = p_cols
    df_p2.columns = p_cols
    df_p3.columns = p_cols

    df_p_all.loc[:, 'Key-values ID'] = 'Unknown'
    df_p1.loc[:, 'Placement'] = 'MM_BPCOM_HBS_Placement'
    df_p2.loc[:, 'Placement'] = 'MM_BPES_HBS_Placement'
    df_p3.loc[:, 'Placement'] = 'MM_DE_HBS_Placement'
    # Converting from USD to EUR
    df_p3['Revenue (€)'] *= 0.85

    # Concatinating dataframes
    df = pd.concat([df_p_all, df_p1, df_p2, df_p3], ignore_index=True)
    if df[df.duplicated(subset=['Date', 'Advertiser', 'Placement'], keep='first')].shape[0] == 0:
        print('No duplicates detected')
    else:
        print('Duplicates detected!')
        exit()

    df['Date'] = pd.to_datetime(df['Date'])
    df['Day_of_the_month'] = df['Date']
    #####################################
    # Exporting summary table
    #####################################
    summary_table = df.groupby(['Placement', 'Advertiser']).agg([
        'count', 'min', 'mean', 'max', 'std', 'sum']).round(2)
    summary_table.to_csv(path + '\\' + file_name + '_summary_table.csv')

    #####################################
    # Normalizing revenue values (0 to 1)
    # Storing them to norm_revenue column
    # Fitting linear regression models
    #####################################
    all_models_param = pd.DataFrame(
        columns=['Placement', 'Advertiser', 'degree', 'F-statistics', 'r_2', 'r_2_adj',
                 'p-val of F', 'AIC', 'BIC'])
    top_models_param = pd.DataFrame(
        columns=['Placement', 'Advertiser', 'degree', 'F-statistics', 'r_2', 'r_2_adj',
                 'p-val of F', 'AIC', 'BIC'])

    placements = df['Placement'].unique()
    advertisers = df['Advertiser'].unique()

    df['norm_revenue'] = df['Revenue (€)']
    df['Day_of_the_month'] = df['Day_of_the_month'].dt.day

    for i in placements:
        grouped = df.groupby(['Placement'])
        grouped = grouped.get_group(i)

        for j in advertisers:
            if grouped[grouped['Advertiser'] == j].shape[0] < 2:
                continue

            grouped_j = grouped.groupby('Advertiser')
            grouped_j = grouped_j.get_group(j)

            grouped_j.loc[:, 'norm_revenue'] = grouped_j.loc[:, 'norm_revenue'].transform(lambda z: z / z.max())

            df.loc[grouped_j.index.to_list(), 'norm_revenue'] = grouped_j['norm_revenue']

            ########################################################################
            # Fitting (1-3rd order models)
            ########################################################################

            model_param = pd.DataFrame(
                columns=['Placement', 'Advertiser', 'degree', 'F-statistics', 'r_2', 'r_2_adj',
                         'p-val of F', 'AIC', 'BIC'])

            x = grouped_j['Day_of_the_month'][:, np.newaxis]
            y = grouped_j.loc[:, 'norm_revenue'][:, np.newaxis]

            inds = x.ravel().argsort()
            x = x.ravel()[inds].reshape(-1, 1)
            y = y[inds]

            for k in range(1, 4):
                polynomial_features = PolynomialFeatures(degree=k)
                xp = polynomial_features.fit_transform(x)

                model = sm.OLS(y, xp).fit()

                ########################################################################
                # Getting parameters of a model
                ########################################################################

                degree = int(model.summary2().tables[0].iloc[4, 1])
                r_2 = float(model.summary2().tables[0].iloc[6, 1])
                r_2_adj = float(model.summary2().tables[0].iloc[0, 3])
                F = float(model.summary2().tables[0].iloc[4, 3])
                pF = float(model.summary2().tables[0].iloc[5, 3])
                aic = float(model.summary2().tables[0].iloc[1, 3])
                bic = float(model.summary2().tables[0].iloc[2, 3])

                model_param = model_param.append(
                    {'Placement': i, 'Advertiser': j, 'r_2': r_2, 'r_2_adj': r_2_adj, 'degree': degree,
                     'F-statistics': F,
                     'p-val of F': pF, 'AIC': aic, 'BIC': bic}, ignore_index=True)
            # Storing all calculated models
            all_models_param = all_models_param.append(model_param)

            # Picking the degree of the best fit model and saving it to the main df
            # The choice is based on the F value significance and value of adjusted R squared
            # The best model gets its degree and adjusted R squared value placed in the main dataframe (df)
            best_model_param = model_param[model_param['p-val of F'] < 0.05]
            if best_model_param.shape[0] > 0:

                order_group = best_model_param.loc[
                    best_model_param['r_2_adj'] == max(best_model_param['r_2_adj']), 'degree'].values[0]
                df.loc[grouped_j.index.to_list(), 'order_group'] = order_group

                r_2_adjusted = best_model_param.loc[
                    best_model_param['r_2_adj'] == max(best_model_param['r_2_adj']), 'r_2_adj'].values[0]
                df.loc[grouped_j.index.to_list(), 'r_2_adj'] = r_2_adjusted

                # Joining chosen model's statistics together
                top_models_param = top_models_param.append(best_model_param, ignore_index=True)
            else:
                continue
    top_models_param.to_csv((path + '\\' + file_name + '_top_models_param.csv'), index=False)
    df.to_csv(path + '\\' + file_name + '_clean.csv')

    #############################################
    # Daily Revenue/Total impressions vs Advertisers
    #############################################

    placements = ['MM_BPCOM_HBS_Placement', 'MM_BPES_HBS_Placement', 'MM_DE_HBS_Placement']
    variable = ['Revenue (€)', 'Total impressions']

    for j in variable:
        f, axes = plt.subplots(1, 3, figsize=(20, 20))
        plt.subplots_adjust(bottom=0.25)
        for i in placements:
            ax = sns.boxplot(y=j, x="Advertiser", data=df[df['Placement'] == i], orient='v',
                             ax=axes[placements.index(i)])
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.title.set_text(i)
        f.suptitle('Daily ' + j + ' in different placements', fontsize=20, va='center', y=0.95)
        plt.savefig(path + '\\' + file_name + '_placements_' + j + '_.png')
        plt.show()
    ###########################################
    # Plotting significantly fitting models
    ###########################################
    df = df.sort_values(by='order_group')
    fig = px.scatter(df[df['order_group'].notna()], x='Day_of_the_month', y='norm_revenue', facet_col="Placement",
                     facet_row='order_group', color="Advertiser", trendline='lowess',
                     labels={'norm_revenue': 'Normalised revenues', 'order_group': 'Polynomial degree',
                             'Advertiser': 'Advertisers \n (double click to select)'},
                     title='Monthly revenues of Advertisers in different placements (Cols) by fitted regression model '
                           '(Rows)')
    fig.update_layout(
        font_family="Courier New",
        title_font_size=16)
    fig.show()

    print('Finished! You may now stop the testing process')


schedule.every(5).seconds.do(excited_bear)
while True:
    schedule.run_pending()
    time.sleep(1)
