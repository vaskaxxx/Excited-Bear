(NOTE: the Medium_Data_Extraction_Key.json file must be stored in the same directory as the .py file)

# Portfolio
AIM: Create a program that downloads data from google sheets, cleans it and plots it every day, automatically.

Program downloads data from google sheets then cleans and plots it every day. In addition,
it shows the trends of advertiser's monthly revenues (linear regression models). It graphs
only significant models, and the rest of regression models are stored in separated csv file.

Some advertiser data has insufficient sample size for kurtosis test, therefore, program prints
out a warning. However, this warning would disappear as the data updates.

The data came with a recruitment task that I got from one company - therefore, the data is 
incomplete. It is a raw advertisement data that comes from 3 sources (Placements) and is updated everyday.
