(NOTE: the google API key is provided upon reuqest)

# Excited Bear
AIM: Create a program that downloads data from google sheets, cleans it and plots it every day, automatically.
DATA: A raw internet advertisement data of revenues and impression counts that come from 3 sources (Placements) that is updated everyday.

Program downloads data from google sheets, cleans and plots it every day. In addition,
it classifies ressults into 1-3rd degree polynomials of advertiser's monthly revenues. It graphs
local regression of significant models. Details of all regression models are stored in separated csv file.

Some advertiser data has insufficient sample size for kurtosis test, therefore, program prints
out a warning. However, this warning would disappear as the data updates.
