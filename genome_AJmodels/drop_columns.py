from sys import argv
import pandas as pd

'''
Takes a tab delimited input file with column names.
Prints a tab delimited files without the columns containing the provided string
'''

file = argv[1] # file to remove columns from
string = argv[2] # string in columns to be removed
outfile = argv[3]

df_drop = pd.read_csv(file, sep = '\t', low_memory=False).dropna(axis=0, how='any')

df_drop[df_drop.columns.drop(list(df_drop.filter(regex=string)))].dropna(axis=0, how='any').to_csv(outfile, sep='\t', index=False)