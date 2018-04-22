from sys import argv
import os
import pandas as pd
# from ggplot import *
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True, style="ticks")
import numpy as np
# import rpy2
# %load_ext rpy2.ipython


def my_fun(word):
    words = '{} and {}'.format(word, word)
    return words


def read_abc_config(abc_config_name):
    """
    Get input files used for ABC and results files output by ABC.
    :param abc_config_name: The configuration file used to run ABCtoolbox 
    :return: simName:
    :return: obsName:
    :return: outputPrefix:
    """

    simName = ""
    obsName = ""
    outputPrefix = ""

    if os.path.isfile(abc_config_name):
        print(abc_config_name)
        abc_config = open(abc_config_name, 'r')
        for line in abc_config:
            line_lst = line.split()
            arg = line_lst[0]
            if arg == "simName":
                simName = line.split()[1]
            if arg == "obsName":
                obsName = line.split()[1]
            if arg == "outputPrefix":
                outputPrefix = line.split()[1]
            if arg == "params":
                params_str = line.split()[1]
        abc_config.close()
    else:
        print('{} does not exist'.format(abc_config_name))
        exit()

    if not simName:
        print("simName not included in ABCtoolbox config file")
        exit()
    if not obsName:
        print("obsName not included in ABCtoolbox config file")
        exit()
    if not outputPrefix:
        print("outputPrefix not included in ABCtoolbox config file")
        exit()

    return [simName, obsName, outputPrefix, params_str]


def get_results_files(outputPrefix):
    """
    Define names of ABCtoolbox estimate results files
    :param outputPrefix: the output prefix provided in the ABCtoolbox config file
    :return: names of ABCtoolbox output files
    """

    BestSimsParamStats_name = '{}model0_BestSimsParamStats_Obs0.txt'.format(outputPrefix)
    MarginalPosteriorDensities_name = '{}model0_MarginalPosteriorDensities_Obs0.txt'.format(outputPrefix)
    MarginalPosteriorCharacteristics_name = '{}model0_MarginalPosteriorCharacteristics.txt'.format(outputPrefix)
    jointPosterior_name = '{}model0_jointPosterior_8_9_Obs0.txt'.format(outputPrefix)
    MarginalPosteriorCharacteristics_reformat_name = '{}model0_MarginalPosteriorCharacteristicsReformat.txt'.format(
        outputPrefix)
    modelFit_name = '{}modelFit.txt'.format(outputPrefix)

    return [BestSimsParamStats_name,
            MarginalPosteriorDensities_name,
            MarginalPosteriorCharacteristics_name,
            jointPosterior_name,
            MarginalPosteriorCharacteristics_reformat_name,
            modelFit_name]

def get_results_files_reducedParams(outputPrefix):
    """
    Define names of ABCtoolbox estimate results files
    :param outputPrefix: the output prefix provided in the ABCtoolbox config file
    :return: names of ABCtoolbox output files
    """

    BestSimsParamStats_name = '{}model0_BestSimsParamStats_Obs0.txt'.format(outputPrefix)
    MarginalPosteriorDensities_name = '{}model0_MarginalPosteriorDensities_Obs0.txt'.format(outputPrefix)
    MarginalPosteriorCharacteristics_name = '{}model0_MarginalPosteriorCharacteristics.txt'.format(outputPrefix)
    jointPosterior_name = '{}model0_jointPosterior_1_2_Obs0.txt'.format(outputPrefix)
    MarginalPosteriorCharacteristics_reformat_name = '{}model0_MarginalPosteriorCharacteristicsReformat.txt'.format(
        outputPrefix)
    modelFit_name = '{}modelFit.txt'.format(outputPrefix)

    return [BestSimsParamStats_name,
            MarginalPosteriorDensities_name,
            MarginalPosteriorCharacteristics_name,
            jointPosterior_name,
            MarginalPosteriorCharacteristics_reformat_name,
            modelFit_name]


def get_modelchocie_files(outputPrefix):
    """
    Define names of ABCtoolbox model choice
    :param outputPrefix: the output prefix provided in the ABCtoolbox config file
    :return: names of ABCtoolbox output files
    """

    confusionMatrix_name = '{}confusionMatrix.txt'.format(outputPrefix)
    modelChoiceValidation_name = '{}modelChoiceValidation.txt'.format(outputPrefix)
    modelFit_name = '{}modelFit.txt'.format(outputPrefix)

    return [confusionMatrix_name,
            modelChoiceValidation_name,
            modelFit_name]


def reformat_Characteristics(MarginalPosteriorCharacteristics_name):
    """
    reformat the ABCtoolbox output file MarginalPosteriorCharacteristics to a table with parameter as the rows and
     posterior density characteristics as columns.
    :param MarginalPosteriorCharacteristics_name: file name of ABCtoolbox output file with characteristics of posterior
    density.
    :return: df_table: pandas dataframe with parameters as rows and posterior density characteristics as columns
    """

    characteristics = ['mode', 'mean', 'median', 'q50_lower', 'q50_upper', 'q90_lower', 'q90_upper', 'q95_lower',
                       'q95_upper',
                       'q99_lower', 'q99_upper', 'HDI50_lower', 'HDI50_upper', 'HDI90_lower', 'HDI90_upper',
                       'HDI95_lower',
                       'HDI95_upper', 'HDI99_lower', 'HDI99_upper']
    n_chars = len(characteristics)

    if os.path.isfile(MarginalPosteriorCharacteristics_name):
        print('parsing {}'.format(MarginalPosteriorCharacteristics_name))

        df = pd.read_csv(MarginalPosteriorCharacteristics_name, sep='\t').drop('dataSet', 1)
        header = list(df)

        df_list = []
        start = 0
        for i in range(1, int(len(df.columns) / n_chars)):
            param = header[start].split(characteristics[0])[0].strip('_')
            df_param = df.loc[:, header[start]:header[start + n_chars - 1]]
            df_param.columns = characteristics
            df_param['param'] = param
            df_param.set_index('param')
            df_list.append(df_param)
            start = n_chars * i
        df_table = pd.concat(df_list)

    else:
        print('{} does not exist'.format(MarginalPosteriorCharacteristics_name))
        print('Did you run ABCtoolbox in this directory?')
        exit()

    return df_table


def get_param_indexes(params_str):
    """
    Convert string of parameter column numbers into list of ints
    :param params_str: string of parameter column numbers from the ABCtoolbox config file, of the form: 1-5,9-12
    :return: get_param_indexes: list of ints
    """
    param_indexes = []
    ranges = params_str.split(',')
    for values in ranges:
        if '-' in values:
            x = int(values.split('-')[0].strip()) - 1
            y = int(values.split('-')[1].strip())
            param_indexes.extend(list(range(x, y)))
        else:
            param_indexes.append(int(values) -1 )
    return param_indexes


def get_col_names(ints, df):
    """
    Get list of column names of a dataframe from list of ints
    :param ints: list of integers
    :param df: dataframe with column names
    :return: names: list of column names
    """

    names = []
    for i in ints:
        names.append(list(df)[i])
    return names


def plot_param_densities(posterior, truncated, prior, df_chrs_reformat, param):
    """
    Use matplotlib and seaborn to plot prior, truncated, and posterior distributions of parameter.
    :param posterior: dataframe of ABCtoolbox *MarginalPosteriorDensities_Obs0.txt
    :param truncated: dataframe of ABCtoolbox *BestSimsParamStats_Obs0.txt
    :param prior: dataframe of simulation input for ABCtoolbox
    :param param: string parameter name, which is the column name in the dataframes
    """

    mode = float(df_chrs_reformat['mode'].loc[df_chrs_reformat['param'] == param])
    HDI90_lower = float(df_chrs_reformat['HDI90_lower'].loc[df_chrs_reformat['param'] == param])
    HDI90_upper = float(df_chrs_reformat['HDI90_upper'].loc[df_chrs_reformat['param'] == param])

    plt.figure()
    plt.xlabel(param);
    plt.plot(posterior[param], posterior['{}.density'.format(param)], label='posterior');
    sns.kdeplot(truncated[param], label='truncated prior');
    sns.kdeplot(prior[param], color='grey', label='prior');
    ymin, ymax = plt.ylim()
    plt.vlines(mode, ymin, ymax, colors='black');
    plt.vlines(HDI90_lower, ymin, ymax, colors='black', linestyle='dotted');
    plt.vlines(HDI90_upper, ymin, ymax, colors='black', linestyle='dotted');
    return


def create_joint_df(jointPosterior_name):
    if os.path.isfile(jointPosterior_name):
        joint_NEA_NWA_df = pd.read_csv(jointPosterior_name, sep='\t')
    else:
        print('{} does not exist'.format(jointPosterior_name))
        print('Did you run ABCtoolbox in this directory?')
        exit()
    return joint_NEA_NWA_df


def get_prob_NEA_grtr_NWA(joint_NEA_NWA_df):
    total_density = sum(joint_NEA_NWA_df['density'])
    NEA_grtr_density = joint_NEA_NWA_df[joint_NEA_NWA_df['Log10_NEA'] > joint_NEA_NWA_df['Log10_NWA']]['density']
    prob = sum(NEA_grtr_density) / total_density
    return prob


def plot_joint_mtpltlb(joint_NEA_NWA_df, df_chrs_reformat):
    # density map
    NWA, NEA, z = joint_NEA_NWA_df['Log10_NWA'], joint_NEA_NWA_df['Log10_NEA'], joint_NEA_NWA_df['density']
    NWA = np.unique(NWA)
    NEA = np.unique(NEA)
    X, Y = np.meshgrid(NWA, NEA)
    Z = z.reshape(len(NEA), len(NWA))
    plt.pcolormesh(X, Y, Z, cmap='viridis')
    colorbar = plt.colorbar()
    colorbar.set_label('Density')

    # y = x line
    plt.plot(NWA, NWA, color='black')

    # Scatterplot point
    NEA_mode = df_chrs_reformat.loc[df_chrs_reformat['param'] == 'Log10_NEA']['mode']
    NWA_mode = df_chrs_reformat.loc[df_chrs_reformat['param'] == 'Log10_NWA']['mode']
    plt.scatter(NWA_mode, NEA_mode, marker='*', facecolor='black', edgecolor='none')

    # Axes limits and labels
    plt.xlim(np.min(NWA), np.max(NWA))
    plt.xlabel('$\log_{10}$ NWA')

    plt.ylabel('$\log_{10}$ NEA')
    plt.ylim(min(NEA), max(NEA))
    plt.show()
    return
