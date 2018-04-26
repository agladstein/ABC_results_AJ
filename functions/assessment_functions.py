from sys import argv
import os
import pandas as pd
from ggplot import *
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


def create_combined_df(file_name):
    combined_df = pd.read_csv(file_name, sep='\t')
    combined_df['obs_str'] = combined_df['obs'].astype(str)
    if 'chr' in combined_df:
        combined_df['chr_str'] = combined_df['chr'].astype(str)
    else:
        combined_df['chr'] = 0
        combined_df['chr_str'] = combined_df['chr'].astype(str)
    return combined_df


def calc_estimate_dist(param, combined_PosteriorCharacteristics_observed_df):
    estimate = '{}_mode'.format(param)
    estimate_dist_name = '{}_estimate_dist'.format(param)
    combined_PosteriorCharacteristics_observed_df[estimate_dist_name] = (combined_PosteriorCharacteristics_observed_df[estimate] - combined_PosteriorCharacteristics_observed_df[param])**2
    combined_PosteriorCharacteristics_observed_df.head()
    return combined_PosteriorCharacteristics_observed_df


def calc_HPDI_dist(param, combined_PosteriorCharacteristics_observed_df):
    HDI95_upper_name = '{}_HDI95_upper'.format(param)
    HDI95_lower_name = '{}_HDI95_lower'.format(param)
    HPDI_dist_name = '{}_HPDI_dist'.format(param)
    HDI95_upper = combined_PosteriorCharacteristics_observed_df[HDI95_upper_name]
    HDI95_lower = combined_PosteriorCharacteristics_observed_df[HDI95_lower_name]
    true = combined_PosteriorCharacteristics_observed_df[param]
    
    combined_PosteriorCharacteristics_observed_df[HPDI_dist_name] = np.where((true < HDI95_lower) & (true > HDI95_upper), ((true - HDI95_lower)**2 + (true - HDI95_upper)**2)*(-1), (true - HDI95_lower)**2 + (true - HDI95_upper)**2)    
    return combined_PosteriorCharacteristics_observed_df


def calc_estimate_ratio(param, combined_PosteriorCharacteristics_observed_df):
    ratio_name = '{}_estimate_ratio'.format(param)
    estimate = '{}_mode'.format(param)
    combined_PosteriorCharacteristics_observed_df[ratio_name] = combined_PosteriorCharacteristics_observed_df[estimate]/combined_PosteriorCharacteristics_observed_df[param]
    return combined_PosteriorCharacteristics_observed_df


def calc_HPDI_ratio(param, combined_PosteriorCharacteristics_observed_df):
    ratio_upper_name = '{}_HPDI95_upper_ratio'.format(param)
    ratio_lower_name = '{}_HPDI95_lower_ratio'.format(param)
    dist_ratio_name = '{}_HPDI95_dist_ratio'.format(param)
    HDI95_upper_name = '{}_HDI95_upper'.format(param)
    HDI95_lower_name = '{}_HDI95_lower'.format(param)
    combined_PosteriorCharacteristics_observed_df[ratio_upper_name] = combined_PosteriorCharacteristics_observed_df[HDI95_upper_name]/combined_PosteriorCharacteristics_observed_df[param]
    combined_PosteriorCharacteristics_observed_df[ratio_lower_name] = combined_PosteriorCharacteristics_observed_df[HDI95_lower_name]/combined_PosteriorCharacteristics_observed_df[param]
    combined_PosteriorCharacteristics_observed_df[dist_ratio_name] = combined_PosteriorCharacteristics_observed_df[ratio_upper_name] - combined_PosteriorCharacteristics_observed_df[ratio_lower_name]
    return combined_PosteriorCharacteristics_observed_df


def proportion_smaller_chr1(df, dist_column, chrom, param):
    proportion_smaller_dict = {}
    chr1 = df[dist_column].loc[(df['chr_str'] == '1')].reset_index(drop=True)
    other_chr = df[dist_column].loc[(df['chr_str'] == str(chrom))].reset_index(drop=True)

    proportion = (float(sum(chr1 > other_chr) - sum(chr1 < 0)))/100
    if chrom == 'genome':
        proportion_smaller_dict['chr'] = 0
    else:
        proportion_smaller_dict['chr'] = int(chrom)
    proportion_smaller_dict['param'] = param
    proportion_smaller_dict['proportion'] = proportion
    return proportion_smaller_dict


def gg_boxplot_estimate_dist(param, combined_PosteriorCharacteristics_observed_df, y_axis_name):
    plot = ggplot(aes(x = 'chr', y = y_axis_name), data = combined_PosteriorCharacteristics_observed_df) + \
        geom_boxplot() + \
        theme_bw()
    return plot


def gg_lineplot(param, combined_PosteriorCharacteristics_observed_df, y_axis_name):
    plot = ggplot(aes(x = 'chr', y = y_axis_name, colour='obs_str'), data = combined_PosteriorCharacteristics_observed_df) + \
        geom_point() + \
        geom_line() + \
        theme_bw()
    return plot


def regplot(y_axis_name, df):
    plot = sns.lmplot(x="chr", y=y_axis_name, 
               size=5, 
               data=df, 
               lowess=True)
    return plot


def gg_distribution_plot(param, combined_PosteriorCharacteristics_observed_df, x_axis_name):
    plot = ggplot(aes(x = x_axis_name, colour='factor(chr)'), data = combined_PosteriorCharacteristics_observed_df) + \
        geom_density() + \
        geom_vline(x = 1) + \
        scale_color_brewer(type='div', palette=2) + \
        theme_bw()
    return plot


def gg_density_plot(param, PosteriorDensities_df, true_value):
    density = list(PosteriorDensities_obs1_df)[PosteriorDensities_obs1_df.columns.get_loc(param)+1]
    plot = ggplot(aes(x = param, y = density, colour = 'chr_str'), data = PosteriorDensities_df) + \
        geom_line(size = 2) + \
        geom_vline(x = true_value, size = 3, colour = 'black') + \
        scale_color_brewer(type='div', palette=2) + \
        theme_bw()
    return plot


def gg_lineplot2(df):
    plot = ggplot(aes(x = 'chr', y = 'proportion', colour='param'), data = df) + \
        geom_point() + \
        geom_line(size = 2) + \
        geom_hline(y = 0.5, linetype = 'dashed', color = 'black', size = 2) + \
        ylim(0,1) + \
        scale_color_brewer(type='div', palette=2) + \
        theme_bw()
    return plot


def calc_confusion_matrix(df, chrom, prob):
    confusion = {}
    act_B_ls_C = df['B-C'] < 0
    act_B_gr_C = df['B-C'] > 0
    pred_B_gr_C = df['B_C_prob'] > prob
    pred_B_ls_C = df['B_C_prob'] < 1 - prob
        
    confusion['act_B_ls_C_pred_B_gr_C'] = len(df[(df.chr == chrom) & act_B_ls_C  & pred_B_gr_C])
    confusion['act_B_ls_C_pred_B_ls_C'] = len(df[(df.chr == chrom) & act_B_ls_C  & pred_B_ls_C])
    confusion['act_B_gr_C_pred_B_gr_C'] = len(df[(df.chr == chrom) & act_B_gr_C  & pred_B_gr_C])
    confusion['act_B_gr_C_pred_B_ls_C'] = len(df[(df.chr == chrom) & act_B_gr_C  & pred_B_ls_C])   
    confusion['chr'] = chrom
    confusion['prob'] = str(prob)
        
    return confusion