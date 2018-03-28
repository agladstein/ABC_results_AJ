import pandas as pd
import os


def create_PosteriorCharacteristics_df(observed_file_name, PosteriorCharacteristics_file_name, obs):
    observed_df = pd.read_csv(observed_file_name, sep='\t')
    observed_df.drop([col for col in observed_df.columns if 'CGI' in col], axis=1, inplace=True)
    PosteriorCharacteristics_df = pd.read_csv(PosteriorCharacteristics_file_name, sep='\t')
    
    PosteriorCharacteristics_observed_df = pd.concat([observed_df, PosteriorCharacteristics_df], axis=1)
    PosteriorCharacteristics_observed_df['obs'] = str(obs)
    if 'dataSet' in PosteriorCharacteristics_observed_df:
        PosteriorCharacteristics_observed_df.drop('dataSet', axis=1, inplace=True)
    return PosteriorCharacteristics_observed_df


def create_PosteriorDensities_df(PosteriorDensities_file_name, obs):
    PosteriorDensities_df = pd.read_csv(PosteriorDensities_file_name, sep='\t')
    PosteriorDensities_df['obs'] = str(obs)
    return PosteriorDensities_df


def combine_obs_files(results_path, obs_list):
    combined_PosteriorCharacteristics_observed_df = pd.DataFrame()
    combined_PosteriorDensities_df = pd.DataFrame()
    for obs in obs_list:
        observed_file_name = '{}/obs{}/results_param_observed.txt'.format(results_path, obs)
        PosteriorCharacteristics_file_name = '{}/obs{}/ABC_test_genome_estimate_10pls_100ret_model0_MarginalPosteriorCharacteristics.txt'.format(results_path, obs)
        PosteriorDensities_file_name = '{}/obs{}/ABC_test_genome_estimate_10pls_100ret_model0_MarginalPosteriorDensities_Obs0.txt'.format(results_path, obs)

        PosteriorCharacteristics_observed_df = create_PosteriorCharacteristics_df(observed_file_name, PosteriorCharacteristics_file_name, obs)
        PosteriorDensities_df = create_PosteriorDensities_df(PosteriorDensities_file_name, obs)

        combined_PosteriorCharacteristics_observed_df = pd.concat([combined_PosteriorCharacteristics_observed_df, PosteriorCharacteristics_observed_df])
        combined_PosteriorDensities_df = pd.concat([combined_PosteriorDensities_df, PosteriorDensities_df])
    return [combined_PosteriorCharacteristics_observed_df, combined_PosteriorDensities_df]


def main():
    print('This script is hardcoded to be run on an system with the file structure /vol_c/ABC_test_genome')
    
    results_path = '/vol_c/ABC_test_genome'
    combined_PosteriorCharacteristics_observed_name = '{}/ABC_update_estimate_10pls_100ret_model0_MarginalPosteriorCharacteristics_combined.txt'.format(results_path)
    combined_PosteriorDensities_name = '{}/ABC_update_estimate_10pls_100ret_model0_MarginalPosteriorDensities_Obs0_combined.txt'.format(results_path)

    obs_list = list(range(1, 101))

    if os.path.isfile(combined_PosteriorCharacteristics_observed_name) and os.path.isfile(combined_PosteriorDensities_name):
        print('{} and {} already exist.'.format(combined_PosteriorDensities_name, combined_PosteriorCharacteristics_observed_name))
    else:
        print('{} do not {} already exist.'.format(combined_PosteriorDensities_name, combined_PosteriorCharacteristics_observed_name))
        print('Creating files.')
        [combined_PosteriorCharacteristics_observed_df, combined_PosteriorDensities_df] = combine_obs_files(results_path, obs_list)
        combined_PosteriorCharacteristics_observed_df.to_csv(combined_PosteriorCharacteristics_observed_name, sep='\t', index=False)
        combined_PosteriorDensities_df.to_csv(combined_PosteriorDensities_name, sep='\t', index=False)
    
if __name__ == '__main__':
    main()
