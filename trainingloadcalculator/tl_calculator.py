from __future__ import print_function
import workout_file_parser
import os
import numpy as np
import pandas as pd
import argparse


class WrongFileError(Exception):
    def __init__(self, message, errors):
        super(WrongFileError, self).__init__(message)


class ParseError(Exception):
    def __init__(self, message, errors):
        super(ParseError, self).__init__(message)


class MissingDataError(Exception):
    def __init__(self, message, errors):
        super(MissingDataError, self).__init__(message)


class WrongDateTimeFormat(Exception):
    def __init__(self, message, errors):
        super(MissingDataError, self).__init__(message)


class NoDistanceError(Exception):
    def __init__(self, message, errors):
        super(MissingDataError, self).__init__(message)


class NoHeartRateError(Exception):
    def __init__(self, message, errors):
        super(MissingDataError, self).__init__(message)


def calculate_y(a, b, x):
    return a * np.exp(b * x)


def fill_missing_seconds(df, time_column, hr_column, distance_column):
    df = df.reset_index()
    df = df.drop_duplicates(subset=time_column)
    date_range = pd.date_range(df[time_column].min(),
                               df[time_column].max(), freq='1S')
    df = df.set_index(time_column)

    df = df.reindex(date_range.values, fill_value=np.NaN)

    df[hr_column] = df[hr_column].interpolate(method='time')
    if distance_column is not None:
        df[distance_column] = df[distance_column].interpolate(method='time')

    return df


def remove_idle_time(df, distance_column, threshold=0.005):
    return df[(df[distance_column] - df[distance_column].shift(1)) > threshold]


def check_file_extension(file_name):
    return (file_name.endswith('gpx') or file_name.endswith('tcx') or
            file_name.endswith('srm') or file_name.endswith('fit') or
            file_name.endswith('pwx'))


def parse_workout_file(file_name):
    try:
        workout_parser = workout_file_parser.WorkoutFileParser()
        parsed_file = workout_parser.parse_file(file_name)
        sample_df = parsed_file[0]['samples']
    except:
        raise ParseError('Something went wrong while parsing the workout file!!')

    if not len(sample_df):
        raise MissingDataError('Empty workout file!')

    return sample_df


def get_column_names(sample_df):
    hr_column, time_column = None, None
    if 'hr' in sample_df.columns:
        hr_column = 'hr'
        time_column = 'time'
    elif 'heartrate' in sample_df.columns:
        hr_column = 'heartrate'
        time_column = 'time'
    elif 'heart_rate' in sample_df.columns:
        hr_column = 'heart_rate'
        time_column = 'timestamp'

    if hr_column is None:
        raise NoHeartRateError('Did not find a heart rate column in the data')

    distance_column = None
    if 'distance' in sample_df.columns:
        distance_column = 'distance'
    elif 'dist' in sample_df.columns:
        distance_column = 'dist'

    return hr_column, time_column, distance_column


def check_spinning_or_rolls(df, distance_column):
    return (sum(np.isnan(df[distance_column].values)) == len(df) 
            or sum(np.isnan(df['lat'].values)) == len(df)
            or sum(np.isnan(df['lon']).values) == len(df))


def calculate_TRIMP(workout_file, a, b, rustHR, maxHR, is_male=True):
    # First, let's make sure we can read the file
    print('Checking file extension...')
    if check_file_extension(workout_file):
        print('Found a unparsed workout-file. Parsing it... ', end='')
        try:
            sample_df = parse_workout_file(workout_file)
        except:
            raise   # ParseError or MissingDataError
    else:
        try:   #  Final attempt: maybe the workout_file is a csv and doesn't need to be parsed
            print('Found a CSV file. Reading it... ', end='')
            sample_df = pd.read_csv(workout_file)
        except:
            raise WrongFileError('\nMake sure the file has one of the following extensions: gpx, tcx, srm, fit, pwx or csv.')
    print('OK')

    # Every format has different names for the columns in the dataframe
    print('Searching for a column with heart rate, time and distance + checking whether the training was a spinning session or on rolls... ', end='')
    spinning_or_rolls = False
    try:
        hr_column, time_column, distance_column = get_column_names(sample_df)
        if distance_column is None:
            spinning_or_rolls = True
    except NoHeartRateError:
        raise

    if not spinning_or_rolls: 
        # Convert the columns to their corresponding datatype
        sample_df[hr_column] = pd.to_numeric(sample_df[hr_column])
        sample_df[distance_column] = pd.to_numeric(sample_df[distance_column])
        sample_df['lat'] = pd.to_numeric(sample_df['lat'])
        sample_df['lon'] = pd.to_numeric(sample_df['lon'])
        try:
            sample_df[time_column] = pd.to_datetime(sample_df[time_column])
            sample_df[time_column] = sample_df[time_column].apply(lambda x: x.replace(microsecond=0))   # Remove microseconds!
        except:
            raise WrongDateTimeFormat('Could not parse the time column!')
        spinning_or_rolls = check_spinning_or_rolls(sample_df, distance_column)
    print('OK')

    # Make sure we have a record for every second. Linear interpolation for the missing seconds
    print('Filling in the missing seconds... ', end='')
    sample_df = fill_missing_seconds(sample_df, time_column, hr_column, distance_column)
    print('OK')

    if not spinning_or_rolls:
        # If the athlete barely moved, we remove the record
        print('Removing idle time... ', end='')
        old_length = len(sample_df)
        sample_df = remove_idle_time(sample_df, distance_column)
        print('Removed', str(old_length - len(sample_df)), 'records')

    print('Calculating the different training load metrics...')
    print('-'*100)
    print('\tDuration:\t\t', len(sample_df), 'seconds')
    sample_df['Hrres'] = (sample_df[hr_column] - rustHR) / (maxHR - rustHR)
    sample_df['Ratio'] = sample_df['Hrres'].apply(lambda x: calculate_y(a, b, x))
    sample_df['TRIMP'] = sample_df['Hrres'] * sample_df['Ratio']
    print('\tiTRIMP:\t\t\t', str(sample_df['TRIMP'].sum() / 60))
    if not is_male:
        a = 0.86
        b = 1.67
    else:
        a = 0.64
        b = 1.92
    b_trimp = (len(sample_df) / 60) * sample_df['Hrres'].mean() * calculate_y(a, b, sample_df['Hrres'].mean())
    print('\tbTRIMP ('+['female', 'male'][is_male]+'):\t'+'\t'*(is_male), str(b_trimp))
    print('-'*100)

parser = argparse.ArgumentParser(description='Calculate the TRIMP for a given workout file')
parser.add_argument('workout_file', metavar='workout_file', type=str, nargs=1, help='the path of the workout_file')
parser.add_argument('a', metavar='a', type=float, nargs=1, help='lactate curve: a+exp(b*x)')
parser.add_argument('b', metavar='b', type=float, nargs=1, help='lactate curve: a+exp(b*x)')
parser.add_argument('rustHR', metavar='rustHR', type=int, nargs=1, help='heart rate in rest')
parser.add_argument('maxHR', metavar='maxHR', type=int, nargs=1, help='maximum heart rate')
parser.add_argument('-gender', metavar='gender', default=1, type=int, help='the gender of the athlete', required=False)

args = parser.parse_args()
print(args.workout_file)
calculate_TRIMP(args.workout_file[0], args.a[0], args.b[0], args.rustHR[0], args.maxHR[0], args.gender)