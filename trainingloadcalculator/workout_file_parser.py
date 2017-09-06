"""
Contains code to parse files of the following formats:
    * .fit: wrapper around the python fitparse library
    * .pwx: custom code that parses xml
    * .tcx: custom code that parses xml
    * .gpx: using activityio
    * .srm: using activityio
"""

from __future__ import print_function

import xml.etree.ElementTree as ET
import arrow
import pandas as pd
import fitparse
from fitparse import Activity
import numpy as np
import os
from subprocess import call
import os
import glob
import zipfile
import gzip
import shutil
import random
import json


class WorkoutFileParser(object):
    def __init__(self):
        pass

    def parse_file(self, file_path):
        """
        Parses a file, located at `file_path` into one or more `pandas.DataFrame`
        :param file_path: string with the path of the file
        :return: one or more DataFrames (pandas)
        """
        file_type = file_path.split('.')[-1]
        file_parsers = {'srm': self.parse_srm_file, 'pwx': self.parse_pwx_file,
                        'fit': self.parse_fit_file, 'gpx': self.parse_gpx_file,
                        'tcx': self.parse_tcx_file}
        return file_parsers[file_type](file_path), file_type

    @staticmethod
    def _fit_get_dataframe_by_type(activity, _type):
        records = []
        for definition in activity.get_records_by_type(_type):
            record = {}
            for f in definition.fields:
                record[f.field.name] = f.data
            records.append(record)
        if len(records):
            return pd.DataFrame(records).drop_duplicates()
        else:
            return None

    @staticmethod
    def parse_fit_file(file_path):
        activity = Activity(file_path)
        activity.parse()

        # Parse the session data, this contains some aggregate statistics about the workout (e.g. avg/max_speed)
        sessions_df = WorkoutFileParser._fit_get_dataframe_by_type(activity, 'session')

        # Parse bike_profile data, contains some information the bike and about the sensors on the bike
        bike_profiles_df = WorkoutFileParser._fit_get_dataframe_by_type(activity, 'bike_profile')

        # Parse activity data
        activities_df = WorkoutFileParser._fit_get_dataframe_by_type(activity, 'activity')

        # Parse lap data: these are aggregate statistics about segments within the workout
        laps_df = WorkoutFileParser._fit_get_dataframe_by_type(activity, 'lap')

        # Parse event data: contains the timestamp of certain events (such as stopping and starting)
        events_df = WorkoutFileParser._fit_get_dataframe_by_type(activity, 'event')

        # Parse user data: contains information about the athlete
        user_df = WorkoutFileParser._fit_get_dataframe_by_type(activity, 'user_profile')

        # Parse body metrics such as weight, percent_fat, ..
        body_metrics_df = WorkoutFileParser._fit_get_dataframe_by_type(activity, 'weight_scale')

        # Parse file data: which sensor/device produced this data?
        file_data_df = WorkoutFileParser._fit_get_dataframe_by_type(activity, 'file_id')

        # Parse the sample data: certain parameters (distance, heart rate, ...) are measured at certain freqs
        sample_df = WorkoutFileParser._fit_get_dataframe_by_type(activity, 'record')

        return {'sessions': sessions_df, 'bike_profiles': bike_profiles_df, 'activities': activities_df,
                'laps': laps_df, 'events': events_df, 'user': user_df, 'body_metrics': body_metrics_df,
                'file_data': file_data_df, 'samples': sample_df}

    @staticmethod
    def parse_pwx_file(file_path):
        tree = ET.parse(file_path)

        ns = {'n': 'http://www.peaksware.com/PWX/1/0'}

        workouts = tree.getroot().findall('.//n:workout', ns)
        for workout in workouts:
            sport_type = workout.find('.//n:sportType', ns).text
            summary = workout.find('.//n:summarydata', ns)
            device = workout.find('.//n:device', ns)
            athlete = workout.find('.//n:athlete', ns)
            time = workout.find('.//n:time', ns).text
            parsed_time = arrow.get(time)

            events = workout.findall('.//n:event', ns)
            samples = workout.findall('.//n:sample', ns)
            segments = workout.findall('.//n:segment', ns)

            sample_dicts = []
            event_dicts = []
            segment_dicts = []
            device_dict = {}
            summary_dict = {}
            athlete_dict = {}

            if device:
                for child in device:
                    if child.tag.split('}')[1] != 'extension':
                        device_dict[child.tag.split('}')[1]] = child.text
                    else:
                        for child2 in child:
                            device_dict[child2.tag.split('}')[1]] = child2.text

            if athlete:
                for child in athlete:
                    athlete_dict[child.tag.split('}')[1]] = child.text

            if summary:
                for child in summary:
                    if child.text is None:
                        for attr in child.attrib:
                            summary_dict[child.tag.split('}')[1] + attr] = float(child.attrib[attr])
                    else:
                        summary_dict[child.tag.split('}')[1]] = float(child.text)

            if samples:
                for sample in samples:
                    sample_dict = {}
                    for child in sample:
                        sample_dict[child.tag.split('}')[1]] = float(child.text)
                    sample_dicts.append(sample_dict)

            sample_df = pd.DataFrame.from_records(sample_dicts)
            if len(sample_df):
                sample_df['Time'] = sample_df['timeoffset'].apply(lambda x: parsed_time.replace(seconds=+x).datetime)
                sample_df = sample_df.drop('timeoffset', axis=1)

            if events:
                for event in events:
                    event_dict = {}
                    for child in event:
                        if child.text.isdigit():
                            event_dict[child.tag.split('}')[1]] = float(child.text)
                        else:
                            event_dict[child.tag.split('}')[1]] = child.text

                    event_dicts.append(event_dict)

            events_df = pd.DataFrame.from_records(event_dicts)

            if segments:
                for i, segment in enumerate(segments):
                    segment_dict = {}
                    if segment.find('.//n:name', ns):
                        segment_dict['name'] = segment.find('.//n:name', ns).text
                    else:
                        segment_dict['name'] = str(i)

                    for child in segment.find('.//n:summarydata', ns):
                        if child.text is None:
                            for attr in child.attrib:
                                segment_dict[child.tag.split('}')[1] + attr] = float(child.attrib[attr])
                        else:
                            segment_dict[child.tag.split('}')[1]] = float(child.text)

                    segment_dicts.append(segment_dict)

            segments_df = pd.DataFrame.from_records(segment_dicts)

            return {'sessions': pd.DataFrame([summary_dict]), 'type': sport_type,
                    'laps': segments_df, 'events': events_df, 'user': pd.DataFrame([athlete_dict]),
                    'file_data': pd.DataFrame([device_dict]), 'samples': sample_df}

    # @staticmethod
    # def parse_fit_file(path):
    #     activity = Activity(path)
    #     activity.parse()
    #     records = activity.get_records_by_type('record')
    #     d = []
    #     for record in records:
    #         d.append(record.as_dict())
    #
    #     return pd.DataFrame.from_records(d)

    @staticmethod
    def get_zip_gz_file(directory, opened_files):
        for file in os.listdir(directory):
            if file.endswith('.zip'):
                if directory + os.sep + file not in opened_files:
                    return directory + os.sep + file, 'zip'
            if file.endswith('.gz'):
                if directory + os.sep + file not in opened_files:
                    return directory + os.sep + file, 'gz'
        return None, None

    @staticmethod
    def give_files_unique_name(directory):
        for file in os.listdir(directory):
            if file[-4:] != 'json':
                filename, extension = '.'.join(file.split('.')[:-1]), '.' + file.split('.')[-1]
                suffix = filename.split('_')[-1]
                if extension == '.zip' or extension == '.gz' or (suffix.isdigit() and int(suffix) > 1000):
                    continue
                else:
                    print(directory + os.sep + filename + '_' + str(random.randint(1000, 10000000000)) + extension)
                    os.rename(directory + os.sep + file,
                              directory + os.sep + filename + '_' + str(random.randint(1000, 10000000000)) + extension)

    @staticmethod
    def extract_files(directory):
        opened_files = []
        file, _type = WorkoutFileParser.get_zip_gz_file(directory, opened_files)

        while file:

            if _type == 'zip':
                zip_ref = zipfile.ZipFile(file)
                zip_ref.extractall(directory)

            elif _type == 'gz':
                with open('.'.join(file.split('.')[:-1]), 'wb') as f_out, gzip.open(file, 'rb') as f_in:
                    shutil.copyfileobj(f_in, f_out)

            opened_files.append(file)
            file, _type = WorkoutFileParser.get_zip_gz_file(directory, opened_files)

            WorkoutFileParser.give_files_unique_name(directory)

    @staticmethod
    def delete_files(directory):
        for file in os.listdir(directory):
            if file.endswith('.csv') or file.endswith('.tcx') or file.endswith('.gpx') or file.endswith('.pwx') \
                    or file.endswith('.srm') or file.endswith('.fit'):
                print(directory + os.sep + file)
                os.remove(directory + os.sep + file)

    @staticmethod
    def parse_tcx_file(file_path):
        tree = ET.parse(file_path)
        # print tree.xpath('namespace-uri(.)')
        ns = {'n': tree.getroot().tag.split('}')[0].strip('{'),
              'k': 'http://www.garmin.com/xmlschemas/ActivityExtension/v2'}

        if tree.getroot().find('.//n:Activity', ns) is None: return {}

        sport_type = tree.getroot().find('.//n:Activity', ns).attrib['Sport']

        laps = tree.getroot().findall('.//n:Lap', ns)
        lap_values = []
        for lap in laps:
            vector = {}
            if lap.find('.//n:TotalTimeSeconds', ns) is not None:
                vector['duration'] = lap.find('.//n:TotalTimeSeconds', ns).text
            else:
                vector['duration'] = np.NaN

            if lap.find('.//n:DistanceMeters', ns) is not None:
                vector['distance'] = lap.find('.//n:DistanceMeters', ns).text
            else:
                vector['distance'] = np.NaN

            if lap.find('.//n:MaximumSpeed', ns) is not None:
                vector['maxSpeed'] = lap.find('.//n:MaximumSpeed', ns).text
            else:
                vector['maxSpeed'] = np.NaN

            if lap.find('.//n:Calories', ns) is not None:
                vector['calories'] = lap.find('.//n:Calories', ns).text
            else:
                vector['calories'] = np.NaN

            if lap.find('.//n:AverageHeartRateBpm/Value', ns) is not None:
                vector['avgHr'] = lap.find('.//n:AverageHeartRateBpm/Value', ns).text
            else:
                vector['avgHr'] = np.NaN

            if lap.find('.//n:MaximumHeartRateBpm/Value', ns) is not None:
                vector['maxHr'] = lap.find('.//n:MaximumHeartRateBpm/Value', ns).text
            else:
                vector['maxHr'] = np.NaN

            if lap.find('.//n:Cadence', ns) is not None:
                vector['cadence'] = lap.find('.//n:Cadence', ns).text
            else:
                vector['cadence'] = np.NaN
            lap_values.append(vector)

        trackpoints_values = []
        trackpoints = tree.getroot().findall('.//n:Trackpoint', ns)
        for trackpoint in trackpoints:
            vector = {}
            vector['time'] = trackpoint.find('.//n:Time', ns).text

            if trackpoint.find('.//n:AltitudeMeters', ns) is not None:
                vector['altitude'] = trackpoint.find('.//n:AltitudeMeters', ns).text
            else:
                vector['altitude'] = np.NaN

            if trackpoint.find('.//n:DistanceMeters', ns) is not None:
                vector['distance'] = trackpoint.find('.//n:DistanceMeters', ns).text
            else:
                vector['distance'] = np.NaN

            if trackpoint.find('.//n:HeartRateBpm', ns) is not None:
                vector['hr'] = trackpoint.find('.//n:HeartRateBpm/n:Value', ns).text
            else:
                vector['hr'] = np.NaN

            if trackpoint.find('.//n:Position', ns) is not None:
                vector['lat'] = trackpoint.find('.//n:Position/n:LatitudeDegrees', ns).text
                vector['lon'] = trackpoint.find('.//n:Position/n:LongitudeDegrees', ns).text
            else:
                vector['lat'] = np.NaN
                vector['lon'] = np.NaN

            if trackpoint.find('.//n:Cadence', ns) is not None:
                vector['cadence'] = trackpoint.find('.//n:Cadence', ns).text
            else:
                vector['cadence'] = np.NaN

            if trackpoint.find('.//n:Extensions/k:TPX/k:Speed', ns) is not None:
                vector['speed'] = trackpoint.find('.//n:Extensions/k:TPX/k:Speed', ns).text
            else:
                vector['speed'] = np.NaN

            if trackpoint.find('.//n:Extensions/k:TPX/k:Watts', ns) is not None:
                vector['watt'] = trackpoint.find('.//n:Extensions/k:TPX/k:Watts', ns).text
            else:
                vector['watt'] = np.NaN

            trackpoints_values.append(vector)

        sample_df = pd.DataFrame.from_records(trackpoints_values)

        file_data = {}
        if tree.getroot().find('.//n:Creator/n:Name', ns) is not None:
            file_data['device'] = tree.getroot().find('.//n:Creator/n:Name', ns).text
        else:
            file_data['device'] = np.NaN

        file_data_df = pd.DataFrame.from_records([file_data])

        laps_df = pd.DataFrame.from_records(lap_values)

        return {'type': sport_type, 'laps': laps_df,
                'file_data': file_data_df, 'samples': sample_df}

    @staticmethod
    def parse_gpx_file(file_path):
        from subprocess import call
        python_code = """
import sys
sys.path.append(\"""" + os.getcwd() + '/activityio' + """\")
from activityio import gpx
import pandas as pd
pd.DataFrame(gpx.read(\"""" + file_path + """\")).to_csv("gpx_temp.csv")
        """
        call('python3.5 -c \'' + python_code + '\'', shell=True)
        return {'samples': pd.read_csv('gpx_temp.csv')}

    @staticmethod
    def parse_srm_file(file_path):
        python_code = """
import sys
sys.path.append(\"""" + os.getcwd() + '/activityio' + """\")
from activityio import srm
import pandas as pd
pd.DataFrame(srm.read(\"""" + file_path + """\")).to_csv("srm_temp.csv")
        """
        call('python3.5 -c \'' + python_code + '\'', shell=True)
        return {'samples': pd.read_csv('srm_temp.csv')}