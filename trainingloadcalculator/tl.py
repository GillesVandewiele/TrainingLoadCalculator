from __future__ import print_function
import tl_calculator

parser = argparse.ArgumentParser(description='Calculate the TRIMP for a given workout file')
parser.add_argument('workout_file', metavar='workout_file', type=str, nargs=1, help='the path of the workout_file')
parser.add_argument('a', metavar='a', type=float, nargs=1, help='lactate curve: a+exp(b*x)')
parser.add_argument('b', metavar='b', type=float, nargs=1, help='lactate curve: a+exp(b*x)')
parser.add_argument('rustHR', metavar='rustHR', type=int, nargs=1, help='heart rate in rest')
parser.add_argument('maxHR', metavar='maxHR', type=int, nargs=1, help='maximum heart rate')
parser.add_argument('-gender', metavar='gender', default=1, type=int, help='the gender of the athlete', required=False)

args = parser.parse_args()
print(args.workout_file)
tl_calculator.calculate_TRIMP(args.workout_file[0], args.a[0], args.b[0], args.rustHR[0], args.maxHR[0], args.gender)