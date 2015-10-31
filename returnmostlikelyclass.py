__author__ = 'coenjonker'

import sys

from primitive.functions import *


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print 'Usage: python returnmostlikelyclass.py <traindata> <testdata> <outputfile>'
        sys.exit(1)
    train = sys.argv[1]
    test = sys.argv[2]
    output = sys.argv[3]

    data = read_data(train)

    counts = pd.DataFrame(data.groupby('Category').count()['Dates'])
    counts['Ratio'] = counts / counts.sum()

    classes = [x[0] for x in counts['Ratio'].iteritems()]
    ratios = [round(x[1], 4) for x in counts['Ratio'].iteritems()]

    with open(output, 'w') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(['id'] + classes)
        with open(test, 'r') as in_f:
            reader = csv.reader(in_f)
            reader.next()
            for row in reader:
                rid = row[0]
                result = [rid] + ratios
                writer.writerow(result)















