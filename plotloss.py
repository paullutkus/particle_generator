import matplotlib.pyplot as plt
import csv
import sys

fname = sys.argv[1]

x = []
y = []

csvfile = open(fname, 'r')
plots = csv.reader(csvfile, delimiter=',')
iterplots = iter(plots)
next(iterplots)
for row in iterplots:
    y.append(float(row[0]))
csvfile.close()

csvfile = open(fname, 'r')
plots = csv.reader(csvfile, delimiter=',')
data_list = list(plots)

plt.plot(y)
plt.xlabel('It.')
plt.ylabel('Loss')
plt.title("AE Loss")
text = "final loss: " + str(round(float(data_list[-1][0]), 3))
plt.text(len(data_list)-50000, 0.4, text)

csvfile.close()

plt.show()
