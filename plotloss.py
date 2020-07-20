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
plt.yscale('log')
plt.xlabel('It.')
plt.ylabel('Loss')
plt.title("AE Loss")
text = "Final Loss: " + str(round(float(data_list[-1][0]), 3))
plt.text(len(data_list)-9000, 0.1, text)

csvfile.close()

plt.show()
