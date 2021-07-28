'''Tutorial for plotting data.'''

import matplotlib.pyplot as plt
import pandas as pd

x = [1, 2, 3, 4, 5]
y = [2, 3, 4, 8, 9]
z = [1, 5, 6, 7, 8]
'''plt.plot(x, y)
plt.plot(x, z)
plt.xlabel("x")
plt.ylabel("y and z")
plt.title("Test\nkxdjf\nlkjdf")
plt.show()'''
sam_da = pd.read_csv("sample_data.csv")
print(sam_da)
print(type(sam_da))
print(sam_da.column_b.iloc[2])
plt.plot(sam_da.column_a, sam_da.column_c, 'o')
plt.plot(sam_da.column_a, sam_da.column_b, '*')
plt.show()