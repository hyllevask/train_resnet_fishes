# Split data into training/validation/test sets.
# Only splits the csv file not the image folders

import pandas as pd

df = pd.read_csv("./combined_images/annotations.csv")

validation = df.sample(n = 1000)
df = df.drop(validation.index)

test = df.sample(n=1000)

train = df.drop(test.index)


validation.to_csv("./combined_images/validation.csv")
test.to_csv("./combined_images/test.csv")
train.to_csv("./combined_images/train.csv")

