import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(r"C:\Users\anish\Downloads\SMA 1-3 Dataset.csv")

# Count the frequency of each location
location_counts = df['Location'].value_counts()

print(location_counts)

# Visualize the top 10 locations
top_locations = location_counts.head(10)

# Plotting the top 10 most common locations
plt.figure(figsize=(10, 6))
top_locations.plot(kind='bar', color='skyblue')
plt.title('Top 10 Most Common Locations')
plt.xlabel('Location')
plt.ylabel('Frequency')
plt.xticks(rotation=45)  # rotate x-axis tick labels by 45
plt.show()
