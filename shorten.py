import pickle

# Load the original data
with open('data.pkl', 'rb') as file:
    data = pickle.load(file)

# Select a smaller subset (e.g., first 3 samples)
small_data = data[:3]

# Save the smaller subset to a new file
with open('/mnt/data/small_data.pkl', 'wb') as file:
    pickle.dump(small_data, file)

