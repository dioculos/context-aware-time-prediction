import pandas as pd
import os

# Read your CSV file into a pandas DataFrame
df = pd.read_csv(os.getcwd()+"/datasets/bpic/bpic.csv", sep="|")

print (df.columns)

# Find the maximum sequence length for CaseID
max_seq_length = df.groupby('CaseID')['ActivityID'].transform(len).max()

# Create a new DataFrame to store the padded sequences
padded_df = pd.DataFrame()

# Iterate over unique CaseIDs
for case_id, group in df.groupby('CaseID'):
    activities = group['ActivityID'].tolist()
    padding = max_seq_length - len(activities)
    padded_activities = ['4'] * padding + activities
    first_timestamp = group['CompleteTimestamp'].iloc[0]  # Get the timestamp of the first occurrence
    timestamps = [first_timestamp] * padding + group['CompleteTimestamp'].tolist()  # Apply the timestamp to all entries
    padded_df = pd.concat([padded_df, pd.DataFrame({'CaseID': [case_id] * max_seq_length, 'ActivityID': padded_activities, 'CompleteTimestamp': timestamps})])

# Save the padded DataFrame to a new CSV file
padded_df.to_csv('padded_event_log.csv', index=False, sep="|")