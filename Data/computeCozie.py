from influxdb import InfluxDBClient, DataFrameClient
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.cluster import KMeans
import seaborn as sns
import requests
import datetime
import time
import json
import numpy as np

# added later 
import pytz

import credentials as cd # Credentials are store in a GIT IGNORE'ed file


#################### TODOs############################
'''


'''

################################################################################

##Global
# Influx authentication
client = DataFrameClient(cd.host, cd.port, cd.usr, cd.passwd, cd.db, ssl=True, verify_ssl=True)

def main():
	# Get current time and time x=52 weeks ago for querrzing influxdb
	now_str, from_time_str = get_current_time(weeks=100)

	# Get comfort data from learnign trail for the last 15min

	cozie_df = query_cozie(to_time_str = now_str, from_time_str = from_time_str)

	#plot_by_user(cozie_df)

	#plot_by_hour(cozie_df)

	#plot_by_week_hour(cozie_df)

	#plot_response_rate(cozie_df)

	#plot_by_heartrate(cozie_df)

	# get mbient data
	mbient_df = query_mbient(to_time_str = now_str, from_time_str = from_time_str)
	# mapped mbient data to specific users
	mapped_mbient_df = mbientMapping(mbient_df)
	# combine both dataframes
	cozie_mbient_df = combineCozieMbient(cozie_df, mapped_mbient_df)

	plot_mbient_location_comparison(mapped_mbient_df)

def get_current_time(weeks):
	#current time, not sure if I should be using Sing time or now
	now = datetime.datetime.utcnow()
	#the time 1 year ago
	time_ago = now - datetime.timedelta(weeks=weeks)

	#convert time to a format for querrying influxdb
	now_str = now.strftime('%Y-%m-%dT%H:%M:%SZ')
	from_time_str = time_ago.strftime('%Y-%m-%dT%H:%M:%SZ')

	#returning as global variable - TODO tidy this up to avoid future issues
	return(now_str, from_time_str)


def query_cozie(to_time_str, from_time_str):
	print("querying learnign trail database on influx db ")

	# Query Influx
	result = client.query("SELECT thermal, heartRate FROM people.autogen.coolbit WHERE time > '{}' AND time < '{}' GROUP BY userid".format(from_time_str,to_time_str))

	# Create emtpy dataframe
	cozie_df = pd.DataFrame()

		# Iterate through groups (rooms, and users)
	for key in result:
		# Get data frame belonging to a group
		current_df = result[key]
		# Set the location and user id to the data
		current_df["user_id"] =  key[1][0][1]
		# Concat this sub dataframe to the main result data frame
		cozie_df = pd.concat([cozie_df, current_df], sort=False)


	# Clean Dataframe
	# remove nulls
	cozie_df = cozie_df[cozie_df.user_id != "null"]
	# Remove Claytons fucking typo ;)
	cozie_df = cozie_df[cozie_df.user_id != "66NZJD"]

	#Convert Datetime to Singapore
	cozie_df.index = cozie_df.index.tz_convert('Asia/Singapore')

	cozie_df.to_csv("../data/cozie.csv")

	return cozie_df

def plot_by_heartrate(cozie_df):
	
	#Filter out null heart rate data
	cozie_df = cozie_df[cozie_df.heartRate>0]
	print(cozie_df)

	g = sns.FacetGrid(cozie_df, col="thermal")
	# bins = np.linspace(0, 60, 13)
	g.map(plt.hist, "heartRate", color="steelblue")
	axes = g.axes.flatten()
	axes[0].set_title("Preferm Warmer")
	axes[0].set_ylabel("Count")
	axes[1].set_title("Comfortable")
	axes[2].set_title("Prefer Cooler")
	plt.savefig("../figures/heartHist.pdf", format="pdf")
	plt.show()

def plot_by_user(cozie_df):



	#Reset index for plotting purporses
	cozie_df.reset_index(inplace = True)

	#Restrict HeartRate
	# cozie_df = cozie_df[cozie_df.heartRate>0]
	# cozie_df = cozie_df[cozie_df.heartRate<100]

	#Split to hot, commfy and cold dataframes and count votes by user
	hot_df = cozie_df[cozie_df.thermal == 11].groupby('user_id').count()
	comfy_df = cozie_df[cozie_df.thermal == 10].groupby('user_id').count()
	cold_df = cozie_df[cozie_df.thermal == 9].groupby('user_id').count()

	#Rename the column name from thermal to corresponding value
	hot_df.rename(columns={"thermal": "Prefer Cooler"}, inplace=True)
	comfy_df.rename(columns={"thermal": "Comfortable"}, inplace=True)
	cold_df.rename(columns={"thermal": "Prefer Wamer"}, inplace=True)


	#consolidate the data into a single dataframe
	consolodated_df = pd.concat([hot_df["Prefer Cooler"],comfy_df["Comfortable"], cold_df["Prefer Wamer"] ], axis=1, sort=False)


	#Fill NaN with zeros
	consolodated_df.fillna(value=0, inplace=True)
	consolodated_df["Total"] = consolodated_df.sum(axis=1)

	# Filter users with less than 10 data points
	consolodated_df = consolodated_df[consolodated_df.Total >10.0]

	# normalise Data frame and drop total
	normalised_df = consolodated_df.div(consolodated_df.Total, axis=0)
	normalised_df.drop(columns = "Total", inplace=True)

	# Do some keans clustering NOTE: This is now redundant as the sns clustermap automatically does the same analysis 
	kmeans = KMeans(n_clusters=3)
	kmeans.fit(normalised_df)
	labels = kmeans.predict(normalised_df)
	centroids = kmeans.cluster_centers_
	centroid_df = pd.DataFrame(centroids, columns= ["Prefer Cooler", "Comfortable", "Prefer Wamer"])
	clustered_df = pd.concat([normalised_df, centroid_df])

	print(consolodated_df)
	print(normalised_df)

	#Generate new anonymous data labels for users
	user_numbers = range(1,normalised_df.index.size+1)
	user_list = ["User " + str(x) for x in user_numbers]
	print(user_list)

	normalised_df.index = user_list

	# Plot the normalised df
	ax = sns.clustermap(normalised_df, cmap="Blues", metric="euclidean", method="single", annot=True, fmt='g', col_cluster=False)
	#ax.ax_heatmap.axes.get_yaxis().set_visible(False)
	plt.savefig("../figures/cozie_users.pdf", format='pdf')
	plt.show()


def plot_by_hour(cozie_df):

	cozie_df['day']=cozie_df.index.dayofweek
	cozie_df['hour'] = cozie_df.index.hour
	cozie_df['hour_of_week'] = cozie_df.day*24 + cozie_df.hour
	print(cozie_df)

	# Split to hot, commfy and cold dataframes and count votes by user
	hot_df = cozie_df[cozie_df.thermal == 11].groupby('hour').count()
	comfy_df = cozie_df[cozie_df.thermal == 10].groupby('hour').count()
	cold_df = cozie_df[cozie_df.thermal == 9].groupby('hour').count()

	#Rename the column name from thermal to corresponding value
	hot_df.rename(columns={"thermal": "Prefer Cooler"}, inplace=True)
	comfy_df.rename(columns={"thermal": "Comfortable"}, inplace=True)
	cold_df.rename(columns={"thermal": "Prefer Wamer"}, inplace=True)


	#consolidate the data into a single dataframe
	consolodated_df = pd.concat([hot_df["Prefer Cooler"],comfy_df["Comfortable"], cold_df["Prefer Wamer"] ], axis=1, sort=False)

	#Fill NaN with zeros
	consolodated_df.fillna(value=0, inplace=True)
	print(consolodated_df)

	# Get Total
	consolodated_df["Total"] = consolodated_df.sum(axis=1)

	# normalise Data frame and drop total
	normalised_df = consolodated_df.div(consolodated_df.Total, axis=0)
	normalised_df.drop(columns = "Total", inplace=True)

	#normalised_df.index=["Mon", "Tue", "Wed", "Thur", "Fri", "Sat", "Sun"]

	# Plot the normalised df
	ax = sns.heatmap(normalised_df, cmap="Blues", annot=consolodated_df[["Prefer Cooler", "Comfortable", "Prefer Wamer"]], fmt='g', annot_kws={"size": 8})
	plt.ylabel('Hour of the Day')
	plt.savefig('../figures/hourPlot.pdf', format='pdf')
	plt.show()

def plot_by_week_hour(cozie_df):



	cozie_df['day']=cozie_df.index.dayofweek
	cozie_df['hour'] = cozie_df.index.hour
	cozie_df['hours_of_week'] = cozie_df.day*24 + cozie_df.hour
	print(cozie_df)


	# Split to hot, commfy and cold dataframes and count votes by user
	hot_df = cozie_df[cozie_df.thermal == 11].groupby('hours_of_week').count()
	comfy_df = cozie_df[cozie_df.thermal == 10].groupby('hours_of_week').count()
	cold_df = cozie_df[cozie_df.thermal == 9].groupby('hours_of_week').count()

	#Rename the column name from thermal to corresponding value
	hot_df.rename(columns={"thermal": "Prefer Cooler"}, inplace=True)
	comfy_df.rename(columns={"thermal": "Comfortable"}, inplace=True)
	cold_df.rename(columns={"thermal": "Prefer Wamer"}, inplace=True)


	#consolidate the data into a single dataframe
	consolodated_df = pd.concat([hot_df["Prefer Cooler"],comfy_df["Comfortable"], cold_df["Prefer Wamer"] ], axis=1, sort=False)

	#Fill NaN with zeros
	consolodated_df.fillna(value=0, inplace=True)
	print(consolodated_df)

	# Get Total
	consolodated_df["Total"] = consolodated_df.sum(axis=1)

	# normalise Data frame and drop total
	normalised_df = consolodated_df.div(consolodated_df.Total, axis=0)
	normalised_df.drop(columns = "Total", inplace=True)

	#Reset index for plotting purporses
	consolodated_df.reset_index(inplace = True)

	#normalised_df.index=["Mon", "Tue", "Wed", "Thur", "Fri", "Sat", "Sun"]

	# Plot the normalised df
	ax = sns.lineplot(x="hours_of_week", y="Total", data=consolodated_df)
	plt.xlabel("Response Rate Based on Hour of the Week")
	plt.show()

def plot_response_rate(cozie_df):
		print(cozie_df)
		time_group_df = cozie_df.resample('D').agg({"thermal":'count'})
		print(time_group_df)
		time_group_df = time_group_df[time_group_df.index > '03-30-2019']
		time_group_df.reset_index(inplace=True)
		print(time_group_df)
		ax = sns.barplot(x="index", y="thermal", data=time_group_df)
		ax.xaxis.set_major_locator(plt.MaxNLocator(5))
		ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
		plt.ylabel("Nmber of Responses")
		plt.xticks(rotation='vertical')
		plt.xlabel("Days Since Start of Experiment")
		plt.savefig("../figures/response_rate.pdf", format="pdf")
		plt.show()

def query_mbient(to_time_str, from_time_str):
	print("querying mbient database on influx db ")

	# Query Influx
	result = client.query("SELECT illuminance, relative_humidity, temperature FROM people.autogen.mbient WHERE time > '{}' AND time < '{}' GROUP BY sensor_id, sensor_location".format(from_time_str,to_time_str)) 
	# Create emtpy dataframe
	ambient_df = pd.DataFrame()

	# Iterate through groups (sensor_id, sensor_location)
	for key in result:
		# Get data frame belonging to a group
		current_df = result[key]
		# Set the sensor_id and location to the data
		current_df["sensor_id"] =  key[1][0][1]
		current_df["sensor_location"] = key[1][1][1]
		# Concat this sub dataframe to the main result data frame
		ambient_df = pd.concat([ambient_df, current_df], sort=False)

	#Convert Datetime to Singapore
	ambient_df.index = ambient_df.index.tz_convert('Asia/Singapore')
	ambient_df.to_csv("ambient.csv")

	return ambient_df

"""
Mapped mbient dataframe to the user who wore it in a given timeframe based on a mapping csv
"""
def mbientMapping(dataframe):
	mapped_df = pd.DataFrame()
	users_map = pd.read_csv("mbientMapping.csv") # load mapping of users and mbient
	timezone = pytz.timezone("Asia/Singapore") # auxiliary variable for timezone convertion of naive dates

	for _, row in users_map.iterrows():
		currSensorId = row['sensor_id']
		currStartDate = timezone.localize(datetime.datetime.strptime(row['start_date'], "%d.%m.%Y"))
		currEndDate = timezone.localize(datetime.datetime.strptime(row['end_date'], "%d.%m.%Y"))
		# filter rows based on parameters
		curr_mapped = dataframe[(dataframe['sensor_id'] == currSensorId) & 
								(dataframe.index >= currStartDate) & 
								(dataframe.index <= currEndDate)]
		curr_mapped['user_id'] = row['participantID']
		mapped_df = mapped_df.append(curr_mapped)
		
	mapped_df.to_csv("ambientMapped.csv")
	
	print("ambientMapped counts:")
	countInstances(mapped_df, "mappedMbient")
	return mapped_df

"""
Merged the cozie responses with the already mapped mbiente responses based on the user_id and timestamp
"""
def combineCozieMbient(cozie_df, mapped_mbient_df):
	merged_df = pd.DataFrame()
	cozie_df['time'] = cozie_df.index
	cozie_df = cozie_df.sort_index()
	mapped_mbient_df['time'] = mapped_mbient_df.index
	mapped_mbient_df = mapped_mbient_df.sort_index()

	# generate list of users
	user_list = mapped_mbient_df['user_id'].unique()

	# for each existing user
	for user in user_list:
		# first filter by user
		curr_cozie_df = cozie_df[cozie_df['user_id'] == user]
		curr_mapped_mbient_df = mapped_mbient_df[mapped_mbient_df['user_id'] == user]
		# then do the time aligment
		curr_merged = pd.merge_asof(curr_cozie_df, curr_mapped_mbient_df, left_on='time', 
                                                right_on='time', 
                                                tolerance=pd.Timedelta('1m'))
		merged_df = merged_df.append(curr_merged)

	# remove nulls
	merged_df.dropna(axis=0, how='any', inplace=True)

	cozie_df.to_csv("cozie.csv") # TODO: debugging
	merged_df.to_csv("test.csv") # TODO: debugging


	countInstances(merged_df, "merged mbient and cozie")
	return merged_df

"""
Different plots of mbient data comparing the values when worn on wrist or as a clip
"""
def plot_mbient_location_comparison(dataframe): # TODO: add illuminance
	wrist_df = dataframe[dataframe['sensor_location'] == 'wrist']
	clip_df = dataframe[dataframe['sensor_location'] == 'clip']

	wrist_temp_df = wrist_df['temperature']
	clip_temp_df = clip_df['temperature']
	
	wrist_light_df = wrist_df['illuminance']
	clip_light_df = clip_df['illuminance']

	# get rid of nulls for boxplots
	wrist_temp_df.dropna(axis=0, how='any', inplace=True)
	clip_temp_df.dropna(axis=0, how='any', inplace=True)
	wrist_light_df.dropna(axis=0, how='any', inplace=True)
	clip_light_df.dropna(axis=0, how='any', inplace=True)

	# boxp plot of temperature
	fig1, ax1 = plt.subplots(figsize=(14,12))
	df = [wrist_temp_df, clip_temp_df]
	ax1.boxplot(df)
	ax1.set_title('Temperature distribution')
	ax1.set_ylabel('Temperature (C)')
	ax1.set_xticklabels(['wrist', 'clip'])

	fig2, ax2 = plt.subplots(figsize=(14,12))
	df = [wrist_light_df, clip_light_df]	
	ax2.boxplot(df)
	ax2.set_title('Illuminance distribution')
	ax2.set_ylabel('Illuminance (Lux)')
	ax2.set_xticklabels(['wrist', 'clip'])

	# temperature time series comparison
	user_list = dataframe['user_id'].unique() # generate list of users
	for user in user_list: # for each existing user
		currWrist_df = wrist_df[wrist_df['user_id'] == user]
		currClip_df = clip_df[clip_df['user_id'] == user]



	plt.show()

	return

"""
Count number of instances per user
"""
def countInstances(dataframe, titleName):
	if 'merge' in titleName:
		count_df = dataframe.groupby(['user_id_x'], sort=False).size().reset_index(name='Count')
	else:
		count_df = dataframe.groupby(['user_id'], sort=False).size().reset_index(name='Count')
	print(count_df)

	ax = count_df.plot(kind='bar',figsize=(14,12), title=titleName)
	ax.set_xlabel("Users")
	ax.set_ylabel("Frequency")
	
	if 'merge' in titleName:
		ax.set_xticklabels(count_df['user_id_x'])
	else:
		ax.set_xticklabels(count_df['user_id'])
	return
	


#Just in case we end up importing funcitons from this file
if __name__ == "__main__":

	main()
