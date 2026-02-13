import pandas as pd

drivers = pd.read_csv("data/input/drivers.csv")
rides_1 = pd.read_csv("data/input/rides_1.csv")
rides_2 = pd.read_csv("data/input/rides_2.csv")
rides_3 = pd.read_csv("data/input/rides_3.csv")
rides_4 = pd.read_csv("data/input/rides_4.csv")
cars = pd.read_csv("data/input/cars.csv")
incidents = pd.read_csv("data/input/incidents.csv")

rides = pd.concat([rides_1, rides_2, rides_3, rides_4], ignore_index=True)

# Number of days passed from the last inspection of the driver's car
today = pd.to_datetime("05-15-2023")
cars["last_inspection_date"] = pd.to_datetime(cars["last_inspection_date"])
cars["days_since_inspection"] = (today - cars["last_inspection_date"]).dt.days.astype(int)

# Experience of the driver
drivers["experience"] = (today.year - drivers["started_driving_year"]).astype(int)

#  == Work on rides data ==
#First step: calculate the total number of upvotes per ride
upvotes_col = [
    "car_cleanness_upvote_given",
    "politeness_upvote_given",
    "communication_upvote_given",
    "punctuality_upvote_given"
]

rides["upvotes_total"] = (
    rides[upvotes_col]
    .fillna(0)
    .sum(axis=1)
    .astype(int)
) 

#Second step: indicate whether a ride is rejected using a boolean variable
rides["is_rejected"] = (rides["status"] == "Rejected by the driver")

#Third step: aggregate data by driver id
rides_stats = ( 
    rides
    .groupby("driver_id", as_index = False)
    .agg(
        number_of_rejected_rides = ("is_rejected", "sum"),
        number_of_upvotes = ("upvotes_total", "sum"),
        number_of_complaints = ("complaint_given", "sum")
    )
)

#Incidents dataset
incident_stats = (
    incidents
    .groupby("driver_id", as_index = False)
    .aggregate(
        number_of_incidents = ("incident_id", "count")
    )
)

#Merge all dataset
result = drivers.merge(rides_stats, on = "driver_id", how = "left")
result = result.merge(cars, on = "car_id", how = "left")
result = result.merge(incident_stats, on = "driver_id", how = "left")

#Rename cols to simplify understanding of the dataset
result = result.rename(columns={"model":"car_model", "manufacture_year":"car_manufacture_year"})

#Select only cols we'll use for preprocessing and modelling
final_cols = [
    "driver_id",
    "car_model",
    "car_manufacture_year",
    "days_since_inspection",
    "age",
    "experience",
    "second_language",
    "rating",
    "net_worth_of_tips",
    "number_of_rejected_rides",
    "number_of_upvotes",
    "number_of_complaints",
    "number_of_incidents",
    "driver_class"
]

result[final_cols].to_csv("data/processed/collected.csv", index = False)