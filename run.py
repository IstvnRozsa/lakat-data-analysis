import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pylab as p
from scipy import integrate

def generate_labels_data(data_dict):
    # Data to plot
    labels = []
    sizes = []

    for x, y in data_dict.items():
        labels.append(x)
        sizes.append(y)
    return labels, sizes

'''
Task 4. Real Epidemiology data
'''
def last_data(url:str, selected_country:str, selected_date:str):
    df = pd.read_csv(url)
    df = df.melt(id_vars=["Province/State", "Country/Region", "Lat", "Long"], 
        var_name="Date", 
        value_name="Value")
    df.sort_values(by=['Value'])
    df2 = df[(df["Date"]==selected_date) & (df["Country/Region"] == selected_country)] 
    max = df2['Value'].groupby(df2["Date"]).sum().max()
    return max, df2



print("Task 4. Real Epidemiology data")
confirmed_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
deaths_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
recovered_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"

selected_country = "Hungary"
selected_date = "8/4/21" # The last day when recovered cases was administrated
confirmed, confirmed_df = last_data(confirmed_url, selected_country, selected_date)
deaths, deaths_df = last_data(deaths_url, selected_country, selected_date)
recovered, recovered_df = last_data(recovered_url, selected_country, selected_date)

sick = confirmed - deaths - recovered

print("The number of actually confirmed people in " + selected_country + " on "+selected_date+":", f"{confirmed:,}")
print("The number of actually recovered people in " + selected_country + " on "+selected_date+":", f"{recovered:,}")
print("The number of actually dead people in " + selected_country + " on "+selected_date+":", f"{deaths:,}")
print("-"*4)
print("The number of actually sick people in " + selected_country + " on "+selected_date+":", f"{sick:,}")


covid_dict = {
    "confirmed": confirmed,
    "recovered": recovered,
    "deaths": deaths
}

covid_labels, covid_sizes = generate_labels_data(covid_dict)
# Plot
plt.bar(covid_labels, covid_sizes)
 
plt.xlabel("Case registered")
plt.ylabel("No. of people")
title = "Task 4. Sick people in " + selected_country + " on "+selected_date+": " + f"{sick:,}"
plt.title(title)
plt.show()




'''
Task 5. Running cost for your experiment
'''
df = pd.read_csv("supplierprices.txt", sep='\t')


needs = {
    "distilled water":100,
    "alcohol (96%)": 1000,
    "rabbit litter": 5,
    "fruit fly food":50
    }

needs_price = 0
needs_with_price = {}
for n in needs:
    df2 = df[df["Reagent name"]==n]
    if df2["Quantity"].max() >= needs[n]:
        price = df2["Price"].min() * needs[n] 
    else:
        price = df2["Price"].min() * df2["Quantity"].max()
        print("The item is out of stock. You can only order " + str(df2["Quantity"].max()) + " pieces")
        
    print("The price of " + n +": "+ str(price) + " USD")
    needs_price += price
    print("-"*50)
    needs_with_price[n] = price


# Data to plot
label_needs, prices = generate_labels_data(needs_with_price)

# Plot
plt.pie(prices, labels=label_needs)
plt.title("Task 5. Running cost")
plt.axis('equal')
plt.show()



'''
Task 6.1 What is the busiest hour during the day?
'''
print()
pizza_order_df = pd.read_csv("pizzaorder.txt", sep='\t', names=["Time", "Pizza Type", "Distance", "Delivery Guy"])
pizza_order_df["Hour"] = pizza_order_df["Time"].str[:2]


task_6_1 = pizza_order_df.groupby(["Hour"][:2]).count()
task_6_1_maximum = task_6_1.max()
task_6_1 = task_6_1[task_6_1["Pizza Type"]==task_6_1_maximum["Pizza Type"]].reset_index()
print("Task 6.1: The busiest hour during the day: " + str(task_6_1["Hour"].max()))





'''
Task 6.2: What is your revenue if Margherite cost 1000, Salami 1200, Funghi 1100, Quatro formaggi 1250, Calzone 1250, Frutti di Mare 1500, Hawaii 1250 ?
'''
print()
# create a list of our conditions
task_6_2_conditions = [
    (pizza_order_df['Pizza Type'] == "Margherita"),
    (pizza_order_df['Pizza Type'] == "Salami"),
    (pizza_order_df['Pizza Type'] == "Funghi"),
    (pizza_order_df['Pizza Type'] == "Quattro Formaggi"),
    (pizza_order_df['Pizza Type'] == "Calzone"),
    (pizza_order_df['Pizza Type'] == "Frutti di Mare"),
    (pizza_order_df['Pizza Type'] == "Hawaii")
    ]

# create a list of the values we want to assign for each condition
task_6_2_values = [1000,1200,1100,1250,1250,1500,1250]

pizza_order_df['Price'] = np.select(task_6_2_conditions, task_6_2_values)

print("Task 6.2: Revenue for today: " + str(pizza_order_df["Price"].sum()) + " HUF")



'''
Task 6.3: How many pizzas were ordered before noon?
'''
print()
print("-"*15,"Task 6.3. How many pizzas were ordered before noon?","-"*15)
task_6_3 = pizza_order_df[pizza_order_df["Hour"].astype(int) < 12]
print("Task 6.3: " + str(task_6_3["Pizza Type"].count()) + " pizzas were ordered before noon")



'''
Task 6.4: Which pizza was order the most? (and how much)
'''
print()
task_6_4 = pizza_order_df.groupby(["Pizza Type"]).count()
task_6_4 = task_6_4.reset_index()
task_6_4_maximum = task_6_4.max()
task_6_4_m = task_6_4[task_6_4["Time"]==task_6_4_maximum["Time"]].reset_index()
print("Task 6.4: The most popular pizza: " + task_6_4_m["Pizza Type"].max() + ", sold: " + str(task_6_4_m["Price"].max()))
# Plot
plt.pie(task_6_4["Price"], labels=task_6_4["Pizza Type"])
plt.axis('equal')
plt.title("Task 6.4: Which pizza was order the most?")
plt.show()



'''
Task 6.5: Who travelled the most that day?
'''
print()
task_6_5 = pizza_order_df.groupby(["Time"])
task_6_5 = task_6_5.first().reset_index()

task_6_5 = task_6_5.groupby(["Delivery Guy"]).sum("Distance").reset_index()[["Delivery Guy", "Distance"]]
max_travelled = task_6_5[task_6_5["Distance"]==task_6_5.max()["Distance"]]
print("Task 6.5: " + max_travelled["Delivery Guy"].max() +" travelled the most (" + str(max_travelled["Distance"].max()) +" km) today")

# Plot
plt.pie(task_6_5["Distance"], labels=task_6_5["Delivery Guy"])
plt.axis('equal')
plt.title("Task 6.5: Who travelled the most that day?")
plt.show()


'''
Task 6.6: 
'''
print()
price_df = pd.read_csv("pizzaprices.txt")

pizza_order_w_price_df = pd.merge(pizza_order_df, price_df, on='Pizza Type', how="left", indicator=True)
pizza_order_w_price_df = pizza_order_w_price_df.groupby("Hour").sum('Price from File').reset_index()

max_hourly_revenue = pizza_order_w_price_df[pizza_order_w_price_df["Price from File"] == pizza_order_w_price_df.max()["Price from File"]]


# Plot
plt.bar(pizza_order_w_price_df["Hour"], pizza_order_w_price_df["Price from File"])
 
plt.xlabel("Hours")
plt.ylabel("Revenue")
title_6_6 = "The highest revenue was " + str(max_hourly_revenue["Price from File"].sum()) + " at " + max_hourly_revenue["Hour"].max()
plt.title(title_6_6)
plt.show()



'''
Task Lotka Voltera
'''


a = 1.
b = 0.1
c = 1.5
d = 0.75
def dX_dt(X, t=0):
    """ Return the growth rate of fox and rabbit populations. """
    return np.array([ a*X[0] -   b*X[0]*X[1] , -c*X[1] + d*b*X[0]*X[1] ])


X_f0 = np.array([     0. ,  0.])
X_f1 = np.array([ c/(d*b), a/b])
all(dX_dt(X_f0) == np.zeros(2) ) and all(dX_dt(X_f1) == np.zeros(2)) # => True


def d2X_dt2(X, t=0):
    """ Return the Jacobian matrix evaluated in X. """
    return np.array([[a -b*X[1],   -b*X[0]     ],
                  [b*d*X[1] ,   -c +b*d*X[0]] ])

A_f0 = d2X_dt2(X_f0)


A_f1 = d2X_dt2(X_f1)                    # >>> array([[ 0.  , -2.  ],
                                        #            [ 0.75,  0.  ]])
# whose eigenvalues are +/- sqrt(c*a).j:
lambda1, lambda2 = np.linalg.eigvals(A_f1) # >>> (1.22474j, -1.22474j)
# They are imaginary numbers. The fox and rabbit populations are periodic as follows from further
# analysis. Their period is given by:
T_f1 = 2*np.pi/abs(lambda1)                # >>> 5.130199


t = np.linspace(0, 15,  1000)              # time
X0 = np.array([10, 5])                     # initials conditions: 10 rabbits and 5 foxes
X, infodict = integrate.odeint(dX_dt, X0, t, full_output=True)
infodict['message']                     # >>> 'Integration successful.'


rabbits, foxes = X.T
f1 = p.figure()
p.plot(t, rabbits, 'r-', label='Rabbits')
p.plot(t, foxes  , 'b-', label='Foxes')
p.grid()
p.legend(loc='best')
p.xlabel('time')
p.ylabel('population')
p.title('Evolution of fox and rabbit populations')
p.show()
f1.savefig('rabbits_and_foxes_1.png')