# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import streamlit as st
    import requests
    import pandas as pd

    st.title("COVID-19 Data Analysis and Prediction")

    response = requests.get("https://disease.sh/v3/covid-19/countries")
    covid = response.json()
    covid_19 = pd.DataFrame(covid)

    from sklearn.preprocessing import LabelEncoder

    # Apply label encoding
    encoder = LabelEncoder()
    covid_19["country"] = encoder.fit_transform(covid_19["country"])
    encoded_labels = list(covid_19["country"])
    # Get corresponding original values for encoded labels
    original_labels = encoder.inverse_transform(encoded_labels)

    # Print the original labels
    result_list_of_country = {k: v for k, v in zip(original_labels, encoded_labels)}

    from sklearn.preprocessing import StandardScaler

    X = covid_19.drop(columns=['countryInfo', 'continent', 'deaths'])
    x_stand = StandardScaler().fit_transform(X)

    cov19 = pd.DataFrame(x_stand, columns=['updated', 'country', 'cases', 'todayCases', 'todayDeaths', 'recovered',
                                           'todayRecovered', 'active', 'critical', 'casesPerOneMillion',
                                           'deathsPerOneMillion ', 'tests', 'testsPerOneMillion', 'population',
                                           'oneCasePerPeople', 'oneDeathPerPeople', 'oneTestPerPeople',
                                           'activePerOneMillion', 'recoveredPerOneMillion', 'criticalPerOneMillion'])
    cov19['deaths'] = covid_19['deaths']

    # Calculate the correlation matrix
    correlation_matrix = cov19.corr()

    # Get the absolute correlation values with the target variable
    correlation_with_target = abs(correlation_matrix['deaths'])

    # Select the features with correlation greater than 0.5
    selected_features = correlation_with_target[correlation_with_target > 0.5].index.tolist()
    selected_features.remove("deaths")

    # Importing libraries
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn import metrics

    X = abs(cov19[selected_features])
    y = covid_19['deaths']

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.35,
                                                        random_state=40)  # splitting data with test size of 35%
    model = LinearRegression()  # build linear regression model
    model.fit(x_train, y_train)  # fitting the training data
    predicted = model.predict(x_test)  # testing our model’s performance
    mean_squared_error(y_test, predicted)
    metrics.r2_score(y_test, predicted)

    input_value1 = st.number_input("Please enter the  number of cases: ")

    input_value2 = st.number_input("Please enter the  number of recovered cases: ")

    input_value3 = st.number_input("Please enter the  number of critical cases: ")

    input_value4 = st.number_input("Please enter the  number of tests: ")

    parameter_default_values = [input_value1, input_value2, input_value3, input_value4]

    test = pd.DataFrame([parameter_default_values],
                        columns=['cases', 'recovered', 'critical', 'tests'],
                        dtype=float)

    if st.button("Prediction"):
        prediction = model.predict(test)  # testing our model
        st.text("Number of deaths predicted: ")
        st.text(prediction)
