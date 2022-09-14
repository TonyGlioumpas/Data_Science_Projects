In this assignment, a predictive model to predict a churn target (customer retention setting) is created.

The **data set** consists of customer information of a Belgian bank-insurer for retail customers above 18 years old (i.e. not companies but individuals).   
Some data anonymization was applied to e.g. remove personally identifying information.   
The main goal of the data set is to predict which of the customers will churn.  
The target was constructed as follows:  
The main goal of this exercise was to construct a predictive model to predict churn (in short, customer who "leave").   

The available data files describe the customer base, randomly split (stratified) in a train and test part:
* **train_month_3_with_target.csv**: state of the applicable customer base at 2018-TT (the "current" point in time), with the target included (determined using 2018-(TT+4), 2018-(TT+5) and 2018-(TT+6) as described above)
* **train_month_2.csv**: state of the applicable customer base at 2018-(TT-1) (can be used for additional feature engineering; note that balance on account is stable compared to 2018-TT)
* **train_month_1.csv**: state of the applicable customer base at 2018-(TT-2) (can be used for additional feature engineering; note that balance on account is stable compared to 2018-TT)
	
* **test_month_3.csv**: state of the applicable customer base at 2018-TT for the test set customers
* **test_month_2.csv**: state of the applicable customer base at 2018-(TT-1) for the test set customers (can be used for additional feature engineering; note that balance on account is stable compared to 2018-TT)
* **test_month_1.csv**: state of the applicable customer base at 2018-(TT-2) for the test set customers (can be used for additional feature engineering; note that balance on account is stable compared to 2018-TT)

Features:
* client_id: unique identifier per customer (anonymized hash); not to be used as a feature
*  homebanking_active: whether the customer used / logged into home banking (Internet or Mobile app) this month
*  has_homebanking: whether the customer has activated home banking (Internet of Mobile app)
*  has_insurance_21: whether the customer owns "tak 21" life insurance products
*  has_insurance_23: whether the customer owns "tak 23" life insurance products
*  has_life_insurance_fixed_cap: whether the customer owns life insurance with fixed capital
*  has_life_insurance_decreasing_cap: whether the customer owns life insurance with decreasing capital
*  has_fire_car_other_insurance: whether the customer has fire/car/other insurance
*  has_personal_loan: whether the customer has an outstanding personal loan
*  has_mortgage_loan: whether the customer has an outstanding mortgage loan
*  has_current_account: whether the customer has a current (checkings) account
*  has_pension_saving: whether the customer has a pension (retirement) savings account
*  has_savings_account: whether the customer has a savings account
*  has_current_account_starter: whether the customer has a "starter" (a product offering some discounts and typically offered to new or younger customers) current (checkings) account
*  has_savings_account_starter: whether the customer has a "starter" (a product offering some discounts and typically offered to new or younger customers) savings account
*  bal_insurance_21: balance on "tak 21" life insurance
*  bal_insurance_23: balance on "tak 23" life insurance
*  cap_life_insurance_fixed_cap: capital for life insurance with fixed capital
*  cap_life_insurance_decreasing_cap: capital for life insurance with decreasing capital
*  prem_fire_car_other_insurance: premiums paid for fire/car/other insurance
*  bal_personal_loan: outstanding balance on personal loans
*  bal_mortgage_loan: outstanding balance on mortgage loans
*  bal_current_account: balance on current (checkings) accounts
*  bal_pension_saving: balance on pension (retirement) savings accounts
*  bal_savings_account: balance on savings accounts
*  bal_current_account_starter: balance on starter current (checkings) accounts
*  bal_savings_account_starter: balance on starter savings accounts
*  visits_distinct_so: how many different sales offices were visited in the past month by this customer (number of visits per office is not available)
*  visits_distinct_so_areas: how many different sales office areas were visited in the past month by this customer
*  customer_since_all: since when has the customer been a client for either bank or insurance products (YYYY-MM)
*  customer_since_bank: since when has the customer been a client for bank products (YYYY-MM)
*  customer_gender: gender code
*  customer_birth_date: date of birth (YYYY-MM)
*  customer_postal_code: postal code where the customer lives
*  customer_occupation_code: occupation (job) code
*  customer_self_employed: whether the customer is self-employed
*  customer_education: code of education level
*  customer_children: this describes family situation: either no children (no), one baby or toddler (onebaby), preschool child(ren) (preschool), children in secondary school (young), children in high school (adolescent), children between 18-24 (mature), children older than 24 (grownup), or children but without further details (yes)
*  customer_relationship: this describes the marital situation: single (single), living together or married (couple). 

Note that divorced persons and widowers are also indicated as single
	• target: target (1 or 0)

Monetary values were rounded at tens of euros. Extremely affluent customers or customers with large negative balances were not included. Missing values are represented as "NA". Given the real-life nature of the data set, expect other data quality issues as well.
