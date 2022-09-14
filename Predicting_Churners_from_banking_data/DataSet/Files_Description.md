The available data files describe the customer base, randomly split (stratified) in a train and test part:
* **train_month_3_with_target.csv**: state of the applicable customer base at 2018-TT (the "current" point in time), with the target included (determined using 2018-(TT+4), 2018-(TT+5) and 2018-(TT+6) as described above)
* **train_month_2.csv**: state of the applicable customer base at 2018-(TT-1) (can be used for additional feature engineering; note that balance on account is stable compared to 2018-TT)
* **train_month_1.csv**: state of the applicable customer base at 2018-(TT-2) (can be used for additional feature engineering; note that balance on account is stable compared to 2018-TT)
	
* **test_month_3.csv**: state of the applicable customer base at 2018-TT for the test set customers
* **test_month_2.csv**: state of the applicable customer base at 2018-(TT-1) for the test set customers (can be used for additional feature engineering; note that balance on account is stable compared to 2018-TT)
* **test_month_1.csv**: state of the applicable customer base at 2018-(TT-2) for the test set customers (can be used for additional feature engineering; note that balance on account is stable compared to 2018-TT)
