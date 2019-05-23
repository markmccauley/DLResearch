from selenium import webdriver

from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
import requests
from selenium.webdriver.common.keys import Keys
import time
import json
import csv

#Make tickers input from csv
#with open('companylist.csv', 'r') as f:
#  reader = csv.reader(f)
#  tickers = list(reader)

#change to exact location of PhantomJS.exe
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("headless")
driver = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver",options = chrome_options)

tickers = ["AAPL", "NFLX"]

for ticker in tickers:
    try:
        #ticker = ticker[0]
        print("Processing ticker: " + ticker +"\n")
        Company_Dict = {}
        Company_Dict["Ticker"] = ticker

        r  = requests.get("https://finance.yahoo.com/quote/"+ticker+"/profile?p="+ticker)
        data = r.text
        soup = BeautifulSoup(data, features = "html5lib")
		
        company_name = soup.find(attrs = {"class" : "Fz(m) Mb(10px)"})
        Company_Dict["Name"] = company_name.get_text()
       
        r = requests.get("https://finance.yahoo.com/quote/"+ticker+"/community/")
        data = r.text
        soup = BeautifulSoup(data, features = "html5lib")

        comments = []

        body = soup.find_all(attrs = {'class' : 'Fz(14px)'})
        for b in body:
            line = b.get_text()
            if len(line) > 20:
                comments.append(line)

        if len(comments) < 1:
            print "No Conversations for:" + ticker
        else:
            Company_Dict["Conversations"] = comments
            with open(Company_Dict["Ticker"]+".json", 'w') as data_file:
                json_string = json.dump(Company_Dict, data_file, indent = 2)
        
    except:
        print("Yahoo Does not have ticker: " + ticker)
