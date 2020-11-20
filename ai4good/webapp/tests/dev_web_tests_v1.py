from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.keys import Keys
import re
import argparse
import sys


# Get command line parameters to pass to Zalenium
parser = argparse.ArgumentParser()
parser.add_argument('--zaluser', default='none')
parser.add_argument('--zalpassword', default='none')
parser.add_argument('--zalhost', default='none')

args = parser.parse_args()


driver = webdriver.Remote(
   command_executor='http://'+args.zaluser+':'+args.zalpassword+'@'+args.zalhost+'/wd/hub',
   desired_capabilities=DesiredCapabilities.CHROME)

driver.get("http://ai4good-dev.azurewebsites.net/sim/run_model")

#% test code
assert driver.title in ("AI4Good COVID-19 Model Server", "Updating...")
driver.set_window_size(1898, 1133)
driver.find_element(By.CSS_SELECTOR, "#react-select-2--value > .Select-placeholder").click()
driver.find_element(By.CSS_SELECTOR, ".VirtualizedSelectFocusedOption").click()
driver.find_element(By.CSS_SELECTOR, "#react-select-3--value > .Select-placeholder").click()
driver.find_element(By.CSS_SELECTOR, ".VirtualizedSelectFocusedOption").click()
driver.find_element(By.CSS_SELECTOR, ".Select-placeholder").click()
driver.find_element(By.CSS_SELECTOR, ".VirtualizedSelectOption").click()
driver.find_element(By.ID, "run_model_button").click()
driver.close()
