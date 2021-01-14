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

home_url = "http://ai4good-dev.azurewebsites.net/sim/run_model"
run_model_url = home_url + "sim/run_model"
validate_model_url = home_url + "sim/validate_model"
driver.get(home_url)

# test code
driver.set_window_size(1920, 1029)
assert driver.title in ("AI4Good COVID-19 Model Server authentication", "Updating...")
driver.find_element(By.ID, "login-email").click()
driver.find_element(By.ID, "login-email").send_keys("test@test.com")
driver.find_element(By.ID, "login-password").send_keys("test123")
driver.find_element(By.ID, "login-submit-button").click()
driver.execute_script("window.scrollTo(0,0)")
driver.find_element(By.ID, "landing-button").click()
driver.execute_script("window.scrollTo(0,0)")
driver.get(run_model_url)
assert driver.title in ("AI4Good COVID-19 Model Server", "Updating...")
driver.set_window_size(1898, 1133)
driver.find_element(By.CSS_SELECTOR, "#react-select-2--value > .Select-placeholder").click()
driver.find_element(By.CSS_SELECTOR, ".VirtualizedSelectFocusedOption").click()
driver.find_element(By.CSS_SELECTOR, "#react-select-3--value > .Select-placeholder").click()
driver.find_element(By.CSS_SELECTOR, ".VirtualizedSelectFocusedOption").click()
driver.find_element(By.CSS_SELECTOR, ".Select-placeholder").click()
driver.find_element(By.CSS_SELECTOR, ".VirtualizedSelectOption").click()
driver.find_element(By.ID, "run_model_button").click()
driver.execute_script("window.scrollTo(0,0)")
driver.get(validate_model_url)
driver.execute_script("window.scrollTo(0,0)")
assert driver.title in ("AI4Good COVID-19 Model Server", "Updating...")
driver.close()
