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

driver.get("http://ai4good-dev.azurewebsites.net/")

# assert "Python" in driver.title
#elem = driver.find_element_by_id("react-entry-point")
# elem.clear()
# elem.send_keys("pycon")
# #elem.send_keys(Keys.RETURN)
# assert "No results found." not in driver.page_source
driver.set_window_size(1920, 1029)
assert "AI4Good COVID-19 Model Server" in driver.title # check for correct page title
driver.find_element(By.ID, "landing-button").click()
driver.find_element(By.ID, "login-email").click()
driver.find_element(By.ID, "login-email").send_keys("test@test.com")
driver.find_element(By.ID, "login-password").send_keys("test123")
driver.find_element(By.ID, "login-button").click()
driver.find_element(By.ID, "name-camp").click()
driver.find_element(By.ID, "name-camp").send_keys("Moira")
driver.find_element(By.ID, "location").click()
driver.find_element(By.ID, "location").send_keys("Some region")
driver.find_element(By.ID, "total-area").click()
driver.find_element(By.ID, "total-area").send_keys("100000")
driver.find_element(By.ID, "page-1-button").click()
driver.find_element(By.CSS_SELECTOR, ".justify-content-center").click()
driver.find_element(By.ID, "accommodation-area-type1").click()
driver.find_element(By.ID, "accommodation-area-type1").send_keys("10000")
driver.find_element(By.ID, "accommodation-no-unit-type1").click()
driver.find_element(By.ID, "accommodation-no-unit-type1").send_keys("20")
driver.find_element(By.ID, "accommodation-no-person-type1").click()
driver.find_element(By.ID, "accommodation-no-person-type1").send_keys("5000")
driver.find_element(By.ID, "page-2-button").click()
driver.execute_script("window.scrollTo(0,0)")
driver.find_element(By.ID, "available-ICU-beds").click()
driver.find_element(By.ID, "available-ICU-beds").send_keys("100")
driver.find_element(By.CSS_SELECTOR, "#remove-high-risk-off-site > .custom-radio:nth-child(3) > .custom-control-label").click()
driver.find_element(By.ID, "isolation-centre-capacity").click()
driver.find_element(By.ID, "isolation-centre-capacity").send_keys("300")
driver.find_element(By.CSS_SELECTOR, "#community-shielding > .custom-radio:nth-child(1) > .custom-control-label").click()
driver.find_element(By.CSS_SELECTOR, "#community-surveillance-program > .custom-radio:nth-child(2) > .custom-control-label").click()
driver.find_element(By.ID, "page-3-button").click()
driver.execute_script("window.scrollTo(0,0)")
driver.find_element(By.CSS_SELECTOR, "#radio-intervene-social > .custom-radio:nth-child(3) > .custom-control-label").click()
driver.find_element(By.CSS_SELECTOR, "#tabs-health-intervent > .nav-item:nth-child(2)").click()
driver.find_element(By.CSS_SELECTOR, "#slider-health-intervent > .rc-slider").click()
driver.find_element(By.ID, "activity-no-place-admin").click()
driver.find_element(By.ID, "activity-no-place-admin").send_keys("5")
driver.find_element(By.ID, "activity-no-person-admin").click()
driver.find_element(By.ID, "activity-no-person-admin").click()
driver.find_element(By.ID, "activity-no-person-admin").send_keys("100")
driver.find_element(By.ID, "activity-no-visit-admin").click()
driver.find_element(By.ID, "activity-no-visit-admin").send_keys("2000")
driver.find_element(By.ID, "page-4-button").click()
driver.execute_script("window.scrollTo(0,0)")
driver.close()
