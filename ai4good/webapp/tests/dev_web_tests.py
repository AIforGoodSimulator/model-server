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

# assert "Python" in driver.title
#elem = driver.find_element_by_id("react-entry-point")
# elem.clear()
# elem.send_keys("pycon")
# #elem.send_keys(Keys.RETURN)
# assert "No results found." not in driver.page_source
driver.set_window_size(1898, 1133)
driver.find_element(By.CSS_SELECTOR, "#react-select-2--value > .Select-placeholder").click()
driver.find_element(By.CSS_SELECTOR, ".VirtualizedSelectFocusedOption").click()
driver.find_element(By.CSS_SELECTOR, "#react-select-3--value > .Select-placeholder").click()
driver.find_element(By.CSS_SELECTOR, ".VirtualizedSelectFocusedOption").click()
driver.find_element(By.CSS_SELECTOR, ".Select-placeholder").click()
driver.find_element(By.CSS_SELECTOR, ".VirtualizedSelectOption").click()
element = driver.find_element(By.ID, "run_model_button")
actions = ActionChains(driver)
actions.move_to_element(element).perform()
driver.find_element(By.ID, "run_model_button").click()
element = driver.find_element(By.CSS_SELECTOR, "body")
actions = ActionChains(driver)
assert "AI4Good COVID-19 Model Server" in driver.title

#actions.move_to_element(element, 0, 0).perform()
#vars["window_handles"] = driver.window_handles
window_before = driver.window_handles[0]
driver.find_element(By.ID, "model_results_button").click()
window_after = driver.window_handles[1]

#vars["win1142"] = wait_for_window(2000)
#driver.switch_to.window(vars["win1142"])
driver.switch_to.window(window_after)

driver.implicitly_wait(120)

#assert re.search(r'Model Results', driver.page_source)
assert "AI4Good COVID-19 Model Server" in window_before.page_source
print(window_after)

driver.close()


