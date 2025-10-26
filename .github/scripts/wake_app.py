import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ==========================================================
# IMPORTANT: REPLACE WITH YOUR STREAMLIT APP URL
# ==========================================================
STREAMLIT_URL = "https://tangerang-house-price-pipeline.streamlit.app/"


def wake_up_app(url):
    """
    Uses Selenium to open a Streamlit app and click the 
    "Yes, get this app back up!" button if it exists.
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=chrome_options)
    
    print(f"Attempting to wake up app at: {url}")
    driver.get(url)

    try:
        # Wait a max of 30 seconds for the button to appear
        button_xpath = "//button[contains(., 'Yes, get this app back up!')]"
        button = WebDriverWait(driver, 30).until(
            EC.element_to_be_clickable((By.XPATH, button_xpath))
        )
        
        # If button is found, click it
        print("App is sleeping. Found wake-up button. Clicking it...")
        button.click()
        
        # Wait a few seconds for the app to load
        time.sleep(15) 
        print("App should be awake now.")
        
    except Exception as e:
        # If button is not found after 30s, app is already awake
        print("App is already awake (wake-up button not found).")

    finally:
        driver.quit()
        print("Browser closed. Job finished.")


if __name__ == "__main__":
    wake_up_app(STREAMLIT_URL)