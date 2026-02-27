# Save this as test_playwright.py and run it
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("https://www.google.com")
    title = page.title()
    print(f"Page title: {title}")
    browser.close()
