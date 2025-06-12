from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from tqdm import tqdm
import json
import time
import os

def scrape_tds_site():
    all_content = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("https://tds.s-anand.net/#/")
        time.sleep(5)

        # Expand all folders and files
        page.wait_for_selector("aside.sidebar")
        expandable_items = page.query_selector_all("li.folder.level-1:not(.open), li.file > div:not(.open)")
        for item in expandable_items:
            try:
                item.click()
                time.sleep(0.3)
            except:
                continue

        # Collect all hrefs (not ElementHandles)
        link_elements = page.query_selector_all("li.file a")
        link_info = [
            {
                "title": link.inner_text(),
                "href": link.get_attribute("href")
            }
            for link in link_elements
        ]

        print(f"🔗 Found {len(link_info)} total topic links.")
        for i, item in enumerate(tqdm(link_info, desc="Scraping TDS topics")):
            try:
                full_url = f"https://tds.s-anand.net/{item['href']}"
                page.goto(full_url)
                page.wait_for_selector(".content", timeout=10000)  # wait max 10s

                # Get the actual text content via Playwright (not BeautifulSoup)
                text = page.locator(".content").inner_text()

                # Optional: print preview of content
                print(f"\n📘 {item['title']}:\n{text[:200]}...\n")

                all_content.append({
                    "title": item["title"],
                    "content": text.strip(),
                    "url": full_url
                })

            except Exception as e:
                print(f"⚠️ Error at topic {i+1} ({item['title']}): {e}")

        browser.close()

    os.makedirs("data", exist_ok=True)
    with open("data/tds_content.json", "w", encoding="utf-8") as f:
        json.dump(all_content, f, ensure_ascii=False, indent=2)

    print("✅ TDS scraping complete! Saved to data/tds_content.json")

if __name__ == "__main__":
    scrape_tds_site()