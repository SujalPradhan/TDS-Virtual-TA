from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from datetime import datetime
import json
import time
import os

BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
LOGIN_URL = f"{BASE_URL}/login"
CATEGORY_URL = f"{BASE_URL}/c/courses/tds-kb/34"
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 4, 14)

def parse_created_at(date_str):
    try:
        return datetime.strptime(date_str, "%b %d, %Y %I:%M %p")
    except ValueError:
        return None

def is_within_range(dt):
    return dt and START_DATE <= dt <= END_DATE

def scrape_forum():
    with sync_playwright() as p:
        user_data_dir = "playwright_user_data"
        os.makedirs(user_data_dir, exist_ok=True)
        context = p.chromium.launch_persistent_context(user_data_dir, headless=False)
        page = context.pages[0] if context.pages else context.new_page()

        # Go to homepage
        page.goto(BASE_URL)
        time.sleep(3)

        # Pause for manual login
        if LOGIN_URL in page.url:
            print("🔐 Please log in manually in the browser window.")
            input("✅ After logging in and seeing the homepage, press Enter here...")

        result = []

        for page_num in range(1, 10):  # Adjust upper range if needed
            paged_url = f"{CATEGORY_URL}?page={page_num}"
            print(f"\n📄 Scraping category page: {paged_url}")
            page.goto(paged_url)
            time.sleep(3)

            soup = BeautifulSoup(page.content(), "html.parser")
            topics = soup.select("tr.topic-list-item")
            if not topics:
                print(f"❌ No topics found on page {page_num}. Stopping.")
                break

            for topic_row in topics:
                # Extract topic URL
                title_tag = topic_row.select_one("a.title.raw-link")
                if not title_tag:
                    continue
                href = title_tag.get("href")
                if not href or not href.startswith("/t/"):
                    continue
                topic_url = BASE_URL + href

                # Extract created date from the <td> with title attribute
                age_td = topic_row.select_one("td.activity.num.topic-list-data.age")
                created_dt = None
                if age_td and age_td.has_attr("title"):
                    # title looks like: "Created: May 23, 2025 3:06 am\nLatest: May 27, 2025 1:14 pm"
                    title_attr = age_td["title"]
                    # Extract the "Created: ..." part
                    created_line = [line for line in title_attr.split("\n") if line.startswith("Created:")]
                    if created_line:
                        created_str = created_line[0].replace("Created:", "").strip()
                        created_dt = parse_created_at(created_str)

                if not is_within_range(created_dt):
                    print(f"Skipping topic {topic_url} created at {created_dt} (outside range)")
                    continue

                print(f"🧵 Fetching topic: {topic_url} (created at {created_dt})")
                page.goto(topic_url)
                time.sleep(2)

                # Scroll to load all posts
                for _ in range(5):
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    time.sleep(2)

                topic_html = page.content()
                topic_soup = BeautifulSoup(topic_html, "html.parser")
                posts = topic_soup.select("div.topic-post")

                print(f"   💬 Found {len(posts)} posts")
                for post in posts:
                    # Wait for relative-date spans to load in JS-rendered DOM
                    page.wait_for_selector("span.relative-date", timeout=5000)

                    created_at = None
                    created_at_tag = post.select_one("span.relative-date")

                    if created_at_tag:
                        created_at = created_at_tag.get("title")
                        if created_at:
                            print(f"🕒 Post date: {created_at}")
                        else:
                            print("⚠️ Found date tag but no title attribute")
                    else:
                        print("⚠️ No date tag found")

                    if not created_at or not is_within_range(parse_created_at(created_at)):
                        continue

                    author = post.get("data-user-card") or "unknown"
                    content_div = post.select_one(".cooked")
                    content_text = content_div.get_text(separator="\n", strip=True) if content_div else ""
                    print(f"   ✍️ Author: {author}, Content length: {len(content_text)}")

                    result.append({
                        "topic_url": topic_url,
                        "author": author,
                        "created_at": created_at,
                        "content": content_text
                    })

        os.makedirs("data", exist_ok=True)
        with open("data/discourse_forum_posts.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print("\n✅ Discourse scraping complete!")
        
        # Keep browser open until user decides to close it
        print("\n🌐 Browser window remains open for your inspection.")
        input("Press Enter when you're ready to close the browser and finish...")
        context.close()

if __name__ == "__main__":
    scrape_forum()