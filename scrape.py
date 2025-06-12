"""
Discourse Forum Content Collector
This module extracts course-related discussions from the IITM online platform
"""
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from datetime import datetime
import json
import time
import os
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("forum-collector")

# Platform configuration
PLATFORM_ROOT = "https://discourse.onlinedegree.iitm.ac.in"
AUTHENTICATION_URL = f"{PLATFORM_ROOT}/login"
COURSE_CONTENT_URL = f"{PLATFORM_ROOT}/c/courses/tds-kb/34"

# Content date filtering parameters
COLLECTION_START = datetime(2025, 1, 1)
COLLECTION_END = datetime(2025, 4, 14)

def normalize_timestamp(timestamp_text: str) -> Optional[datetime]:
    """Convert text date to datetime object"""
    try:
        return datetime.strptime(timestamp_text, "%b %d, %Y %I:%M %p")
    except ValueError:
        logger.warning(f"Could not parse date: {timestamp_text}")
        return None

def is_date_in_target_range(dt: Optional[datetime]) -> bool:
    """Check if date is within collection period"""
    return dt and COLLECTION_START <= dt <= COLLECTION_END

def collect_forum_content():
    """Primary function to extract and save forum content"""
    collected_posts = []
    
    with sync_playwright() as browser_factory:
        # Initialize persistent browser session
        user_session_dir = "playwright_user_data"
        os.makedirs(user_session_dir, exist_ok=True)
        
        browser_context = browser_factory.chromium.launch_persistent_context(
            user_session_dir, 
            headless=False,
            viewport={"width": 1280, "height": 800}
        )
        
        browser_page = browser_context.pages[0] if browser_context.pages else browser_context.new_page()
        
        # Initialize platform access
        logger.info("🌐 Accessing educational platform...")
        browser_page.goto(PLATFORM_ROOT)
        time.sleep(2)
        
        # Handle authentication if needed
        if AUTHENTICATION_URL in browser_page.url:
            logger.info("🔒 Authentication required - Please log in manually")
            input("✅ Press Enter after completing authentication...")
        
        # Iterate through forum pages
        for page_index in range(1, 10):
            current_page_url = f"{COURSE_CONTENT_URL}?page={page_index}"
            logger.info(f"📋 Processing discussion page {page_index}")
            
            browser_page.goto(current_page_url)
            time.sleep(2)
            
            # Extract topics from page
            page_content = BeautifulSoup(browser_page.content(), "html.parser")
            topic_elements = page_content.select("tr.topic-list-item")
            
            if not topic_elements:
                logger.info(f"🛑 No more topics found on page {page_index}. Collection complete.")
                break
            
            # Process each topic in the current page
            for topic_element in topic_elements:
                # Find topic link
                title_element = topic_element.select_one("a.title.raw-link")
                if not title_element:
                    continue
                    
                topic_path = title_element.get("href")
                if not topic_path or not topic_path.startswith("/t/"):
                    continue
                    
                topic_url = f"{PLATFORM_ROOT}{topic_path}"
                
                # Extract topic creation timestamp
                date_element = topic_element.select_one("td.activity.num.topic-list-data.age")
                creation_date = None
                
                if date_element and date_element.has_attr("title"):
                    date_text = date_element["title"]
                    created_prefix = [line for line in date_text.split("\n") if "Created:" in line]
                    
                    if created_prefix:
                        date_text = created_prefix[0].replace("Created:", "").strip()
                        creation_date = normalize_timestamp(date_text)
                
                # Skip content outside collection period
                if not is_date_in_target_range(creation_date):
                    logger.debug(f"⏭️ Skipping topic from {creation_date} (outside collection period)")
                    continue
                
                # Process individual topic
                logger.info(f"📄 Retrieving discussion: {topic_url}")
                browser_page.goto(topic_url)
                
                # Ensure dynamic content loads by scrolling
                for _ in range(3):
                    browser_page.evaluate("window.scrollBy(0, window.innerHeight)")
                    time.sleep(1.5)
                
                # Extract posts from topic
                topic_content = BeautifulSoup(browser_page.content(), "html.parser")
                post_elements = topic_content.select("div.topic-post")
                
                logger.info(f"   Found {len(post_elements)} contributions")
                
                # Process each post in topic
                for post_element in post_elements:
                    # Ensure dynamic date elements are loaded
                    try:
                        browser_page.wait_for_selector("span.relative-date", timeout=3000)
                    except:
                        logger.warning("   ⚠️ Timeout waiting for date elements")
                    
                    post_date_element = post_element.select_one("span.relative-date")
                    post_date = None
                    
                    if post_date_element and post_date_element.has_attr("title"):
                        post_date = post_date_element.get("title")
                    
                    # Skip posts outside collection period
                    if not post_date or not is_date_in_target_range(normalize_timestamp(post_date)):
                        continue
                    
                    # Extract post metadata and content
                    contributor = post_element.get("data-user-card") or "anonymous"
                    content_container = post_element.select_one(".cooked")
                    content = content_container.get_text(separator="\n", strip=True) if content_container else ""
                    
                    if content:
                        collected_posts.append({
                            "topic_url": topic_url,
                            "author": contributor,
                            "created_at": post_date,
                            "content": content
                        })
        
        # Save collected content
        os.makedirs("data", exist_ok=True)
        output_file = "data/discourse_forum_posts.json"
        
        with open(output_file, "w", encoding="utf-8") as json_file:
            json.dump(collected_posts, json_file, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Collection complete. Saved {len(collected_posts)} posts to {output_file}")
        
        # Allow manual inspection before closing
        input("Press Enter to close browser and complete the process...")
        browser_context.close()

if __name__ == "__main__":
    collect_forum_content()